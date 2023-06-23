import argparse
import csv
import os
import pickle
import random
from collections import defaultdict
from typing import List, Dict

import torch
from tqdm import tqdm
import multiprocessing as mp
import itertools
import functools
import numpy as np

from transformers import AutoTokenizer, set_seed, pipeline, PreTrainedTokenizerFast, PhrasalConstraint, \
    StoppingCriteriaList, AutoModelForCausalLM
from datasets import Dataset

from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids, get_user_positives, TOKENIZER_DIR
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
from pathlm.utils import get_pid_to_eid, get_eid_to_name_map, get_data_dir, get_set, check_dir,SEED

from transformers import LogitsProcessorList

def evaluate_bert(model, args):
    """
    Recommendation evaluation
    """
    # random_baseline(args)
    dataset_name = args.data
    custom_model_name = model.name_or_path.split("/")[-1]
    test_set = get_set(dataset_name, set_str='test')

    # Generate paths for the test users
    # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok
    if False:
        topks = pickle.load(open(f"./results/{dataset_name}/{custom_model_name}/topks.pkl", "rb"))
    elif custom_model_name.startswith('ft'):
        topks = []
        check_dir(f"./results/{dataset_name}/{custom_model_name}")
        pickle.dump(topks, open(f"./results/{dataset_name}/{custom_model_name}/topks.pkl", "wb"))
    else:
        topks = generate_topks_withWordLevel(model, list(test_set.keys()), args)
        check_dir(f"./results/{dataset_name}/{custom_model_name}")
        pickle.dump(topks, open(f"./results/{dataset_name}/{custom_model_name}/topks.pkl", "wb"))
    metrics = {"ndcg": [], "mmr": []}
    for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
        hits = []
        for recommended_item in topk:
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
        ndcg = ndcg_at_k(hits, len(hits))
        mmr = mmr_at_k(hits, len(hits))
        metrics["ndcg"].append(ndcg)
        metrics["mmr"].append(mmr)

    print(f"no of users: {len(test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")


def generate_topks_withWordLevel(model, uids: List[str], args: argparse.Namespace):
    """
    Recommendation and explanation generation
    """
    # set_seed(SEED)
    dataset_name = args.data
    data_dir = f"data/{dataset_name}"
    tokenizer_dir = f'./tokenizers/{dataset_name}'
    TOKENIZER_TYPE = "WordLevel"

    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    # Load user negatives
    user_positives = get_user_positives(dataset_name)

    fill_masker = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

    #init_condition_fn = lambda uid: tokenizer.encode(f"U{uid} R-1 [MASK] [MASK] [MASK] [MASK] [MASK]")
#
    #sequences = [init_condition_fn(uid) for uid in uids]
    #dataset = Dataset.from_dict({'uid': uids, 'sequence': sequences})
    #output = model(**dataset['sequence'])
    #
    #preds = fill_masker(dataset['sequence'], top_k=30)
    #topks = {}
    #for uid, pred in enumerate(preds):
    #    uid = str(uid + 1)
    #    topks[uid] = [x['token_str'][1:] for x in pred[-1]]
    #    topks[uid] = list(set(topks[uid]) - set(user_positives[uid]))[:10]
    #return topks

    init_condition_fn = lambda uid: f"U{uid} R-1 [MASK] [MASK] [MASK] [MASK] [MASK]"

    sequences = [init_condition_fn(uid) for uid in uids]
    dataset = Dataset.from_dict({'uid': uids, 'sequence': sequences})

    topks = {}
    for uid, sequence in zip(uids, dataset['sequence']):
        # Tokenize the sequence and send the tensors to the same device as your model
        inputs = tokenizer(sequence, return_tensors="pt").to("cuda")

        with torch.no_grad():  # Deactivate gradients for the following code block
            # Get the model's predictions
            outputs = model(**inputs)
            entity_predictions, relation_predictions = outputs.logits

        # We assume the position of the last [MASK] token is -1. Adjust if needed.
        mask_position = -1

        # Select top-k predictions from each head for the last MASK
        top_k_entities = torch.topk(entity_predictions[0, mask_position], 30).indices

        # Convert token IDs to tokens
        top_k_entities = [tokenizer.decode([idx]) for idx in top_k_entities]

        topks[str(uid)] = list(set(top_k_entities) - set(user_positives[str(uid)]))[:10]

    return topks

def random_baseline(args: argparse.Namespace):
    """
    Recommendation evaluation
    """
    dataset_name = args.data
    test_set = get_set(dataset_name, set_str='test')

    def get_user_negatives(dataset_name: str) -> Dict[str, List[str]]:
        data_dir = f"data/{dataset_name}"
        ikg_ids = set(get_pid_to_eid(data_dir).values())
        uid_negatives = {}
        # Generate paths for the test set
        train_set = get_set(dataset_name, set_str='train')
        for uid, items in tqdm(train_set.items(), desc="Calculating user negatives", colour="green"):
            uid_negatives[uid] = list(ikg_ids - set(items))
        return uid_negatives

    user_negatives = get_user_negatives(dataset_name)
    topk = {}
    metrics = {"ndcg": [], "mmr": []}
    for uid in tqdm(list(test_set.keys()), desc="Evaluating", colour="green"):
        topk[uid] = random.sample(user_negatives[uid], 10)
        hits = []
        for recommended_item in topk[uid]:
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
        ndcg = ndcg_at_k(hits, len(hits))
        mmr = mmr_at_k(hits, len(hits))
        metrics["ndcg"].append(ndcg)
        metrics["mmr"].append(mmr)
    print("Random baseline:")
    print(f"no of users: {len(test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")

