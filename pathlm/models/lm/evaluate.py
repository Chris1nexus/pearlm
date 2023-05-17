import argparse
import csv
import random
from typing import List, Dict
from tqdm import tqdm

import numpy as np
from transformers import AutoTokenizer, set_seed, pipeline, PreTrainedTokenizerFast, PhrasalConstraint

from pathlm.models.lm.generation_constraints import ForceLastTokenLogitsProcessorWordLevel, \
    ForceTokenAtWordPositionLogitsProcessorBPE
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
from pathlm.utils import get_pid_to_eid, get_eid_to_name_map, get_data_dir, get_set

from transformers import LogitsProcessorList

def generate_topks_withWordLevel(model, uids: List[str], args: argparse.Namespace):
    """
    Recommendation and explanation generation
    """
    dataset_name = args.data
    model_name = args.model
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"./tokenizers/{args.data}/WordLevel.json", max_len=256,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    user_negatives = get_user_negatives_tokens_ids(dataset_name, tokenizer)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(args.seed)
    topk = {}
    metrics = {"ndcg": [], "mmr": []}
    for uid in tqdm(uids, desc="Generating topks", colour="green"):
        # Define the logits processor
        logits_processor = LogitsProcessorList([
            ForceLastTokenLogitsProcessorWordLevel(user_negatives[uid], total_length=6)  # 7 = 2 input token + 5 generated tokens
        ])
        uid = str(int(uid) + 1)  # user_id starts from 1 in the augmented graph starts from 0
        outputs = generator(f"U{uid} watched",
                            max_length=7,  # 7 = 2 input token + 5 generated tokens
                            num_return_sequences=10,
                            logits_processor=logits_processor)
        # Convert tokens to entity names
        topk[uid] = []
        for output in outputs:
            output = output['generated_text'].split(" ")
            recommended_item = output[-1][1:]
            topk[uid].append(recommended_item)

    return topk

def generate_topks_withBPE(model, uids: List[str], args: argparse.Namespace):
    """
    Recommendation and explanation generation
    """
    dataset_name = args.data
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    eid2name = get_eid_to_name_map(f"data/{dataset_name}")
    name2id = {v: k for k, v in eid2name.items()}

    user_negatives = get_user_negatives_tokens_ids(dataset_name, tokenizer)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(args.seed)
    topks = {}
    for uid in tqdm(uids, desc="Generating topks", colour="green"):
        # Define the logits processor
        logits_processor = LogitsProcessorList([
            ForceTokenAtWordPositionLogitsProcessorBPE(tokenizer, user_negatives[uid], word_position=6)  # 7 = 2 input token + 5 generated tokens
        ])
        uid = str(int(uid) + 1)  # user_id starts from 1 in the augmented graph starts from 0
        outputs = generator(f"U{uid} <word_end> watched",
                            max_length=70,  # word != token so we need to increase the max_length
                            num_return_sequences=20,
                            logits_processor=logits_processor)
        # Convert tokens to entity names
        topks[uid] = []
        for output in outputs:
            if len(topks[uid]) == 10:
                continue
            output = output['generated_text'].split("<word_end>")[:7] #Hop 3
            recommended_item = output[-1].strip()
            try:
                id = name2id[recommended_item]
            except:
                continue
            topks[uid].append(id)

    return topks

def evaluate(model, args: argparse.Namespace):
    """
    Recommendation evaluation
    """
    random_baseline(args)
    dataset_name = args.data
    custom_model_name = model.name_or_path.split("/")[-1]
    test_set = get_set(dataset_name, set_str='test')

    # Generate paths for the test users
    # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok
    if custom_model_name.startswith('ft'):
        topks = generate_topks_withBPE(model, list(test_set.keys()), args)
    else:
        topks = generate_topks_withWordLevel(model, list(test_set.keys()), args)

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
