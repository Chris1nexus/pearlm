import argparse
import csv
from collections import defaultdict
from typing import List, Dict

import numpy as np
from transformers import AutoTokenizer, set_seed, pipeline

from pathlm.utils import get_pid_to_eid, get_eid_to_name_map


def dcg_at_k(hit_list: List[int], k: int, method: int=1) -> float:
    r = np.asfarray(hit_list)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(hit_list: List[int], k: int, method=0) -> float:
    dcg_max = dcg_at_k(sorted(hit_list, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(hit_list, k, method) / dcg_max

def mmr_at_k(hit_list: List[int], k: int) -> float:
    r = np.asfarray(hit_list)[:k]
    hit_idxs = np.nonzero(r)
    if len(hit_idxs[0]) > 0:
        return 1 / (hit_idxs[0][0] + 1)
    return 0.

def get_set(dataset_name: str, set: str='test') -> Dict[str, list[int]]:
    data_dir = f"data/{dataset_name}"
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    i2kg = get_pid_to_eid(data_dir)

    # Generate paths for the test set
    test_set = defaultdict(list)
    with open(f"{data_dir}/preprocessed/{set}.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            user_id = str(int(user_id) - 1)  # user_id starts from 1 in the augmented graph starts from 0
            item_id = i2kg[item_id]  # Converting dataset id to eid
            test_set[user_id].append(item_id)
    f.close()
    return test_set


def get_entity_vocab(dataset_name: str, model_name: str) -> List[int]:
    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    entity_list = get_eid_to_name_map(dataset_name).values()
    return fast_tokenizer.convert_tokens_to_ids(entity_list)

def evaluate(model, args: argparse.Namespace):
    """
    Recommendation evaluation
    """
    dataset_name = args.data
    model_name = args.model

    test_set = get_set(dataset_name, set='test')
    entity_vocab = get_entity_vocab(dataset_name)

    # Generate paths for the test users
    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    generator = pipeline('text-generation', model=model, tokenizer=fast_tokenizer)
    set_seed(args.seed)
    topk = {}
    metrics = {}
    for uid in test_set.keys():
        outputs = generator(f"{uid} watched ", num_beams=4, force_words_ids=entity_vocab, num_return_sequences=10)
        # Convert tokens to entity names
        topk[uid], hits = [], []
        for output in outputs:
            # TODO: Convert final entity name (we must ensure it is an item) to entity id
            recommended_item = output[-1]
            topk[uid].append(output[-1])
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
        metrics["ndcg"].append(ndcg)
        metrics["mmr"].append(mmr)

    print(f"no of users: {test_set.keys()}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
