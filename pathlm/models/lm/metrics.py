import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
NDCG = "ndcg"
MMR = "mmr"
SERENDIPITY = "serendipity"
COVERAGE = "coverage"
DIVERSITY = "diversity"
NOVELTY = "novelty"
CFAIRNESS = "cfairness"
PFAIRNESS = "pfairness"

REC_QUALITY_METRICS_TOPK = [NDCG, MMR, SERENDIPITY, DIVERSITY,
                            NOVELTY, PFAIRNESS]

def precision_at_k(hit_list: List[int], k: int) -> float:
    r = np.asfarray(hit_list)[:k] != 0
    if r.size != 0:
        return np.mean(r)
    return 0.

def recall_at_k(hit_list: List[int], k: int, test_set_len: int) -> float:
    r = np.asfarray(hit_list)[:k] != 0
    if r.size != 0:
        return np.sum(r) / test_set_len
    return 0.

def F1(pre: float, rec: float) -> float:
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

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

"""
Beyond Accuracy
"""

"""
    Catalog coverage https://dl.acm.org/doi/pdf/10.1145/2926720
"""
def coverage(recommended_items, n_items_in_catalog):
    return len(recommended_items) / n_items_in_catalog

def serendipity_at_k(user_topk, most_pop_topk, k):
    user_topk, most_pop_topk = set(user_topk), set(most_pop_topk)
    intersection = user_topk.intersection(most_pop_topk)
    return (k-len(intersection)) / k


def diversity_at_k(topk_items, pid2genre):
    diversity_items_tok = set([pid2genre[pid] for pid in topk_items]) # set of genres
    return len(diversity_items_tok)/len(topk_items)


def novelty_at_k(topk_items, pid2popularity):
    novelty_items_topk = [1 - pid2popularity[pid] for pid in topk_items]
    return np.mean(novelty_items_topk)


def get_dataset_id2eid(dataset_name, what="user"):
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    file = open(os.path.join(data_dir, f"mapping/{what}.txt"), "r")
    csv_reader = csv.reader(file, delimiter='\t')
    dataset_pid2eid = {}
    next(csv_reader, None)
    for row in csv_reader:
        dataset_pid2eid[row[1]] = row[0]
    file.close()
    return dataset_pid2eid

def get_result_dir(dataset_name):
    return os.path.join('results', dataset_name)

def get_item_genre(dataset_name):
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    dataset_id2model_kg_id = get_dataset_id2eid(dataset_name, "product")
    dataset_id2model_kg_id = dict(zip([int(x) for x in dataset_id2model_kg_id.keys()], dataset_id2model_kg_id.values()))
    item_genre_df = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    item_genre_df.pid = item_genre_df.pid.map(dataset_id2model_kg_id)
    return dict(zip(item_genre_df.pid, item_genre_df.genre))

def get_mostpop_topk(dataset_name, model_name, k):
    most_pop_dir = os.path.join('results', dataset_name, 'most_pop')
    try:
        with open(os.path.join(most_pop_dir, "item_topks.pkl"), 'rb') as f:
            most_pop_topks = pickle.load(f)
        f.close()
    except FileNotFoundError:
        raise FileNotFoundError("Run most_pop.py first to generate most_pop_topk.pkl")

    dataset_pid2model_kg_id = get_dataset_id2eid(dataset_name, "product")
    most_pop_topks = {uid: [dataset_pid2model_kg_id[pid] for pid in topk[:k]]
                      for uid, topk in most_pop_topks.items()}
    return most_pop_topks

def get_item_count(dataset_name):
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    df_items = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    return df_items.pid.unique().shape[0]

def get_item_pop(dataset_name):
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    dataset_id2model_kg_id = get_dataset_id2eid(dataset_name, "product")
    dataset_id2model_kg_id = dict(zip([int(x) for x in dataset_id2model_kg_id.keys()], dataset_id2model_kg_id.values()))
    df_items = pd.read_csv(os.path.join(data_dir, "products.txt"), sep="\t")
    df_items.pid = df_items.pid.map(dataset_id2model_kg_id)
    return dict(zip(df_items.pid, df_items.pop_item))