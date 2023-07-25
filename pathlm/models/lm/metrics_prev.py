from typing import List

import numpy as np


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
def coverage(recommended_items_by_group, n_items_in_catalog):
    group_metric_value = {}
    for group, item_set in recommended_items_by_group.items():
        group_metric_value[group] = len(item_set) / n_items_in_catalog
    return group_metric_value


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