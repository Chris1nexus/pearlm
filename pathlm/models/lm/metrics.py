import csv
import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from pathlm.models.lm.lm_utils import get_user_positives
from collections import Counter


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

def get_mostpop_topk(dataset_name, k):
    most_pop_dir = os.path.join('resources', dataset_name, 'most_pop')
    os.makedirs(most_pop_dir, exist_ok=True)
    mostpop_topk_path = os.path.join(most_pop_dir, f"item_topks_{k}.pkl" )
    if os.path.exists(mostpop_topk_path):
        with open(mostpop_topk_path, 'rb') as f:
            most_pop_topks = pickle.load(f)
        f.close()
    else:
        most_pop_topks = compute_mostpop_topk(dataset_name, k)
        
        with open(mostpop_topk_path, 'wb') as f:
            pickle.dump(most_pop_topks, f)
        f.close()        


    #dataset_pid2model_kg_id = get_dataset_id2eid(dataset_name, "product")
    most_pop_topks = {uid: [pid for pid in topk[:k]]
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


def compute_mostpop_topk(dataset_name,  k=10):
        #def mostpop_topk(train_user_dict, positive_item_user_dict,  k=10):
        '''
        path = os.path.join('./data', dataset_name, 'preprocessed'  )
        uid_inter_test = os.path.join(path,  f'test.txt')
        uid_inter_valid = os.path.join(path,  f'valid.txt')
        uid_inter_train = os.path.join(path,  f'train.txt')
        item_list_file = os.path.join(path, 'i2kg_map.txt')
        kg_filepath = os.path.join(path,  f'kg_final.txt')                                      
        pid_mapping_filepath = os.path.join(path,  f'i2kg_map.txt')
        rel_mapping_filepath = os.path.join(path,  f'r_map.txt')
        rel_df = pd.read_csv(rel_mapping_filepath, sep='\t')
        pid_df = pd.read_csv(pid_mapping_filepath, sep='\t')
        
        pid2eid = { pid : eid for pid,eid in zip(pid_df.pid.values.tolist(), pid_df.eid.values.tolist())  }
        rel_id2type = { int(i) : rel_name for i,rel_name in zip(rel_df.id.values.tolist(), rel_df.name.values.tolist())  } 
        rel_id2type[int(LiteralPath.interaction_rel_id)] = INTERACTION[dataset_name]
        
        rel_type2id = { v:k for k,v in rel_id2type.items() }

        train_user_dict = get_set(dataset_name, set_str='train')
        valid_user_dict = get_set(dataset_name, set_str='valid')
        test_user_dict = get_set(dataset_name, set_str='test')
        user_negatives = get_user_negatives(dataset_name)
        '''
        positive_item_user_dict = get_user_positives(dataset_name)
        interacted_items = []
        for uid in positive_item_user_dict:
            interacted_items.extend(positive_item_user_dict[uid])
        item_frequency = sorted(Counter(interacted_items).items(), key=lambda x: x[1], reverse=True)
        topks = dict()
        for uid in positive_item_user_dict:
            topks[uid] = []
            positive_items = set(positive_item_user_dict[uid])
            topk_items = set()
            for pid, freq in item_frequency:
                if pid in positive_items:
                    continue
                if pid in topk_items:
                    continue
                topks[uid].append(pid) 
                topk_items.add(pid)
                if len(topks[uid]) >= k:
                    break
        return topks
