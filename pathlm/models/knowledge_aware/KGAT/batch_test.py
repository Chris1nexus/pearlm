'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import heapq
from torch.utils.data import DataLoader, RandomSampler
import torch

from pathlm.datasets.kgat_dataset import KGATStyleDataset
from pathlm.evaluation.eval_metrics import evaluate_rec_quality

from loader_kgat import KGAT_loader
from pathlm.models.knowledge_aware.KGAT.parser import parse_args
from utils import *

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    return K_max_item_score

def ranklist_by_sorted(test_items, model_scores, Ks, save_topk=True):
    item_score = {i: model_scores[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    return K_max_item_score

def test(model, users_to_test, drop_flag=False, batch_test_flag=False):
    test_users = list(users_to_test.keys())

    if args.model_type in ['ripple']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    topks = {}

    DATASET_KEY = 'A_dataset' if args.model_type == 'cke' else 'dataset'
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        # Assuming you have a method in your model to get the ratings
        rate_batch = model.get_ratings(user_batch, item_batch, drop_flag=drop_flag)
        rate_batch = rate_batch.reshape((-1, len(item_batch)))

        for i, user in enumerate(user_batch):
            user_ratings = rate_batch[i]
            user_pos_test = data_generator['dataset'].test_user_dict[user]
            if args.test_flag == 'part':
                pids = ranklist_by_heapq(user_pos_test, item_batch, user_ratings, Ks)
            else:
                pids = ranklist_by_sorted(user_pos_test, item_batch, user_ratings, Ks)
            topks[user] = pids

    rec_quality_metrics, avg_rec_quality_metrics = evaluate_rec_quality(args.dataset, topks, data_generator['dataset'].test_user_dict)

    return rec_quality_metrics, avg_rec_quality_metrics

if __name__ == '__main__':
    args = parse_args()
    Ks = eval(args.Ks)

    data_generator = {}

    MANUAL_SEED = 2023

    torch.manual_seed(MANUAL_SEED)
    g = torch.Generator(device='cpu')
    g.manual_seed(MANUAL_SEED)

    kgat_a_ds = KGAT_loader(args=args, path=DATA_DIR[args.dataset])
    kgat_ds = KGATStyleDataset(args=args, path=DATA_DIR[args.dataset])
    data_generator['A_dataset'] = kgat_a_ds
    data_generator['dataset'] = kgat_ds
    data_generator['A_loader'] = DataLoader(kgat_a_ds,
                                            batch_size=kgat_a_ds.batch_size_kg,
                                            sampler=RandomSampler(kgat_a_ds,
                                                                  replacement=True,
                                                                  generator=g) if args.with_replacement else None,
                                            shuffle=False if args.with_replacement else True,
                                            drop_last=True,
                                            persistent_workers=True
                                            )
    data_generator['loader'] = DataLoader(kgat_ds,
                                          batch_size=kgat_ds.batch_size,
                                          sampler=RandomSampler(kgat_ds,
                                                                replacement=True,
                                                                generator=g) if args.with_replacement else None,
                                          shuffle=False if args.with_replacement else True,
                                          drop_last=True,
                                          persistent_workers=True
                                          )
    batch_test_flag = False

    USR_NUM, ITEM_NUM = data_generator['dataset'].n_users, data_generator['dataset'].n_items
    N_TRAIN, N_TEST = data_generator['dataset'].n_train, data_generator['dataset'].n_test
    BATCH_SIZE = args.batch_size