'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import heapq
from torch.utils.data import DataLoader, RandomSampler
import torch

from pathlm.datasets.data_utils import get_user_negatives
from pathlm.datasets.kgat_dataset import KGATStyleDataset
from pathlm.evaluation.eval_metrics import evaluate_rec_quality

from dataloader_kgat import KGATLoader
from pathlm.models.knowledge_aware.KGAT.parser import parse_args
from utils import *

def ranklist_by_heapq(test_items, rating, Ks):
    item_score = {i: rating[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    return K_max_item_score

def ranklist_by_sorted(test_items, model_scores, Ks):
    item_score = {i: model_scores[i] for i in test_items}
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    return K_max_item_score

def test(args, model, users_to_test, data_generator):
    Ks = eval(args.Ks)
    u_batch_size = args.test_batch_size * 2
    n_test_users = len(users_to_test)
    n_user_batches = n_test_users // u_batch_size + 1
    topks = {}
    user_negatives = get_user_negatives(args.dataset)
    model.eval()  # Set the model to evaluation mode

    for u_batch_id in range(n_user_batches):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = users_to_test[start: end]
        item_batch = list(range(0, model.n_items+1))
        feed_dict = data_generator['kg_augmented_dataset'].prepare_test_data(user_batch, item_batch)
        # Forward pass through the model
        rate_batch = model(feed_dict, 'eval')  # Assuming model's forward pass returns the required ratings.
        rate_batch = rate_batch.detach().cpu().numpy()  # Convert to numpy for subsequent operations

        for i, user in enumerate(user_batch):
            user_ratings = rate_batch[i]
            candidate_items = user_negatives[user]
            if args.test_flag == 'part':
                pids = ranklist_by_heapq(candidate_items, user_ratings, Ks)
            else:
                pids = ranklist_by_sorted(candidate_items, user_ratings, Ks)
            topks[user] = pids

    avg_metrics_dict = {k: evaluate_rec_quality(args.dataset, topks, data_generator['dataset'].test_user_dict, k)[1] for k in Ks}

    return avg_metrics_dict, topks

if __name__ == '__main__':
    args = parse_args()
    Ks = eval(args.Ks)

    data_generator = {}

    MANUAL_SEED = 2023

    torch.manual_seed(MANUAL_SEED)
    g = torch.Generator(device='cpu')
    g.manual_seed(MANUAL_SEED)

    kgat_a_ds = KGATLoader(args=args, path=DATA_DIR[args.dataset])
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