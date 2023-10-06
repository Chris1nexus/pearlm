'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import math
import multiprocessing
import pickle

import numpy as np
from tqdm import tqdm

from pathlm.models.wadb_utils import MetricsLogger
from batch_test import *
from time import time
from KGAT import KGAT

import sys
from utils import *




if __name__ == '__main__':
    # Set random seeds
    torch.manual_seed(2023)
    np.random.seed(2023)

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else (( torch.device('mps') if hasattr(torch.backends,'mps') and torch.backends.mps.is_available()\
        else 'cpu')  )
    args.device = device

    os.makedirs(TMP_DIR[args.dataset], exist_ok=True)

    metrics = MetricsLogger(args.wandb_entity if args.wandb else None, 
                            f'{MODEL}_{args.dataset}',
                            config=args)
    metrics.register('train_loss')
    metrics.register('train_base_loss')
    metrics.register('train_reg_loss')
    metrics.register('train_kge_loss')
    metrics.register('ndcg')
    metrics.register('hit')
    metrics.register('recall')     
    metrics.register('precision')
   
    """
    *********************************************************
    Load Data from data_generator function.
    """
    train_cores = multiprocessing.cpu_count()
    test_cores = multiprocessing.cpu_count() // 2
    data_generator = {}
    g = torch.Generator(device='cpu')
    g.manual_seed(2023)

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
                                            num_workers=train_cores,
                                            drop_last=True,
                                            persistent_workers=True
                                            )
    data_generator['loader'] = DataLoader(kgat_ds,
                                          batch_size=kgat_ds.batch_size,
                                          sampler=RandomSampler(kgat_ds,
                                                                replacement=True,
                                                                generator=g) if args.with_replacement else None,
                                          shuffle=False if args.with_replacement else True,
                                          num_workers=train_cores,
                                          drop_last=True,
                                          persistent_workers=True
                                          )
    batch_test_flag = False

    USR_NUM, ITEM_NUM = data_generator['dataset'].n_users, data_generator['dataset'].n_items
    N_TRAIN, N_TEST = data_generator['dataset'].n_train, data_generator['dataset'].n_test
    BATCH_SIZE = args.batch_size

    config = {
        'n_users': data_generator['dataset'].n_users,
        'n_items': data_generator['dataset'].n_items,
        'n_relations': data_generator['dataset'].n_relations,
        'n_entities': data_generator['dataset'].n_entities
    }

    if args.model_type in ['kgat', 'cfkg']:
        key = 'A_dataset' if args.model_type == 'kgat' else 'dataset'
        config['A_in'] = sum(data_generator[key].lap_list)
        config['all_h_list'] = data_generator[key].all_h_list
        config['all_r_list'] = data_generator[key].all_r_list
        config['all_t_list'] = data_generator[key].all_t_list
        config['all_v_list'] = data_generator[key].all_v_list
        config['n_relations'] = data_generator[key].n_relations

    model = KGAT(data_config=config, pretrain_data=None, args=args).to(device)

    # Use PyTorch's save and load API
    weights_save_path = os.path.join(TMP_DIR[args.dataset], "weights")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0.
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    train_time = 0
    for epoch in tqdm(range(args.epoch)):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator['dataset'].n_train // args.batch_size + 1

        loader_iter = iter(data_generator['loader'])
        loader_A_iter = iter(data_generator['A_loader']) if 'A_loader' in data_generator else None

        for idx in range(n_batch):
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(data_generator['loader'])
                batch_data = next(loader_iter)

            users, pos_items, neg_items = data_generator['dataset'].prepare_train_data(batch_data)

            optimizer.zero_grad()
            batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model(users, pos_items, neg_items)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            base_loss += batch_base_loss.item()
            kge_loss += batch_kge_loss.item()
            reg_loss += batch_reg_loss.item()

        if math.isnan(loss):
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        n_A_batch = len(data_generator['A_dataset'].all_h_list) // args.batch_size_kg + 1

        if args.use_kge is True:
            # using KGE method (knowledge graph embedding).
            train_start = time()
            loader_A_iter = iter(data_generator['A_loader'])
            for idx in range(n_A_batch):
                try:
                    A_batch_data = next(loader_A_iter)
                except StopIteration:
                    loader_A_iter = iter(data_generator['A_loader'])
                    A_batch_data = next(loader_A_iter)

                feed_dict = data_generator['A_dataset'].as_train_A_feed_dict(model, A_batch_data)

                optimizer.zero_grad()
                batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(feed_dict)
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()
                kge_loss += batch_kge_loss.item()
                reg_loss += batch_reg_loss.item()

            train_time += time() - train_start

        if args.use_att is True:
            # updating attentive laplacian matrix.
            model.update_attentive_A()

        if math.isnan(loss):
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        show_step = 1
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = f'Epoch {epoch} [{time() - t1:.1f}s]: train==[{loss:.5f}={base_loss:.5f} + {kge_loss:.5f} + {reg_loss:.5f}]'
                print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator['dataset'].test_user_dict.keys())

        ret, top_k = test(users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
        os.makedirs(LOG_DATASET_DIR[args.dataset], exist_ok=True)
        topk_path = f'{LOG_DATASET_DIR[args.dataset]}/item_topk.pkl'
        with open(topk_path, 'wb') as f:
            pickle.dump(top_k, f)
            print(f'Saved topK to: {topk_path}')

        """
        *********************************************************
        Performance logging.
        """
        test_time = 0
        t3 = time()
        metrics.log('train_loss', loss.item())
        metrics.log('train_base_loss', base_loss.item())
        metrics.log('train_kge_loss', kge_loss.item())
        metrics.log('train_reg_loss', reg_loss.item())
        metrics.log('valid_ndcg', ret['ndcg'].item())
        metrics.log('valid_hit', ret['hit_ratio'].item())
        metrics.log('valid_recall', ret['recall'].item())
        metrics.log('valid_precision', ret['precision'].item())
        metrics.push(['train_loss', 'train_base_loss', 'train_kge_loss', 'train_reg_loss',
                      'valid_ndcg', 'valid_hit', 'valid_recall', 'valid_precision'])

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        metrics_logs = {}
        for metric_name, metric_values in ret.items():
            if metric_name != 'auc':
                for idx, k in enumerate(Ks):
                    metrics_logs[f'{metric_name}@{k}'] = metric_values[idx]
        test_time += t3 - t2
        metrics_logs['test_time'] = test_time

        if args.verbose > 0:
            perf_str = f'Epoch {epoch} [{t2 - t1:.1f}s + {t3 - t2:.1f}s]: train==[{loss:.5f}={base_loss:.5f} + {kge_loss:.5f} + {reg_loss:.5f}], recall=[{ret["recall"][0]:.5f}, {ret["recall"][-1]:.5f}], precision=[{ret["precision"][0]:.5f}, {ret["precision"][-1]:.5f}], hit=[{ret["hit_ratio"][0]:.5f}, {ret["hit_ratio"][-1]:.5f}], ndcg=[{ret["ndcg"][0]:.5f}, {ret["ndcg"][-1]:.5f}]'
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=1000)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        torch.save(model.state_dict(), os.path.join(weights_save_path, f'weights_epoch_{epoch}.pth'))
        print(f'save the weights in path: {weights_save_path}')

        metrics.write(TEST_METRICS_FILE_PATH[args.dataset])

        metrics.close_wandb()
