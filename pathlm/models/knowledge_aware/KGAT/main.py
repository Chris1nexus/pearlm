'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import math
import multiprocessing
import pickle
from itertools import cycle

import numpy as np
from torch.utils.data import default_collate

from pathlm.utils import SEED
from tqdm import tqdm

from pathlm.models.traditional.traditional_utils import early_stopping
from pathlm.models.wadb_utils import MetricsLogger
from batch_test import *
from time import time
from KGAT import KGAT

import sys
from utils import *

def move_to_gpu(batch):
    return {key: value.to('cuda') for key, value in default_collate(batch).items()}


if __name__ == '__main__':
    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else (( torch.device('mps') if hasattr(torch.backends,'mps') and torch.backends.mps.is_available()\
        else 'cpu')  )
    args.device = device
    torch.multiprocessing.set_start_method('spawn')

    os.makedirs(TMP_DIR[args.dataset], exist_ok=True)
   
    """
    *********************************************************
    Load Data from data_generator function.
    """
    train_cores = multiprocessing.cpu_count()
    test_cores = multiprocessing.cpu_count() // 2
    data_generator = {}
    g = torch.Generator(device='cpu').manual_seed(SEED)

    data_generator['kg_augmented_dataset'] = KGATLoader(args=args, path=DATA_DIR[args.dataset])
    data_generator['dataset'] = KGATStyleDataset(args=args, path=DATA_DIR[args.dataset])
    data_generator['kg_augmented_dataloader'] = DataLoader(data_generator['kg_augmented_dataset'],
                                                           batch_size=data_generator[
                                                               'kg_augmented_dataset'].batch_size_kg,
                                                           sampler=RandomSampler(data_generator['kg_augmented_dataset'],
                                                                                 replacement=True,
                                                                                 generator=g) if args.with_replacement else None,
                                                           shuffle=False if args.with_replacement else True,
                                                           num_workers=0,
                                                           drop_last=True,
                                                           persistent_workers=False,
                                                              collate_fn=move_to_gpu
                                                           )
    data_generator['loader'] = DataLoader(data_generator['dataset'],
                                          batch_size=data_generator['dataset'].batch_size,
                                          sampler=RandomSampler(data_generator['dataset'],
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

    key =  'kg_augmented_dataset'
    def sum_sparse_tensors(tensor_list):
        # Initialize with the first tensor
        result = tensor_list[0].clone()
        # Iterate over the rest of the tensors and add them
        for tensor in tensor_list[1:]:
            result = result + tensor
        return result

    def sum_sparse_tensors_(sparse_tensor_list):
        stacked_tensors = torch.stack(sparse_tensor_list, dim=0)
        summed_tensor = torch.sparse.sum(stacked_tensors, dim=0)
        return summed_tensor

    config['A_in'] = sum_sparse_tensors_(data_generator[key].lap_list)
    config['all_h_list'] = data_generator[key].all_h_list
    config['all_r_list'] = data_generator[key].all_r_list
    config['all_t_list'] = data_generator[key].all_t_list
    config['all_v_list'] = data_generator[key].all_v_list
    config['n_relations'] = data_generator[key].n_relations

    model = KGAT(data_config=config, pretrain_data=None, args=args).to(device)
    # Use PyTorch's save and load API
    weights_save_path = os.path.join("weights")

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

        loader_cycle = cycle(data_generator['loader'])
        for idx in range(n_batch):
            batch_data = next(loader_cycle)
            batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train_step(batch_data, mode='rec')
            loss += batch_loss.item()
            base_loss += batch_base_loss.item()
            kge_loss += batch_kge_loss.item()
            reg_loss += batch_reg_loss.item()
#
        if math.isnan(loss):
            print('ERROR: loss@phase1 is nan.')
            sys.exit()
        # Compute average losses over all batches
        avg_loss = loss / n_batch
        avg_base_loss = base_loss / n_batch
        avg_kge_loss = kge_loss / n_batch
        avg_reg_loss = reg_loss / n_batch
        print(f"Epoch {epoch} [{time() - t1:.1f}s]: train==[Cumulative loss: {avg_loss:.5f}= base loss: {avg_base_loss:.5f} + kge_loss: {avg_kge_loss:.5f} + reg_loss: {avg_reg_loss:.5f}]")


        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        n_A_batch = args.batch_size_kg
        if args.use_kge is True:
            print('Use KGE method (knowledge graph embedding).')
            # using KGE method (knowledge graph embedding).
            train_start = time()
            loader_A_cycle = cycle(data_generator['kg_augmented_dataloader'])
            for idx in range(n_A_batch):
                A_batch_data = next(loader_A_cycle)
                batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train_step(A_batch_data, mode='kge')

                loss += batch_loss.item()
                kge_loss += batch_kge_loss.item()
                reg_loss += batch_reg_loss.item()

            train_time += time() - train_start

        if args.use_att is True:
            print("Updating attentive laplacian matrix.")
            # updating attentive laplacian matrix.
            model(None, mode='update_att')

        if math.isnan(loss):
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        show_step = 1
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = f'Epoch {epoch} [{time() - t1:.1f}s]: train==[Cumulative loss: {loss:.5f}= base loss: {base_loss:.5f} + kge_loss: {kge_loss:.5f} + reg_loss: {reg_loss:.5f}]'
                print(perf_str)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator['dataset'].test_user_dict.keys())

        ret, top_k = test(args, model, users_to_test, data_generator)
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
        #metrics.log('train_loss', loss)
        #metrics.log('train_base_loss', base_loss)
        #metrics.log('train_kge_loss', kge_loss)
        #metrics.log('train_reg_loss', reg_loss)
        #metrics.log('valid_ndcg', ret['ndcg'])
        #metrics.log('valid_hit', ret['hit_ratio'])
        #metrics.log('valid_recall', ret['recall'])
        #metrics.log('valid_precision', ret['precision'])
        #metrics.push(['train_loss', 'train_base_loss', 'train_kge_loss', 'train_reg_loss',
        #              'valid_ndcg', 'valid_hit', 'valid_recall', 'valid_precision'])
        Ks = eval(args.Ks)
        loss_loger.append(loss)
        rec_loger.append(ret[Ks[0]]['recall'])
        pre_loger.append(ret[Ks[0]]['precision'])
        ndcg_loger.append(ret[Ks[0]]['ndcg'])
        #hit_loger.append(ret['hit_ratio'])
        metrics_logs = {}

        for idx, k in enumerate(Ks):
            for metric_name, metric_value in ret[k].items():
                if metric_name != 'auc':
                    metrics_logs[f'{metric_name}@{k}'] = metric_value
        test_time += t3 - t2
        metrics_logs['test_time'] = test_time

        if args.verbose > 0:
            ndcg = ret[k]['ndcg']
            perf_str = (f'Epoch {epoch} [{t2 - t1:.1f}s + {t3 - t2:.1f}s]: train==[{loss:.5f}={base_loss:.5f} + '
                        f'{kge_loss:.5f} + {reg_loss:.5f} + {ndcg:.2f}')
            print(perf_str)

        ##cur_best_pre_0, stopping_step, should_stop = early_stopping(ret[Ks[0]]['ndcg'], cur_best_pre_0,
        ##                                                            stopping_step, expected_order='acc',
         #                                                           flag_step=1000)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        #torch.save(model.state_dict(), os.path.join(weights_save_path, f'weights_epoch_{epoch}.pth'))
        #print(f'save the weights in path: {weights_save_path}')

        #metrics.write(TEST_METRICS_FILE_PATH[args.dataset])

        #metrics.close_wandb()
