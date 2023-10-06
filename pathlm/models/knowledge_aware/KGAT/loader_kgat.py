'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import random

import numpy as np
import torch
from time import time
import scipy.sparse as sp
import collections

from pathlm.datasets.kgat_dataset import KGATStyleDataset


class KGAT_loader(KGATStyleDataset):
    def __init__(self, args, path, batch_style='list'):
        super().__init__(args, path, batch_style)

        # Generate the sparse adjacency matrices for user-item interaction & relational kg data.
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()

        # Generate the sparse laplacian matrices.
        self.lap_list = self._get_relational_lap_list()

        # Generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()
        self.exist_heads = list(self.all_kg_dict.keys())
        self.N_exist_heads = len(self.exist_heads)

        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        n_all = self.n_users + self.n_entities

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.extend([R, R_inv])
        adj_r_list.extend([0, self.n_relations + 1])

        for r_id, data in self.relation_dict.items():
            K, K_inv = _np_mat2sp_adj(np.array(data), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.extend([K, K_inv])
            adj_r_list.extend([r_id + 1, r_id + 2 + self.n_relations])

        print(f'\tConverted {len(adj_mat_list)} relational triples into adj mat in {time()-t1:.4f}s')
        self.n_relations = len(adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        if self.args.adj_type == 'bi':
            lap_list = [self._bi_norm_lap(adj) for adj in self.adj_list]
            print('\tGenerated bi-normalized adjacency matrix.')
        else:
            lap_list = [self._si_norm_lap(adj) for adj in self.adj_list]
            print('\tGenerated si-normalized adjacency matrix.')
        return lap_list

    def _bi_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):
            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        all_h_list, all_t_list, all_r_list, all_v_list = [], [], [], []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list.extend(lap.row)
            all_t_list.extend(lap.col)
            all_v_list.extend(lap.data)
            all_r_list.extend([self.adj_r_list[l_id]] * len(lap.row))

        assert len(all_h_list) == sum(len(lap.data) for lap in self.lap_list)

        # Organize data by head entity
        org_h_dict = collections.defaultdict(lambda: [[], [], []])
        for idx, h in enumerate(all_h_list):
            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])

        # Sort data by tail entity for each head
        sorted_h_dict = {}
        for h, (t_list, r_list, v_list) in org_h_dict.items():
            sorted_indices = np.argsort(t_list)
            sorted_h_dict[h] = [np.array(t_list)[sorted_indices],
                                np.array(r_list)[sorted_indices],
                                np.array(v_list)[sorted_indices]]

        # Flatten sorted data
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []
        for h, (sorted_t_list, sorted_r_list, sorted_v_list) in sorted(sorted_h_dict.items()):
            new_h_list.extend([h] * len(sorted_t_list))
            new_t_list.extend(sorted_t_list)
            new_r_list.extend(sorted_r_list)
            new_v_list.extend(sorted_v_list)

        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)

        return new_h_list, new_r_list, new_t_list, new_v_list



    def __len__(self):
        # number of existing users after the preprocessing described in the paper,
        # determines the length of the training dataset, for which a positive an negative are extracted
        return self.N_exist_heads

    def __getitem__(self, idx):
        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            sampled_triples = random.sample(pos_triples, num)
            pos_rs, pos_ts = zip(*sampled_triples)
            return list(pos_rs), list(pos_ts)

        def sample_neg_triples_for_h(h, r, num):
            all_possible_tails = set(range(self.n_users + self.n_entities))
            existing_tails = {t for t, rel in self.all_kg_dict[h] if rel == r}
            neg_ts = np.random.choice(list(all_possible_tails - existing_tails), num, replace=False)
            return list(neg_ts)

        h = self.exist_heads[idx]
        pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
        neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)

        if self.batch_style_id == 0:
            return h, pos_rs[0], pos_ts[0], neg_ts[0]
        else:
            return {'heads': h, 'relations': pos_rs[0], 'pos_tails': pos_ts[0], 'neg_tails': neg_ts[0]}

    def as_test_feed_dict(self, model, user_batch, item_batch):
        return {
            model.users: torch.tensor(user_batch),
            model.pos_items: torch.tensor(item_batch),
            model.mess_dropout: torch.zeros(len(eval(self.args.layer_size))),
            model.node_dropout: torch.zeros(len(eval(self.args.layer_size))),
        }

    def as_train_feed_dict(self, model, batch_data):
        if self.batch_style_id == 0:
            users, pos_items, neg_items = batch_data
            batch_data = {'users': users, 'pos_items': pos_items, 'neg_items': neg_items}

        return {
            model.users: torch.tensor(batch_data['users']),
            model.pos_items: torch.tensor(batch_data['pos_items']),
            model.neg_items: torch.tensor(batch_data['neg_items']),
            model.mess_dropout: torch.tensor(eval(self.args.mess_dropout)),
            model.node_dropout: torch.tensor(eval(self.args.node_dropout)),
        }

    def as_train_A_feed_dict(self, model, batch_data):
        if self.batch_style_id == 0:
            heads, relations, pos_tails, neg_tails = batch_data
            batch_data = {'heads': heads, 'relations': relations, 'pos_tails': pos_tails, 'neg_tails': neg_tails}

        return {
            model.h: torch.tensor(batch_data['heads']),
            model.r: torch.tensor(batch_data['relations']),
            model.pos_t: torch.tensor(batch_data['pos_tails']),
            model.neg_t: torch.tensor(batch_data['neg_tails']),
        }