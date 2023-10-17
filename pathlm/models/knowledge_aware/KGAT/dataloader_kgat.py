'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import os
import random
from typing import Tuple, List, DefaultDict, Union, Dict

import numpy as np
import torch
from time import time
import collections

from pathlm.datasets.kgat_dataset import KGATStyleDataset


class KGATLoader(KGATStyleDataset):
    def __init__(self, args, path, batch_style='map'):
        super().__init__(args, path, batch_style)
        self.batch_size_kg = args.batch_size_kg
        # Generate the sparse adjacency matrices for user-item interaction & relational kg data.
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()
        # Generate the sparse laplacian matrices.
        self.lap_list = self._get_relational_lap_list()

        # Generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()
        self.exist_heads = list(self.all_kg_dict.keys())
        self.N_exist_heads = len(self.exist_heads)

        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

    def _get_relational_adj_list(self) -> Tuple[List[torch.sparse.FloatTensor], List[int]]:
        """
        Construct adjacency matrices for ratings and relational triples.

        Returns:
            tuple: A tuple containing two lists - adjacency matrices list and relation list.
        """
        t1 = time()  # Start recording time
        adj_mat_list = []  # List to store adjacency matrices
        adj_r_list = []  # List to store relations

        def _np_mat2sp_adj(np_mat: np.ndarray, row_pre: int, col_pre: int) -> Tuple[
            torch.sparse.FloatTensor, torch.sparse.FloatTensor]:
            """
            Convert a numpy matrix to two PyTorch sparse adjacency tensors.

            Args:
                np_mat (np.ndarray): Input numpy matrix.
                row_pre (int): Row offset.
                col_pre (int): Column offset.

            Returns:
                tuple: Two PyTorch sparse tensors.
            """
            n_all = self.n_users + self.n_entities

            # Single-direction conversion
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = torch.ones(len(a_rows))

            b_rows = a_cols
            b_cols = a_rows
            b_vals = torch.ones(len(b_rows))

            # Construct PyTorch sparse tensors
            indices_a = torch.tensor(np.stack([a_rows, a_cols]), dtype=torch.long)
            indices_b = torch.tensor(np.stack([b_rows, b_cols]), dtype=torch.long)
            a_adj = torch.sparse_coo_tensor(indices=indices_a, values=a_vals, size=(n_all, n_all)).to(self.args.device)
            b_adj = torch.sparse_coo_tensor(indices=indices_b, values=b_vals, size=(n_all, n_all)).to(self.args.device)

            return a_adj, b_adj

        # Convert ratings to adjacency matrices
        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.extend([R, R_inv])
        adj_r_list.extend([0, self.n_relations + 1])
        print('\tconvert ratings into adj mat done.')

        # Convert relational triples to adjacency matrices
        for r_id in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.extend([K, K_inv])
            adj_r_list.extend([r_id + 1, r_id + 2 + self.n_relations])
        print('\tconvert %d relational triples into adj mat done. @%.4fs' % (len(adj_mat_list), time() - t1))

        self.n_relations = len(adj_r_list)  # Update the number of relations
        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self) -> List[torch.sparse_coo_tensor]:
        """
        Generate a list of normalized laplacian matrices based on the type of normalization (bi or si).

        Returns:
            list: List of normalized adjacency matrices.
        """

        def _bi_norm_lap(adj: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
            """
            Compute the bi-normalized laplacian of a sparse matrix.

            Args:
                adj (torch.sparse_coo_tensor): Sparse adjacency matrix.

            Returns:
                torch.sparse_coo_tensor: Bi-normalized laplacian matrix.
            """
            rowsum = torch.sparse.sum(adj, dim=1).to_dense()

            d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

            d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()
            bi_lap = torch.sparse.mm(torch.sparse.mm(adj, d_mat_inv_sqrt).t(), d_mat_inv_sqrt)
            return bi_lap

        def _si_norm_lap(adj: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
            """
            Compute the si-normalized laplacian of a sparse matrix.

            Args:
                adj (torch.sparse_coo_tensor): Sparse adjacency matrix.

            Returns:
                torch.sparse_coo_tensor: Si-normalized laplacian matrix.
            """
            rowsum = torch.sparse.sum(adj, dim=1).to_dense()

            d_inv = torch.pow(rowsum, -1).flatten()
            d_inv[torch.isinf(d_inv)] = 0.

            d_mat_inv = torch.diag(d_inv).to_sparse()
            norm_adj = torch.sparse.mm(d_mat_inv, adj)
            return norm_adj

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self) -> DefaultDict[int, List[Tuple[int, int]]]:
        """
        Create a dictionary where keys are head entities and values are lists of tail entities and relations.

        Returns:
            collections.defaultdict: Dictionary of head entities and their associated tail entities and relations.
        """
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):
            rows = lap.indices()[0].cpu().numpy()
            cols = lap.indices()[1].cpu().numpy()

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self) -> Tuple[List[int], List[int], List[int], List[float]]:
        """
        Organize and sort knowledge graph data based on head entities and tail entities.

        Returns:
            tuple: Lists of head entities, relations, tail entities, and values.
        """
        all_h_list, all_t_list, all_r_list, all_v_list = [], [], [], []

        for l_id, lap in enumerate(self.lap_list):
            rows = lap.indices()[0].cpu().numpy()
            cols = lap.indices()[1].cpu().numpy()
            data = lap.values().cpu().numpy()

            all_h_list.extend(rows)
            all_t_list.extend(cols)
            all_v_list.extend(data)
            all_r_list.extend([self.adj_r_list[l_id]] * len(rows))

        assert len(all_h_list) == sum(len(lap.values()) for lap in self.lap_list)

        org_h_dict = collections.defaultdict(lambda: [[], [], []])
        for idx, h in enumerate(all_h_list):
            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])

        sorted_h_dict = {}
        for h, (t_list, r_list, v_list) in org_h_dict.items():
            sorted_indices = np.argsort(t_list)
            sorted_h_dict[h] = [np.array(t_list)[sorted_indices],
                                np.array(r_list)[sorted_indices],
                                np.array(v_list)[sorted_indices]]

        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []
        for h, (sorted_t_list, sorted_r_list, sorted_v_list) in sorted_h_dict.items():
            new_h_list.extend([h] * len(sorted_t_list))
            new_t_list.extend(sorted_t_list.tolist())
            new_r_list.extend(sorted_r_list.tolist())
            new_v_list.extend(sorted_v_list.tolist())

        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)

        return new_h_list, new_r_list, new_t_list, new_v_list

    def __len__(self) -> int:
        """
        Determine the number of existing users after the preprocessing.
        It defines the length of the training dataset, for which a positive and negative are extracted.

        Returns:
            int: Number of existing users.
        """
        return self.N_exist_heads

    def __getitem__(self, idx: int) -> Union[Tuple[int, int, int, int], Dict[str, int]]:
        """
        Fetch a data sample based on the given index. This data sample includes
        head entities, positive relations, positive tails, and negative tails.

        Args:
            idx (int): Index of the desired data sample.

        Returns:
            Union[Tuple[int, int, int, int], Dict[str, int]]:
            Data sample either in tuple format or dictionary format based on `self.batch_style_id`.
        """
        def sample_pos_triples_for_h(h: int, num: int) -> Tuple[List[int], List[int]]:
            """
            Sample positive triples for the given head entity.

            Args:
                h (int): Head entity.
                num (int): Number of triples to sample.

            Returns:
                tuple: List of relations and list of tails.
            """
            pos_triples = self.all_kg_dict[h]
            sampled_triples = random.sample(pos_triples, num)
            pos_ts, pos_rs = zip(*sampled_triples)
            return list(pos_rs), list(pos_ts)

        def sample_neg_triples_for_h(h: int, r: int, num: int) -> List[int]:
            """
            Sample negative tails for the given head entity and relation.

            Args:
                h (int): Head entity.
                r (int): Relation for which negative tails are sampled.
                num (int): Number of tails to sample.

            Returns:
                list: List of negative tails.
            """
            all_possible_tails = set(range(self.n_users + self.n_entities))
            existing_tails = {t for t, rel in self.all_kg_dict[h] if rel == r}
            neg_ts = np.random.choice(list(all_possible_tails - existing_tails), num, replace=False)
            return list(neg_ts)

        h = self.exist_heads[idx]
        pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
        neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)

        return {
            'heads': h,
            'relations': pos_rs[0],
            'pos_tails': pos_ts[0],
            'neg_tails': neg_ts[0]
        }

    def get_sparsity_split(self):
        split_file = os.path.join(self.path, 'sparsity.split')
        split_uids, split_state = [], []

        try:
            with open(split_file, 'r') as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                else:
                    split_uids.append(list(map(int, line.strip().split(' '))))
            print('Loaded sparsity split.')

        except Exception:
            split_uids, split_state = self.load_or_create_sparsity_split()
            with open(split_file, 'w') as f:
                for state, uids in zip(split_state, split_uids):
                    f.write(state + '\n')
                    f.write(' '.join(map(str, uids)) + '\n')
            print('Created sparsity split.')

        return split_uids, split_state

    def load_or_create_sparsity_split(self):
        """Load the sparsity split from file or create a new one if it doesn't exist."""
        split_file = os.path.join(self.path, 'sparsity.split')
        split_uids, split_states = [], []

        # Try loading from file
        try:
            with open(split_file, 'r') as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_states.append(line.strip())
                else:
                    split_uids.append(list(map(int, line.strip().split(' '))))
            print('Loaded sparsity split.')

        # If loading fails, create a new split
        except Exception:
            split_uids, split_states = self.generate_sparsity_split()
            with open(split_file, 'w') as f:
                for state, uids in zip(split_states, split_uids):
                    f.write(f"{state}\n{' '.join(map(str, uids))}\n")
            print('Created sparsity split.')

        return split_uids, split_states


    def generate_sparsity_split(self):
        """Generate sparsity split based on user data."""
        all_users_to_test = list(self.test_user_dict.keys())
        user_item_counts = {uid: len(self.train_user_dict[uid]) + len(self.test_user_dict[uid])
                            for uid in all_users_to_test}

        # Group users by number of interactions
        grouped_users = {}
        for n_iids, uid in user_item_counts.items():
            grouped_users.setdefault(n_iids, []).append(uid)

        split_uids, split_states, temp_users, total_interactions, fold_count = [], [], [], 0, 4

        for n_iids, users in sorted(grouped_users.items(), key=lambda x: x[0]):
            temp_users.extend(users)
            total_interactions += n_iids * len(users)

            if total_interactions >= 0.25 * (self.n_train + self.n_test) or fold_count == 0:
                split_uids.append(temp_users)
                state_info = f"#interactions/user<={n_iids}, #users={len(temp_users)}, #total interactions={total_interactions}"
                split_states.append(state_info)
                print(state_info)

                temp_users, total_interactions = [], 0
                fold_count -= 1

        return split_uids, split_states

    def get_feed_dict_for_training(self, batch_data):
        """Prepare the training data as a feed dictionary."""
        if self.batch_style_id != 0:
            return batch_data
        feed_dict = {}
        if isinstance(batch_data, tuple) and len(batch_data) == 3: #For inference
            feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'] = batch_data
        elif isinstance(batch_data, tuple) and len(batch_data) == 4: #For training
            feed_dict['heads'], feed_dict['relations'], feed_dict['pos_tails'], feed_dict['neg_tails'] = batch_data
        return feed_dict

    def prepare_test_data(self, user_batch, item_batch):
        """Prepare the test data based on given user and item batches."""
        return {
            'users': user_batch,
            'pos_items': item_batch,
            'mess_dropout': [0.] * len(eval(self.args.layer_size)),
            'node_dropout': [0.] * len(eval(self.args.layer_size)),
        }
