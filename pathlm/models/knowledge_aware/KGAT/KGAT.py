'''
Created on Dec 03, 2023
Pytorch Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
from scipy.sparse import coo_matrix

class KGAT(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(KGAT, self).__init__()
        self._parse_args(data_config, pretrain_data, args)
        """
        *********************************************************
        Create Model Parameters for CF & KGE parts.
        """
        self._build_weights()
        self._statistics_params()

    def _build_weights(self):
        if self.pretrain_data is None:
            self.user_embed = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.n_users, self.emb_dim)), requires_grad=True)
            self.entity_embed = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.n_entities, self.emb_dim)), requires_grad=True)
            print('using xavier initialization, for user and entity embed')
        else:
            self.user_embed = nn.Parameter(
                torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32))

            item_embed = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32)
            other_embed = torch.nn.init.xavier_uniform_(torch.empty(self.n_entities - self.n_items, self.emb_dim))

            self.entity_embed = nn.Parameter(torch.cat([item_embed, other_embed], dim=0))
            print('using pretrained initialization')

        self.relation_embed = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.kge_dim)))
        self.trans_W = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.emb_dim, self.kge_dim)))

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            setattr(self, f'W_gc_{k}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]))))
            setattr(self, f'b_gc_{k}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]))))
            setattr(self, f'W_bi_{k}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]))))
            setattr(self, f'b_bi_{k}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]))))
            setattr(self, f'W_mlp_{k}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(2 * self.weight_size_list[k], self.weight_size_list[k + 1]))))
            setattr(self, f'b_mlp_{k}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]))))

    def train_step(self, batch_data, mode=None):
        batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = self.forward(batch_data, mode)
        if mode == 'rec':
            opt = self.recommendation_opt
        elif mode == 'kge':
            opt = self.kge_opt

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        return batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss

    def forward(self, batch_data, mode=None):
        # Initialize default values for losses
        batch_loss = torch.tensor(0.0).float().to(self.device)  # Ensure tensor is on the correct device
        batch_base_loss = torch.tensor(0.0).float().to(self.device)
        batch_kge_loss = torch.tensor(0.0).float().to(self.device)
        batch_reg_loss = torch.tensor(0.0).float().to(self.device)

        # Recommendation mode
        if mode == 'rec':
            self.u_e, self.pos_i_e, self.neg_i_e = self._forward_phase_I(batch_data['users'],
                                                                         batch_data['pos_items'],
                                                                         batch_data['neg_items'])
            self._bpr_loss()
            batch_base_loss = self.base_loss
            batch_reg_loss = self.reg_loss
            batch_loss = self.base_loss + self.reg_loss

        # Knowledge Graph Embedding mode
        elif mode == 'kge':
            h = torch.tensor(batch_data['heads'])
            r = torch.tensor(batch_data['relations'])
            pos_t = torch.tensor(batch_data['pos_tails'])
            neg_t = torch.tensor(batch_data['neg_tails'])

            self.h_e, self.r_e, self.pos_t_e, self.neg_t_e, _, _ = self._forward_phase_II(h, r, pos_t, neg_t)
            self._kge_loss()
            batch_kge_loss = self.kge_loss
            batch_reg_loss = self.reg_loss
            batch_loss = self.kge_loss

        else:
            raise ValueError("Invalid mode")

        return batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss
    '''
        def forward(self, users, pos_items, neg_items, h=None, r=None, pos_t=None, neg_t=None):
        # Phase I: CF
        self.users, self.pos_items, self.neg_items = users, pos_items, neg_items
        self.u_e, self.pos_i_e, self.neg_i_e = self._forward_phase_I(users, pos_items, neg_items)
        pos_scores = torch.sum(self.u_e * self.pos_i_e, dim=1)
        neg_scores = torch.sum(self.u_e * self.neg_i_e, dim=1)
        base_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        # Initialize KGE loss to 0, in case it's not used
        kge_loss = torch.tensor(0.0, requires_grad=True)

        # Phase II: KGE
        if h is not None and r is not None and pos_t is not None and neg_t is not None:
            self.h_e, self.r_e, self.pos_t_e, self.neg_t_e, self.A_kg_score, self.A_out = \
                self._forward_phase_II(h, r, pos_t,neg_t)
            pos_kg_score = self._get_kg_score(self.h_e, self.r_e, self.pos_t_e)
            neg_kg_score = self._get_kg_score(self.h_e, self.r_e, self.neg_t_e)
            kge_loss = F.softplus(-(neg_kg_score - pos_kg_score)).mean()

        reg_loss = self._regularization_loss()

        # Compute total loss
        batch_loss = base_loss + kge_loss + reg_loss

        return batch_loss, base_loss, kge_loss, reg_loss

    '''

    """
    *********************************************************
    Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
    Different Convolutional Layers:
        1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
        2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
        3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
    """
    def _forward_phase_I(self, users, pos_items, neg_items):
        if self.alg_type in ['bi', 'kgat']:
            ua_embeddings, ea_embeddings = self._create_bi_interaction_embed()
        elif self.alg_type in ['gcn']:
            ua_embeddings, ea_embeddings = self._create_gcn_embed()
        elif self.alg_type in ['graphsage']:
            ua_embeddings, ea_embeddings = self._create_graphsage_embed()
        else:
            raise NotImplementedError

        u_e = ua_embeddings[users]
        pos_i_e = ea_embeddings[pos_items]
        neg_i_e = ea_embeddings[neg_items]

        return u_e, pos_i_e, neg_i_e

    """
    *********************************************************
    Compute Knowledge Graph Embeddings via TransR.
    """
    def _forward_phase_II(self, h, r, pos_t,neg_t):
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(h, r, pos_t, neg_t)
        self.A_kg_score = self._generate_transE_score(h=h, t=pos_t, r=r)
        self.A_out = self._create_attentive_A_out()

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        embeddings = torch.cat([self.user_embed, self.entity_embed], dim=0)
        embeddings = embeddings.unsqueeze(1)

        # head & tail entity embeddings: batch_size * 1 * emb_dim
        h_e = embeddings[h]
        pos_t_e = embeddings[pos_t]
        neg_t_e = embeddings[neg_t]

        # relation embeddings: batch_size * kge_dim
        r_e = self.relation_embed[r]

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = self.trans_W[r]

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = torch.bmm(h_e, trans_M).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, trans_M).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, trans_M).squeeze(1)

        # Uncomment below if you want to apply l2 normalization
        # h_e = nn.functional.normalize(h_e, p=2, dim=1)
        # r_e = nn.functional.normalize(r_e, p=2, dim=1)
        # pos_t_e = nn.functional.normalize(pos_t_e, p=2, dim=1)
        # neg_t_e = nn.functional.normalize(neg_t_e, p=2, dim=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _regularization_loss(self):
        regularizer = torch.norm(self.u_e, p=2) + torch.norm(self.pos_i_e, p=2) + torch.norm(self.neg_i_e, p=2)
        regularizer = regularizer / self.batch_size
        reg_loss = self.regs[0] * regularizer
        return reg_loss

    """
    Optimize Recommendation (CF) Part via BPR Loss.
    """
    def _bpr_loss(self):
        pos_scores = torch.sum(self.u_e * self.pos_i_e, dim=1)
        neg_scores = torch.sum(self.u_e * self.neg_i_e, dim=1)

        regularizer = self.u_e.norm(p=2).pow(2) + self.pos_i_e.norm(p=2).pow(2) + self.neg_i_e.norm(p=2).pow(2)
        regularizer = regularizer / self.batch_size

        # Using the softplus as BPR loss to avoid the nan error.
        base_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        self.base_loss = base_loss
        self.kge_loss = torch.tensor(0.0).float()
        self.reg_loss = self.regs[0] * regularizer
        self.recommendation_loss = self.base_loss + self.kge_loss + self.reg_loss

        # Optimization process.
        self.recommendation_opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _kge_loss(self):
        def _get_kg_score(h_e, r_e, t_e):
            kg_score = torch.sum((h_e + r_e - t_e).pow(2), dim=1, keepdim=True)
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e)

        # Using the softplus as BPR loss to avoid the nan error.
        kg_loss = F.softplus(-(neg_kg_score - pos_kg_score)).mean()

        kg_reg_loss = self.h_e.norm(p=2).pow(2) + self.r_e.norm(p=2).pow(2) + \
                      self.pos_t_e.norm(p=2).pow(2) + self.neg_t_e.norm(p=2).pow(2)
        kg_reg_loss = kg_reg_loss / self.batch_size_kg

        self.reg_loss = self.regs[1] * kg_reg_loss
        self.kge_loss = kg_loss + self.reg_loss

        # Optimization process.
        self.kge_opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _create_bi_interaction_embed(self):
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(self.A_in)
        ego_embeddings = torch.cat([self.user_embed, self.entity_embed], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = torch.cat(temp_embed, 0)

            add_embeddings = ego_embeddings + side_embeddings

            # transformed sum messages of neighbors.
            sum_embeddings = F.leaky_relu(
                torch.mm(add_embeddings, getattr(self, 'W_gc_%d' % k) + getattr(self, 'b_gc_%d' % k))
            )

            # bi messages of neighbors.
            bi_embeddings = ego_embeddings * side_embeddings
            # transformed bi messages of neighbors.
            bi_embeddings = F.leaky_relu(
                torch.mm(bi_embeddings, getattr(self, 'W_bi_%d' % k)) + getattr(self, 'b_bi_%d' % k))

            ego_embeddings = bi_embeddings + sum_embeddings
            # message dropout.
            ego_embeddings = F.dropout(ego_embeddings, self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = all_embeddings.split([self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_gcn_embed(self):
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(self.A_in)

        embeddings = torch.cat([self.user_embed, self.entity_embed], dim=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], embeddings))

            embeddings = torch.cat(temp_embed, 0)
            embeddings = F.leaky_relu(
                torch.mm(embeddings, getattr(self, 'W_gc_%d' % k) + getattr(self, 'b_gc_%d' % k)))
            embeddings = F.dropout(embeddings, self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = all_embeddings.split([self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_graphsage_embed(self):
        # Generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(self.A_in)

        pre_embeddings = torch.cat([self.user_embed, self.entity_embed], dim=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)

            embeddings = torch.cat([pre_embeddings, embeddings], 1)
            pre_embeddings = F.relu(
                torch.mm(embeddings, getattr(self, 'W_mlp_%d' % k)) + getattr(self, 'b_mlp_%d' % k))

            pre_embeddings = F.dropout(pre_embeddings, self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = all_embeddings.split([self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        X_coo = coo_matrix(X)
        values = X_coo.data
        indices = np.vstack((X_coo.row, X_coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = X_coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)

    def _create_attentive_A_out(self):
        indices = torch.LongTensor([self.all_h_list, self.all_t_list]).t()
        A_values = F.softmax(torch.sparse.FloatTensor(indices.t(), self.A_values, self.A_in.shape), dim=1)
        return A_values

    """
    Update the attentive laplacian matrix.
    """
    def update_attentive_A(self):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            h_tensor = torch.tensor(self.all_h_list[start:end], dtype=torch.long)
            r_tensor = torch.tensor(self.all_r_list[start:end], dtype=torch.long)
            pos_t_tensor = torch.tensor(self.all_t_list[start:end], dtype=torch.long)

            A_kg_score = self.A_kg_score(h_tensor, r_tensor,
                                         pos_t_tensor)  # Assuming A_kg_score is a method in the class
            kg_score.extend(A_kg_score.tolist())

        kg_score = torch.tensor(kg_score, dtype=torch.float32)

        # Assuming A_out is a method in the class
        new_A = self._create_attentive_A_out()  # Use the provided function directly
        new_A_values = new_A.values().numpy()
        new_A_indices = new_A.indices().numpy()

        rows = new_A_indices[0]
        cols = new_A_indices[1]
        self.A_in = coo_matrix((new_A_values, (rows, cols)),
                               shape=(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.)

    def _generate_transE_score(self, h, t, r):
        embeddings = torch.cat([self.user_embed, self.entity_embed], dim=0)
        embeddings = embeddings.unsqueeze(1)

        h_e = torch.index_select(embeddings, 0, h)
        t_e = torch.index_select(embeddings, 0, t)

        # relation embeddings: batch_size * kge_dim
        r_e = torch.index_select(self.relation_embed, 0, r)

        # relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = torch.index_select(self.trans_W, 0, r)

        # batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = torch.matmul(h_e, trans_M).squeeze(1)
        t_e = torch.matmul(t_e, trans_M).squeeze(1)

        # l2-normalize
        # h_e = F.normalize(h_e, p=2, dim=1)
        # r_e = F.normalize(r_e, p=2, dim=1)
        # t_e = F.normalize(t_e, p=2, dim=1)

        kg_score = torch.sum(t_e * torch.tanh(h_e + r_e), dim=1)

        return kg_score

    def _statistics_params(self):
        # number of params
        total_parameters = sum(p.numel() for p in self.parameters())
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'kgat'
        self.device = args.device
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100

        # initialize the attentive matrix A for phase I.
        self.A_in = data_config['A_in']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)
        self.mess_dropout = eval(args.mess_dropout)
        self.regs = eval(args.regs)
        self.verbose = args.verbose

