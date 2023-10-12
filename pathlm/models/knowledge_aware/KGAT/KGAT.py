'''
Created on Dec 03, 2023
Pytorch Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch import nn


class KGAT(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(KGAT, self).__init__()
        self._parse_args(data_config, pretrain_data, args)
        """
        *********************************************************
        Create Model Parameters for CF & KGE parts.
        """
        self._build_weights()
        self.aggregation_fun = None
        if self.alg_type in ['bi', 'kgat']:
            self.aggregation_fun = self.bi_interaction
        elif self.alg_type in ['gcn']:
            self.aggregation_fun = self._create_gcn_embed
        elif self.alg_type in ['graphsage']:
            self.aggregation_fun = self._create_graphsage_embed
        else:
            raise NotImplementedError
        self._statistics_params()

    def _define_layer(self, in_size, out_size, name):
        setattr(self, f'W_{name}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_size, out_size))))
        setattr(self, f'b_{name}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, out_size))))

    def _build_weights(self):
        if self.pretrain_data is None:
            self.user_embed = torch.nn.Embedding(self.n_users, self.emb_dim)
            self.entity_embed = torch.nn.Embedding(self.n_entities, self.emb_dim)
            torch.nn.init.xavier_uniform_(self.user_embed.weight.data)
            torch.nn.init.xavier_uniform_(self.entity_embed.weight.data)
            print('using xavier initialization, for user and entity embed')
        else:
            self.user_embed = nn.Parameter(
                torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32))

            item_embed = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32)
            other_embed = torch.nn.init.xavier_uniform_(torch.empty(self.n_entities - self.n_items, self.emb_dim))

            self.entity_embed = nn.Parameter(torch.cat([item_embed, other_embed], dim=0))
            print('using pretrained initialization')

        self.relation_embed = nn.Embedding(
            torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.kge_dim)))
        self.trans_W = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.emb_dim, self.kge_dim)))
        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            setattr(self, f'W_gc_{k}', nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]))))
            setattr(self, f'b_gc_{k}',
                    nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]))))
            setattr(self, f'W_bi_{k}', nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]))))
            setattr(self, f'b_bi_{k}',
                    nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]))))
            setattr(self, f'W_mlp_{k}', nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(2 * self.weight_size_list[k], self.weight_size_list[k + 1]))))
            setattr(self, f'b_mlp_{k}',
                    nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]))))
        self.A_values = torch.FloatTensor(len(self.all_h_list)).to(self.device)

    def train_step(self, batch_data, mode=None):
        base_loss = torch.tensor(0.0).float().to(self.device)
        kge_loss = torch.tensor(0.0).float().to(self.device)
        reg_loss = torch.tensor(0.0).float().to(self.device)
        batch_loss = torch.tensor(0.0).float().to(self.device)
        if mode == 'rec':
            u_e, pos_i_e, neg_i_e = self.forward(batch_data, mode)
            base_loss, reg_loss, recommendation_loss = self._bpr_loss(u_e, pos_i_e, neg_i_e)
            batch_loss = recommendation_loss
            opt = self.recommendation_opt
        elif mode == 'kge':
            h_e, r_e, pos_t_e, neg_t_e = self.forward(batch_data, mode)
            kge_loss, reg_loss = self._kge_loss(h_e, r_e, pos_t_e, neg_t_e)
            batch_loss = kge_loss
            opt = self.kge_opt
        else:
            raise ValueError("Invalid mode")

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

        return batch_loss, base_loss, kge_loss, reg_loss

    def forward(self, batch_data=None, mode=None):
        if mode == 'rec':
            u_e, pos_i_e, neg_i_e = self._forward_phase_I(batch_data['users'],
                                                          batch_data['pos_items'],
                                                          batch_data['neg_items'])
            return u_e, pos_i_e, neg_i_e
        elif mode == 'kge':
            return self._forward_phase_II(batch_data['heads'], batch_data['relations'], batch_data['pos_tails'],
                                          batch_data['neg_tails'])
        elif mode == 'update_att':
            self.update_attentive_A()
            return
        elif mode == 'eval':
            u_e, pos_i_e, _ = self._forward_phase_I(batch_data['users'], batch_data['pos_items'])
            batch_predictions = torch.matmul(u_e, pos_i_e.t())  # Transpose pos_i_e for matrix multiplication
            return batch_predictions
        else:
            raise ValueError("Invalid mode")

    """
    *********************************************************
    Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
    Different Convolutional Layers:
        1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
        2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
        3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
    """
    def _forward_phase_I(self, users, pos_items, neg_items=None):
        ua_embeddings, ea_embeddings = self.aggregation_fun()
        u_e, pos_i_e, neg_i_e = ua_embeddings[users], ea_embeddings[pos_items], ea_embeddings[neg_items]
        return u_e, pos_i_e, neg_i_e
    """
    *********************************************************
    Compute Knowledge Graph Embeddings via TransR.
    """

    def _forward_phase_II(self, h, r, pos_t, neg_t):
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_inference(h, r, pos_t, neg_t)
        A_kg_score = self._generate_transE_score(h=h, t=pos_t, r=r)
        self.A_out = self._create_attentive_A_out(A_kg_score, h, pos_t)
        return h_e, r_e, pos_t_e, neg_t_e

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
        return h_e, r_e, pos_t_e, neg_t_e

    def _regularization_loss(self):
        regularizer = torch.norm(self.u_e, p=2) + torch.norm(self.pos_i_e, p=2) + torch.norm(self.neg_i_e, p=2)
        regularizer = regularizer / self.batch_size
        reg_loss = self.regs[0] * regularizer
        return reg_loss

    """
    Optimize Recommendation (CF) Part via BPR Loss.
    """

    def _bpr_loss(self, u_e, pos_i_e, neg_i_e):
        pos_scores = torch.sum(u_e * pos_i_e, dim=1)
        neg_scores = torch.sum(u_e * neg_i_e, dim=1)

        regularizer = u_e.norm(p=2).pow(2) + pos_i_e.norm(p=2).pow(2) + neg_i_e.norm(p=2).pow(2)
        regularizer = regularizer / self.batch_size

        # Using the softplus as BPR loss to avoid the nan error.
        base_loss = F.softplus(-(pos_scores - neg_scores)).mean()
        kge_loss = torch.tensor(0.0).float()
        reg_loss = self.regs[0] * regularizer
        recommendation_loss = base_loss + kge_loss + reg_loss
        # Optimization process.
        self.recommendation_opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return base_loss, reg_loss, recommendation_loss

    def _kge_loss(self, h_e, r_e, pos_t_e, neg_t_e):
        pos_kg_score = torch.sum((h_e + r_e - pos_t_e).pow(2), dim=1, keepdim=True)
        neg_kg_score = torch.sum((h_e + r_e - neg_t_e).pow(2), dim=1, keepdim=True)
        # Using the softplus as BPR loss to avoid the nan error.
        kg_loss = F.softplus(-(neg_kg_score - pos_kg_score)).mean()
        kg_reg_loss = h_e.norm(p=2).pow(2) + r_e.norm(p=2).pow(2) + \
                      pos_t_e.norm(p=2).pow(2) + neg_t_e.norm(p=2).pow(2)
        kg_reg_loss = kg_reg_loss / self.batch_size_kg
        reg_loss = self.regs[1] * kg_reg_loss
        kge_loss = kg_loss + reg_loss
        # Optimization process.
        self.kge_opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return kge_loss, reg_loss

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            # Mask for selecting rows in the desired range
            mask = (X._indices()[0] >= start) & (X._indices()[0] < end)

            # Extract the values and adjust the row indices
            values = X._values()[mask]
            indices = X._indices()[:, mask]
            indices[0, :] -= start  # adjust row indices

            # Create a new sparse tensor using the masked values and adjusted indices
            A_fold = torch.sparse_coo_tensor(indices, values, size=(end - start, X.size()[1]))
            A_fold_hat.append(A_fold)

        return A_fold_hat

    def _create_attentive_A_out(self, kg_score, batch_h_list=None, batch_t_list=None):
        if batch_h_list is None or batch_t_list is None:
            indices = torch.LongTensor([self.all_h_list, self.all_t_list]).transpose(0, 1).to(self.device)
        else:
            indices = torch.stack((batch_h_list, batch_t_list), dim=1).to(self.device)
        size = (self.n_users + self.n_entities, self.n_users + self.n_entities)

        # Create a sparse tensor using kg_score as values
        sparse_tensor = torch.sparse.FloatTensor(indices.t(), kg_score, size).coalesce()

        # Apply sparse softmax
        softmax_sparse = torch.sparse.softmax(sparse_tensor, dim=1)
        return softmax_sparse

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

            h_batch = torch.tensor(self.all_h_list[start:end], dtype=torch.long).to(self.device)
            r_batch = torch.tensor(self.all_r_list[start:end], dtype=torch.long).to(self.device)
            pos_t_batch = torch.tensor(self.all_t_list[start:end], dtype=torch.long).to(self.device)

            A_kg_score = self._generate_transE_score(h_batch, pos_t_batch, r_batch)
            kg_score += list(A_kg_score)

        kg_score = torch.tensor(kg_score, dtype=torch.float).to(self.device)

        # We'll get softmaxed sparse tensor
        softmaxed_A = self._create_attentive_A_out(kg_score)


        # Extract the values and indices from softmaxed_A
        new_A_values = softmaxed_A.values()
        new_A_indices = softmaxed_A.indices()

        # Convert to long tensor to build new A_in
        rows = new_A_indices[0]
        cols = new_A_indices[1]
        values = new_A_values

        self.A_in = torch.sparse_coo_tensor(torch.stack([rows, cols]), values,
                                            size=(self.n_users + self.n_entities, self.n_users + self.n_entities))

        if self.alg_type in ['org', 'gcn']:
            # Set diagonal elements to 1
            indices = torch.arange(0, self.A_in.shape[0], dtype=torch.long).to(self.device)
            self.A_in.index_add_(0, torch.stack([indices, indices]),
                                 torch.ones(self.A_in.shape[0], dtype=torch.float32).to(self.device))

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
