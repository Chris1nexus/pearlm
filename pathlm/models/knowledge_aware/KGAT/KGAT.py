'''
Created on Dec 03, 2023
Pytorch Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from typing import Tuple, Callable, Dict, Union

import torch
import torch.nn.functional as F
from torch import nn


class KGAT(nn.Module):
    def __init__(self, data_config, pretrain_data, args):
        super(KGAT, self).__init__()
        self._parse_args(data_config, pretrain_data, args)
        self.device = args.device
        self._build_weights()
        # Setting the aggregation function based on the alg_type
        method_mapper = {
            "bi": self._bi_interaction_method,
            "kgat": self._bi_interaction_method,
            "gcn": self._gcn_method,
            "graphsage": self._graphsage_method
        }

        if self.alg_type in method_mapper:
            self.aggregation_fun = lambda: self._aggregator(method_mapper[self.alg_type])
        else:
            raise NotImplementedError
        self._statistics_params()

    def _define_layer(self, in_size, out_size, name):
        setattr(self, f'W_{name}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(in_size, out_size))))
        setattr(self, f'b_{name}', nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, out_size))))

    def _build_weights(self):
        if self.pretrain_data is None:
            self.user_embed = torch.nn.Embedding(self.n_users, self.emb_dim).to(self.device)
            self.entity_embed = torch.nn.Embedding(self.n_entities, self.emb_dim).to(self.device)
            torch.nn.init.xavier_uniform_(self.user_embed.weight.data)
            torch.nn.init.xavier_uniform_(self.entity_embed.weight.data)
            print('using xavier initialization, for user and entity embed')
        else:
            self.user_embed = nn.Parameter(
                torch.tensor(self.pretrain_data['user_embed'], dtype=torch.float32)).to(self.device)

            item_embed = torch.tensor(self.pretrain_data['item_embed'], dtype=torch.float32).to(self.device)
            other_embed = torch.nn.init.xavier_uniform_(torch.empty(self.n_entities - self.n_items, self.emb_dim)).to(self.device)

            self.entity_embed = nn.Parameter(torch.cat([item_embed, other_embed], dim=0)).to(self.device)
            print('using pretrained initialization')

        #self.relation_embed = nn.Parameter(
        #    torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.kge_dim)))
        self.relation_embed = torch.nn.Embedding(self.n_relations, self.kge_dim).to(self.device)
        torch.nn.init.xavier_uniform_(self.relation_embed.weight.data)

        self.trans_W = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(self.n_relations, self.emb_dim, self.kge_dim))).to(self.device)
        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            setattr(self, f'W_gc_{k}', nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]).to(self.device))))
            setattr(self, f'b_gc_{k}',
                    nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]).to(self.device))))
            setattr(self, f'W_bi_{k}', nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(self.weight_size_list[k], self.weight_size_list[k + 1]).to(self.device))))
            setattr(self, f'b_bi_{k}',
                    nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]).to(self.device))))
            setattr(self, f'W_mlp_{k}', nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(2 * self.weight_size_list[k], self.weight_size_list[k + 1]).to(self.device))))
            setattr(self, f'b_mlp_{k}',
                    nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(1, self.weight_size_list[k + 1]).to(self.device))))
        self.A_values = torch.FloatTensor(len(self.all_h_list)).to(self.device)

    def _initialize_tensor(self, value: float = 0.0) -> torch.Tensor:
        """Initialize a tensor with the given value."""
        return torch.tensor(value).float().to(self.device)

    def train_step(self, batch_data, mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one training step based on the given mode ('rec' or 'kge').

        Args:
            batch_data: Data required for the training step.
            mode (str): Specifies the mode of operation, either 'rec' for recommendation or 'kge' for knowledge graph embedding.

        Returns:
            Tuple of batch loss, base loss, kge loss, and regularization loss.
        """
        base_loss = self._initialize_tensor()
        kge_loss = self._initialize_tensor()
        reg_loss = self._initialize_tensor()
        batch_loss = self._initialize_tensor()

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

    def forward(self, batch_data: Dict[str, torch.Tensor], mode: str) -> Union[torch.Tensor, None]:
        """
        Forward pass based on the given mode.

        Args:
            batch_data: Data required for the forward pass.
            mode (str): Specifies the mode of operation.

        Returns:
            Result of the forward pass or None.
        """
        if mode == 'rec':
            return self._forward_phase_I(batch_data['users'], batch_data['pos_items'], batch_data.get('neg_items'))
        elif mode == 'kge':
            return self._forward_phase_II(batch_data['heads'], batch_data['relations'], batch_data['pos_tails'],
                                          batch_data['neg_tails'])
        elif mode == 'update_att':
            self.update_attentive_A()
            return
        elif mode == 'eval':
            u_e, pos_i_e, _ = self._forward_phase_I(batch_data['users'], batch_data['pos_items'])
            return torch.matmul(u_e, pos_i_e.t())  # Transpose pos_i_e for matrix multiplication

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

    def _get_kg_inference(self, h: torch.Tensor, r: torch.Tensor, pos_t: torch.Tensor, neg_t: torch.Tensor) -> Tuple[
        torch.Tensor, ...]:
        """
        Retrieve the knowledge graph embeddings for given entities and relations.

        Given tensors of head entities (h), relations (r), positive tail entities (pos_t),
        and negative tail entities (neg_t), this function returns their respective embeddings.

        Args:
            h (torch.Tensor): Tensor of head entities.
            r (torch.Tensor): Tensor of relations.
            pos_t (torch.Tensor): Tensor of positive tail entities.
            neg_t (torch.Tensor): Tensor of negative tail entities.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing transformed embeddings for head entities,
                                      relation embeddings, positive tail entities, and negative tail entities.
        """
        # Get combined embeddings and add an extra dimension for matrix multiplication
        embeddings = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0).unsqueeze(1)

        # Retrieve head, positive tail, and negative tail entity embeddings
        h_e = embeddings[h]
        pos_t_e = embeddings[pos_t]
        neg_t_e = embeddings[neg_t]

        # Get relation embeddings and transformation weights
        self.relation_embed.to(self.device), self.trans_W.to(self.device)
        r_e = self.relation_embed(r)
        trans_M = self.trans_W[r]

        # Transform entity embeddings using the relation transformation weights
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

    def _split_A_hat(self, X: torch.sparse.FloatTensor) -> list:
        """
        Split a sparse matrix into several smaller submatrices along the rows.

        Args:
        - X (torch.sparse.FloatTensor): The sparse matrix to split.

        Returns:
        - A list of submatrices.
        """
        A_fold_hat = []
        fold_len = (self.n_users + self.n_entities) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = (i_fold + 1) * fold_len if i_fold != self.n_fold - 1 else self.n_users + self.n_entities

            # Create a mask to select rows within the current range
            mask = (X._indices()[0] >= start) & (X._indices()[0] < end)

            # Extract the non-zero values and their indices from the masked rows
            values = X._values()[mask]
            indices = X._indices()[:, mask]
            indices[0, :] -= start # Adjust row indices to be local to the current submatrix

            # Create a new sparse tensor for the current submatrix
            A_fold = torch.sparse_coo_tensor(indices, values, size=(end - start, X.size()[1])).to(self.device)

            # Add the new submatrix to the list
            A_fold_hat.append(A_fold)

        return A_fold_hat

    def _create_attentive_A_out(self, kg_score: torch.Tensor, batch_h_list: torch.Tensor = None,
                                batch_t_list: torch.Tensor = None) -> torch.sparse.FloatTensor:
        """
        Create an attentive adjacency matrix using knowledge graph scores.

        Args:
        - kg_score (torch.Tensor): Scores from the knowledge graph.
        - batch_h_list (torch.Tensor, optional): Head entities in batches. Default to None.
        - batch_t_list (torch.Tensor, optional): Tail entities in batches. Default to None.

        Returns:
        - An attentive adjacency matrix in sparse format.
        """
        if batch_h_list is None or batch_t_list is None:
            indices = torch.LongTensor([self.all_h_list, self.all_t_list]).transpose(0, 1).to(self.device)
        else:
            indices = torch.stack((batch_h_list, batch_t_list), dim=1).to(self.device)
        size = (self.n_users + self.n_entities, self.n_users + self.n_entities)

        sparse_tensor = torch.sparse_coo_tensor(indices.t(), kg_score, size).coalesce()
        # Apply sparse softmax
        softmax_sparse = torch.sparse.softmax(sparse_tensor, dim=1)
        return softmax_sparse

    def update_attentive_A(self):
        """
        Update the adjacency matrix A using attentive TransE scores.

        This function computes the attentive adjacency matrix by splitting the data into
        batches, computing the TransE scores for each batch, and then constructing a new
        adjacency matrix using the scores.
        """
        fold_len = len(self.all_h_list) // self.n_fold
        kg_scores = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = len(self.all_h_list) if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len

            h_batch = torch.tensor(self.all_h_list[start:end], dtype=torch.long).to('cpu')
            r_batch = torch.tensor(self.all_r_list[start:end], dtype=torch.long).to('cpu')
            pos_t_batch = torch.tensor(self.all_t_list[start:end], dtype=torch.long).to('cpu')

            A_kg_score = self._generate_transE_score(h_batch, pos_t_batch, r_batch)
            kg_scores.append(A_kg_score)

        kg_scores = torch.cat(kg_scores).to(self.device)

        # We'll get softmaxed sparse tensor
        softmaxed_A = self._create_attentive_A_out(kg_scores)

        # Extract the values and indices from softmaxed_A
        rows, cols = softmaxed_A.indices()
        values = softmaxed_A.values()

        # Construct the updated adjacency matrix on CPU and then transfer to GPU
        self.A_in = torch.sparse_coo_tensor(torch.stack([rows, cols]), values,
                                            size=(self.n_users + self.n_entities, self.n_users + self.n_entities)).to(
            self.device)

        if self.alg_type in ['org', 'gcn']:
            indices = torch.arange(0, self.A_in.shape[0], dtype=torch.long).to(self.device)
            self.A_in.index_add_(0, torch.stack([indices, indices]),
                                 torch.ones(self.A_in.shape[0], dtype=torch.float32).to(self.device))

        # Optional: Free up some memory
        del softmaxed_A, kg_scores, rows, cols, values
        torch.cuda.empty_cache()

    def _generate_transE_score(self, h: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Generate a TransE score for knowledge graph embeddings.

        This function computes the TransE score for the provided head (h), tail (t),
        and relation (r) tensors. The TransE score is a measure of the relationship
        strength in the knowledge graph.

        Args:
            h (torch.Tensor): Head entities tensor.
            t (torch.Tensor): Tail entities tensor.
            r (torch.Tensor): Relations tensor.

        Returns:
            torch.Tensor: Computed TransE scores.
        """
        assert h.device == t.device == r.device
        curr_device = h.device
        # Concatenate user and entity embeddings as in the original code
        embeddings = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0).to(curr_device)
        relation_embed = self.relation_embed.to(curr_device)
        trans_W = self.trans_W.to(curr_device)
        # Ensure h and t are on the same device as embeddings before indexing
        h_e = embeddings[h].unsqueeze(1)
        t_e = embeddings[t].unsqueeze(1)
        r_e = relation_embed(r)

        # Relation transform weights: batch_size * kge_dim * emb_dim
        trans_M = trans_W[r]

        # Batch_size * 1 * kge_dim -> batch_size * kge_dim
        h_e = torch.matmul(h_e, trans_M).squeeze(1)
        t_e = torch.matmul(t_e, trans_M).squeeze(1)

        # L2-normalize (if needed)
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

    def _aggregator(self, unique_aggregation_method: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Common aggregation routine.

        This function follows common initialization and finalization steps and applies
        the unique aggregation method based on the provided parameter.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                ua_embeddings (torch.Tensor): Embeddings for users.
                ea_embeddings (torch.Tensor): Embeddings for entities.
        """
        A_fold_hat = self._split_A_hat(self.A_in)
        embeddings = torch.cat([self.user_embed.weight, self.entity_embed.weight], dim=0)
        all_embeddings = [embeddings]

        for k in range(self.n_layers):
            temp_embed = [torch.sparse.mm(A_fold, embeddings) for A_fold in A_fold_hat]
            embeddings, norm_embeddings = unique_aggregation_method(embeddings, temp_embed, k)
            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)
        ua_embeddings, ea_embeddings = torch.split(all_embeddings, [self.n_users, self.n_entities], 0)

        return ua_embeddings, ea_embeddings

    def _bi_interaction_method(self, embeddings, temp_embed, k) -> Tuple[torch.Tensor, torch.Tensor]:
        side_embeddings = torch.cat(temp_embed, 0)
        add_embeddings = embeddings + side_embeddings
        W_gc, b_gc = getattr(self, f'W_gc_{k}'), getattr(self, f'b_gc_{k}')
        sum_embeddings = F.leaky_relu(torch.mm(add_embeddings, W_gc + b_gc))
        bi_embeddings = embeddings * side_embeddings
        W_bi, b_bi = getattr(self, f'W_bi_{k}'), getattr(self, f'b_bi_{k}')
        bi_embeddings = F.leaky_relu(torch.mm(bi_embeddings, W_bi) + b_bi)
        embeddings = bi_embeddings + sum_embeddings
        embeddings = F.dropout(embeddings, self.mess_dropout[k])
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings, norm_embeddings

    def _gcn_method(self, embeddings, temp_embed, k) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(temp_embed, 0)
        W_gc, b_gc = getattr(self, f'W_gc_{k}'), getattr(self, f'b_gc_{k}')
        embeddings = F.leaky_relu(torch.mm(embeddings, W_gc + b_gc))
        embeddings = F.dropout(embeddings, self.mess_dropout[k])
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings, norm_embeddings

    def _graphsage_method(self, embeddings, temp_embed, k) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([embeddings, torch.cat(temp_embed, 0)], 1)
        W_mlp, b_mlp = getattr(self, f'W_mlp_{k}'), getattr(self, f'b_mlp_{k}')
        embeddings = F.relu(torch.mm(embeddings, W_mlp + b_mlp))
        embeddings = F.dropout(embeddings, self.mess_dropout[k])
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings, norm_embeddings

    def create_embed(self, method: str) -> Tuple[torch.Tensor, torch.Tensor]:
        method_mapper = {
            "bi_interaction": self._bi_interaction_method,
            "gcn": self._gcn_method,
            "graphsage": self._graphsage_method
        }
        return self._aggregator(method_mapper[method])

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
        self.A_in = data_config['A_in'].to(self.device)

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




