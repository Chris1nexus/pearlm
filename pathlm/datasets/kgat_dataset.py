import collections
import numpy as np
import os
from torch.utils.data import Dataset

class KGATStyleDataset(Dataset):
    """
    This dataset is used by the following models: {BPRMF, FM, NFM, CKE, CFKG, KGAT}
    """
    def __init__(self, args, path, batch_style='list'):
        super(KGATStyleDataset).__init__()

        self.batch_styles = {'list': 0, 'map': 1}
        assert batch_style in self.batch_styles, f"Error: got {batch_style} but valid batch styles are {list(self.batch_styles.keys())}"
        self.path = path
        self.args = args
        self.batch_style = batch_style
        self.batch_style_id = self.batch_styles[self.batch_style]

        self.batch_size = args.batch_size

        # Load data
        self.train_data, self.train_user_dict = self._load_ratings(os.path.join(path, 'train.txt'))
        self.valid_data, self.valid_user_dict = self._load_ratings(os.path.join(path, 'valid.txt'))
        self.test_data, self.test_user_dict = self._load_ratings(os.path.join(path, 'test.txt'))

        self.exist_users = list(self.train_user_dict.keys())
        self.N_exist_users = len(self.exist_users)

        self._statistic_ratings()

        # Load KG data
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(os.path.join(path, 'kg_final.txt'))

        # Print dataset info
        self.batch_size_kg = self.n_triples // (self.n_train // args.batch_size_kg)
        self._print_data_info()

        self.layer_size = eval(args.layer_size)[0]
        self.mess_dropout = eval(args.mess_dropout)[0]
        self.node_dropout = eval(args.mess_dropout)[0]

    def _load_ratings(self, file_name):
        user_dict = collections.defaultdict(list)
        inter_mat = []

        with open(file_name, 'r') as file:
            for line in file:
                items = [int(i) for i in line.strip().split()]
                user, pos_items = items[0], items[1:]
                inter_mat.extend([(user, item) for item in pos_items])
                user_dict[user].extend(pos_items)

        return np.array(inter_mat), user_dict

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.valid_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)

    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)
            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print(f'[n_users, n_items]=[{self.n_users}, {self.n_items}]')
        print(f'[n_train, n_test]=[{self.n_train}, {self.n_test}]')
        print(f'[n_entities, n_relations, n_triples]=[{self.n_entities}, {self.n_relations}, {self.n_triples}]')
        print(f'[batch_size, batch_size_kg]=[{self.batch_size}, {self.batch_size_kg}]')

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
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = {}

        for uid in all_users_to_test:
            n_iids = len(self.train_user_dict[uid]) + len(self.test_user_dict[uid])
            user_n_iid.setdefault(n_iids, []).append(uid)

        split_uids, split_state = [], []
        temp, n_rates, fold = [], 0, 4

        for n_iids in sorted(user_n_iid):
            temp.extend(user_n_iid[n_iids])
            n_rates += n_iids * len(user_n_iid[n_iids])

            if n_rates >= 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)
                state = f'#inter per user<=[{n_iids}], #users=[{len(temp)}], #all rates=[{n_rates}]'
                split_state.append(state)
                print(state)

                temp, n_rates = [], 0
                fold -= 1

            if not temp:
                continue

            split_uids.append(temp)
            state = f'#inter per user<=[{n_iids}], #users=[{len(temp)}], #all rates=[{n_rates}]'
            split_state.append(state)
            print(state)

        return split_uids, split_state

    def __len__(self):
        # number of existing users after the preprocessing described in the paper,
        # determines the length of the training dataset, for which a positive an negative are extracted
        return len(self.exist_users)

    ##_generate_train_cf_batch
    def __getitem__(self, idx):
        """
        if self.batch_size <= self.n_users:
            user = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        """
        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items
        """
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        """
        u = self.exist_users[idx]
        pos_item = sample_pos_items_for_u(u, 1)
        neg_item = sample_neg_items_for_u(u, 1)
        if len(pos_item) == 1:
            pos_item = pos_item[0]
        if len(neg_item) == 1:
            neg_item = neg_item[0]


        if self.batch_style_id == 0:
            return u, pos_item, neg_item
        else:
            return {'users': u, 'pos_items': pos_item, 'neg_items':neg_item}#u, pos_item, neg_item #users, pos_items, neg_items


    def prepare_train_data_as_feed_dict(self, batch_data):
        # Ensure batch_data is in dictionary format
        feed_dict = {}
        if self.batch_style_id == 0:
            users, pos_items, neg_items = batch_data
            feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'] = users, pos_items, neg_items
        else:
            return batch_data
        return feed_dict

    def prepare_test_data_as_feed_dict(self, batch_data):
        feed_dict = {}
        if self.batch_style_id == 0:
            users, pos_items, neg_items = batch_data
            feed_dict['users'], feed_dict['pos_items'], feed_dict['neg_items'] = users, pos_items, neg_items
        else:
            return batch_data
        return feed_dict
        '''
        # Ensure batch_data is in dictionary format
        if self.batch_style_id == 0:
            users, pos_items, neg_items = batch_data
        else:
            users = batch_data['users']
            pos_items = batch_data['pos_items']
            neg_items = batch_data['neg_items']
        return users, pos_items, neg_items
        '''


    def prepare_train_data_kge_as_feed_dict(self, batch_data):
        feed_dict = {}
        if self.batch_style_id == 0:
            heads, relations, pos_tails, neg_tails = batch_data
            feed_dict['heads'], feed_dict['relations'], feed_dict['pos_tails'], feed_dict['neg_tails'] = heads, relations, pos_tails, neg_tails
        else:
            return batch_data
        return feed_dict

