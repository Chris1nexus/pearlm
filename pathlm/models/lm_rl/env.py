import gym
import logging
import random
import sys
import torch
from torch import autocast


import pfrl

import os
from typing import List, Any
import argparse
from tqdm import tqdm
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import PreTrainedTokenizerFast, LogitsProcessorList, set_seed

from pathlm.models.lm.perlm import PERLM
from pathlm.models.lm.decoding_constraints import ConstrainedLogitsProcessorWordLevel, PrefixConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel
from pathlm.utils import SEED, get_pid_to_eid, get_eid_to_name_map, get_data_dir, get_set, check_dir
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k 
from pathlm.sampling import KGsampler
from pathlm.sampling.samplers.constants import LiteralPath, TypeMapper

from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids, \
    _initialise_type_masks, \
    get_user_negatives, get_user_positives

from pathlm.models.rl.PGPR.pgpr_utils import get_knowledge_derived_relations, MODEL_DATASET_DIR,INTERACTION, DATASET_INFO_DIR,\
        PRODUCT, USER, ENTITY, RELATION

from pathlm.models.lm_rl.lm_rl_utils import tokenize_augmented_kg



import gym
import logging
import random
import sys
import torch
from torch import autocast


class TextRLEnv(gym.Env):
    def __init__(self, model, tokenizer, observation_input=[], max_length=100, compare_sample=2,
                 unfreeze_layer_from_past=0):
        try:
            tokvocab = tokenizer.get_vocab()
        except:
            tokvocab = tokenizer.vocab
            pass
        vocabs = list(dict(sorted(tokvocab.items(), key=lambda item: item[1])).keys())
        self.action_space = gym.spaces.Discrete(len(vocabs))
        self.actions = vocabs
        self.model = model
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.compare_sample = compare_sample
        self.target_table = {}
        self.unfreeze_layer_from_past = 1 if unfreeze_layer_from_past == 0 else unfreeze_layer_from_past
        self.env_max_length = min(max(self.model.config.max_length, self.tokenizer.model_max_length), max_length)
        self.reset()

        self.gen_stop_toks = []
        logging.disable(sys.maxsize)
        if self.tokenizer.sep_token:
            self.gen_stop_toks.append(self.tokenizer.sep_token)
        if self.tokenizer.eos_token:
            self.gen_stop_toks.append(self.tokenizer.eos_token)
        logging.disable(logging.NOTSET)

    def step(self, action):
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish)
        self.predicted = predicted
        return self._get_obs(predicted), reward, finish, {"predicted_str": predicted_str}

    def get_reward(self, input_item, predicted_list, finish):
        reward = [0] * self.compare_sample
        return reward

    def gat_obs_input(self, input_item):
        return input_item['input']

    @autocast('cuda')
    def reset(self, input_item=None):
        self.predicted = [[]] * self.compare_sample
        self.predicted_end = [False] * self.compare_sample
        self.input_item = {"input": ""}
        if input_item is None:
            self.input_item = random.choice(self.observation_space)
        else:
            self.input_item = input_item
        return self._get_obs(self.predicted)

    @autocast('cuda')
    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            obs_list = []
            for p_text in predicted:
                p_text_str = self.tokenizer.convert_tokens_to_string(p_text)
                if self.model.__class__.__name__ == 'OPTForCausalLM':
                    feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                                  return_tensors='pt',
                                                  add_special_tokens=False).to(self.model.device)
                    with torch.cuda.amp.autocast(enabled=False):
                        prediction = self.model(**feature_dict, output_hidden_states=True)
                    outputs = prediction.hidden_states[-self.unfreeze_layer_from_past][:, -1, :]
                else:
                    if len([k for k, v in self.model.named_parameters() if 'decoder' in k]) > 0:
                        feature_dict = self.tokenizer([self.gat_obs_input(self.input_item)],
                                                      return_tensors='pt',
                                                      add_special_tokens=True).to(self.model.device)
                        if len(p_text) > 0:
                            decoder_input_ids = [self.model.config.decoder_start_token_id] + \
                                                self.tokenizer.convert_tokens_to_ids(p_text)
                            dec_input = torch.tensor([decoder_input_ids]).to(self.model.device)
                            feature_dict['decoder_input_ids'] = dec_input
                        else:
                            feature_dict['decoder_input_ids'] = torch.tensor(
                                [[self.model.config.decoder_start_token_id]]).to(self.model.device)
                        with torch.cuda.amp.autocast(enabled=False):
                            prediction = self.model(**feature_dict, output_hidden_states=True)
                        outputs = prediction.decoder_hidden_states[-self.unfreeze_layer_from_past].squeeze(0)
                    else:
                        if self.model.__class__.__name__ == 'DistributedBloomForCausalLM':
                            with self.model.inference_session(max_length=self.env_max_length) as sess:
                                feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                                              return_tensors='pt',
                                                              add_special_tokens=False).to(self.model.device)
                                embs = self.model.transformer.word_embeddings(feature_dict.input_ids)
                                embs = self.model.transformer.word_embeddings_layernorm(embs)
                                h = sess.step(embs)
                                outputs = self.model.transformer.ln_f(h[:, -1])
                        else:
                            feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                                          return_tensors='pt',
                                                          add_special_tokens=False).to(self.model.device)
                            prediction = self.model(**feature_dict, output_hidden_states=True)
                            
                            outputs = prediction.hidden_states[-self.unfreeze_layer_from_past].squeeze(0)
                obs_list.append(outputs.data[-1])
            return (torch.stack(obs_list))

    def _predict(self, vocab_id):
        predicted_list = {}
        predicted_list_end = {}
        with torch.inference_mode():
            for i, (v_id, predicted, predicted_end) in enumerate(zip(vocab_id, self.predicted, self.predicted_end)):
                predicted_list_end[i] = False
                if not predicted_end:
                    if v_id >= len(self.actions):
                        print('Out of bound action: ', v_id, len(self.actions))

                    pred_word = self.actions[v_id]
                    if pred_word in self.gen_stop_toks \
                            or len(pred_word) < 1 \
                            or len(predicted) > self.env_max_length:
                        predicted_list_end[i] = True
                        predicted_list[i] = [pred_word]
                    else:
                        predicted_list[i] = [pred_word]
                else:
                    predicted_list_end[i] = True
                    predicted_list[i] = ['']

            for i, (l, e) in enumerate(zip(predicted_list.values(), predicted_list_end.values())):
                self.predicted[i] = self.predicted[i] + l
                self.predicted_end[i] = e

            return self.predicted, all(self.predicted_end), [self.tokenizer.convert_tokens_to_string(i) for i in
                                                             self.predicted]




class PathRLenv(TextRLEnv):
    
    TRAIN_SET='train'
    VALID_SET='valid'
    TEST_SET='test'
    
    WRONG_PATH_REWARD = 0
    PRED_POSITIVE_ITEM_REWARD = 1
    PRED_NEGATIVE_ITEM_REWARD = 0 #-1
    TOO_LONG_PENALTY = 0
    
    def __init__(self, model, tokenizer, dataset_name, n_hop, kg, max_sequence_len=None, 
                 n_sequences_per_user=10, n_special_tokens=0,
                wrong_path_reward = 0,
                pred_positive_item_reward = 1,
                pred_negative_item_reward = 0,        
                ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.wrong_path_reward = wrong_path_reward
        self.pred_positive_item_reward = pred_positive_item_reward
        self.pred_negative_item_reward = pred_negative_item_reward          
        tokenized_kg, kg_to_vocab = tokenize_augmented_kg(kg, 
                                                          tokenizer, 
                                                          use_token_ids=False
                                                         )
        self.tokenized_kg = tokenized_kg
        self.kg_to_vocab = kg_to_vocab
        
        self.n_hop = n_hop
        #self.SEQUENCE_LEN =  2 + 2 + n_hop*2 + (n_hop-1)*2  # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
        self.SEQUENCE_LEN =  n_special_tokens + 2*n_hop+1  # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
        
        if max_sequence_len is None:
            max_sequence_len = self.SEQUENCE_LEN
        
        self.N_SEQUENCES_PER_USER = n_sequences_per_user 
        self.n_sequences_per_user = n_sequences_per_user
        
        self.__load_user_data()
        self.__generate_prompts()
        self.__init_sequence_cache()
        
        super().__init__(model, 
                         tokenizer, 
                         observation_input=self.inference_paths, 
                         max_length=max_sequence_len,#self.SEQUENCE_LEN, 
                         compare_sample=self.n_sequences_per_user
                        )        
    def __init_sequence_cache(self):
        self.rewards_cache = [0 for _ in range(self.N_SEQUENCES_PER_USER)]
        self.path_cache = [[] for _ in range(self.N_SEQUENCES_PER_USER)]
        self.validity_cache = [True for _ in range(self.N_SEQUENCES_PER_USER)]
        
    def get_reward(self, input_item, predicted_list, finish):
        #print(predicted_list)
        input_item = input_item['input'].split(' ')[1:]
        
        uid = input_item[0][1:]
        n_cur_tokens = len(predicted_list[0])
        # user, rel -> len == 2
        rewards = [ 0 if n_cur_tokens <= self.SEQUENCE_LEN else  PathRLenv.TOO_LONG_PENALTY   for _ in range(len(predicted_list))]
        
        #print(f'{n_cur_tokens}/{self.SEQUENCE_LEN}', finish, )
        if finish: #or n_cur_tokens+3 >= self.SEQUENCE_LEN:
            for seq_id, sequence in enumerate(predicted_list):
                #print(finish, sequence)
                prev_ent =input_item[0]
                rel = input_item[1]
                cur_ent = None                
                for idx in range(0,len(sequence),2):
                    if idx-1 >= 0:
                        rel = sequence[idx-1]
                    cur_ent = sequence[idx]
                    #print('aaa', rel not in tokenized_kg[prev_ent], cur_ent not in tokenized_kg[prev_ent][rel])
                    
                    if prev_ent not in self.kg_to_vocab or rel not in self.tokenized_kg[prev_ent] or cur_ent not in self.tokenized_kg[prev_ent][rel]:
                        # check path correctness
                        rewards[seq_id] += self.wrong_path_reward #PathRLenv.WRONG_PATH_REWARD
                    #self.SEQUENCE_LEN-1 \
                    if idx == len(sequence)-2\
                            and rewards[seq_id] >= 0 \
                            and cur_ent[0] == LiteralPath.prod_type:
                        # check positive item if non negative reward(means that path is correct)
                        pid = cur_ent[1:]
                        cur_reward = 0.
                        if pid in self.user_positives[PathRLenv.TRAIN_SET][uid]:
                            cur_reward = self.pred_positive_item_reward #PathRLenv.PRED_POSITIVE_ITEM_REWARD
                        else:
                            cur_reward = self.pred_negative_item_reward #PathRLenv.PRED_NEGATIVE_ITEM_REWARD
                        rewards[seq_id] += cur_reward
                    prev_ent = cur_ent
            #print(rewards)
            #reward = [0]  # calculate reward score based on predicted_list
        return rewards    
    def step(self, action):
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish)
        self.predicted = predicted
        return self._get_obs(predicted), reward, finish, {"predicted_str": predicted_str}
    
    
    @autocast('cuda')
    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            input_ids_list = []
            obs_list = []
            for p_text in predicted:
                p_text_str = self.tokenizer.convert_tokens_to_string(p_text)
                
                feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                              return_tensors='pt',
                                              add_special_tokens=False).to(self.model.device)

                prediction = self.model(**feature_dict, output_hidden_states=True)

                outputs = prediction.hidden_states[-self.unfreeze_layer_from_past].squeeze(0)
                #print(prediction.hidden_states[-self.unfreeze_layer_from_past].shape)
                input_ids_list.append(feature_dict['input_ids'][-1])
                
                obs_list.append(outputs.data[-1])
                #print(feature_dict['input_ids'].shape)
                #print(feature_dict['input_ids'][-1].shape)
                #print(outputs.data.shape)
                #print(outputs.data[-1].shape)
                #print()

            return (torch.stack(obs_list)), (torch.stack(input_ids_list))
        
        
        
    def __generate_prompts(self):  
        init_condition_fn = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths = [{'input': init_condition_fn(uid)} for uid in self.uids ]
        
    def __load_user_data(self):
        self.train_set = get_set(self.dataset_name, set_str=PathRLenv.TRAIN_SET)
        self.valid_set = get_set(self.dataset_name, set_str=PathRLenv.VALID_SET)
        self.test_set = get_set(self.dataset_name, set_str=PathRLenv.TEST_SET)
        self.uids = list(self.train_set.keys())
        # Load user negatives (all interactions not in train)
        self.user_negatives = self.__load_user_negatives()
        # Interactions occurring in train, val, test, each in separate groups
        self.user_positives = self.__load_user_positives()
        self.id_to_uid_token_map = {self.tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in self.uids}        
    
    def __load_user_negatives(self):
        #user_negatives = get_user_negatives_tokens_ids(self.dataset_name, self.tokenizer)
        user_negatives = get_user_negatives(self.dataset_name)
        return user_negatives
    
    def __load_user_positives(self):
        user_positives = dict()
        user_positives[PathRLenv.TRAIN_SET] = self.train_set
        user_positives[PathRLenv.VALID_SET] = self.valid_set
        user_positives[PathRLenv.TEST_SET] = self.test_set
        return user_positives
        
    


    
    



'''
class PathRLenv(TextRLEnv):
    
    TRAIN_SET='train'
    VALID_SET='valid'
    TEST_SET='test'
    
    WRONG_PATH_REWARD = 0
    PRED_POSITIVE_ITEM_REWARD = 1
    PRED_NEGATIVE_ITEM_REWARD = -1
    TOO_LONG_PENALTY = 0
    
    def __init__(self, model, tokenizer, dataset_name, n_hop, kg, max_sequence_len=None, 
                 n_sequences_per_user=10, n_special_tokens=0,
                
                ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        tokenized_kg, kg_to_vocab = tokenize_augmented_kg(kg, 
                                                          tokenizer, 
                                                          use_token_ids=False
                                                         )
        self.tokenized_kg = tokenized_kg
        self.kg_to_vocab = kg_to_vocab
        
        self.n_hop = n_hop
        #self.SEQUENCE_LEN =  2 + 2 + n_hop*2 + (n_hop-1)*2  # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
        self.SEQUENCE_LEN =  n_special_tokens + 2*n_hop+1  # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
        
        if max_sequence_len is None:
            max_sequence_len = self.SEQUENCE_LEN
        
        self.N_SEQUENCES_PER_USER = n_sequences_per_user 
        self.n_sequences_per_user = n_sequences_per_user
        
        self.__load_user_data()
        self.__generate_prompts()
        self.__init_sequence_cache()
        
        super().__init__(model, 
                         tokenizer, 
                         observation_input=self.inference_paths, 
                         max_length=max_sequence_len,#self.SEQUENCE_LEN, 
                         compare_sample=self.n_sequences_per_user
                        )        
    def __init_sequence_cache(self):
        self.rewards_cache = [0 for _ in range(self.N_SEQUENCES_PER_USER)]
        self.path_cache = [[] for _ in range(self.N_SEQUENCES_PER_USER)]
        self.validity_cache = [True for _ in range(self.N_SEQUENCES_PER_USER)]
        
    def get_reward(self, input_item, predicted_list, finish):
        #print(predicted_list)
        input_item = input_item['input'].split(' ')[1:]
        
        uid = input_item[0][1:]
        n_cur_tokens = len(predicted_list[0])
        # user, rel -> len == 2
        rewards = [ 0 if n_cur_tokens <= self.SEQUENCE_LEN else  PathRLenv.TOO_LONG_PENALTY   for _ in range(len(predicted_list))]
        
        #print(f'{n_cur_tokens}/{self.SEQUENCE_LEN}', finish, )
        if finish: #or n_cur_tokens+3 >= self.SEQUENCE_LEN:
            for seq_id, sequence in enumerate(predicted_list):
                #print(finish, sequence)
                prev_ent =input_item[0]
                rel = input_item[1]
                cur_ent = None                
                for idx in range(0,len(sequence),2):
                    if idx-1 >= 0:
                        rel = sequence[idx-1]
                    cur_ent = sequence[idx]
                    #print('aaa', rel not in tokenized_kg[prev_ent], cur_ent not in tokenized_kg[prev_ent][rel])
                    
                    if prev_ent not in self.kg_to_vocab or rel not in self.tokenized_kg[prev_ent] or cur_ent not in self.tokenized_kg[prev_ent][rel]:
                        # check path correctness
                        rewards[seq_id] += PathRLenv.WRONG_PATH_REWARD
                    #self.SEQUENCE_LEN-1 \
                    if idx == len(sequence)-2\
                            and rewards[seq_id] >= 0 \
                            and cur_ent[0] == LiteralPath.prod_type:
                        # check positive item if non negative reward(means that path is correct)
                        pid = cur_ent[1:]
                        cur_reward = 0.
                        if pid in self.user_positives[PathRLenv.TRAIN_SET][uid]:
                            cur_reward = PathRLenv.PRED_POSITIVE_ITEM_REWARD
                        else:
                            cur_reward = PathRLenv.PRED_NEGATIVE_ITEM_REWARD
                        rewards[seq_id] += cur_reward
                    prev_ent = cur_ent
            #print(rewards)
            #reward = [0]  # calculate reward score based on predicted_list
        return rewards    
    def step(self, action):
        predicted, finish, predicted_str = self._predict(vocab_id=action)
        reward = self.get_reward(self.input_item, predicted, finish)
        self.predicted = predicted
        return self._get_obs(predicted), reward, finish, {"predicted_str": predicted_str}
 
    
    @autocast('cuda')
    def _get_obs(self, predicted=[]):
        with torch.inference_mode():
            input_ids_list = []
            obs_list = []
            for p_text in predicted:
                p_text_str = self.tokenizer.convert_tokens_to_string(p_text)
                
                feature_dict = self.tokenizer([[self.gat_obs_input(self.input_item), p_text_str]],
                                              return_tensors='pt',
                                              add_special_tokens=False).to(self.model.device)

                prediction = self.model(**feature_dict, output_hidden_states=True)
                outputs = prediction.hidden_states[-self.unfreeze_layer_from_past].squeeze(0)
                
                input_ids_list.append(feature_dict['input_ids'][-1])
                
                obs_list.append(outputs.data[-1])

            return (torch.stack(obs_list)), (torch.stack(input_ids_list))
        
        
        
    def __generate_prompts(self):  
        init_condition_fn = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths = [{'input': init_condition_fn(uid)} for uid in self.uids ]
        
    def __load_user_data(self):
        self.train_set = get_set(self.dataset_name, set_str=PathRLenv.TRAIN_SET)
        self.valid_set = get_set(self.dataset_name, set_str=PathRLenv.VALID_SET)
        self.test_set = get_set(self.dataset_name, set_str=PathRLenv.TEST_SET)
        self.uids = list(self.train_set.keys())
        # Load user negatives (all interactions not in train)
        self.user_negatives = self.__load_user_negatives()
        # Interactions occurring in train, val, test, each in separate groups
        self.user_positives = self.__load_user_positives()
        self.id_to_uid_token_map = {self.tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in self.uids}        
    
    def __load_user_negatives(self):
        #user_negatives = get_user_negatives_tokens_ids(self.dataset_name, self.tokenizer)
        user_negatives = get_user_negatives(self.dataset_name)
        return user_negatives
    
    def __load_user_positives(self):
        user_positives = dict()
        user_positives[PathRLenv.TRAIN_SET] = self.train_set
        user_positives[PathRLenv.VALID_SET] = self.valid_set
        user_positives[PathRLenv.TEST_SET] = self.test_set
        return user_positives
'''
    