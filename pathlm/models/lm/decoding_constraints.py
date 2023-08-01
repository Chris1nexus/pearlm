import numpy as np
import torch
from collections import defaultdict
import math

from transformers import LogitsProcessor

from pathlm.models.lm.decoding_cache import LFUCache, LRUCache

"""
Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage this means
to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
"""
class ConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, tokenized_kg, force_token_map, total_length, tokenizer, num_return_sequences,
                 id_to_uid_token_map, eos_token_ids, mask_cache_size=3*10**4, cand_cache_size=1*10**5, **kwargs):
        super().__init__(**kwargs)
        self.kg = tokenized_kg
        self.force_token_map = force_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.id_to_uid_token_map = id_to_uid_token_map
        self.call_counter_to_process_finished_sequence = 0
        self.eos_token_ids = eos_token_ids
        self.vocab_tokens = [i for i in range(len(self.tokenizer.get_vocab()))]  
        self.cache = LFUCache(cand_cache_size)#dict()
        #self.mask_cache = dict()
        self.mask_cache = LFUCache(mask_cache_size)


    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        min_score = float("-inf")
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:,:] = -math.inf   #scores[:,[i for i in range(num_tokens) if i not in self.eos_token_ids] ] # float("-Inf") #min_score  # float("-Inf")
            for i in self.eos_token_ids: 
                scores[:, i] = 0.
        else:
            mask_list = []
            def init_mask(vocab_size, candidate_tokens):
                    banned_mask = torch.ones(vocab_size, dtype=torch.bool)
                    banned_mask[candidate_tokens] = False   
                    return banned_mask             
            def convert_iterable(iterable):
                #return torch.LongTensor(list(iterable) ).to(scores.device)
                return list(iterable)
            def cache_ent_rel_cand(key,k1,k2):
                    if k1 in self.kg and k2 in self.kg[k1]:
                        self.cache.put(key,  convert_iterable(self.kg[k1][k2]) ) 
                    else:
                        self.cache.put(key, convert_iterable([]) ) 
            def cache_rel_cand(key,k1):
                    if k1 in self.kg:
                        self.cache.put(key, convert_iterable(self.kg[k1].keys()) )
                    else:
                        self.cache.put(key, convert_iterable([]) )                                                 
            #masked_scores = torch.full_like(scores, -math.inf).to(scores.device)
            for idx in range(scores.shape[0]):
                cur_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
                
                candidate_tokens = None
                if cur_len % 2 == 1:
                    # parse ent -----> candidate relations
                    k1 = input_ids[idx, -2].item()
                    k2 = input_ids[idx, -1].item()
                    key = k1, k2
                    
                    candidate_tokens = self.cache.get(key)
                    if candidate_tokens is None:
                        cache_ent_rel_cand(key, k1, k2)
                    if cur_len == self.total_length - 1: # Remove from candidates products not in user negatives
                        uid_cond_key = cur_uid, *key
                        candidate_tokens = self.cache.get(uid_cond_key)
                        if candidate_tokens is None:
                            candidates = self.cache.get(key)
                            if candidates is None:
                                cache_ent_rel_cand(key, k1, k2)
                            self.cache.put(uid_cond_key, convert_iterable(
                                set(candidates).intersection(set(self.force_token_map[cur_uid]))
                                )
                            ) 
                        
                        key = uid_cond_key
                        #self.cache[key] = list(
                        #    set(self.cache[key]).intersection(set(self.force_token_map[cur_uid])))
                else:
                    # parse ent->rel    -----> candidates
                    k1 = input_ids[idx, -1].item()
                    key = k1
                    candidate_tokens = self.cache.get(key)
                    if candidate_tokens is None:
                        cache_rel_cand(key, k1)

                candidate_tokens = self.cache.get(key)
                #masked_scores[idx, candidate_tokens] = scores[idx, candidate_tokens] #.scatter_(dim=-1, index=candidate_tokens, src=scores[idx] )
                #scores[idx].index_fill_(-1, torch.LongTensor(candidate_tokens), -math.inf)
                #'''
                #if key not in self.mask_cache:
                banned_mask = self.mask_cache.get(key)
                if banned_mask is None:
                    banned_mask = np.isin(self.vocab_tokens, candidate_tokens, invert=True) #init_mask(len(self.vocab_tokens), candidate_tokens)
                    #self.mask_cache[key] = banned_mask #np.isin(self.vocab_tokens, candidate_tokens)
                    self.mask_cache.put(key, banned_mask)
                #banned_mask = self.mask_cache[key]
                #scores[idx, mask] = -math.inf  
                mask_list.append(banned_mask)
                #'''
            #scores = masked_scores
            #'''
            #mask = np.vstack(mask_list)
            #scores[~mask] = min_score
            banned_tokens_mask = np.vstack(mask_list)#.to(scores.device)
            scores[banned_tokens_mask] = -math.inf   
            #'''
            
            #print(scores.device, banned_tokens_mask.device)
            #scores[banned_tokens_mask] = -math.inf          
            #Set to min score the tokens which are not in force_tokens_map[cur_uid] at this position
        return scores


class PrefixConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, tokenized_kg, force_token_map, total_length, tokenizer, num_return_sequences,
                 id_to_uid_token_map, eos_token_ids, mask_cache_size=3*10**4, cand_cache_size=1*10**5, **kwargs):
        super().__init__(**kwargs)
        self.kg = tokenized_kg
        self.force_token_map = force_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.id_to_uid_token_map = id_to_uid_token_map
        self.call_counter_to_process_finished_sequence = 0
        self.eos_token_ids = eos_token_ids
        self.vocab_tokens = [i for i in range(len(self.tokenizer.get_vocab()))]
        self.cache = dict()
        self.mask_cache = dict()
        self.cache = LFUCache(cand_cache_size)#dict()
        #self.mask_cache = dict()
        self.mask_cache = LFUCache(mask_cache_size)
    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_ids]] = float("-Inf")
            for i in self.eos_token_ids:
                scores[:, i] = 0.  
        else:
            #mask = torch.full_like(scores, -math.inf)
            masked_scores = torch.full_like(scores, -math.inf)
            indices = []
            for idx in range(scores.shape[0]):
                if cur_len % 2 == 1:
                    # parse ent -----> candidate relations
                    cur_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
                    k1 = input_ids[idx, -2].item()
                    k2 = input_ids[idx, -1].item()
                    key = k1, k2
                    if key not in self.cache:
                        if k1 in self.kg and k2 in self.kg[k1]:
                            self.cache[key] = torch.LongTensor(list(self.kg[k1][k2]) )
                        else:
                            self.cache[key] = torch.LongTensor([])
                    if cur_len == self.total_length - 1: # Remove from candidates products not in user negatives

                        uid_cond_key = cur_uid, *key
                        self.cache[uid_cond_key] = torch.LongTensor(list(
                            set(self.cache[key]).intersection(set(self.force_token_map[cur_uid])))
                        )
                        key = uid_cond_key

                else:
                    # parse ent->rel    -----> candidates
                    k1 = input_ids[idx, -1].item()
                    key = k1
                    if key not in self.cache:
                        if k1 in self.kg:
                            self.cache[key] = torch.LongTensor(list(self.kg[k1].keys()) )
                        else:
                            self.cache[key] = torch.LongTensor([])

                candidate_tokens = self.cache[key].to(scores.device)
                indices.append(candidate_tokens)
                masked_scores[idx].scatter_(dim=-1, index=candidate_tokens, src=scores[idx] )
                #print('s', masked_scores[idx, candidate_tokens])
                #print(candidate_tokens)
                #mask[idx, candidate_tokens] = 0.
            #masked_scores.scatter(dim=1, index=torch.vstack(indices).to(scores.device), src=scores)
            #scores = scores + mask
            
            scores = masked_scores 
            #Set to min score the tokens which are not in force_tokens_map[cur_uid] at this position
        return scores


class PLMLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, tokenized_kg, force_token_map, total_length, tokenizer, num_return_sequences,
                 id_to_uid_token_map, eos_token_ids, ent_mask, rel_mask, token_id_to_token,  **kwargs):
        super().__init__(**kwargs)
        self.ent_ids = [idx for idx, elem in enumerate(ent_mask) if elem > 0]
        self.rel_ids = [idx for idx, elem in enumerate(rel_mask) if elem > 0]
        

        self.ent_mask = [elem > 0 for idx, elem in enumerate(ent_mask) ]
        self.rel_mask = [elem > 0 for idx, elem in enumerate(rel_mask) ]
        
        self.token_id_to_token = token_id_to_token
        self.kg = tokenized_kg
        self.force_token_map = force_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.id_to_uid_token_map = id_to_uid_token_map
        self.call_counter_to_process_finished_sequence = 0
        self.eos_token_ids = eos_token_ids
        self.vocab_tokens = [i for i in range(len(self.tokenizer.get_vocab()))]


    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:,:] = -math.inf   #scores[:,[i for i in range(num_tokens) if i not in self.eos_token_ids] ] # float("-Inf") #min_score  # float("-Inf")
            for i in self.eos_token_ids: 
                scores[:, i] = 0.
        else:
            mask = torch.full_like(scores, -math.inf)
            for idx in range(scores.shape[0]):
                cur_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
                if cur_len % 2 == 1:
                        # parse ent -----> candidate relations
                        candidate_tokens = self.ent_ids
                        if cur_len == self.total_length - 1: # Remove from candidates products not in user negatives
                            candidate_tokens = self.force_token_map[cur_uid]
                        key = cur_uid,idx
                else:
                    candidate_tokens = self.rel_ids
                    key = idx

                #if key not in self.mask_cache:
                #    mask = np.isin(self.vocab_tokens, candidate_tokens)
                #    self.mask_cache[key] = mask

                #candidate_tokens = self.mask_cache[key]
                #mask[idx, candidate_tokens] = 0.
                mask[idx, candidate_tokens] = scores[idx, candidate_tokens]
            #scores = scores + mask
        '''
        min_score = scores.min()
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_ids]] = min_score  # float("-Inf")
            for i in self.eos_token_ids:
                scores[:, i] = 0.
        else:
            mask_list = []
            
            for idx in range(scores.shape[0]):
                cur_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
                if cur_len % 2 == 1:
                    # parse ent -----> candidate relations
                    candidate_tokens = self.ent_ids
                    if cur_len == self.total_length - 1: # Remove from candidates products not in user negatives
                        candidate_tokens = self.force_token_map[cur_uid]
                    key = cur_uid,idx

                else:
                    candidate_tokens = self.rel_ids
                    key = idx
                if key not in self.mask_cache:
                    mask = np.isin(self.vocab_tokens, candidate_tokens)
                    self.mask_cache[key] = mask
                mask = self.mask_cache[key]
                mask_list.append(mask)

            mask = np.vstack(mask_list)
            scores[~mask] = min_score
        '''
        #Set to min score the tokens which are not in force_tokens_map[cur_uid] at this position
        return scores
    

"""

class TypifiedForceLastTokenLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, force_token_map, total_length, tokenizer, num_return_sequences, id_to_uid_token_map,
                 eos_token_ids, **kwargs):
        super().__init__(**kwargs)
        self.force_token_map = force_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.id_to_uid_token_map = id_to_uid_token_map
        self.call_counter_to_process_finished_sequence = 0
        self.eos_token_ids = eos_token_ids

    def __call__(self, input_ids, scores):
        # print(input_ids)
        # print(input_ids.shape)
        # print(input_ids.shape)
        cur_len = input_ids.shape[-1]
        # print(input_ids.shape, self.__decode(input_ids[0]))
        # print(input_ids.shape, scores.shape)
        # print(cur_len, input_ids[0])
        '''
        BLOCK = True
        if cur_len == self.total_length-2 and not BLOCK:

            #print(input_ids[0])
            #print(scores.shape)
            force_tokens = None
            uid = None
            user_tokens = None
            UID_POS = 0
            #print()
            min_score = scores.min()
            for idx in range(scores.shape[0]):
                #if idx % self.num_return_sequences == 0:
                #    #user_tokens = self.__decode([input_ids[idx,UID_POS]])
                #    #user_tokens[-1][1:]
                #     self.force_token_map[uid]   
                uid = self.id_to_uid_token_map[input_ids[idx,UID_POS].item()] # user_tokens[-1][1:]
                force_tokens = self.force_token_map[uid]                

                #Compute min score in scores tensor
                mask = np.isin(range(scores.shape[-1]), force_tokens)
                used_mask = np.isin(range(scores.shape[-1]), self.used_tokens)
                #print(used_mask.sum(), used_mask.shape)
                mask = mask & ~used_mask  # Remove already used tokens from the mask
                # Set to the smallest representable number
                scores[idx, ~mask] = min_score#float('-Inf')
        elif cur_len == self.total_length-1:
            if not BLOCK:
                num_tokens = scores.shape[1]
                scores[:, [i for i in range(num_tokens) if i not in self.eos_token_ids]] = -float("inf")
                for i in self.eos_token_ids:
                    scores[:, i] = 0     

            #for seq in input_ids:   
            #    print(self.__decode(seq))
            #print()
        '''
        return scores

    def __decode(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def process_finished_sequence(self, input_ids):
        # Remember the last token of the finished sequence
        self.call_counter_to_process_finished_sequence += 1
        self.used_tokens.append(input_ids[-1].item())

class ForceTokenAtWordPositionLogitsProcessorBPE(LogitsProcessor):
    def __init__(self, tokenizer, force_tokens, word_position, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.force_tokens = force_tokens
        self.word_position = word_position

    def __call__(self, input_ids, scores):
        decoded_text = self.tokenizer.decode(input_ids[0])
        word_count = decoded_text.count('<end_word>')
        if word_count == self.word_position - 1:  # Subtract 1 because word count is 0-based
            mask = np.isin(range(scores.shape[-1]), self.force_tokens)
            scores[:, ~mask] = -float('Inf')
        return scores
        
"""