import numpy as np
import torch
from collections import defaultdict

from transformers import LogitsProcessor

"""
Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage this means
to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
"""
class ConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, tokenized_kg, force_token_map, total_length, tokenizer, num_return_sequences,
                 id_to_uid_token_map, eos_token_ids, **kwargs):
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

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        min_score = scores.min()
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_ids]] = min_score  # float("-Inf")
            for i in self.eos_token_ids:
                scores[:, i] = 1.
        else:
            mask_list = []
            for idx in range(scores.shape[0]):
                if cur_len % 2 == 1:
                    # parse ent -----> candidate relations
                    cur_uid = self.id_to_uid_token_map[input_ids[idx, 1].item()]
                    k1 = input_ids[idx, -2].item()
                    k2 = input_ids[idx, -1].item()
                    key = k1, k2
                    if key not in self.cache:
                        if k1 in self.kg and k2 in self.kg[k1]:
                            self.cache[key] = list(self.kg[k1][k2])
                        else:
                            self.cache[key] = []
                    if cur_len == self.total_length - 1: # Remove from candidates products not in user negatives
                        self.cache[key] = list(set(self.cache[key]).intersection(set(self.force_token_map[cur_uid])))
                else:
                    # parse ent->rel    -----> candidates
                    k1 = input_ids[idx, -1].item()
                    key = k1
                    if key not in self.cache:
                        if k1 in self.kg:
                            self.cache[key] = list(self.kg[k1].keys())
                        else:
                            self.cache[key] = []

                candidate_tokens = self.cache[key]
                if key not in self.mask_cache:
                    self.mask_cache[key] = np.isin(self.vocab_tokens, candidate_tokens)
                mask = self.mask_cache[key]
                mask_list.append(mask)

            mask = np.vstack(mask_list)
            scores[~mask] = min_score
            #Set to min score the tokens which are not in force_tokens_map[cur_uid] at this position
        return scores

    def __decode(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)

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