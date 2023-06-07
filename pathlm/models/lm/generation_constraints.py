import numpy as np
import torch
from collections import defaultdict

from transformers import LogitsProcessor


"""
Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage this means
to limit the hop size. This is a word-level constraint, does not work with piece tokenizers.
"""
class ForceLastTokenLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, force_tokens, total_length, **kwargs):
        super().__init__(**kwargs)
        self.force_tokens = force_tokens
        self.total_length = total_length
        self.used_tokens = []


    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.total_length-1:
            #Compute min score in scores tensor

            mask = np.isin(range(scores.shape[-1]), self.force_tokens)
            used_mask = np.isin(range(scores.shape[-1]), self.used_tokens)
            mask = mask & ~used_mask  # Remove already used tokens from the mask
            # Set to the smallest representable number
            scores[:, ~mask] = float('-Inf')
        return scores

    def process_finished_sequence(self, input_ids):
        # Remember the last token of the finished sequence
        self.used_tokens.append(input_ids[-1].item())


class TypifiedForceLastTokenLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, force_token_map, total_length, tokenizer, num_return_sequences, id_to_uid_token_map, **kwargs):
        super().__init__(**kwargs)
        self.force_token_map = force_token_map
        self.total_length = total_length
        self.tokenizer = tokenizer
        self.used_tokens = defaultdict(list)
        self.num_return_sequences = num_return_sequences
        self.id_to_uid_token_map = id_to_uid_token_map
        self.call_counter_to_process_finished_sequence = 0

    def __call__(self, input_ids, scores):
        #print(input_ids)
        #print(input_ids.shape)
        
        cur_len = input_ids.shape[-1]
        #print(input_ids.shape, self.__decode(input_ids[0]))
        #print(input_ids.shape, scores.shape)
        if cur_len == self.total_length-1:

            #print(input_ids[0])
            #print(scores.shape)
            force_tokens = None
            uid = None
            user_tokens = None
            UID_POS = 1 
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
        return scores

    def __decode(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)


    def process_finished_sequence(self, input_ids):
        # Remember the last token of the finished sequence
        self.call_counter_to_process_finished_sequence += 1
        self.used_tokens.append(input_ids[-1].item())
"""
Force the last token to be one of the force_tokens if the total length is reached, in the path generation stage this means
to limit the hop size. This is a BPE-level constraint, works with piece tokenizers.
"""
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