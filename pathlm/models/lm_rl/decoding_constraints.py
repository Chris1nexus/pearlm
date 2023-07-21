import torch
import numpy as np
from collections import defaultdict
import math
from transformers import LogitsProcessor



class RLConstrainedLogitsProcessorWordLevel(LogitsProcessor):
    def __init__(self, tokenized_kg, force_token_map, total_length, tokenizer, num_return_sequences,
                 id_to_uid_token_map, eos_token_ids, vocab_size=None,  **kwargs):
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
        if vocab_size is None: 
            self.vocab_size = len(self.tokenizer.get_vocab())
        else:
            self.vocab_size = vocab_size
        self.vocab_tokens = [i for i in range(self.vocab_size)]
        self.cache = dict()
        self.mask_cache = dict()

    def __call__(self, input_ids, scores, is_inference=False):
        cur_len = input_ids.shape[-1]
        min_score = scores.min()
        if cur_len == self.total_length:
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i not in self.eos_token_ids]] = -math.inf
            for i in self.eos_token_ids:
                scores[:, i] = 0.
        else:
            #mask_list = []
            mask = torch.full_like(scores, -math.inf)
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
                    if cur_len == self.total_length - 1 and is_inference: # Remove from candidates products not in user negatives
                        uid_cond_key = cur_uid,*key
                        self.cache[uid_cond_key] = list(set(self.cache[key]).intersection(set(self.force_token_map[cur_uid])))
                        key = uid_cond_key
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
                mask[idx, candidate_tokens] = 0.
                if len(candidate_tokens) == 0:
                    mask[:, self.tokenizer.pad_token_id] = 0.                    
                
                #if key not in self.mask_cache:
                #    self.mask_cache[key] = np.isin(self.vocab_tokens, candidate_tokens)
                #mask = self.mask_cache[key]
                #mask_list.append(mask)                
                
            scores = scores + mask
            #mask = torch.BoolTensor(np.vstack(mask_list))
            #scores[~mask] = min_score
            
            
            #Set to min score the tokens which are not in force_tokens_map[cur_uid] at this position
        return scores

    def __decode(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)
