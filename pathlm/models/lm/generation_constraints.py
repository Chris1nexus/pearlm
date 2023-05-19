import numpy as np
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

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.total_length:
            mask = np.isin(range(scores.shape[-1]), self.force_tokens)
            scores[:, ~mask] = -float('Inf')
        return scores

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