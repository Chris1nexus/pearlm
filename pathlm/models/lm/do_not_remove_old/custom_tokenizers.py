import os
import argparse
from datasets import load_from_disk
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import LlamaTokenizer


def main(args: argparse.Namespace) -> None:
    def get_training_corpus():
        for i in range(0, len(dataset), 1000):
            yield dataset[i: i + 1000]['text']

    # Check if directory exists otherwise create it
    if not os.path.exists(f'./{args.tokenizer}tokenizer'):
        os.makedirs(f'./{args.tokenizer}tokenizer')

    # Load dataset
    dataset = load_from_disk(f'data/{args.data}/hf_dataset.hf')
    # Create tokenizer object based on args
    if args.tokenizer == "BPE":
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.pre_tokenizer = BertPreTokenizer()
        # Customize training
        tokenizer.train_from_iterator(get_training_corpus(), vocab_size=25_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        tokenizer.save_model(f'./{args.tokenizer}tokenizer')

    elif args.tokenizer == "LLama":
        #Create a tokenizer object for Llama
        #tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        # Customize training
        tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=25_000, min_frequency=2, special_tokens_map={
                                          "bos_token": "<s>",
                                          "pad_token": "<pad>",
                                          "eos_token": "</s>",
                                          "unk_token": "<unk>"
        })
        tokenizer.save_pretrained(f'./{args.tokenizer}tokenizer')


#Main function that reads parameter
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--tokenizer", type=str, default="LLama", help="Tokenizer to use")
    args = parser.parse_args()
    main(args)