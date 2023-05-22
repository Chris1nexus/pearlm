import argparse
import os
from typing import List

import numpy as np
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import check_dir
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from pathlm.models.lm.lm_utils import TOKENIZER_DIR

def tokenize_function(examples: str):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)

def get_training_corpus():
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        yield samples["path"]

def group_texts(examples: List[str], block_size=256):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def stratified_sampling(dataset, valid_size: float=0.05):
    # Extract user_ids
    uid_to_idxs = {}
    for idx, path in enumerate(dataset['path']):
        uid = path.split(' ')[0]
        if uid not in uid_to_idxs:
            uid_to_idxs[uid] = []
        uid_to_idxs[uid].append(idx)

    # Create indices for stratified split
    train_indices, test_indices = [], []

    for uid, idxs in uid_to_idxs.items():
        np.random.shuffle(idxs)  # randomize user specific indices

        split_point = int(len(idxs) * valid_size)  # calculate split point

        # Append to the respective lists
        test_indices.extend(idxs[:split_point])
        train_indices.extend(idxs[split_point:])

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset.select(train_indices),
        'test': dataset.select(test_indices),
    })
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model", type=str, default="roberta-large",
                        help="Model to load tokenizer from if train_from_old is True")
    parser.add_argument("--train_from_old", type=bool, default=False, help="Whether you are creating a new tokenizer"
                                                                           "or you are training from an old one for fine-tuning"
                                                                           "and existing model")
    parser.add_argument("--context_length", type=int, default=32,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducibility")
    parser.add_argument("--nproc", type=int, default=4, help="Number of processes for dataset mapping")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.data

    """
    Load dataset and process paths
    """
    data_dir = f"data/{dataset_name}"
    plain_text_path = False

    print("Loading and processing path sequences...")
    dataset = PathDataset(dataset_name, data_dir, plain_text_path=plain_text_path)
    if plain_text_path:
        dataset.dataset = dataset.dataset.map(
            lambda x: {"path": [dataset.convert_numeric_path_to_textual_path(elem) for elem in x["path"]]},
            batched=True, num_proc=args.nproc)
    else:
        dataset.dataset = dataset.dataset.map(
            lambda x: {"path": [dataset.keep_numeric(elem) for elem in x["path"]]},
            batched=True, num_proc=args.nproc)
    dataset.show_random_examples()
    dataset = dataset.dataset
    # Load the specified tokenizer
    print("Train/Validation split...")
    dataset_split = stratified_sampling(dataset, 0.05)

    tokenizer_dir = f'{TOKENIZER_DIR}/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)

    """
    Create/Load tokenizer
    """
    # Handle creating a new tokenizer or training from an old one
    if args.train_from_old == False:
        TOKENIZER_TYPE = "WordLevel"
        # Handle just WordLevel from scratch since it is the only one that needs to be trained from scratch
        tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
        context_length = args.context_length

        # Word level tokenizer
        print("Creating new tokenizer...")
        tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
        print("Training tokenizer...")
        tokenizer.train_from_iterator(dataset["path"], trainer=trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[EOS]:0 $A:0 [BOS]:0",
            special_tokens=[("[EOS]", tokenizer.token_to_id("[EOS]")), ("[BOS]", tokenizer.token_to_id("[BOS]"))]
        )

        tokenizer.save(tokenizer_file)
        print("Tokenizer saved to disk")

        # Tokenizer and tokenization function
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)
        print("Tokenizing dataset...")
        tokenized_dataset = dataset_split.map(tokenize_function,
                                              batched=True,
                                              num_proc=args.nproc,
                                              remove_columns=["path"]
                                              )
        # Group texts into chunks of block_size tokens
        tokenized_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=args.nproc,
        )
        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{model_name}/from_scratch_tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
        print("Tokenized dataset saved to disk")

    # If we are using a tokenizer that has already been trained
    else:
        TOKENIZER_TYPE = "BPE"
        #Load pretrained tokenizer from huggingface
        print("Loading pretrained tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print("Training tokenizer...")
        tokenizer = tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=tokenizer.vocab_size)
        tokenizer.save(f"./tokenizers/{dataset_name}/ft-{model_name}-{TOKENIZER_TYPE}.json")
        print("Tokenizer saved to disk")

        print("Tokenizing dataset...")
        tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=4, remove_columns=["path"])

        # Group texts into chunks of block_size tokens
        tokenized_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{model_name}/tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(f"data/{dataset_name}/{model_name}/tokenized_dataset.hf")
        print("Tokenized dataset saved to disk")

