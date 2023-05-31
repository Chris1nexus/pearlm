import argparse
import math
import os
from typing import List

import numpy as np
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, EncoderDecoderModel, EncoderDecoderConfig,
    set_seed
from pathlm.models.lm.evaluate import evaluate
from pathlm.models.lm.lm_utils import MLM_MODELS, TOKENIZER_DIR
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import check_dir, SEED
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)



# Read an example and return the tokenized version


def train_encoder_decoder_two_stage(tokenizer, tokenized_dataset, args: argparse.Namespace):
    pass

def train_encoder_decoder_one_stage(tokenizer, tokenized_dataset, args: argparse.Namespace):
    context_length = args.context_length
    encoder_name = args.encoder
    decoder_name = args.decoder

    # Initializing the selected model style configuration
    encoder_config = AutoConfig.from_pretrained(
        encoder_name,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoder_config = AutoConfig.from_pretrained(
        decoder_name,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Initializing a model from the configuration
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    config.decoder_start_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    model = EncoderDecoderModel(config=config)
    # Training arguments
    custom_name = f"from_scratch-{args.data}-{args.encoder}-{args.decoder}"

    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"encoder-decoder-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=10000,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=True,
        # use_mps_device=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=1000,  # number of warmup steps for learning rate
        save_steps=10000,
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=SEED,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.data}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.data}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--encoder", type=str, default="roberta-base", help="")
    parser.add_argument("--decoder", type=str, default="distilgpt2", help="")
    parser.add_argument("--one_stage", type=bool, default=True, help="")
    parser.add_argument("--context_length", type=int, default=32,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--nproc", type=int, default=4, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Test batch size")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    args = parser.parse_args()

    set_seed(SEED)
    TOKENIZER_TYPE = "WordLevel"
    model_name = "encoder-decoder"
    dataset_name = args.data

    tokenizer_dir = f'{TOKENIZER_DIR}/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    tokenized_dataset = load_from_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    # Train the model
    if args.load_model:
        # Training arguments
        custom_name = f"from_scratch-{args.data}-{args.encoder_config}-{args.decoder_config}"
        model = AutoModelForCausalLM.from_pretrained(f"models-weights/{dataset_name}/{model_name}/{custom_name}")
    else:
        model = train_encoder_decoder_one_stage(tokenizer, tokenized_dataset, args)
    evaluate(model, args)




