import argparse
import os
import pickle
from collections import defaultdict
from typing import List, Dict
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from datasets import load_from_disk, DatasetDict, Dataset
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,\
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, LogitsProcessorList,\
    set_seed, GPT2LMHeadModel, GPT2Model, EarlyStoppingCallback
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import is_torch_tpu_available
import pandas as pd
from os.path import join

from pathlm.models.lm.plmrec import PLMRec
from pathlm.models.lm.perlm import PERLM 
from pathlm.models.lm.decoding_constraints import ConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids, \
                    _initialise_type_masks, \
                    _initialise_weights_from_kge, \
                    tokenize_augmented_kg, \
                    get_user_negatives
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.tools.mapper import EmbeddingMapper
from pathlm.utils import SEED, get_pid_to_eid, get_set, check_dir


from multiprocessing import Pool
from datasets import Dataset
from os import listdir
from os.path import isfile, join
import pandas as pd
import math
from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map
from pathlm.models.lm.trainer import PathCLMTrainer












# Read an example and return the tokenized version
def tokenize_function(examples: str, context_length: int = 200):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)


def fine_tune(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
         'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    pretrain_model = f'./clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}/checkpoint-{args.pretrain_ckpt}'
    # Initializing the selected model style configuration
    config = AutoConfig.from_pretrained(
        f'clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}/checkpoint-{args.pretrain_ckpt}',
        **config_kwargs
    )

    print('Model config: ', config)
    config.update({'num_hops': args.n_hop,
                   'sample_size_pretrain': args.sample_size_pretrain,
                   'sample_size_finetune': args.sample_size_finetune,
                   'sample_size_hop': args.sample_size_hop,
                   'task': args.task,
                   'train_batch_size': args.batch_size,
                   'test_batch_size': args.infer_batch_size,
                   'ent_mask': ent_mask,
                   'rel_mask': rel_mask,
                   'token_id_to_token': token_id_to_token,
                   })

    # Initializing a model from the configuration
    if args.task == 'finetune' and args.pretrain_ckpt is not None:
        print('Loading from checkpoint for finetuning: ', args.pretrain_ckpt)
        model = PERLM.from_pretrained(pretrain_model,
                                              config=config)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)

    # Training arguments
    custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.n_beams}-{args.n_seq_infer}-{args.logit_processor_type}"

    STEP_INTERVAL = 50
    EVAL_STEP_INTERVAL = 500

    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=EVAL_STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=3e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,  # number of warmup steps for learning rate
        save_steps=EVAL_STEP_INTERVAL,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        #no_cuda=True,
        seed=SEED,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = PathCLMTrainer(
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        n_beams=args.n_beams,        
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        experiment_name=custom_name,
        data_collator=data_collator)

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


def train_end_to_end(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
        #'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    embeds=None

    # Initializing a model from the configuration
    if args.continue_training and args.pretrain_ckpt is not None:
        pretrain_model = f'./clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.logit_processor_type}/checkpoint-{args.pretrain_ckpt}'
        config = AutoConfig.from_pretrained(
            pretrain_model,
            **config_kwargs
        )
        print('Loading from checkpoint for continuing traning: ', args.pretrain_ckpt)
        model = PERLM.from_pretrained(pretrain_model,
                                              config=config)
        print(model.config)
    else:
        print('TRAINING NEW MODEL')
        if 'plm-rec' in model_name:
            model_class, model_subname = model_name.split('@')
            if len(args.emb_filename) > 0:
                    embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
                    try:
                        embeds = pickle.load(open(embed_filepath, 'rb'))
                    except:
                        embeds = None
                    if embeds:
                        print('Using embeddings: ',args.emb_filename)
                        config_kwargs.update({
                        'hidden_size':int(args.emb_size),
                        'num_attention_heads':int(args.emb_size)//10
                        })

            config = AutoConfig.from_pretrained(
                model_class,
                **config_kwargs
            )

            model_cls = PLMRec
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                **config_kwargs
            )

            model_cls = PERLM
    print('Model config: ', config)
    config.update({'num_hops': args.n_hop,
                   'sample_size_pretrain': args.sample_size_pretrain,
                   'sample_size_finetune': args.sample_size_finetune,
                   'sample_size_hop': args.sample_size_hop,
                   'task': args.task,
                   'train_batch_size': args.batch_size,
                   'test_batch_size': args.infer_batch_size,
                   'ent_mask': ent_mask,
                   'rel_mask': rel_mask,
                   'token_id_to_token': token_id_to_token})

    model = model_cls(config)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)

    if embeds:
        mapper = EmbeddingMapper(tokenizer, kg, embeds)      
        mapper.init_with_embedding(model.transformer.wte.weight)
        print(f'Model {model_name} initialized with custom embeddings of size: ', model.transformer.wte.weight.shape)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)

    # Training arguments
    #custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.n_beams}-{args.n_seq_infer}-{args.logit_processor_type}"
    custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.logit_processor_type}"


    STEP_INTERVAL = 100
    EVAL_STEP_INTERVAL = 1000
    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=EVAL_STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=2e-4,
        weight_decay=0.01,
        bf16=False,
        fp16=True,#True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,  # number of warmup steps for learning rate
        save_steps=EVAL_STEP_INTERVAL,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,
        #no_cuda=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = PathCLMTrainer(
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        n_beams=args.n_beams,         
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        experiment_name=custom_name,
        logit_processor_type=args.logit_processor_type,
        data_collator=data_collator)

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


def train_pretraining(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)

    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
        # 'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    pretrain_model = f'./clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}-{args.logit_processor_type}/checkpoint-{args.pretrain_ckpt}'
    config = AutoConfig.from_pretrained(
        pretrain_model,
        **config_kwargs
    )

    # Initializing a model from the configuration
    if args.continue_training and args.pretrain_ckpt is not None:
        print('Loading from checkpoint for continuing traning: ', args.pretrain_ckpt)
        model = PERLM.from_pretrained(pretrain_model,
                                              config=config)
        print(model.config)
    else:
        print('TRAINING NEW MODEL')
        config = AutoConfig.from_pretrained(
            model_name,
            **config_kwargs
        )
        model = PERLM(config)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = f'{dirpath}/mapping/'
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)
    try:
        embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
        embeds = pickle.load(open(embed_filepath, 'rb'))
        print('TRANSE LOADED')
        model, config = _initialise_weights_from_kge(embeds, tokenizer, kg, config_kwargs, model, args)
    except:
        pass

    print('Model config: ', config)
    config.update({'num_hops': args.n_hop,
                   'sample_size_pretrain': args.sample_size_pretrain,
                   'sample_size_finetune': args.sample_size_finetune,
                   'sample_size_hop': args.sample_size_hop,
                   'task': args.task,
                   'train_batch_size': args.batch_size,
                   'test_batch_size': args.infer_batch_size,
                   'ent_mask': ent_mask,
                   'rel_mask': rel_mask,
                   'token_id_to_token': token_id_to_token})

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)

    # Training arguments
    custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}"

    STEP_INTERVAL = 100
    EVAL_STEP_INTERVAL = 1000
    training_args = TrainingArguments(
        f"clm-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=EVAL_STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=2e-4,#3e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=20,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,  # number of warmup steps for learning rate
        save_steps=EVAL_STEP_INTERVAL,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        load_best_model_at_end=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--task", type=str, default="end-to-end", help="{pretrain, finetune, end-to-end}")
    parser.add_argument("--sample_size_pretrain", type=str, default="1500",
                        help="Which sample size dataset to use for pretraining")
    parser.add_argument("--sample_size_finetune", type=str, default="500",
                        help="Which sample size dataset to use for fine-tuning/end-to-end")
    parser.add_argument("--sample_size_hop", type=str, default="3",
                        help="Which number of hops dataset to use for fine-tuning/end-to-end")
    parser.add_argument("--logit_processor_type", type=str, default="gcd",
                        help="Path sequence deconding method: default to Graph Constrained Decoding")    


    # Model arguments
    parser.add_argument("--model", type=str, default="gpt2-large",
                        help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--nproc", type=int, default=8, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=256, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="Test batch size")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")
    parser.add_argument("--load_data", type=bool, default=False, help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default='1', help="")
    parser.add_argument("--infer_batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=30,
                        help="Number of sequences generated for each user")
    parser.add_argument("--n_beams", type=int, default=30,
                        help="Number of sequences generated for each user")    

    # Parameter relative to resume training
    parser.add_argument("--continue_training", type=bool, default=False,
                        help="Whether to continue training from a checkpoint or not")
    parser.add_argument("--pretrain_ckpt", type=none_or_int, default='8500',
                        help="Checkpoint from which to resume training of the model (default to starting from scratch)")
    # Parameter relative to weight initialization
    parser.add_argument("--embedding_root_dir", type=str, default="./embedding-weights",
                        help="default: ./embedding-weights")
    parser.add_argument("--emb_filename", type=str, default='', help="default: 'transe_embed.pkl'")
    parser.add_argument("--emb_size", type=int, default=100,
                        help="Transformer Embedding size (must match external embedding size, if chosen)")

    args = parser.parse_args()

    set_seed(SEED)

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    sample_size = args.sample_size_pretrain if args.task == 'pretrain' else args.sample_size_finetune
    dataset_hop_size = args.sample_size_hop

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if args.load_data:
        task = args.task
        if task == 'finetune':  # They share the same dataset
            task = 'end-to-end'
        tokenized_dataset = load_from_disk(
            f"data/{dataset_name}/{TOKENIZER_TYPE}/{task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)
    else:
        # Load the dataset
        data_dir = f"data/{dataset_name}"
        plain_text_path = True

        print("Loading and processing path sequences...")
        dataset = PathDataset(dataset_name, data_dir, task=args.task, sample_size=sample_size, n_hop=dataset_hop_size,
                              plain_text_path=plain_text_path)

        dataset.show_random_examples()
        dataset = dataset.dataset
        print(type(dataset))
        if not os.path.exists(tokenizer_file):
            # Word level tokenizer
            print("Training tokenizer...")
            tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
            special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
            tokenizer.train_from_iterator(dataset["path"], trainer=trainer)
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[BOS]:0 $A:0 [EOS]:0",
                special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]"))]
            )

            tokenizer.save(tokenizer_file)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
                                                eos_token="[EOS]", bos_token="[BOS]",
                                                pad_token="[PAD]", unk_token="[UNK]",
                                                mask_token="[MASK]", use_fast=True)
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                                eos_token="[EOS]", bos_token="[BOS]",
                                                pad_token="[PAD]", unk_token="[UNK]",
                                                mask_token="[MASK]", use_fast=True)

        if args.task == "pretrain":
            # Load the specified tokenizer
            print("Tokenizing dataset...")

            tokenized_dataset = dataset.map(tokenize_function,
                                            batched=True,
                                            num_proc=args.nproc,
                                            remove_columns=["path"]
                                            )
            print("Train/Validation split...")
            tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.10)
            print(tokenized_dataset['train'][0])
        elif args.task == "finetune" or args.task == "end-to-end":
            print("Tokenizing dataset...")
            tokenized_dataset = dataset.map(tokenize_function,
                                            batched=True,
                                            num_proc=args.nproc,
                                            remove_columns=["path"]
                                            )
            tokenized_dataset = DatasetDict({
                "train": tokenized_dataset,
            })

        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(
            f"data/{dataset_name}/{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")

    # Train the model
    if args.load_model: #ENSURE IS WORKING
        # Training arguments
        curr_sample_size = args.sample_size_pretrain if args.task == 'pretrain' else args.sample_size_finetune
        custom_name = f'clm-{args.task}-{args.dataset}-{args.model}-{curr_sample_size}-{args.sample_size_hop}-{args.logit_processor_type}/checkpoint-{args.eval_ckpt_iter}'  # f"clm-from_scratch-{args.dataset}-{args.model}"
        model = AutoModelForCausalLM.from_pretrained(
            custom_name)  
    else:
        if args.task == "pretrain":
            model = train_pretraining(model_name, tokenizer, tokenized_dataset, args.context_length, args)
        elif args.task == "finetune":
            model = fine_tune(model_name, tokenizer, tokenized_dataset, args.context_length, args)
        elif args.task == "end-to-end":
            model = train_end_to_end(model_name, tokenizer, tokenized_dataset, args.context_length, args)
