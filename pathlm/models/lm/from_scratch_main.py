import argparse
import math
import os
from typing import List
import pickle
import random
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm
import multiprocessing as mp
import itertools
import functools
from transformers.utils import is_torch_tpu_available
import torch

import numpy as np
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, PhrasalConstraint, LogitsProcessorList, set_seed, pipeline \

from pathlm.models.lm.generation_constraints import TypifiedForceLastTokenLogitsProcessorWordLevel
from pathlm.models.lm.evaluate import evaluate
from pathlm.models.lm.lm_utils import MLM_MODELS
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import SEED, get_pid_to_eid, get_eid_to_name_map, get_data_dir, get_set, check_dir
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k 




from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)






class CustomTrainer(Trainer):

    def __init__(
        self,
        dataset_name=None,
        n_hop=3,
        infer_batch_size=1,
        n_sequences_per_user=10,
        tokenizer=None,
        eval_device='cpu',
        **kwargs
    ):   
        super().__init__(**kwargs)


        data_dir = f"data/{dataset_name}"
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.n_hop = n_hop
        self.eval_device = eval_device

        self.SEQUENCE_LEN =  2 + 2 + n_hop*2 + (n_hop-1)*2 + 1 # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
        self.LAST_TOKEN_POS = self.SEQUENCE_LEN-1
        self.INFERENCE_BATCH_SIZE = args.infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user
        print('Sequence length: ',self.SEQUENCE_LEN)
        print('Last token position: ',self.LAST_TOKEN_POS)



        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_pid_to_eid(data_dir).values()])
        self.user_negatives = get_user_negatives_tokens_ids(dataset_name, tokenizer)

        #self.generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=eval_device)
        
        topk = defaultdict(list)
        non_product_count = 0

        self.id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in uids}

        #'''

        init_condition_fn = lambda uid: f"Us U{uid} Rf R-1"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in uids] }
        

        
        self.logits_processor = LogitsProcessorList([
            TypifiedForceLastTokenLogitsProcessorWordLevel(force_token_map=self.user_negatives, 
                        tokenizer=tokenizer, 
                        total_length=self.SEQUENCE_LEN,#LAST_TOKEN_POS,
                        num_return_sequences=self.N_SEQUENCES_PER_USER,
                        id_to_uid_token_map=self.id_to_uid_token_map)#6)
            #ForceLastTokenLogitsProcessorWordLevel(user_negatives, tokenizer=tokenizer, total_length=LAST_TOKEN_POS)
            # 7 = 2 input token + 5 generated tokens
        ])

        self.test_dataset = Dataset.from_dict(self.inference_paths)

    def __lazy_load_data(dataset):
            for row in dataset:
                    yield row["uid"]

    def __generate_topks_withWordLevel(self, model):
        self.generator = pipeline('text-generation', model=model, tokenizer=self.tokenizer, device=self.eval_device)

        outputs = self.generator(CustomTrainer.__lazy_load_data(self.test_dataset),#f"Us U{uid} Rf R-1",
                                max_length=self.SEQUENCE_LEN,#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                logits_processor=self.logits_processor,
                                batch_size=self.INFERENCE_BATCH_SIZE,
        )  
        topk = defaultdict(list)
        non_product_count = 0
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)  ) as pbar:
            for output_batch in outputs:
                    for output in output_batch:
                        output = output['generated_text'].split(" ")
                        uid = output[1][1:]

                        recommended_token = output[-1]
                        recommended_item = recommended_token[1:]
                        if len(recommended_token) < 2  or not recommended_token.startswith("P"):

                            non_product_count += 1
                            continue
                        #print(output)
                        topk[uid].append(recommended_item)
                    pbar.update(1)
        print(f"Non product count: {non_product_count}")        
        return topk

    def evaluate(self, model):
        # Generate paths for the test users
        # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok

        topks = self.__generate_topks_withWordLevel(model)
        check_dir(f"./results/{self.dataset_name}/{self.custom_model_name}")
        pickle.dump(topks, open(f"./results/{self.dataset_name}/{self.custom_model_name}/topks.pkl", "wb"))
        metrics = {"ndcg": [], "mmr": [],  }
        for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
            hits = []
            for recommended_item in topk:
                if recommended_item in self.test_set[uid]:
                    hits.append(1)
                else:
                    hits.append(0)
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
            metrics["ndcg"].append(ndcg)
            metrics["mmr"].append(mmr)

        print(f"no of users: {len(self.test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
        metrics_ = dict()
        for k in metrics:
            metrics_[f'eval_{k}'] = np.mean(metrics[k])
        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            '''
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            '''
            metrics = self.evaluate(model.module)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)



# Read an example and return the tokenized version
def tokenize_function(examples: str, context_length: int=100):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)

def group_texts(examples: List[str], block_size=256):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train_from_scratch(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):

    # Initializing the selected model style configuration
    config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Initializing a model from the configuration
    model = AutoModelForCausalLM.from_config(config)

    # Training arguments
    custom_name = f"from_scratch-{args.data}-{args.model}"

    
    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
            evaluation_strategy="steps",
            save_strategy='steps',
        eval_steps=1000,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=False,
        no_cuda=True,
        logging_first_step=True,
        #use_mps_device=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=1000,  # number of warmup steps for learning rate
        save_steps=500,
        save_total_limit=2,
        #load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,
    )


    if model_name in MLM_MODELS:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #trainer = Trainer(
    #    model=model,
    #    args=training_args,
    #    train_dataset=tokenized_dataset["train"],
    #    eval_dataset=tokenized_dataset["test"],
    #    data_collator=data_collator,
    #)
    trainer = CustomTrainer(
        dataset_name=args.data,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        tokenizer=tokenizer,
        eval_device=args.eval_device,
                model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator)


    # Train model
    trainer.train()
    # Evaluate model
    #eval_results = trainer.evaluate()
    #print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Save model
    weight_path = f"./models-weights/{args.data}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.data}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model

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


def add_user_id(example):
    example["user_id"] = example['path'].split(' ')[0]
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--nproc", type=int, default=2, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=24, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=24, help="Test batch size")
    parser.add_argument("--context_length", type=int, default=100,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")    
    parser.add_argument("--load_data", type=bool, default=False, help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default='10000', help="")
    parser.add_argument("--infer_batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=10, help="Number of sequences generated for each user at inference time")
    args = parser.parse_args()


    set_seed(SEED)

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.data

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if args.load_data:
        tokenized_dataset = load_from_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)        
    else:
        # Load the dataset
        data_dir = f"data/{dataset_name}"
        plain_text_path=True

        print("Loading and processing path sequences...")
        dataset = PathDataset(dataset_name, data_dir, plain_text_path=plain_text_path)
        def convert_and_add_uid(paths_dict, convert_fn):
            batch_dict = {"path":[], "user_id":[]}

            paths_list = batch_dict['path']
            user_list = batch_dict['user_id'] 

            for elem in paths_dict["path"]:
                paths_list.append(convert_fn(elem))
                user_list.append(elem.split(' ')[0])
            return batch_dict        
        def convert_typed_path_and_add_uid(paths_dict, convert_fn):
            batch_dict = {"path":[], "user_id":[]}

            paths_list = batch_dict['path']
            user_list = batch_dict['user_id'] 

            for elem in paths_dict["path"]:
                paths_list.append(convert_fn(elem))
                user_list.append(elem.split(' ')[1][1:]  )
            return batch_dict                    
        if plain_text_path:
            convert_fn = dataset.identity_op#dataset.convert_numeric_path_to_textual_path
            #dataset.dataset = dataset.dataset.map(lambda x: {"path": [dataset.convert_numeric_path_to_textual_path(elem) for elem in x["path"] ] },
            #                batched=True, num_proc=args.nproc)
        else:
            convert_fn = dataset.keep_numeric_typed#keep_numeric
            #dataset.dataset = dataset.dataset.map(lambda x: {"path": [dataset.keep_numeric(elem) for elem in x["path"]]  },
            #                            batched=True, num_proc=args.nproc)     
        
        dataset.dataset = dataset.dataset.map(lambda x: convert_typed_path_and_add_uid(x,convert_fn),#convert_and_add_uid(x, convert_fn),
                                        batched=True, num_proc=args.nproc)    
        dataset.dataset = dataset.dataset.class_encode_column('user_id')                                            
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
                single="[EOS]:0 $A:0 [BOS]:0",
                special_tokens=[("[EOS]", tokenizer.token_to_id("[EOS]")), ("[BOS]", tokenizer.token_to_id("[BOS]"))]
            )
            
            tokenizer.save(tokenizer_file)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)            
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=args.context_length,
                                                eos_token="[EOS]", bos_token="[BOS]",
                                                pad_token="[PAD]", unk_token="[UNK]",
                                                mask_token="[MASK]", use_fast=True)                

        #Check correctness of the encoding
        #print(dataset["path"][0], tokenizer.encode(dataset["path"][0]).tokens)

        # Load the specified tokenizer
        print("Train/Validation split...")
        # Add 'user_id' to the dataset
        #dataset_split = dataset.map(add_user_id, num_proc=args.nproc)
        # Now, we'll stratify by 'user_id'
        dataset_split = dataset.train_test_split(test_size=0.05, stratify_by_column='user_id')
        # Convert DatasetDict to desired format
        dataset_split = DatasetDict({
            'train': dataset_split['train'].remove_columns('user_id'),
            'test': dataset_split['test'].remove_columns('user_id'),
        })

        # Tokenizer and tokenization function
        #tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=args.context_length,
        #                            eos_token="[EOS]", bos_token="[BOS]",
        #                            pad_token="[PAD]", unk_token="[UNK]",
        #                            mask_token="[MASK]", use_fast=True)   
        #tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
        #                                    eos_token="[EOS]", bos_token="[BOS]",
        #                                    pad_token="[PAD]", unk_token="[UNK]",
        #                                    mask_token="[MASK]", use_fast=True)
        print("Tokenizing dataset...")
        pre1 = f"data/{dataset_name}/{TOKENIZER_TYPE}/pre1_from_scratch_tokenized_dataset.hf"

        
        if not os.path.exists(pre1):
            tokenized_dataset = dataset_split.map(tokenize_function, 
                batched=True, 
                num_proc=args.nproc,
                remove_columns=["path"]
            )
            check_dir(pre1)
            tokenized_dataset.save_to_disk(pre1)
        else:
            tokenized_dataset = load_from_disk(pre1)
        
        # Group texts into chunks of block_size tokens
        pre2 = f"data/{dataset_name}/{TOKENIZER_TYPE}/pre2_from_scratch_tokenized_dataset.hf"        
        
        if not os.path.exists(pre2):        
            tokenized_dataset = tokenized_dataset.map(
                group_texts,
                batched=True,
                batch_size=1000,
                num_proc=args.nproc,
            )
            check_dir(pre2)
            tokenized_dataset.save_to_disk(pre2)
        else:
            tokenized_dataset = load_from_disk(pre2)
        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")


    # Train the model
    if args.load_model:
        # Training arguments
        custom_name = f'clm-from_scratch-{args.data}-{args.model}/checkpoint-{args.eval_ckpt_iter}'#f"clm-from_scratch-{args.data}-{args.model}"
        #custom_name = 'distilgpt2-checkpoint-10000'
        model = AutoModelForCausalLM.from_pretrained(custom_name)#f'models-weights/{dataset_name}/{model_name}/{custom_name}')
    else:
        model = train_from_scratch(model_name, tokenizer, tokenized_dataset, args.context_length, args)
    evaluate(model, args)




