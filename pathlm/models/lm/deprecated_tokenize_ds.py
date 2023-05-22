import argparse
import csv
import math
from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from torch.nn import CrossEntropyLoss
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from datasets import load_from_disk, load_dataset


from pathlm.models.lm.do_not_remove_old.path_dataset import PathDataset, PathLMDataset
from pathlm.utils import check_dir, get_pid_to_eid
from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map


# Read an example an return the tokenized version
def tokenize_function(examples: str):
    return fast_tokenizer(examples["path"])

MLM_MODELS = ["bert-large", "roberta-large"]
CLM_MODELS = ['WordLevel', 'gpt2-xl', "stabilityai/stablelm-base-alpha-3b"]
"""
def train(model_name: str, dataset, tokenizer, data_dir, args: argparse.Namespace):
    '''
    # Load the specified model
    if model_name in MLM_MODELS:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    elif model_name in CLM_MODELS:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Training arguments
    custom_name = f"ft-{args.data}-{args.model}"

    # Training arguments for Masked Language Model task
    if model_name in MLM_MODELS:
        training_args = TrainingArguments(
            f"mlm-{custom_name}",
            evaluation_strategy="epoch",
            save_strategy='epoch',
            learning_rate=2e-5,
            weight_decay=0.01,
            #use_mps_device=True,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            load_best_model_at_end=True,
            seed=args.seed,
            
        )
    # Training arguments for Causal Language Model task
    else:
        training_args = TrainingArguments(
            f"clm-{custom_name}",
            evaluation_strategy="epoch",
            save_strategy='epoch',
            learning_rate=2e-5,
            weight_decay=0.01,
            #use_mps_device=True,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            load_best_model_at_end=True,
            seed=args.seed,
        )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        #tokenizer=tokenizer,
    )

    if model_name in MLM_MODELS:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            #tokenizer=tokenizer,
        )

    # Train model
    trainer.train()

    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Save model
    weight_path = f"./models-weights/{args.data}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.data}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)

    '''
    print('Setup torch datasets')
    train_dataset=dataset['train']
    test_dataset=dataset['test']
    train_dataset = PathLMDataset(train_dataset['path'], tokenizer, data_dir)
    test_dataset = PathLMDataset(test_dataset['path'], tokenizer, data_dir)
    def collate_tokenize(text_batch):
          #text_batch = [element["text"] for element in data]
          tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
          return tokenized
    print('Preparing dataloaders')
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16          
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=0, collate_fn=collate_tokenize)
    eval_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,num_workers=0, collate_fn=collate_tokenize)
    print(train_dataset[0])
    #print(train_dataset)
    print('Loading the model')
    model = AutoModelForCausalLM.from_pretrained(model_name)



    optimizer = AdamW(model.parameters(), lr=5e-5)

    accelerator = Accelerator(fp16=True)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps,
    )
    output_dir = f'results/{model_name}__{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    gradient_accumulation_steps = 8
    eval_steps = 5_000
    model.train()
    completed_steps = 0
    for epoch in range(num_train_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1), total=num_training_steps
        ):
            logits = model(batch["input_ids"]).logits
            loss = clm_loss(batch["input_ids"], logits)
            #loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
            if step % 100 == 0:
                accelerator.print(
                    {
                        "lr": lr_scheduler.get_lr(),
                        #"samples": step * samples_per_step,
                        "steps": completed_steps,
                        "loss/train": loss.item() * gradient_accumulation_steps,
                    }
                )
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                #eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
                #accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)

    eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
    accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)    
    model.train()
    return model
"""
def get_training_corpus(raw_datasets, path_dataset):
    return (
        raw_datasets["train"][i : i + 1000]["path"].apply(lambda path: PathDataset.convert_numeric_path_to_textual_path(path, path_dataset.eid2name, path_dataset.rid2name) )
        for i in range(0, len(raw_datasets["train"]), 1000)
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--root-dir", type=str, default="./", help="Root directory where the dataset is stored")

    parser.add_argument("--model", type=str, default="WordLevel", help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducibility")
    parser.add_argument("--from_scratch", type=bool, default=False, help="")
    args = parser.parse_args()
    ROOT_DIR = args.root_dir
    model_name = args.model
    dataset_name = args.data
    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch

    EXPERIMENT_PATH = os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}' )

    os.makedirs(EXPERIMENT_PATH,exist_ok=True  )


    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    fast_tokenizer.train_new_from_iterator
    #print(fast_tokenizer.eos_token)
    #fast_tokenizer.pad_token = fast_tokenizer.eos_token
    # Load the dataset
    data_dir = os.path.join(ROOT_DIR,f"data/{dataset_name}")
    dataset = load_dataset(os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}/dataset.hf' ), streaming=True)
    eid2name = get_eid_to_name_map(data_dir)
    rid2name = get_rid_to_name_map(data_dir)

    print('Creating tokenized dataset')
    def tokenize(examples):
        tmp = [PathDataset.convert_numeric_path_to_textual_path(path, eid2name, rid2name) for path in examples['path']]
        return fast_tokenizer(tmp)
    tokenized_dataset = dataset.map(
        tokenize, batched=True)

    '''
    path_dataset = PathDataset(dataset_name, fast_tokenizer, data_dir)

    path_dataset.show_random_examples()
    dataset = path_dataset.dataset
        #dataset = dataset.map(lambda x: {'path': [PathDataset.convert_numeric_path_to_textual_path(path, path_dataset.eid2name, path_dataset.rid2name) for path in x['path']] },
    #             batched=True, num_proc=8, )
    # Load the specified tokenizer
    dataset_split = dataset.train_test_split(test_size=0.1)
    training_corpus = get_training_corpus()
    tokenizer = fast_tokenizer.train_new_from_iterator(training_corpus, 52000)
    '''
    # Tokenizer and tokenization function
    #fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    #tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=16, remove_columns=["path"])

    # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
    #check_dir(os.path.join(ROOT_DIR,f"{data_dir}/{model_name}/tokenized_dataset.hf"))
    #tokenized_dataset.save_to_disk(os.path.join(ROOT_DIR,f"data/{dataset_name}/{model_name}/tokenized_dataset.hf"))




def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )