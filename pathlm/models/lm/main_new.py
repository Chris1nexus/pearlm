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
from datasets import load_from_disk
from tqdm import tqdm

from pathlm.models.lm.do_not_remove_old.path_dataset import PathDataset, PathLMDataset
from pathlm.utils import check_dir, get_pid_to_eid
from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map


def clm_loss(inputs, logits, alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # Calculate and scale weighting
    #weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
    #    axis=[0, 2]
    #)
    #weights = alpha * (1.0 + weights)
    weights = alpha
    # Calculate weighted average
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss

# Read an example an return the tokenized version
def tokenize_function(examples):
    return fast_tokenizer(examples["path"])

MLM_MODELS = ["bert-large", "roberta-large"]
CLM_MODELS = ['WordLevel', 'gpt2-xl', "stabilityai/stablelm-base-alpha-3b"]
def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                mlm=model_name in MLM_MODELS,
                                mlm_probability=0.15)
    if model_name in MLM_MODELS:
        
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
    #train_dataset=dataset['train']
    #test_dataset=dataset['test']
    #train_dataset = PathLMDataset(train_dataset['path'], tokenizer, data_dir)
    #test_dataset = PathLMDataset(test_dataset['path'], tokenizer, data_dir)
    def collate_tokenize(text_batch):
          #text_batch = [element["text"] for element in data]
          tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
          return tokenized
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')
    print('Preparing dataloaders')
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64         
    train_dataloader = DataLoader(dataset['train'], batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=0,collate_fn=data_collator.torch_call)#, collate_fn=collate_tokenize)
    eval_dataloader = DataLoader(dataset['test'], batch_size=EVAL_BATCH_SIZE, shuffle=False,num_workers=0,collate_fn=data_collator.torch_call)#, collate_fn=collate_tokenize)
    #print(train_dataset[0])
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
    os.makedirs(os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}' ),exist_ok=True  )


    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(fast_tokenizer.eos_token)
    fast_tokenizer.pad_token = fast_tokenizer.eos_token
    data_dir = os.path.join(ROOT_DIR,f"data/{dataset_name}")


    if not  os.path.exists(os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}/dataset.hf') ):
        path_dataset = PathDataset(dataset_name, fast_tokenizer, data_dir)
        path_dataset.show_random_examples()
        dataset = path_dataset.dataset
        dataset_split = dataset.train_test_split(test_size=0.1)
        # save the dataset
        dataset_split.save_to_disk(os.path.join(ROOT_DIR,f"data/{dataset_name}/{model_name}/dataset.hf"))




    if os.path.exists(os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}/tokenized_dataset.hf') ):
        tokenized_dataset = load_from_disk(os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}/tokenized_dataset.hf' ))
    else:
        dataset = load_from_disk(os.path.join(ROOT_DIR,f'data/{dataset_name}/{model_name}/dataset.hf' ))
        
        eid2name = get_eid_to_name_map(data_dir)
        rid2name = get_rid_to_name_map(data_dir)

        print('Creating tokenized dataset')
        def tokenize(examples):
            tmp = [PathDataset.convert_numeric_path_to_textual_path(path, eid2name, rid2name) for path in examples['path']]
            return fast_tokenizer(tmp)
        tokenized_dataset = dataset.map(
            tokenize, batched=True, num_proc=16, remove_columns=dataset["train"].column_names
        )

        tokenized_dataset.save_to_disk(os.path.join(ROOT_DIR,f"data/{dataset_name}/{model_name}/tokenized_dataset.hf"))
        

    # Tokenizer and tokenization function
    #fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    #tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=16, remove_columns=["path"])

    # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
    #check_dir(os.path.join(ROOT_DIR,f"{data_dir}/{model_name}/tokenized_dataset.hf"))
    #tokenized_dataset.save_to_disk(os.path.join(ROOT_DIR,f"data/{dataset_name}/{model_name}/tokenized_dataset.hf"))

    # Train the model
    model = train(model_name, tokenized_dataset,fast_tokenizer, data_dir, args)

    """
    Recommendation evaluation
    """
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    i2kg = get_pid_to_eid(data_dir)

    # Generate paths for the test set
    test_set = defaultdict(list)
    with open(os.path.join({ROOT_DIR},f"{data_dir}/preprocessed/test.txt", "r")) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            user_id = user_id-1 # user_id starts from 1 in the augmented graph starts from 0
            item_id = i2kg[item_id] #Converting dataset id to eid
            test_set[user_id].append(item_id)
    f.close()

    #exit()

    # Generate paths for the test users
    generator = pipeline('text-generation', model=model)
    set_seed(args.seed)
    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    for uid in test_set.keys():
        outputs = generator(f"{uid}", num_beams=4, do_sample=True)
        # Convert tokens to entity names
        fast_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        #TODO: Convert final entity name (we must ensure it is an item) to entity id

        #TODO: Get the top 10 items from the graph

        #TODO: Evaluate the recommendation
