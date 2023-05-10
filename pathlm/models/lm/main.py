import argparse
import csv
import math
from collections import defaultdict

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import check_dir, get_pid_to_eid


# Read an example an return the tokenized version
def tokenize_function(examples: str):
    return fast_tokenizer(examples["path"])

MLM_MODELS = ["bert-large", "roberta-large"]
CLM_MODELS = ['distilgpt2', 'gpt2-xl', "stabilityai/stablelm-base-alpha-3b"]

def train(model_name: str, args: argparse.Namespace):
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
            learning_rate=2e-5,
            weight_decay=0.01,
            use_mps_device=True,
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
            learning_rate=2e-5,
            weight_decay=0.01,
            use_mps_device=True,
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
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    if model_name in MLM_MODELS:
        data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer, mlm_probability=0.15)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
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
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducibility")
    parser.add_argument("--from_scratch", type=bool, default=False, help="")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.data
    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    try:
        tokenized_dataset = load_from_disk(f'data/{dataset_name}/{model_name}/tokenized_dataset.hf')
    except:
        # Load the dataset
        data_dir = f"data/{dataset_name}"
        dataset = PathDataset(dataset_name, data_dir)
        dataset.show_random_examples()
        dataset = dataset.dataset

        # Load the specified tokenizer
        dataset_split = dataset.train_test_split(test_size=0.1)

        # Tokenizer and tokenization function
        fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=4, remove_columns=["path"])

        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{model_name}/tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(f"data/{dataset_name}/{model_name}/tokenized_dataset.hf")

    # Train the model
    model = train()

    """
    Recommendation evaluation
    """
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    i2kg = get_pid_to_eid(data_dir)

    # Generate paths for the test set
    test_set = defaultdict(list)
    with open(f"{data_dir}/preprocessed/test.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            user_id = user_id-1 # user_id starts from 1 in the augmented graph starts from 0
            item_id = i2kg[item_id] #Converting dataset id to eid
            test_set[user_id].append(item_id)
    f.close()

    exit()

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
