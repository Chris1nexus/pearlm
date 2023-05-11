import argparse
import math
from typing import List
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from pathlm.models.lm.evaluate import evaluate
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import check_dir


# Read an example and return the tokenized version
def tokenize_function(examples: str):
    return fast_tokenizer(examples["path"], truncation=True, max_length=256)

def group_texts(examples: List[str], block_size=128):
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
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
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
            #load_best_model_at_end=True,
            seed=args.seed,
        )

    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if model_name in MLM_MODELS:
        data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer, mlm_probability=0.15)
    else:
        fast_tokenizer.pad_token = fast_tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer, mlm=False)

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

    # Train the model
    try:
        model = AutoModelForCausalLM.from_pretrained(f"models-weights/ml1m/distilgpt2/ft-ml1m-distilgpt2")
    except:
        model = train(model_name, args)
    evaluate(model, args)




