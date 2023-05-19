import argparse
import math
import os
from typing import List
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast
from pathlm.models.lm.evaluate import evaluate
from pathlm.models.lm.lm_utils import MLM_MODELS
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import check_dir
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


WORD_LEVEL_TOKENIZER = "./tokenizers/ml1m/WordLevel.json"
# Read an example and return the tokenized version
def tokenize_function(examples: str):
    return tokenizer(examples["path"],
        truncation=True,
        max_length=context_length)

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

    BATCH_SIZE = 64
    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
            evaluation_strategy="epoch",
            save_strategy='epoch',
        eval_steps=100000,
        learning_rate=2e-5,
        weight_decay=0.01,
        #use_mps_device=True,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=1000,  # number of warmup steps for learning rate
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=args.seed,
    )


    if model_name in MLM_MODELS:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    else:
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
    parser.add_argument("--nproc", type=int, default=4, help="Number of processes for dataset mapping")
    parser.add_argument("--load_data", type=bool, default=False, help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    args = parser.parse_args()


    model_name = args.model
    dataset_name = args.data

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"WordLevel.json")
    context_length = 256
    

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if args.load_data:
        tokenized_dataset = load_from_disk(f"data/{dataset_name}/{model_name}/from_scratch_tokenized_dataset.hf")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=256,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)        
    else:
        # Load the dataset
        data_dir = f"data/{dataset_name}"
        plain_text_path=False
        
        dataset = PathDataset(dataset_name, data_dir, plain_text_path=plain_text_path)
        if plain_text_path:
            dataset.dataset = dataset.dataset.map(lambda x: {"path": [dataset.convert_numeric_path_to_textual_path(elem) for elem in x["path"] ] },
                            batched=True, num_proc=args.nproc)
        else:
            dataset.dataset = dataset.dataset.map(lambda x: {"path": [dataset.keep_numeric(elem) for elem in x["path"] ]  }, 
                                        batched=True, num_proc=args.nproc)        
        dataset.show_random_examples()
        dataset = dataset.dataset

        # Word level tokenizer
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

        #Check correctness of the encoding
        #print(dataset["path"][0], tokenizer.encode(dataset["path"][0]).tokens)

        # Load the specified tokenizer
        dataset_split = dataset.train_test_split(test_size=0.1)

        # Tokenizer and tokenization function
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)
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
        tokenized_dataset.save_to_disk(f"data/{dataset_name}/{model_name}/from_scratch_tokenized_dataset.hf")



    # Train the model
    if args.load_model:
        # Training arguments
        custom_name = f"from_scratch-{args.data}-{args.model}"
        model = AutoModelForCausalLM.from_pretrained(f"models-weights/{dataset_name}/{model_name}/{custom_name}")
    else:
        model = train_from_scratch(model_name, tokenizer, tokenized_dataset, context_length, args)
    evaluate(model, args)



