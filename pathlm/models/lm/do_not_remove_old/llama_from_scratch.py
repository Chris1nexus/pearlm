import argparse
import math
from datasets import load_from_disk
from transformers import LlamaTokenizerFast, TrainingArguments, Trainer, AutoModelForCausalLM
from transformers import LlamaConfig


def main(args):
    def tokenize_function(examples):
        """
        :param examples:
        :return:
        """
        return fast_tokenizer(examples["text"], truncation=True)

    # Load processed dataset
    dataset = load_from_disk(f'data/{args.data}/hf_dataset.hf')

    # Split dataset into train and test
    dataset_split = dataset.train_test_split(test_size=0.1)

    # Load tokenizer trained from scratch
    fast_tokenizer = LlamaTokenizerFast.from_pretrained('./tokenizers/LLamatokenizer')

    # Tokenize dataset
    tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    def group_texts(examples, block_size=128):
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

    # Group texts into chunks of block_size tokens
    lm_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    # Initializing a LLaMA llama-7b style configuration
    config = LlamaConfig(
        vocab_size=25000,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_act="silu",
        max_position_embeddings=512,
    )

    # Initializing a model from the llama-7b style configuration
    model = AutoModelForCausalLM.from_config(config)

    # Number of parameters
    print(f"No of parameters: {sum(p.numel() for p in model.parameters())}")

    # Training arguments
    training_args = TrainingArguments(
        "test-clm",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        use_mps_device=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
    )

    # Train model
    trainer.train()

    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Save model
    trainer.save_model("./models-weights/path-llama-from-scratch")


# Main that reads arguments with argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ml1m", help="{ml1m, lfm1m}")
    args = parser.parse_args()
    main(args)