import math
import random as random

import pandas as pd
from datasets import load_dataset, load_from_disk, ClassLabel
from ipywidgets import HTML
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM, AutoModelForMaskedLM, AutoTokenizer, \
    RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast

def tokenize_function(examples):
    return fast_tokenizer(examples["text"])

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(dataset['text'][pick])

    for pick in picks:
        print(f"N of tokens: {len(slow_tokenizer.encode(pick))}")
        print(slow_tokenizer.convert_ids_to_tokens(slow_tokenizer.encode(pick)))

dataset = load_from_disk('data/ml1m/hf_dataset.hf')
dataset_split = dataset.train_test_split(test_size=0.1)

#slow_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
#show_random_elements(dataset, 30)

fast_tokenizer = AutoTokenizer.from_pretrained("roberta-large")

tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

model = AutoModelForMaskedLM.from_pretrained("roberta-large")

training_args = TrainingArguments(
    "ft-test-mlm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    use_mps_device=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)


data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.save_model("ft-test-mlm")