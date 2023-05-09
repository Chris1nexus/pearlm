import math
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast

def tokenize_function(examples):
    return fast_tokenizer(examples["text"])

dataset = load_from_disk('data/ml1m/hf_dataset.hf')
dataset_split = dataset.train_test_split(test_size=0.1)
fast_tokenizer = RobertaTokenizerFast.from_pretrained('./BPEtokenizer', max_len=512)

tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)

training_args = TrainingArguments(
    "test-mlm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    use_mps_device=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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
trainer.save_model("test-mlm")