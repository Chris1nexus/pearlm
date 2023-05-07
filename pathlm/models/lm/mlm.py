import math

from custom_tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, \
    PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AutoConfig, AutoModelForMaskedLM


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]['text']

def tokenize_function(examples):
    return fast_tokenizer(examples["text"])


dataset = load_dataset('csv', data_files='data/ml1m/path_with_names.csv', split='train')
dataset_split = dataset.train_test_split(test_size=0.1)


#print(dataset)
#dataset = dataset.map(lambda x: x if x['text'] != '' else None, remove_columns=['text'])
#dataset = dataset.filter(lambda x: x['text'] != None and x['text'] != '')
#print(dataset)

tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordLevelTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

#print(tokenizer.encode("The_Fiendish_Plot_of_Dr._Fu_Manchu belong_to_category Category:English-language_films belong_to_category As_Good_as_It_Gets").tokens)

#train.save_to_disk('data/ml1m/train.hf')
#valid.save_to_disk('data/ml1m/valid.hf')

model_checkpoint = "bert-base-cased"
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, mask_token="[MASK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]", unk_token="[UNK]")

tokenized_dataset = dataset_split.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_config(config)
training_args = TrainingArguments(
    "test-mlm",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    use_mps_device=True,
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