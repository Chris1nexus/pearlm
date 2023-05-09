
tokenized_dataset = load_from_disk(f'data/{args.data}/tokenized_dataset.hf')

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
        seed=123,
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        seed=123,
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
trainer.save_model("/models-weights/path-llama-from-scratch")