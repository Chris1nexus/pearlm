import argparse
import os
import pickle

import torch.multiprocessing as mp
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig, \
    PreTrainedTokenizerFast

from pathlm.models.lm.evaluate_mlm import evaluate_bert
from pathlm.models.lm.lm_utils import TOKENIZER_DIR
from pathlm.models.lm.perlmlm import RobertaForMaskedLMWithTypeEmb
from pathlm.models.lm.trainer import PathMLMTrainer
from pathlm.sampling.container.constants import LiteralPath
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.tools.mapper import EmbeddingMapper
from pathlm.utils import check_dir, SEED


def train_encoder(tokenizer, tokenized_dataset, args: argparse.Namespace):
    context_length = args.context_length
    encoder_name = args.model_name

    ent_mask = []
    rel_mask = []
    class_weight = 1.
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
        if token[0] == LiteralPath.rel_type or not token[0].isalpha():
            rel_mask.append(class_weight)
        else:
            rel_mask.append(0)
        if token[0] != LiteralPath.rel_type:
            ent_mask.append(class_weight)
        else:
            ent_mask.append(0)

    embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
    try:
        embeds = pickle.load(open(embed_filepath, 'rb'))
    except:
        embeds = None

    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
        'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    if embeds:
        config_kwargs.update({
            'hidden_size': args.emb_size,
            'num_attention_heads': args.emb_size // 10
        })

    # Initializing the selected model style configuration
    encoder_config = AutoConfig.from_pretrained(
        encoder_name,
        **config_kwargs
    )

    encoder_config.update({'num_hops': args.n_hop,
                           'train_batch_size': args.batch_size,
                           'test_batch_size': args.infer_batch_size,
                           'ent_mask': ent_mask,
                           'rel_mask': rel_mask})

    # Initializing a model from the configuration
    model = RobertaForMaskedLMWithTypeEmb(encoder_config)

    if embeds:
        ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
        # Dataset directories.
        dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

        data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
        kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)

        mapper = EmbeddingMapper(tokenizer, kg, embeds)
        mapper.init_with_embedding(model.roberta.embeddings.word_embeddings.weight)

    # Training arguments
    custom_name = f"from_scratch-{args.dataset}-{args.model_name}"

    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"encoder-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=5000,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=2500,  # number of warmup steps for learning rate
        save_steps=500,
        save_total_limit=2,
        # load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = PathMLMTrainer(
        dataset_name=dataset_name,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        context_length=context_length,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")
    parser.add_argument("--nproc", type=int, default=4, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=1024, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Test batch size")
    parser.add_argument("--eval_ckpt_iter", type=int, default='161500', help="")
    parser.add_argument("--infer_batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    # KGE Parameters
    parser.add_argument("--embedding_root_dir", type=str, default="./embedding-weights",
                        help="default: ./embedding-weights")
    parser.add_argument("--emb_filename", type=str, default='transe_embed.pkl', help="default: 'transe_embed.pkl'")
    parser.add_argument("--emb_size", type=int, default=100,
                        help="Transformer Embedding size (must match external embedding size, if chosen)")

    args = parser.parse_args()

    mp.set_start_method('spawn')
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model_name
    dataset_name = args.dataset

    tokenizer_dir = f'{TOKENIZER_DIR}/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    tokenized_dataset = load_from_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    # Train the model
    if args.load_model:
        # Training arguments
        custom_name = f'from_scratch-{args.dataset}-{args.model_name}/checkpoint-{args.eval_ckpt_iter}'
        model = AutoModelForCausalLM.from_pretrained(f"models-weights/{dataset_name}/{model_name}/{custom_name}")
    else:
        model = train_encoder(tokenizer, tokenized_dataset, args)
    evaluate_bert(model, args)
