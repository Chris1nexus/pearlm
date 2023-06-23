import argparse
import functools
import heapq
import math
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool, set_start_method
from random import random
from typing import List, Dict
import torch.nn.functional as F

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_from_disk, DatasetDict, Dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, pipeline, is_torch_tpu_available, \
    RobertaModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaPreTrainedModel
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from pathlm.models.lm.evaluate_mlm import evaluate_bert
from pathlm.models.lm.lm_utils import MLM_MODELS, TOKENIZER_DIR, get_user_negatives_tokens_ids, get_user_positives
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
from pathlm.sampling.container.constants import LiteralPath
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.tools.mapper import EmbeddingMapper
from pathlm.utils import check_dir, SEED, set_seed, get_pid_to_eid, get_set


class RobertaForMaskedLMDoubleHead(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = None

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        self.ent_mask = config.ent_mask
        self.rel_mask = config.rel_mask
        self.ent_mask = torch.FloatTensor(self.ent_mask)
        self.rel_mask = torch.FloatTensor(self.rel_mask)

        self.num_kg_types = len(RobertaForMaskedLMDoubleHead.kg_categories)

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=self.num_kg_types,
                                                  embedding_dim=config.hidden_size)  # for entities, relations, and special tokens

        self.type_ids_cache = dict()
        self.type_embeds_cache = dict()

        # Create an additional linear layer for the second prediction head
        self.lm_entity_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.lm_relation_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Create ReLU layer
        self.relu = torch.nn.ReLU()

    def __init_type_embeddings(self, batch_size, num_hops):
        # num_hops = self.config.num_hops
        n_tokens = num_hops  # num_hops + 1 + num_hops + 2
        type_ids = torch.ones((batch_size, n_tokens), dtype=torch.long)

        for i in range(n_tokens):
            if i == 0 or i == n_tokens - 1:
                type_ids[:, i] = RobertaForMaskedLMDoubleHead.SPECIAL_ID
            elif i % 2 == 1:
                type_ids[:, i] = RobertaForMaskedLMDoubleHead.ENTITY_ID
            elif i % 2 == 0:
                type_ids[:, i] = RobertaForMaskedLMDoubleHead.RELATION_ID
        type_ids = type_ids.to(self.type_embeddings.weight.device)
        type_embeds = self.type_embeddings(type_ids)
        return type_ids, type_embeds

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.entity_head_lm = self.entity_head_lm.to(self.transformer.first_device)
        self.relation_head_lm = self.relation_head_lm.to(self.transformer.first_device)
        self.type_embeddings = self.type_embeddings.to(self.transformer.first_device)
        self.type_ids = self.type_ids.to(self.transformer.first_device)

        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.entity_head_lm = self.entity_head_lm.to("cpu")
        self.relation_head_lm = self.relation_head_lm.to("cpu")
        self.type_embeddings = self.type_embeddings.to("cpu")
        self.type_ids = self.type_ids.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        batch_size, seq_len = input_ids.shape
        k = (batch_size, seq_len)
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids, type_embeds

        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]
        # if self.model_parallel:
        #    torch.cuda.set_device(inputs_embeds.device)
        #    type_embeds = type_embeds.to(inputs_embeds.device)

        if inputs_embeds is not None:
            inputs_embeds += type_embeds

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        # Get logits from the two heads, first based on entity tokens, then on relation tokens
        entity_predicted_scores = self.lm_entity_head(sequence_output)
        relation_predicted_scores = self.lm_relation_head(sequence_output)

        # Apply ReLU activation
        entity_predicted_scores = self.relu(entity_predicted_scores)
        relation_predicted_scores = self.relu(relation_predicted_scores)

        def compute_loss(logits, labels, class_mask):
            class_mask = class_mask.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=class_mask)
            logits = logits.contiguous()
            labels = labels.contiguous()
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device))
            return lm_loss

        loss = None
        if labels is not None:
            loss = 0.
            # Entity prediction
            logits_mask = (type_ids == RobertaForMaskedLMDoubleHead.ENTITY_ID)
            label_mask = (type_ids == RobertaForMaskedLMDoubleHead.ENTITY_ID)
            loss += compute_loss(entity_predicted_scores[logits_mask], labels[label_mask], self.ent_mask)

            # Relation prediction
            logits_mask = (type_ids == RobertaForMaskedLMDoubleHead.RELATION_ID)
            label_mask = (type_ids == RobertaForMaskedLMDoubleHead.RELATION_ID)
            loss += compute_loss(relation_predicted_scores[logits_mask], labels[label_mask], self.rel_mask)

        if not return_dict:
            output = (entity_predicted_scores, relation_predicted_scores) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=(entity_predicted_scores, relation_predicted_scores),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CustomTrainer(Trainer):

    def __init__(
        self,
        dataset_name=None,
        tokenizer=None,
        eval_device='cpu',
        **kwargs
    ):
        super().__init__(**kwargs)


        data_dir = f"data/{dataset_name}"
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.test_set = get_set(dataset_name, set_str='test')
        self.uids = list(self.test_set.keys())
        self.eval_device = eval_device

        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_pid_to_eid(data_dir).values()])
        self.user_positives = get_user_positives(dataset_name)

        init_condition_fn = lambda uid: f"U{uid} -1 [MASK] [MASK] [MASK] [MASK]"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in self.uids] }

    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        higher_weight = 10
        loss = custom_loss(outputs, labels, higher_weight)

        return (loss, outputs) if return_outputs else loss
    """
    def __generate_topks_withWordLevel(self, model):
        """
        Recommendation and explanation generation
        """
        # set_seed(SEED)
        dataset_name = args.dataset
        data_dir = f"data/{dataset_name}"
        tokenizer_dir = f'./tokenizers/{dataset_name}'
        TOKENIZER_TYPE = "WordLevel"

        tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)

        # Load user negatives

        fill_masker = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

        # init_condition_fn = lambda uid: tokenizer.encode(f"U{uid} R-1 [MASK] [MASK] [MASK] [MASK] [MASK]")
        #
        # sequences = [init_condition_fn(uid) for uid in uids]
        # dataset = Dataset.from_dict({'uid': uids, 'sequence': sequences})
        # output = model(**dataset['sequence'])
        #
        # preds = fill_masker(dataset['sequence'], top_k=30)
        # topks = {}
        # for uid, pred in enumerate(preds):
        #    uid = str(uid + 1)
        #    topks[uid] = [x['token_str'][1:] for x in pred[-1]]
        #    topks[uid] = list(set(topks[uid]) - set(user_positives[uid]))[:10]
        # return topks

        init_condition_fn = lambda uid: f"U{uid} R-1 [MASK] [MASK] [MASK] [MASK] [MASK]"
        user_positives = get_user_positives(dataset_name)
        sequences = [init_condition_fn(uid) for uid in self.uids]
        dataset = Dataset.from_dict({'uid': self.uids, 'sequence': sequences})

        topks = {}
        for uid, sequence in zip(self.uids, dataset['sequence']):
            # Tokenize the sequence and send the tensors to the same device as your model
            inputs = tokenizer(sequence, return_tensors="pt").to("cuda")

            with torch.no_grad():  # Deactivate gradients for the following code block
                # Get the model's predictions
                outputs = model(**inputs)
                entity_predictions, relation_predictions = outputs.logits

            # The position of the last [MASK] token is -2.
            mask_position = -2

            # Select top-k predictions from each head for the last MASK
            top_k_entities = torch.topk(entity_predictions[0, mask_position], 200).indices

            # Convert token IDs to tokens
            top_k_entities = [tokenizer.decode([idx]) for idx in top_k_entities]
            top_k_entities = [x[1:] for x in top_k_entities if x[0] == 'P']
            topks[str(uid)] = list(set(top_k_entities) - set(user_positives[str(uid)]))[:10]

        return topks
    def evaluate(self, model):
        # Generate paths for the test users
        # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok

        topks = self.__generate_topks_withWordLevel(model)
        check_dir(f"./results/{self.dataset_name}/{self.custom_model_name}")
        pickle.dump(topks, open(f"./results/{self.dataset_name}/{self.custom_model_name}/topks.pkl", "wb"))
        metrics = {"ndcg": [], "mmr": [],  }
        for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
            hits = []
            for recommended_item in topk:
                if recommended_item in self.test_set[uid]:
                    hits.append(1)
                else:
                    hits.append(0)
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
            metrics["ndcg"].append(ndcg)
            metrics["mmr"].append(mmr)

        print(f"no of users: {len(self.test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
        metrics_ = dict()
        for k in metrics:
            metrics_[f'eval_{k}'] = np.mean(metrics[k])
        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            '''
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            '''
            metrics = self.evaluate(model)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


# Read an example and return the tokenized version

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
        if token[0] != LiteralPath.rel_type :
            ent_mask.append(class_weight)
        else:
            ent_mask.append(0)

    embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
    try:
        embeds = pickle.load(open(embed_filepath, 'rb'))
    except:
        embeds = None

    config_kwargs ={
        'vocab_size':len(tokenizer),
        'n_ctx':context_length,
        'n_positions': context_length,
        'pad_token_id':tokenizer.pad_token_id,
        'bos_token_id':tokenizer.bos_token_id,
        'eos_token_id':tokenizer.eos_token_id,
    }
    if embeds:
        config_kwargs.update({
        'hidden_size':args.emb_size,
        'num_attention_heads':args.emb_size//10
        })

    # Initializing the selected model style configuration
    encoder_config = AutoConfig.from_pretrained(
        encoder_name,
        **config_kwargs
    )

    encoder_config.update({'num_hops': args.n_hop,
        'train_batch_size':args.batch_size,
        'test_batch_size':args.infer_batch_size,
        'ent_mask': ent_mask,
        'rel_mask': rel_mask})


    # Initializing a model from the configuration
    model = RobertaForMaskedLMDoubleHead(encoder_config)

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
        learning_rate=5e-5,
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

    trainer = CustomTrainer(
        dataset_name=dataset_name,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
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
    parser.add_argument("--context_length", type=int, default=10,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")
    parser.add_argument("--nproc", type=int, default=4, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=1576, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=12, help="Test batch size")
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
