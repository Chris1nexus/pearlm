import argparse
import os
import pickle
import random
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import LogitsProcessorList
from transformers import set_seed, PreTrainedTokenizerFast

from pathlm.models.lm.from_scratch_main import PERLM, PLMRec, _initialise_type_masks
from pathlm.models.lm.generation_constraints import ConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel, \
    PrefixConstrainedLogitsProcessorWordLevel
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
from pathlm.models.rl.PGPR.pgpr_utils import RELATION
from pathlm.sampling.container.constants import TypeMapper, LiteralPath
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.utils import get_pid_to_eid, get_set, check_dir, SEED


def tokenize_augmented_kg(kg, tokenizer, use_token_ids=False):
    type_id_to_subtype_mapping = kg.dataset_info.groupwise_global_eid_to_subtype.copy()
    rel_id2type = kg.rel_id2type.copy()
    type_id_to_subtype_mapping[RELATION] = {int(k): v for k, v in rel_id2type.items()}

    aug_kg = kg.aug_kg

    token_id_to_token = dict()
    kg_to_vocab_mapping = dict()
    tokenized_kg = dict()

    for token, token_id in tokenizer.get_vocab().items():
        if not token[0].isalpha():
            continue

        cur_type = token[0]
        cur_id = int(token[1:])

        type = TypeMapper.mapping[cur_type]
        subtype = type_id_to_subtype_mapping[type][cur_id]
        if cur_type == LiteralPath.rel_type:
            cur_id = None
        value = token
        if use_token_ids:
            value = token_id
        kg_to_vocab_mapping[(subtype, cur_id)] = token_id

    for head_type in aug_kg:
        for head_id in aug_kg[head_type]:
            head_key = head_type, head_id
            if head_key not in kg_to_vocab_mapping:
                continue
            head_ent_token = kg_to_vocab_mapping[head_key]
            tokenized_kg[head_ent_token] = dict()

            for rel in aug_kg[head_type][head_id]:
                rel_token = kg_to_vocab_mapping[rel, None]
                tokenized_kg[head_ent_token][rel_token] = set()

                for tail_type in aug_kg[head_type][head_id][rel]:
                    for tail_id in aug_kg[head_type][head_id][rel][tail_type]:
                        tail_key = tail_type, tail_id
                        if tail_key not in kg_to_vocab_mapping:
                            continue
                        tail_token = kg_to_vocab_mapping[tail_key]
                        tokenized_kg[head_ent_token][rel_token].add(tail_token)

    return tokenized_kg, kg_to_vocab_mapping

def get_user_negatives(dataset_name: str) -> Dict[str, List[str]]:
    NOT_IN_TOKENIZER = {'1871', '831', '1950', '478', '2285'}
    data_dir = f"data/{dataset_name}"
    ikg_ids = set(get_pid_to_eid(data_dir).values())
    uid_negatives = {}
    # Generate paths for the test set
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        uid_negatives[uid] = list(set(ikg_ids - set(train_set[uid]) - set(valid_set[uid])) - NOT_IN_TOKENIZER)
    return uid_negatives

class Evaluator:
    def __init__(
            self,
            dataset_name=None,
            n_hop=3,
            infer_batch_size=1,
            n_sequences_per_user=10,
            tokenizer=None,
            eval_device='cpu',
            tokenized_kg=None,
            custom_model_name=None,
            logit_processor_type='gcd',
                **kwargs
    ):
        super().__init__(**kwargs)
        data_dir = f"data/{dataset_name}"
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.custom_model_name = custom_model_name
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.n_hop = n_hop
        self.eval_device = eval_device

        self.SEQUENCE_LEN = 2 * int(n_hop) + 2  # Special tokens [BOS] included

        self.INFERENCE_BATCH_SIZE = args.infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user
        print('Sequence length: ', self.SEQUENCE_LEN)

        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_pid_to_eid(data_dir).values()])
        self.user_negatives_token_ids = get_user_negatives_tokens_ids(dataset_name, tokenizer)
        self.user_negatives = get_user_negatives(dataset_name)
        self.id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in uids}
        init_condition_fn = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in uids]}

        logit_processor = None
        logit_proc_kwargs ={}
        if logit_processor_type == 'gcd':
            logit_processor_cls = ConstrainedLogitsProcessorWordLevel 
        elif logit_processor_type == 'pgcd':
            logit_processor_cls = PrefixConstrainedLogitsProcessorWordLevel
        else:
            logit_processor_cls = PLMLogitsProcessorWordLevel 
            ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
            logit_proc_kwargs['ent_mask'] = ent_mask
            logit_proc_kwargs['rel_mask'] = rel_mask
            logit_proc_kwargs['token_id_to_token'] = token_id_to_token
        print('Using: ', logit_processor_cls)


        self.logits_processor = LogitsProcessorList([
            logit_processor_cls(tokenized_kg=tokenized_kg,
                                force_token_map=self.user_negatives_token_ids,
                                tokenizer=tokenizer,
                                total_length=self.SEQUENCE_LEN,  # LAST_TOKEN_POS,
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                id_to_uid_token_map=self.id_to_uid_token_map,
                                eos_token_ids=[
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)],
                                **logit_proc_kwargs
                            )
        ])

        self.test_dataset = Dataset.from_dict(self.inference_paths)

    def __generate_topks_withWordLevel(self, model):
        batch_size = self.INFERENCE_BATCH_SIZE
        topk = defaultdict(list)
        topk_sequences = defaultdict(list)
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)) as pbar:
            for i in range(0, len(self.test_dataset), batch_size):
                batch = self.test_dataset[i:i + batch_size]
                inputs = self.tokenizer(batch["uid"], return_tensors='pt', add_special_tokens=False, ).to(self.eval_device)
                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN,
                    min_length=self.SEQUENCE_LEN,
                    num_return_sequences=30,
                    num_beams=30,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    # top_p=0.4,
                    logits_processor=self.logits_processor,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                def normalize_tuple(logits_tuple):
                    # Normalize each tensor in the tuple
                    normalized_tuple = tuple(torch.softmax(logits, dim=-1) for logits in logits_tuple)
                    return normalized_tuple

                def calculate_sequence_scores(normalized_tuple, sequences):
                    # Get the last 5 tokens from each sequence
                    last_5_tokens = sequences[:, -5:]
                    sequence_scores = []
                    # Iterate over each tensor in the normalized tuple
                    for i in range(5):
                        # Get the probabilities corresponding to the ith token in last_5_tokens
                        probs = normalized_tuple[i].gather(1, last_5_tokens[:, i].unsqueeze(1))
                        sequence_scores.append(probs)
                    # Convert the list of tensors into a single tensor
                    sequence_scores = torch.cat(sequence_scores, dim=-1)
                    # Calculate the average score over the last 5 positions for each sequence
                    sequence_scores = sequence_scores.mean(dim=-1)
                    return sequence_scores

                outputs.scores = normalize_tuple(outputs.scores)
                outputs.sequences_scores = calculate_sequence_scores(outputs.scores, outputs.sequences)
                sorted_indices = outputs.sequences_scores.argsort(descending=True)
                sorted_sequences = outputs.sequences[sorted_indices]
                K = 10
                count = 0
                for sequence in sorted_sequences:
                    sequence = self.tokenizer.decode(sequence).split(' ')
                    uid = sequence[1][1:]
                    if len(topk[uid]) >= K:
                        continue
                    recommended_token = sequence[-1]
                    recommended_item = recommended_token[1:]
                    if not recommended_token.startswith("P"):
                        continue
                    if recommended_item not in self.user_negatives[uid]:
                        count = +1
                        continue
                    if recommended_item in topk[uid]:
                        continue
                    topk[uid].append(recommended_item)
                    topk_sequences[uid].append(sequence)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in topk.values()) / len(topk))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        return topk, topk_sequences


    def evaluate(self, model):
        # Generate paths for the test users
        # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok

        topks, topk_sequences = self.__generate_topks_withWordLevel(model)
        custom_model_name = self.custom_model_name.split('/')[:-1]
        check_dir(f"./results/{self.dataset_name}/{custom_model_name}")

        pickle.dump(topks, open(f"./results/{self.dataset_name}/{custom_model_name}/topk_items.pkl", "wb"))
        pickle.dump(topk_sequences, open(f"./results/{self.dataset_name}/{custom_model_name}/pred_paths.pkl", "wb"))   
        metrics = {"ndcg": [], "mmr": [], }
        for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
            hits = []
            for recommended_item in topk:
                if recommended_item in self.test_set[uid]:
                    hits.append(1)
                else:
                    hits.append(0)
            while len(hits) < 10:
                hits.append(0)
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
            metrics["ndcg"].append(ndcg)
            metrics["mmr"].append(mmr)

        print(
            f"no of users: {len(self.test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
        metrics_ = dict()
        for k in metrics:
            metrics_[f'eval_{k}'] = np.mean(metrics[k])
        return metrics_


def random_baseline(args: argparse.Namespace):
    """
    Recommendation evaluation
    """
    dataset_name = args.dataset
    test_set = get_set(dataset_name, set_str='test')

    def get_user_negatives(dataset_name: str) -> Dict[str, List[str]]:
        data_dir = f"data/{dataset_name}"
        ikg_ids = set(get_pid_to_eid(data_dir).values())
        uid_negatives = {}
        # Generate paths for the test set
        train_set = get_set(dataset_name, set_str='train')
        for uid, items in tqdm(train_set.items(), desc="Calculating user negatives", colour="green"):
            uid_negatives[uid] = list(ikg_ids - set(items))
        return uid_negatives

    user_negatives = get_user_negatives(dataset_name)
    topk = {}
    metrics = {"ndcg": [], "mmr": []}
    for uid in tqdm(list(test_set.keys()), desc="Evaluating", colour="green"):
        topk[uid] = random.sample(user_negatives[uid], 10)
        hits = []
        for recommended_item in topk[uid]:
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
        ndcg = ndcg_at_k(hits, len(hits))
        mmr = mmr_at_k(hits, len(hits))
        metrics["ndcg"].append(ndcg)
        metrics["mmr"].append(mmr)
    print("Random baseline:")
    print(f"no of users: {len(test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")



def get_best_checkpoint(model_folder):
    #get the checkpoint with the highest step number in filename
    checkpoints_filenames = [f for f in os.listdir(model_folder) if f.startswith("checkpoint")]
    checkpoints_filenames.sort()
    return checkpoints_filenames[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--task", type=str, default="end-to-end", help="{pretrain, finetune, end-to-end}")
    parser.add_argument("--loading_checkpoint", type=bool, default=True, help="True to load checkpoint False to load from model-weights")
    parser.add_argument("--sample_size", type=str, default="1000",
                        help="Which sample size dataset to use for fine-tuning/end-to-end")
    # Model arguments
    parser.add_argument("--model", type=str, default="gpt2-large",
                        help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--logit_processor_type", type=str, default="gcd",
                        help="Path sequence deconding method: default to Graph Constrained Decoding")    
    parser.add_argument("--n_hop", type=str, default="3",
                        help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--test_batch_size", type=int, default=256, help="Test batch size")
    parser.add_argument("--infer_batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=10,
                        help="Number of sequences generated for each user at inference time")
    
 

    args = parser.parse_args()

    set_seed(SEED)
    
    print(f'sample_size: {args.sample_size}, model: {args.model}, n_hop: {args.n_hop}')
    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset
 
    model_folder = f"clm-{args.task}-{args.dataset}-{args.model}-{args.sample_size}-{args.n_hop}-{args.logit_processor_type}"
    if args.loading_checkpoint:
        model_folder = f"clm-{args.task}-{args.dataset}-{args.model}-{args.sample_size}-{args.n_hop}-{args.logit_processor_type}"
        best_checkpoint = get_best_checkpoint(model_folder)
        model_folder = f"{model_folder}/{best_checkpoint}"
        print(f'loading: {model_folder}')
    else:
        model_folder = f"model-weights/{args.dataset}/{model_folder}"
    print("Loading CKPT...")
    if 'plm-rec' in model_name:
        model = PERLM.from_pretrained(model_folder).to(args.eval_device)
    else:
        model = PLMRec.from_pretrained(model_folder).to(args.eval_device)

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)
    
    print("Evaluating...")
    Evaluator(
        dataset_name = args.dataset,
        tokenized_kg = tokenized_kg,
        n_hop = args.n_hop,
        infer_batch_size = args.infer_batch_size,
        n_sequences_per_user = args.n_seq_infer,
        tokenizer = tokenizer,
        eval_device = args.eval_device,
        custom_model_name = model_folder,
        logit_processor_type=args.logit_processor_type,
    ).evaluate(model)
 
