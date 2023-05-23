from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlm.utils import get_eid_to_name_map, get_data_dir, get_pid_to_eid, get_set

TOKENIZER_DIR = './tokenizers'

MLM_MODELS = ["bert-large", "roberta-large"]
CLM_MODELS = ['WordLevel', 'gpt2-xl', "stabilityai/stablelm-base-alpha-3b"]

WORD_LEVEL_TOKENIZER = "./tokenizers/ml1m/WordLevel.json"

def get_entity_vocab(dataset_name: str, model_name: str) -> List[int]:
    fast_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    entity_list = get_eid_to_name_map(get_data_dir(dataset_name)).values()

    def tokenize_and_get_input_ids(example):
        return fast_tokenizer(example).input_ids

    ans = []
    for entity in entity_list:
        ans.append(tokenize_and_get_input_ids(entity))
    return [item for sublist in ans for item in sublist]

def get_user_negatives_tokens_ids(dataset_name: str, tokenizer) -> Dict[str, List[str]]:
    data_dir = f"data/{dataset_name}"
    ikg_ids = list(get_pid_to_eid(data_dir).values())
    ikg_token_ids = set([tokenizer(f"P{ikg_id}").input_ids[1] for ikg_id in ikg_ids])
    uid_negatives = {}
    # Generate paths for the test set
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        train_items = [tokenizer(f"P{item}").input_ids[1] for item in train_set[uid]]
        val_items = [tokenizer(f"P{item}").input_ids[1] for item in valid_set[uid]]
        uid_negatives[uid] = list(set(ikg_token_ids - set(train_items) - set(val_items) - set([0])))
    return uid_negatives