import csv
from collections import defaultdict
from os.path import join
import os
from typing import Dict, List

from tqdm import tqdm

SEED = 2023

import torch 
import random
import numpy as np




# Check if dir exists and create if not
def check_dir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Clean entities name from _ or previxes
def normalise_name(name: str) -> str:
    if name.startswith("Category:"):
        name = name.replace("Category:", "")
    return name.replace("_", " ")

# Get eid2name dictionary to allow conversions from eid to name
def get_eid_to_name_map(data_dir: str) -> dict:
    e_map_path = join(data_dir, 'preprocessed', 'e_map.txt')
    eid2name = {}
    with open(e_map_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            eid = row[0]
            ename = normalise_name(row[1])
            eid2name[eid] = ename
    f.close()
    return eid2name

def set_seed(seed=SEED, use_deterministic=True):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic)
    np.random.seed(seed) 
    random.seed(seed)

# Get pid2eid dictionary to allow conversions from pid to eid
def get_pid_to_eid(data_dir: str) -> dict:
    i2kg_path = join(data_dir, 'preprocessed', 'i2kg_map.txt')
    pid2eid = {}
    with open(i2kg_path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            eid = row[0]
            pid = row[1]
            pid2eid[pid] = eid
    f.close()
    return pid2eid

# Get rid2name dictionary to allow conversion from rid to name
def get_rid_to_name_map(data_dir: str) -> dict:
    r_map_path = join(data_dir, 'preprocessed', 'r_map.txt')
    rid2name = {}
    with open(r_map_path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            rid = row[0]
            rname = normalise_name(row[-1])
            rid2name[rid] = rname
    f.close()
    return rid2name

def get_data_dir(dataset_name: str) -> str:
    return join('data', dataset_name)

def get_set(dataset_name: str, set_str: str='test') -> Dict[str, List[int]]:
    data_dir = f"data/{dataset_name}"
    # Note that test.txt has uid and pid from the original dataset so a convertion from dataset to entity id must be done
    i2kg = get_pid_to_eid(data_dir)

    # Generate paths for the test set
    curr_set = defaultdict(list)
    with open(f"{data_dir}/preprocessed/{set_str}.txt", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, item_id, rating, timestamp = row
            user_id = user_id  # user_id starts from 1 in the augmented graph starts from 0
            item_id = i2kg[item_id]  # Converting dataset id to eid
            curr_set[user_id].append(item_id)
    f.close()
    return curr_set
