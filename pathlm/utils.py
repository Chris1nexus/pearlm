import csv
from os.path import join
import os

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
    with open(e_map_path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            eid = row[0]
            ename = normalise_name(row[1])
            eid2name[eid] = ename
    f.close()
    return eid2name

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