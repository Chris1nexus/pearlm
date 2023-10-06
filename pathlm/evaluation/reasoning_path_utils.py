from collections import defaultdict

import numpy as np
from utils import *

LIR = "lir"
SEP = "sep"
PTD = "ptd"
LID = "lid"
SED = "sed"
PTC = "ptc"
PPT = "ppt"
FIDELITY = "fidelity"
PATH_QUALITY_METRICS = [LIR, SEP, PTD, LID, SED, PTC, PPT, FIDELITY]


def entity2plain_text(dataset_name, model_name):
    entity2name = entity2plain_text(dataset_name, model_name)
    return entity2name


# (self_loop user 0) (watched movie 2408) (watched user 1953) (watched movie 277) #hop3
# (self_loop user 0) (mention word 2408) (described_as product 1953) (self_loop product 1953) #hop2
def get_linked_interaction_triple(path):
    linked_interaction_id, linked_interaction_rel, linked_interaction_type = path[1][-1], path[1][0], path[1][1]
    return linked_interaction_id, linked_interaction_rel, linked_interaction_type


def get_shared_entity_tuple(path):
    path_type = path[-1][0]
    if path_type == 'self_loop':  # Handle size 3
        shared_entity_id, shared_entity_type = path[-2][-1], path[-2][1]
        return shared_entity_id, shared_entity_type
    shared_entity_id, shared_entity_type = path[-2][-1], path[-2][1]
    return shared_entity_id, shared_entity_type


def get_path_type(path):
    path_type = path[-1][0]
    if path_type == 'self_loop':  # Handle size 3
        path_type = path[-2][0]
    return path_type


def get_path_pattern(path):
    return [path_tuple[0] for path_tuple in path[1:]]


def get_no_path_patterns_in_kg(dataset_name):
    from models.PGPR.pgpr_utils import PATH_PATTERN
    return len(PATH_PATTERN[dataset_name].keys())



def load_SEP_matrix(dataset_name, model_name):
    data_dir = get_data_dir(dataset_name)
    sep_matrix_filepath = os.path.join(data_dir, "SEP_matrix.pkl")
    if os.path.isfile(sep_matrix_filepath):
        print("Loading pre-computed SEP-matrix")
        with open(sep_matrix_filepath, 'rb') as f:
            SEP_matrix = pickle.load(f)
        f.close()
    else:
        print("Generating SEP-matrix")
        SEP_matrix = generate_SEP_matrix(dataset_name, model_name)
        with open(sep_matrix_filepath, 'wb') as f:
            pickle.dump(SEP_matrix, f)
        f.close()
    return SEP_matrix


def generate_SEP_matrix(dataset_name, model_name):
    def normalized_ema(values):
        if max(values) == min(values):
            values = np.array([i for i in range(len(values))])
        else:
            values = np.array([i for i in values])
        values = pd.Series(values)
        ema_vals = values.ewm(span=len(values)).mean().tolist()
        min_res = min(ema_vals)
        max_res = max(ema_vals)
        return [(x - min_res) / (max_res - min_res) for x in ema_vals]

    # Precompute entity distribution
    SEP_matrix = {}
    degrees = load_kg(dataset_name, model_name).degrees
    for type, eid_degree in degrees.items():
        eid_degree_tuples = list(zip(eid_degree.keys(), eid_degree.values()))
        eid_degree_tuples.sort(key=lambda x: x[1])
        ema_es = normalized_ema([x[1] for x in eid_degree_tuples])
        pid_weigth = {}
        for idx in range(len(ema_es)):
            pid = eid_degree_tuples[idx][0]
            pid_weigth[pid] = ema_es[idx]

        SEP_matrix[type] = pid_weigth

    return SEP_matrix

def print_path_quality_metrics(avg_metrics, c_fairness):
    print("\n***---Path Quality---***")
    print("Average for the entire user base:", end=" ")
    for metric, group_value in avg_metrics.items():
        print(f"{metric}: {group_value[OVERALL]:.3f}", end=" | ")
    print("")

    for metric, groups_value in avg_metrics.items():
        print(f"\n--- {metric}---")
        for group, value in groups_value.items():
            print(f"{group}: {value:.3f}", end=" | ")
        print("")
    print("\n")

    print("\n***---Rec CFairness Differences---***")
    for class_group, metric_tuple in c_fairness.items():
        for metric, tuple in metric_tuple.items():
            group_class, avg_value = tuple
            print(f"{metric} Pairwise diff {class_group}: {avg_value:.3f}", end=" | ")
        print("\n")