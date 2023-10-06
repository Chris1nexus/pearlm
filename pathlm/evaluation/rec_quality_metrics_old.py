import numpy as np
from utils import *
from easydict import EasyDict as edict

"""
Implemented evaluation
"""
NDCG = "ndcg"
MMR = "mmr"
SERENDIPITY = "serendipity"
COVERAGE = "coverage"
DIVERSITY = "diversity"
NOVELTY = "novelty"
CFAIRNESS = "cfairness"
PFAIRNESS = "pfairness"

REC_QUALITY_METRICS_TOPK = [NDCG, MMR, SERENDIPITY, DIVERSITY,
                            NOVELTY, PFAIRNESS]  # CHECK EVERYTIME A NEW ONE IS IMPLEMENTED IF IS LOCAL O GLOBAL
REC_QUALITY_METRICS_GLOBAL = [COVERAGE, CFAIRNESS]
"""
Methods
"""

def print_rec_quality_metrics(avg_metrics, c_fairness):
    print("\n***---Recommandation Quality---***")
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

def exposure_pfairness(topk_items, pid2provider_popularity):
    exposure_providers_topk = [pid2provider_popularity[pid] for pid in topk_items]
    return np.mean(exposure_providers_topk)

def consumer_fairness(metrics_distrib, avg_metrics):
    # Compute consumer fairness
    group_name_values = {
        GENDER: ["M", "F"],
        AGE: ["50-55", "45-49", "25-34", "56+", "18-24", "Under 18", "35-44"]
    }
    cfairness_metrics = {}
    c_fairness_rows = []
    for group_class in [GENDER, AGE]:
        cfairness_metrics[group_class] = {}
        for metric, group_values in avg_metrics.items():
            #if len(group_name_values[group_class]) == 2:
            #    group1, group2 = group_name_values[group_class]
            #    #statistically_significant = statistical_test(metrics_distrib[metric][group1], metrics_distrib[metric][group2]) TODO
            #    cfairness_metrics[group_class][metric] = (group1, group2, avg_metrics[metric][group1] -
            #                                           avg_metrics[metric][group2],) #statistically_significant) TODO
            if len(group_name_values[group_class]) >= 2:
                pairwise_diffs = []
                for group1 in group_name_values[group_class]:
                    for group2 in group_name_values[group_class]:
                        if group1 != group2:
                            pairwise_diffs.append(abs(avg_metrics[metric][group1] - avg_metrics[metric][group2]))
                cfairness_metrics[group_class][metric] = (group_class, np.mean(pairwise_diffs), ) #statistically_significant) TODO

    return cfairness_metrics
# REC_QUALITY_METRICS = [NDCG, MMR, SERENDIPITY, COVERAGE, DIVERSITY, NOVELTY]
# FAIRNESS_METRICS = [CFAIRNESS, PFAIRNESS]
