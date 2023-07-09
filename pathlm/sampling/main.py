import os
import argparse
import random

from transformers import set_seed
from pathlm.utils import SEED
from pathlm.models.rl.PGPR.pgpr_utils import * 
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.sampling.scoring.scorer import TransEScorer

def none_or_str(value):
    if value == 'None':
        return None
    return value
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    parser.add_argument('--max_n_paths', type=int, default=100, help='Max number of paths sampled for each user.')
    parser.add_argument('--max_hop', type=none_or_int, default=3, help='Max number of hops.')
    parser.add_argument("--itemset_type", type=str, default='inner', help="Choose whether final entity of a path is a product\nin the train interaction set of a user, outer set, or any reachable item {inner,outer,all} respectively")
    parser.add_argument("--collaborative", type=bool, default=False, help="Wether paths should be sampled considering users as intermediate entities")
    parser.add_argument("--with_type", type=bool, default=False, help="Typified paths")
    parser.add_argument('--nproc', type=int, default=4, help='Number of processes to sample in parallel')
    parser.add_argument("--start_type", type=none_or_str, default=USER, help="Start paths with chosen type")
    parser.add_argument("--end_type", type=none_or_str, default=PRODUCT, help="End paths with chosen type")
    args = parser.parse_args()

    set_seed(SEED)

    ML1M = 'ml1m'
    LFM1M ='lfm1m'
    CELL='cellphones'

    # root dir is current directory (according to the location from where this script is run)
    # e.g. if pathlm/sampling/main.py then ./ translates to pathlm
    ROOT_DIR = './'
    ROOT_DATA_DIR = os.path.join(ROOT_DIR, 'data')
    SAVE_DIR = os.path.join(ROOT_DATA_DIR, 'sampled')
    # Dataset directories.
    DATA_DIR = {
        ML1M: f'{ROOT_DATA_DIR}/{ML1M}/preprocessed',
        LFM1M: f'{ROOT_DATA_DIR}/{LFM1M}/preprocessed',
        CELL: f'{ROOT_DATA_DIR}/{CELL}/preprocessed'
    }
    dataset_name = args.dataset#'ml1m'
    #args.dataset = dataset_name
    dirpath = DATA_DIR[dataset_name]#.replace('ripple', 'kgat')
    #print(dirpath)
    data_dir_mapping = os.path.join(ROOT_DATA_DIR, f'{args.dataset}/preprocessed/mapping/')   
    kg = KGstats(args, args.dataset, dirpath,  save_dir=SAVE_DIR, data_dir=data_dir_mapping)

    MAX_HOP = args.max_hop
    #PROB = 0.01
    N_PATHS = args.max_n_paths
    itemset_type= args.itemset_type
    COLLABORATIVE=args.collaborative
    NPROC = args.nproc
    WITH_TYPE = args.with_type
    print('Closed destination item set: ',itemset_type)
    print('Collaborative filtering: ',args.collaborative)


    #LOGDIR = 'paths_random_walk_typed' + f'__hops_{MAX_HOP}__npaths_{N_PATHS}__closed_{itemset_type}__collaborative_{COLLABORATIVE}__start_type_{args.start_type}__end_type_{args.end_type}__typified_{args.with_type}'
    LOGDIR = f'dataset_{args.dataset}__hops_{MAX_HOP}__npaths_{N_PATHS}'
    

    #ml1m_kg.path_sampler(max_hop=MAX_HOP, p=PROB, logdir=LOGDIR,ignore_rels=set([10]))    
    #embedding_root_dir='../embedding-weights'
    embedding_root_dir= os.path.join(ROOT_DIR, 'embedding-weights')
    scorer=TransEScorer(dataset_name, embedding_root_dir)
    kg.random_walk_sampler(max_hop=MAX_HOP, logdir=LOGDIR,ignore_rels=set( ), max_paths=N_PATHS, itemset_type=itemset_type, 
        collaborative=COLLABORATIVE,
        nproc=NPROC,
        #embedding_root_dir='../embedding-weights',
        #scorer=scorer,
        num_beams=10,
        with_type=WITH_TYPE,
        start_ent_type=args.start_type,
        end_ent_type=args.end_type)


    '''
    dataset_name = 'lfm1m'
    dirpath = DATA_DIR[dataset_name]#.replace('ripple', 'kgat')
    args.dataset = dataset_name
    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')   
    lfm1m_kg = KGstats(args, args.dataset, dirpath,  data_dir=data_dir_mapping)
    scorer=TransEScorer(dataset_name, embedding_root_dir)

    LOGDIR = f'dataset_{args.dataset}__hops_{MAX_HOP}__npaths_{N_PATHS}'
    lfm1m_kg.random_walk_sampler(max_hop=MAX_HOP, logdir=LOGDIR,ignore_rels=set( ), max_paths=N_PATHS, itemset_type=itemset_type, 
        collaborative=COLLABORATIVE,
        nproc=NPROC,
        #embedding_root_dir='../embedding-weights',
        #scorer=scorer,
        num_beams=10,
        with_type=WITH_TYPE,
        start_ent_type=args.start_type,
        end_ent_type=args.end_type        )
    '''
    #ml1m_kg.compute_metapath_frequencies()
    #lfm1m_kg.compute_metapath_frequencies()    

 
    #lfm1m_kg.path_sampler(max_hop=MAX_HOP, p=PROB, logdir=LOGDIR,ignore_rels=set() ) 