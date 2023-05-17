import os
import argparse
from pathlm.models.PGPR.pgpr_utils import * 
from pathlm.sampling.container.kg_analyzer import KGstats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    parser.add_argument('--max_n_paths', type=int, default=100000, help='Max number of paths sampled for each user.')
    parser.add_argument('--max_hop', type=int, default=3, help='Max number of hops.')

    args = parser.parse_args()
    ML1M = 'ml1m'
    LFM1M ='lfm1m'
    CELL='cellphones'
    ROOT_DIR = os.environ('TREX_DATA_ROOT') if 'TREX_DATA_ROOT' in os.environ else '..'
    # Dataset directories.
    DATA_DIR = {
        ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed',
        LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed',
        CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed'
    }
    dataset_name = 'ml1m'
    dirpath = DATA_DIR[dataset_name]#.replace('ripple', 'kgat')
    ml1m_kg = KGstats(dataset_name, dirpath)

    MAX_HOP = args.max_hop
    #PROB = 0.01
    N_PATHS = args.max_n_paths
    LOGDIR = 'paths_random_walk' + f'__hops_{MAX_HOP}__npaths_{N_PATHS}'
    #ml1m_kg.path_sampler(max_hop=MAX_HOP, p=PROB, logdir=LOGDIR,ignore_rels=set([10]))    
    ml1m_kg.random_walk_sampler(max_hop=MAX_HOP, logdir=LOGDIR,ignore_rels=set( ), max_paths=N_PATHS)
    dataset_name = 'lfm1m'
    dirpath = DATA_DIR[dataset_name]#.replace('ripple', 'kgat')
    lfm1m_kg = KGstats(dataset_name, dirpath)	
    lfm1m_kg.random_walk_sampler(max_hop=MAX_HOP, logdir=LOGDIR,ignore_rels=set( ), max_paths=N_PATHS)

    #ml1m_kg.compute_metapath_frequencies()
    #lfm1m_kg.compute_metapath_frequencies()    

 
    #lfm1m_kg.path_sampler(max_hop=MAX_HOP, p=PROB, logdir=LOGDIR,ignore_rels=set() ) 