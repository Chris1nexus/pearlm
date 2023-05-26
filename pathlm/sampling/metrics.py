import os
import argparse
from pathlm.models.PGPR.pgpr_utils import * 
from pathlm.sampling.container.kg_analyzer import KGstats


def item_coverage(stats_kg, path_dataset, end_item=True):
    n_users = len(stats_kg.user_dict)
    n_items = len(stats_kg.items)
    item_ids = set()
    for path in path_dataset:
        if end_item:
            item_ids.add(int(path[-1]) )
        else:
            for idx in range(2,len(path)+1, 4):
                item_ids.add(int(path[idx]))
    return len(item_ids)/n_items

def catalog_coverage(stats_kg, path_dataset,end_item=True):
    n_users = len(stats_kg.user_dict)
    n_items = len(stats_kg.items)

    n_inter = n_users*n_items
    #for uid in stats_kg.user_dict:
    #    inter_pids = stats_kg.user_dict[uid]
    #    n_inter += len(inter_pids)

    item_ids = set()
    for path in path_dataset:
        if end_item:
            item_ids.add(  (int(path[0]), int(path[-1]) )  )
        else:
            for idx in range(2,len(path)+1, 4):
                item_ids.add( (int(path[0]), int(path[idx]) )   )
    return len(item_ids)/n_inter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=LFM1M, help='One of {ml1m, lfm1m}')
    #parser.add_argument('--dataset_path', type=str, help='One of {ml1m, lfm1m}')



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
    dirpath = DATA_DIR[args.dataset]
    randwalk_filepath = os.path.join(*dirpath.split('/')[:-1], 'paths_random_walk', 'paths.txt')

    stats_kg = KGstats(args.dataset, dirpath)


    paths = []
    with open(randwalk_filepath) as f:
        for line in f:
            data = line.rstrip().split(' ')
            paths.append(data)
    print(paths[:5])


    res1 = item_coverage(stats_kg, paths)
    res2 = catalog_coverage(stats_kg, paths)
    print('Item coverage(end item only): ', res1)
    print('Catalog coverage(end item only): ', res2)
    res1 = item_coverage(stats_kg, paths, end_item=False)
    res2 = catalog_coverage(stats_kg, paths,  end_item=False)
    print('Item coverage(all items in path): ', res1)
    print('Catalog coverage(all items in path): ', res2)

