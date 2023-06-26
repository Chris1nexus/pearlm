import multiprocessing as mp
from collections import defaultdict   
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import os
import pickle
import itertools
import functools
import pandas as pd
from collections import deque
import random

from pathlm.models.rl.PGPR.pgpr_utils import *
from pathlm.sampling.container.kg import KnowledgeGraph
from pathlm.sampling.container.path_trie import PathTrie
from pathlm.sampling.container.file_io import PathFileIO
from pathlm.sampling.container.constants import LiteralPath
from pathlm.sampling.scoring.scorer  import TransEScorer
from pathlm.models.rl.PGPR.data_utils import Dataset


def bfs(pid, aug_kg, ptrie, n_hop, product_entity_name, user_entity_name, u2p_rel_name):

    PROD_ENT = product_entity_name
    USER = user_entity_name
    U2P_REL = u2p_rel_name
    trie = ptrie.trie
    pid_to_reachable = set()
    pid_reaches_unique_metapaths = set()
    ptrie_terminal_counts = defaultdict(int)
    seen_entities = defaultdict(set)

    q = [( (pid,PROD_ENT), trie[USER][U2P_REL][PROD_ENT], 0)]
    
    while len(q):
        ((h_id, h_type), cur_trie, hop) = q.pop()
        seen_entities[h_type].add(h_id)
        if hop >= n_hop:
            if h_type == PROD_ENT and ptrie.TERMINATION in cur_trie:
                #cur_trie[self.ptrie.TERMINATION].count += 1
                ptrie_terminal_counts[cur_trie[ptrie.TERMINATION].id] += 1
                pid_to_reachable.add(h_id)
                pid_reaches_unique_metapaths.add(cur_trie[ptrie.TERMINATION].id)
                #pid_to_reachable[pid].add(h_id)
                #pid_reached_from_unique_metapaths[h_id].add(cur_trie[self.ptrie.TERMINATION].id)
                #pid_reaches_unique_metapaths[pid].add(cur_trie[self.ptrie.TERMINATION].id)

                # item found at end of metapath, continue to next possible metapath
            continue
    
        for rel in cur_trie:
            cur = cur_trie[rel]
            if rel not in aug_kg[h_type][h_id]:
                continue
            for t_new in aug_kg[h_type][h_id][rel]:
                t_new_id, t_new_type = t_new
                                      
                #if not all_paths:
                #    if t_new_type in seen_entities and t_new_id in seen_entities[t_new_type]:
                #        continue
                if t_new_type not in cur:
                    continue

                seen_entities[t_new_type].add(t_new_id)
                q.append(((t_new_id, t_new_type), cur[t_new_type], hop+1) )                   

    return pid, pid_to_reachable, pid_reaches_unique_metapaths,ptrie_terminal_counts

'''
def dfs_sample_paths(pid, kg, items, n_hop, KG2T, R2T, PROD_ENT, logdir, ignore_rels=set(), p=0.001, max_paths=4000):
    dirpath = logdir#os.path.join(logdir, 'paths3')
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, f'paths_{pid}.txt' ), 'w') as f:
        
            q = []
            path = []
            q.append(((pid, PROD_ENT) ,1, path) )
            #KG2T = KG_RELATION[self.dataset_name]
            #print(PROD_ENT, KG2T)
            #found_paths = []
            cnt = 0
            while len(q):
                (ent_id,ent_type), dist, cur_path = q.pop()

                if ent_id not in kg:
                    continue
                if n_hop is not None and  dist>1 and ent_id in items :
                    #found_paths.append(cur_path)
                    rand_num = random.random()
                    if rand_num < p:
                        f.write(','.join(cur_path) + '\n' )  
                        cnt += 1
                        if cnt >= max_paths:
                            break
                if dist >= n_hop:
                    continue

                for r, tails in kg[ent_id].items():
                    if r in ignore_rels:
                        continue
                    r_type = R2T[r]
                    for t_id in tails:
                        #print(ent_type, r,'sss', t_id)
                        t_type = KG2T[ent_type][r_type]
                        #if t not in seen:
                        #    q.append( (t,dist+1))
                        #    seen.add(t)
                        #src = (ent_id, ent_type)
                        #dst = (t_id, t_type)
                        path_cont = [x for x in cur_path] 
                        path_cont.append( f'{ent_id} {r} {t_id}')
                        q.append( ((t_id,t_type) ,dist+1, path_cont))#[x for x in cur_path] + [ (src,r,dst) ]  ) )   
    #return found_paths


def random_walk_sample_paths(uid, kg, items, n_hop, KG2T, R2T, PROD_ENT, U2P_REL, logdir, pathIO, user_dict, ignore_rels=set(), max_paths=None):
    dirpath = logdir#os.path.join(logdir, 'paths3')
    os.makedirs(dirpath, exist_ok=True)
    user_products = user_dict[uid]
    unseen_products = list(items - user_products)
    user_products = list(user_products)

    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
        
            cnt = 0
            
            while cnt < max_paths:
                cur_hop = 1
                pid = random.choice(user_products)
                cur_ent_id = pid
                path = [uid, U2P_REL, cur_ent_id]
                #path.append( cur_ent_id)
                while cur_hop < n_hop:
                    if cur_ent_id not in kg:
                        break                    
                    valid_rels = list(kg[cur_ent_id].keys())
                    while True:
                        rel = random.choice(valid_rels )

                        if rel not in ignore_rels:
                            break
                    if cur_hop == n_hop-1:
                        tail_id = random.choice(unseen_products)
                    else:
                        tail_id = random.choice(kg[cur_ent_id][rel] )
                    path.append( rel)
                    path.append( tail_id)
                    cur_ent_id = tail_id
                    
                    cur_hop += 1
                
                if cur_hop >= n_hop:
                    path = [ str(x) for x in path]
                    #path = [x if isinstance(x,int) else x.item()  for x in path]
                    #pathIO.write_to_file(path, n_hop-1, fp)
                    fp.write(' '.join(path) + '\n' )  
                cnt += 1
'''
'''
def random_walk_typed_content_based(uid, kg, items, n_hop, KG2T, R2T, PROD_ENT, U2P_REL, logdir, 
        pathIO, user_dict, uid_ext_ent_mapping, uid_outer_ext_ent_mapping, ignore_rels=set(), max_paths=None, itemset_type='inner'):
    dirpath = logdir#os.path.join(logdir, 'paths3')
    os.makedirs(dirpath, exist_ok=True)

    
    def dfs(prev_ent_id, cur_ent_id, cur_hop, path, prev_rel):
        if cur_hop >= n_hop:
            path = [ str(x) for x in path]
            #path = [x if isinstance(x,int) else x.item()  for x in path]
            #pathIO.write_to_file(path, n_hop-1, fp)
            fp.write(' '.join(path) + '\n' )  
            
            return True



        if cur_ent_id not in kg:
            return False
        valid_rels = list(kg[cur_ent_id].keys())
        random.shuffle(valid_rels)
        for rel in valid_rels:
            if rel in ignore_rels:
                continue
            
            candidates = kg[cur_ent_id][rel] 
            if cur_hop == n_hop-1:
                key = uid,rel,cur_ent_id
                
                if itemset_type == 'inner' and key in uid_ext_ent_mapping:
                    candidates = uid_ext_ent_mapping[key]
                elif itemset_type == 'outer' and key in uid_outer_ext_ent_mapping:
                    candidates = uid_outer_ext_ent_mapping[key]
                # else the default itemset is used (all  items reachable from current entity, which is not an item at the 'n_hop-1' hop)
            if len(candidates) == 0:
                continue             

            if rel == prev_rel:
                path.append( LiteralPath.back_rel )
            else:
                path.append( LiteralPath.fw_rel )
            path.append( rel)
            random.shuffle(candidates)
            

            for next_ent_id in candidates:
                if next_ent_id == prev_ent_id:
                    continue

                if next_ent_id not in items:
                    path.append(LiteralPath.ent)
                else:              
                    if next_ent_id in user_products:  
                        path.append(LiteralPath.recom_prod)
                    else:
                        path.append(LiteralPath.prod)
                path.append(next_ent_id)
                if dfs(cur_ent_id, next_ent_id, cur_hop+1, path):
                    path.pop()
                    path.pop()

                    path.pop()
                    path.pop()                    
                    return True
                path.pop()

                path.pop()
            path.pop()

            path.pop()
        return False
    user_products = list(user_dict[uid])
    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
            cnt = 0
            
            while cnt < max_paths:
                cur_hop = 1
                pid = random.choice(user_products)
                if pid not in kg:
                    continue                
                cur_ent_id = pid
                path = [uid, U2P_REL, cur_ent_id]
                dfs(-1, cur_ent_id, cur_hop, path)
                cnt += 1
'''

def random_walk_typified_beamsearch(uid, dataset_name, kg, items, n_hop, KG2T, R2T, USER_ENT, PROD_ENT, EXT_ENT, U2P_REL, logdir, 
        user_dict, ignore_rels=set(), max_paths=None, itemset_type='inner', REL_TYPE2ID=None, collaborative=True,
        num_beams=10, scorer=None, dataset_info=None,
                 start_ent_type=USER,
         end_ent_type=PRODUCT):
    dirpath = logdir
    os.makedirs(dirpath, exist_ok=True)


    
    REL_TYPE2ID[U2P_REL] = LiteralPath.interaction_rel_id
    u2p_rel_id = LiteralPath.interaction_rel_id
    user_prod_cache = dict()
    unique_path_set = set()
    def find_candidates(uid, candidate_paths):
        
        for cur_hop in range(1, n_hop):
            next_candidates = []
            
            for idx, (path, path_elems, score) in enumerate(candidate_paths):

                    def search_candidate(path_elems, next_candidates, cur_hop, prev_score):
                            prev_ent_t, cur_ent_t, prev_ent_id, cur_ent_id, prev_rel = path_elems
                            #if cur_ent_id not in kg[cur_ent_t]:
                            #    continue
                            valid_rels = list(kg[cur_ent_t][cur_ent_id].keys())
                            random.shuffle(valid_rels)

                            for rel in valid_rels:
                                if rel in ignore_rels:
                                    continue
                                rel_id = REL_TYPE2ID[rel]
                                
                                candidate_types = kg[cur_ent_t][cur_ent_id][rel]



                                random.shuffle(candidate_types)
                                for cand_type in candidate_types:
                                    if not collaborative and cand_type == USER_ENT:
                                        continue 
                                    candidates = kg[cur_ent_t][cur_ent_id][rel][cand_type]

                                    if cur_hop == n_hop-1:

                                        
                                        if itemset_type == 'inner':
                                            if (rel, cand_type) not in user_prod_cache:
                                                candidates = list(user_dict[uid].intersection(set(candidates)))
                                                user_prod_cache[rel,cand_type] = candidates
                                            else:
                                                candidates = user_prod_cache[rel,cand_type]
                                        elif itemset_type == 'outer':
                                            if (rel, cand_type) not in user_prod_cache:
                                                candidates = list(set(candidates) - user_dict[uid]   )
                                                user_prod_cache[rel,cand_type] = candidates
                                            else:
                                                candidates = user_prod_cache[rel,cand_type]                                            
                                        elif itemset_type == 'all':
                                            # candidate set is left unchanged
                                            #print('ZZZZZZZZZ')
                                            pass
                                        else:
                                            #print('YYYYYYYYYYYYY')
                                            continue

                                        #print('AAAAAA ', len(candidates))
                                        ent_t = None
                                        if cur_ent_t == USER_ENT:
                                            ent_t = USER_ENT
                                        else:
                                            ent_t = EXT_ENT
                                        key = uid, rel_id, ent_t, cur_ent_id
                     
                                    if len(candidates) == 0:
                                        continue             
                                    if rel_id == prev_rel:
                                        path.append( LiteralPath.back_rel )
                                    else:
                                        path.append( LiteralPath.fw_rel )
                                    path.append( f'{LiteralPath.rel_type}{rel_id}' )#rel)
                                    random.shuffle(candidates)
                                    

                                    for next_ent_id in candidates:
                                        if next_ent_id == prev_ent_id and prev_ent_t == cand_type:
                                            #print('QQQQQ ', next_ent_id, path)
                                            continue
                                        if cand_type == USER_ENT:
                                            if next_ent_id == uid:
                                                prefix = LiteralPath.main_user
                                            else:
                                                prefix = LiteralPath.oth_user
                                            type_prefix = LiteralPath.user_type

                                        elif cand_type == PROD_ENT:#next_ent_id not in items:
                                            if cur_hop == n_hop-1:
                                                assert next_ent_id in user_dict[uid], f'Error: {next_ent_id} not found for user: {uid} in {user_dict[uid]}' 
                                            if next_ent_id in user_dict[uid]:  
                                                prefix = LiteralPath.recom_prod
                                            else:
                                                prefix = LiteralPath.prod    
                                            type_prefix = LiteralPath.prod_type                  
                                        else:              
                                            prefix = LiteralPath.ent
                                            type_prefix = LiteralPath.ent_type
                                        

                                        path.append(prefix)

                                        path.append(f'{type_prefix}{next_ent_id}')
                                        
                                        path_str = ''.join([x for x in path])
                                        if path_str in unique_path_set:
                                            path.pop()
                                            path.pop()
                                            continue
                                        else:
                                            unique_path_set.add(path_str)
                                        path_elems = [cur_ent_t, cand_type, cur_ent_id, next_ent_id, rel_id]

                                        cur_score = 1.
                                        if scorer is not None:
                                            # TODO compute score
                                            cur_local_eid = dataset_info.global_eid_to_cat_eid[cur_ent_t][cur_ent_id]
                                            next_local_eid = dataset_info.global_eid_to_cat_eid[cand_type][next_ent_id]
 
                                            cur_score = scorer.score(cur_ent_t, cur_local_eid, cand_type, next_local_eid, rel, prev_score=prev_score)                                
                                        
                                        next_candidates.append(([x for x in path], path_elems, cur_score))
                                        path.pop()
                                        path.pop()
                                        path.pop()
                                        path.pop()                                        
                                        return 
                                    path.pop()
                                    path.pop()
                    for i in range(num_beams):
                        search_candidate(path_elems, next_candidates, cur_hop, score)

            # topk filter
            topk_size = num_beams
            #rint('hop: ', cur_hop)
            #if len(next_candidates):
            #    print([ x[0] for x in next_candidates], cur_hop, len(next_candidates[0][0]), len(next_candidates[0]) ) 
            candidate_paths = next_candidates

            candidate_paths = sorted(candidate_paths, key=lambda x: x[-1], reverse=True)[:topk_size]
        #if len(candidate_paths):
        #    print(candidate_paths[0][0], candidate_paths[0][2])
        return candidate_paths
                
                


    user_products = list(user_dict[uid])
    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
            cnt = 0
            #progressbar = tqdm(total=max_paths)
            while cnt < max_paths:
                cur_hop = 1
                candidate_paths = []
                #for i in range(num_beams):
                pid = random.choice(user_products)
                if pid not in kg[PROD_ENT]:
                    continue                
                cur_ent_id = pid

                rel_id = REL_TYPE2ID[U2P_REL]
                path = [LiteralPath.main_user, f'{LiteralPath.user_type}{uid}',
                        LiteralPath.fw_rel, f'{LiteralPath.rel_type}{rel_id}',#U2P_REL, 
                        LiteralPath.recom_prod, f'{LiteralPath.prod_type}{cur_ent_id}']
                
                prev_ent_t = USER_ENT
                cur_ent_t = PROD_ENT
                path_elems = [prev_ent_t, cur_ent_t, uid, pid, rel_id]
                score = 1.
                if scorer is not None:
                    # TODO compute score
                    #print(dataset_info.global_eid_to_cat_eid[prev_ent_t])
                    cur_local_eid = dataset_info.global_eid_to_cat_eid[prev_ent_t][uid]
                    #print( len(dataset_info.global_eid_to_cat_eid[cur_ent_t].keys()), len(dataset_info.global_eid_to_cat_eid[cur_ent_t].values())  )
                    #print(sorted(dataset_info.global_eid_to_cat_eid[cur_ent_t].keys()) )
                    next_local_eid = dataset_info.global_eid_to_cat_eid[cur_ent_t][pid]   
                                             
                    score = scorer.score(prev_ent_t, cur_local_eid, cur_ent_t, next_local_eid, U2P_REL, prev_score=score)      
                candidate_paths.append((path, path_elems, score))
                #print(uid, num_beams, max_paths-cnt)
                candidate_paths = find_candidates(uid, candidate_paths)[: ( min(num_beams, max_paths-cnt) )  ]
                #print(cnt, len(candidate_paths), max_paths)
                for path, path_elems, score in candidate_paths:
                    path = [ str(x) for x in path]
                    fp.write(' '.join(path) + '\n' )  
                #print(len(candidate_paths))
                #print()                
                cnt += max(len(candidate_paths),1)
                #progressbar.update(len(candidate_paths))


    





def random_walk_typified(uid, dataset_name, kg, items, n_hop, KG2T, R2T, USER_ENT, PROD_ENT, EXT_ENT, U2P_REL, logdir, 
         user_dict, ignore_rels=set(), max_paths=None, itemset_type='inner', REL_TYPE2ID=None, 
         collaborative=True,
         num_beams=10, scorer=None,
         dataset_info=None, 
         with_type=True,
         start_ent_type=USER,
         end_ent_type=PRODUCT):
    dirpath = logdir#os.path.join(logdir, 'paths3')
    os.makedirs(dirpath, exist_ok=True)

    REL_TYPE2ID[U2P_REL] = LiteralPath.interaction_rel_id
    u2p_rel_id = LiteralPath.interaction_rel_id
    user_prod_cache = dict()
    unique_path_set = set()    
    def dfs(uid, prev_ent_t, cur_ent_t, prev_ent_id, cur_ent_id, cur_hop, path, prev_rel, n_hop, start_ent_type, end_ent_type):
        if cur_hop >= n_hop:
            path = [ str(x) for x in path]
            #path = [x if isinstance(x,int) else x.item()  for x in path]
            #pathIO.write_to_file(path, n_hop-1, fp)
            path_str = ' '.join(path)
            if path_str in unique_path_set:
                return False
            else:
                unique_path_set.add(path_str)

            fp.write(path_str + '\n' )  
            
            return True



        if cur_ent_id not in kg[cur_ent_t]:
            return False
        valid_rels = list(kg[cur_ent_t][cur_ent_id].keys())
        random.shuffle(valid_rels)
        for rel in valid_rels:
            if rel in ignore_rels:
                continue
            rel_id = REL_TYPE2ID[rel]
            #candidates = kg[cur_ent_t][cur_ent_id][rel] 
            
            candidate_types = kg[cur_ent_t][cur_ent_id][rel]


            random.shuffle(candidate_types)
            for cand_type in candidate_types:
                if not collaborative and cand_type == USER_ENT:
                    continue 
                candidates = kg[cur_ent_t][cur_ent_id][rel][cand_type]

                if cur_hop == n_hop-1 and end_ent_type == PROD_ENT:

                    if itemset_type == 'inner':
                        if (rel, cand_type) not in user_prod_cache:
                            candidates = list(user_dict[uid].intersection(set(candidates)))
                            user_prod_cache[rel,cand_type] = candidates
                        else:
                            candidates = user_prod_cache[rel,cand_type]
                    elif itemset_type == 'outer':
                        if (rel, cand_type) not in user_prod_cache:
                            candidates = list(set(candidates) - user_dict[uid]   )
                            user_prod_cache[rel,cand_type] = candidates
                        else:
                            candidates = user_prod_cache[rel,cand_type]                                            
                    elif itemset_type == 'all':
                        # candidate set is left unchanged
                        pass
                    else:
                        continue

                    ent_t = None
                    if cur_ent_t == USER_ENT:
                        ent_t = USER_ENT
                    else:
                        ent_t = EXT_ENT
                    key = uid, rel_id, ent_t, cur_ent_id

                if len(candidates) == 0:
                    continue         
                if with_type:    
                    if rel_id == prev_rel:
                        path.append( LiteralPath.back_rel )
                    else:
                        path.append( LiteralPath.fw_rel )
                path.append( f'{LiteralPath.rel_type}{rel_id}' )#rel)
                random.shuffle(candidates)
                

                for next_ent_id in candidates:
                    if next_ent_id == prev_ent_id and prev_ent_t == cand_type:
                        continue
                    if cand_type == USER_ENT:
                        if next_ent_id == uid:
                            prefix = LiteralPath.main_user
                        else:
                            prefix = LiteralPath.oth_user
                        type_prefix = LiteralPath.user_type

                    elif cand_type == PROD_ENT:#next_ent_id not in items:
                        if cur_hop == n_hop-1 and end_ent_type == PROD_ENT:
                            assert next_ent_id in user_dict[uid], f'Error: {next_ent_id} not found for user: {uid} in {user_dict[uid]}' 
                        if next_ent_id in user_dict[uid]:  
                            prefix = LiteralPath.recom_prod
                        else:
                            prefix = LiteralPath.prod    
                        type_prefix = LiteralPath.prod_type                  
                    else:              
                        prefix = LiteralPath.ent
                        type_prefix = LiteralPath.ent_type
                    if with_type:
                        path.append(prefix)

                    path.append(f'{type_prefix}{next_ent_id}')
                    #dfs(prev_ent_t, uid, cur_ent_t, cur_ent_id, cur_hop, path)
                    if dfs(uid, cur_ent_t, cand_type, cur_ent_id, next_ent_id, cur_hop+1, path, rel_id, n_hop, start_ent_type, end_ent_type):
                        if with_type:
                            path.pop()
                        path.pop()
                        if with_type:
                            path.pop()
                        path.pop()                    
                        return True

                    if with_type:
                        path.pop()

                    path.pop()
                if with_type:
                    path.pop()
                path.pop()
        return False

    non_prod_entities = set([USER_ENT, EXT_ENT])
    user_products = list(user_dict[uid])
    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
            cnt = 0
            
            while cnt < max_paths:

                if start_ent_type is None:
                    
                    cur_start_ent_type = random.choice([USER, PRODUCT, ENTITY])


                    if cur_start_ent_type == ENTITY:
                        non_ext_entity = set([USER, PRODUCT])
                        candidate_ext_ent_types = [x for x in kg if x not in non_ext_entity ]
                        cur_start_ent_type = random.choice(candidate_ext_ent_types)
                    
                    id = random.choice(list(kg[cur_start_ent_type]) ) 
                else:
                    cur_start_ent_type = start_ent_type
                    id = uid
                if n_hop is None:
                    valid_hop_range = []
                    if end_ent_type is  None:
                        valid_hop_range = [i for i in range(1,50+1)]
                    else:
                        
                        if (cur_start_ent_type in non_prod_entities and end_ent_type not in non_prod_entities) or \
                                (cur_start_ent_type not in non_prod_entities and end_ent_type in non_prod_entities) :
                            # only odd hops
                            valid_hop_range = [i for i in range(1,50+1, 2)]
                        else:
                            # only even hops
                            valid_hop_range = [i for i in range(2,50+1, 2)]
                         
                    cur_n_hop = random.choice(valid_hop_range)
                else:
                    cur_n_hop = n_hop

                

                path = []
                cur_hop = 0
                

                if cur_start_ent_type == USER_ENT:
                    if id == uid:
                        prefix = LiteralPath.main_user
                    else:
                        prefix = LiteralPath.oth_user
                    type_prefix = LiteralPath.user_type
                elif cur_start_ent_type == PROD_ENT:
                    if id in user_dict[uid]:  
                        prefix = LiteralPath.recom_prod
                    else:
                        prefix = LiteralPath.prod    
                    type_prefix = LiteralPath.prod_type                  
                else:              
                    prefix = LiteralPath.ent
                    type_prefix = LiteralPath.ent_type
                if with_type:
                    path.append(prefix)

                path.append(f'{type_prefix}{id}')     

                prev_ent_t = cur_start_ent_type
                cur_ent_t = cur_start_ent_type


                dfs(uid, cur_start_ent_type, cur_start_ent_type, id, id, cur_hop, path, -100, cur_n_hop, cur_start_ent_type, end_ent_type)
                cnt += 1
                '''
                cur_hop = 1
                pid = random.choice(user_products)
                if pid not in kg[PROD_ENT]:
                    continue                
                cur_ent_id = pid

                rel_id = REL_TYPE2ID[U2P_REL]
                if with_type:
                    path = [LiteralPath.main_user, f'{LiteralPath.user_type}{uid}',
                            LiteralPath.fw_rel, f'{LiteralPath.rel_type}{rel_id}',#U2P_REL, 
                            LiteralPath.recom_prod, f'{LiteralPath.prod_type}{cur_ent_id}']
                else:
                    path = [f'{LiteralPath.user_type}{uid}',
                            f'{LiteralPath.rel_type}{rel_id}',#U2P_REL, 
                            f'{LiteralPath.prod_type}{cur_ent_id}']                                            

                prev_ent_t = USER_ENT
                cur_ent_t = PROD_ENT


                dfs(uid, prev_ent_t, cur_ent_t, uid, cur_ent_id, cur_hop, path, rel_id)
                cnt += 1
                '''    


def random_walk_sample_paths_recursive(uid, kg, items, n_hop, KG2T, R2T, PROD_ENT, U2P_REL, logdir, 
     user_dict, ignore_rels=set(), max_paths=None, itemset_type='inner'):
    dirpath = logdir#os.path.join(logdir, 'paths3')
    os.makedirs(dirpath, exist_ok=True)

    
    def dfs(prev_ent_id, cur_ent_id, cur_hop, path):
        if cur_hop >= n_hop:
            path = [ str(x) for x in path]
            #path = [x if isinstance(x,int) else x.item()  for x in path]
            #pathIO.write_to_file(path, n_hop-1, fp)
            fp.write(' '.join(path) + '\n' )  
            
            return True



        if cur_ent_id not in kg:
            return False
        valid_rels = list(kg[cur_ent_id].keys())
        random.shuffle(valid_rels)
        for rel in valid_rels:
            if rel in ignore_rels:
                continue
            
            candidates = kg[cur_ent_id][rel] 
            if cur_hop == n_hop-1:
                key = uid,rel,cur_ent_id
                
                if itemset_type == 'inner' and key in uid_ext_ent_mapping:
                    candidates = uid_ext_ent_mapping[key]
                elif itemset_type == 'outer' and key in uid_outer_ext_ent_mapping:
                    candidates = uid_outer_ext_ent_mapping[key]
                # else the default itemset is used (all  items reachable from current entity, which is not an item at the 'n_hop-1' hop)
            if len(candidates) == 0:
                continue                    

            path.append( rel)
            random.shuffle(candidates)
            

            for next_ent_id in candidates:
                if next_ent_id == prev_ent_id:
                    continue
                path.append(next_ent_id)
                if dfs(cur_ent_id, next_ent_id, cur_hop+1, path):
                    path.pop()
                    path.pop()
                    return True
                path.pop()
            path.pop()
        return False
    user_products = list(user_dict[uid])
    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
            cnt = 0
            
            while cnt < max_paths:
                cur_hop = 1
                pid = random.choice(user_products)
                if pid not in kg:
                    continue                
                cur_ent_id = pid
                path = [uid, U2P_REL, cur_ent_id]
                dfs(-1, cur_ent_id, cur_hop, path)
                cnt += 1
    '''
    iterative sampling
    with open(os.path.join(dirpath, f'paths_{uid}.txt' ), 'w') as fp:
        
            cnt = 0
            
            while cnt < max_paths:
                cur_hop = 1
                pid = random.choice(user_products)
                cur_ent_id = pid
                path = [uid, U2P_REL, cur_ent_id]
                #path.append( cur_ent_id)
                while cur_hop < n_hop:
                    if cur_ent_id not in kg:
                        break                    
                    valid_rels = list(kg[cur_ent_id].keys())
                    while True:
                        rel = random.choice(valid_rels )

                        if rel not in ignore_rels:
                            break
                    if cur_hop == n_hop-1:
                        tail_id = random.choice(unseen_products)
                    else:
                        tail_id = random.choice(kg[cur_ent_id][rel] )
                    path.append( rel)
                    path.append( tail_id)
                    cur_ent_id = tail_id
                    
                    cur_hop += 1
                
                if cur_hop >= n_hop:
                    path = [ str(x) for x in path]
                    #path = [x if isinstance(x,int) else x.item()  for x in path]
                    #pathIO.write_to_file(path, n_hop-1, fp)
                    fp.write(' '.join(path) + '\n' )  
                cnt += 1
    '''
    







def bfs_base_kg(pid, kg, items, n_hop, ignore_rels=set() ):
    q = deque()
    q.append((pid,0) )
    items_in_reach = set()
    while len(q):
        ent, dist = q.popleft()
        if ent in items:
            items_in_reach.add(ent)
        if ent not in kg:
            continue
        if n_hop is not None and dist >= n_hop:
            continue
        for r, tails in kg[ent].items():
            if r in ignore_rels:
                continue
            for t in tails:
                #if t not in seen:
                #    q.append( (t,dist+1))
                #    seen.add(t)
                q.append( (t,dist+1))   
    return pid, items_in_reach

def update_users(uid, user_dict, pid_to_reachable):
    reachable = set()
    for pid in user_dict[uid]:
        reachable.update(pid_to_reachable[pid])
    return uid, reachable

class KGstats:
    def __init__(self, args, dataset_name, path, logdir='statistics', data_dir=None):
        
        os.makedirs(logdir, exist_ok=True)
        self.logdir = os.path.join(logdir, dataset_name)
        os.makedirs(self.logdir, exist_ok=True)

        self.dataset_info = Dataset(args, data_dir=data_dir)
        ptrie = PathTrie(PATH_PATTERN[dataset_name])
        self.ptrie = ptrie

        self.kg2t = KG_RELATION[dataset_name]
        
        self.dataset_name = dataset_name
        print('Loading from ', path, ' the dataset ', dataset_name)
        uid_inter_test = os.path.join(path,  f'test.txt')
        uid_inter_valid = os.path.join(path,  f'valid.txt')
        uid_inter_train = os.path.join(path,  f'train.txt')
        item_list_file = os.path.join(path, 'i2kg_map.txt')#f'item_list.txt')
        kg_filepath = os.path.join(path,  f'kg_final.txt')                                      
        pid_mapping_filepath = os.path.join(path,  f'i2kg_map.txt')
        rel_mapping_filepath = os.path.join(path,  f'r_map.txt')
        rel_df = pd.read_csv(rel_mapping_filepath, sep='\t')
        pid_df = pd.read_csv(pid_mapping_filepath, sep='\t')
        
        self.pid2eid = { pid : eid for pid,eid in zip(pid_df.pid.values.tolist(), pid_df.eid.values.tolist())  }
        self.rel_id2type = { int(i) : rel_name for i,rel_name in zip(rel_df.id.values.tolist(), rel_df.name.values.tolist())  } 
        self.rel_id2type[LiteralPath.interaction_rel_id] = INTERACTION[dataset_name]
        #print(self.rel_id2type)

        self.rel_type2id = { v:k for k,v in self.rel_id2type.items() }
        #print(self.rel_type2id)
        
        self.items = KGstats.load_items(item_list_file)
        
        self.train_user_dict = self.load_user_inter(uid_inter_train)
        self.valid_user_dict = self.load_user_inter(uid_inter_valid)
        self.test_user_dict = self.load_user_inter(uid_inter_test)    
        user_dict = defaultdict(set)
        for uid in self.train_user_dict:
            user_dict[uid].update(self.train_user_dict[uid])
            user_dict[uid].update(self.valid_user_dict[uid])
            user_dict[uid].update(self.test_user_dict[uid])  
        
        self.user_dict = user_dict
        # kg in h,r,t format
        self.kg, self.kg_np = KGstats.load_kg(kg_filepath)
        self.graph_level_stats()
        #self.load_augmented_kg()
        self.load_augmented_kg_V2()
    def graph_level_stats(self):
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.n_relations = max(self.kg_np[:, 1]) + 1
        self.n_entities = max(max(self.kg_np[:, 0]), max(self.kg_np[:, 2])) + 1
        self.n_triples = len(self.kg_np)        
        
    def load_user_inter(self, filepath):
        G = defaultdict(set)
        with open(filepath) as f:
            for line  in f:
                line = line.strip()
                data = [int(v) for v in line.split('\t')]
                
                #print(data)
                uid = data[0]
                pid = int(data[1])
                rating = int(data[2])
                timestamp = int(data[3])
                G[uid].add(self.pid2eid[pid])
        return G        
    
    def load_items(item_file):
        item_ids = set()
        with open(item_file) as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                data = line.strip().rstrip().split('\t')
                #print(data)
                orig_id = int(data[1])
                item_id = int(data[0])
                item_ids.add(item_id)
        return item_ids
    
    def deg(self):
        degs = defaultdict(int)
        for h, rels in  self.kg.items():
            for r,tails in rels.items():
                for tail in tails:
                    degs[h] += 1
        return degs
    
    def load_kg(kg_filepath, undirected=True):
        kg = defaultdict()
        #kg_np = np.loadtxt(kg_filepath, np.uint32)
        kg_np = pd.read_csv(kg_filepath, sep='\t').to_numpy()
        print(kg_np.shape)
        kg_np = np.unique(kg_np, axis=0)
        print(kg_np.shape)
        for triple in kg_np:
            h,r,t = triple
            if h not in kg:
                kg[h] = defaultdict(set)
            if t not in kg:
                kg[t] = defaultdict(set)
            assert h != t, 'Self loop detected'
            kg[h][r].add(t)
            if undirected:
                kg[t][r].add(h)
            
        return kg, kg_np


    def random_walk_sampler(self, ignore_rels=set(), max_hop=None, max_paths=4000, logdir='paths_rand_walk',itemset_type='inner', 
        collaborative=True, num_beams=0, scorer=None, embedding_root_dir='./',
        nproc=8,
        with_type=True,
        start_ent_type=USER,
        end_ent_type=PRODUCT):
        user_dict, items = self.user_dict, self.items
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        if scorer is not None:
            func = random_walk_typified_beamsearch
        else:
            func = random_walk_typified
        
        # undirected knowledge graph hypotesis (for each relation, there exists its inverse)
        '''
        for uid in tqdm(list(self.user_dict)[11:]):       
            func(uid, 
                dataset_name=self.dataset_name,
                kg=self.aug_kg, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, 
                                USER_ENT=USER, PROD_ENT=PROD_ENT, EXT_ENT=ENTITY, 
                                U2P_REL=U2P_REL,
                                logdir=os.path.join(self.logdir, logdir),
                                user_dict=self.train_user_dict,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                REL_TYPE2ID=self.rel_type2id,
                                collaborative=collaborative,
                                num_beams=num_beams,
                                scorer=scorer,
                                dataset_info=self.dataset_info,
                                with_type=with_type,
                                start_ent_type=start_ent_type,
                                end_ent_type=end_ent_type)


        '''

        with mp.Pool(nproc) as pool:
            pool.starmap( functools.partial(func,             
            #func(uid, 
                dataset_name=self.dataset_name,
                kg=self.aug_kg, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, 
                                USER_ENT=USER, PROD_ENT=PROD_ENT, EXT_ENT=ENTITY, 
                                U2P_REL=U2P_REL,
                                logdir=os.path.join(self.logdir, logdir),
                                user_dict=self.train_user_dict,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                REL_TYPE2ID=self.rel_type2id,
                                collaborative=collaborative,
                                num_beams=num_beams,
                                scorer=scorer,
                                dataset_info=self.dataset_info,
                                with_type=with_type,
                                start_ent_type=start_ent_type,
                                end_ent_type=end_ent_type)
            ,
                tqdm([[uid] for uid in self.user_dict ] ))


        #'''

        '''
        with mp.Pool(nproc) as pool:
            pool.starmap( functools.partial(func, 
                dataset_name=self.dataset_name,
                kg=self.aug_kg, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, 
                                USER_ENT=USER, PROD_ENT=PROD_ENT, EXT_ENT=ENTITY, 
                                U2P_REL=U2P_REL,
                                logdir=os.path.join(self.logdir, logdir),
                                user_dict=self.train_user_dict,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                REL_TYPE2ID=self.rel_type2id,
                                collaborative=collaborative,
                                num_beams=num_beams,
                                scorer=scorer,
                                ),
                tqdm([[uid] for uid in self.user_dict ])
                #               #tqdm([[pid] for pid in self.items ])  
                                )   
        '''          

        #kg_copy = dict()
        #for ent in self.kg:
        #    kg_copy[ent] = dict()
        #    for rel in self.kg[ent]:
        #        kg_copy[ent][rel] = list(self.kg[ent][rel] )






        #d = dict()
        #d_outer=dict()
        '''
        uid_ext_ent_filename = f'{self.dataset_name}_aug_uid_ext_ent_mapping.pkl'

        if itemset_type =='inner':
            if not os.path.exists(uid_ext_ent_filename):
                for uid in self.train_user_dict:
                    for pid in self.train_user_dict[uid]:
                        if pid not in self.kg:
                            continue
                        for rel in self.kg[pid]:
                            for ent_id in self.kg[pid][rel]:
                                key = uid,rel, ENTITY, ent_id
                                if key not in d:
                                    d[key] = []
                                d[key].append(pid)

                    interaction_rel_id = LiteralPath.interaction_rel_id #U2P_REL                   
                    for oth_uid in self.train_user_dict:
                        if uid == oth_uid:
                            continue

                        key = uid, interaction_rel_id, USER, oth_uid
                        if key not in d:
                            d[key] = []
                        d[key] = list(self.train_user_dict[uid].intersection(self.train_user_dict[oth_uid])  )     
                                                           

                with open(uid_ext_ent_filename,'wb') as f:
                    pickle.dump(d, f)
            else:
                with open(uid_ext_ent_filename, 'rb') as f:
                    d = pickle.load(f)
        
        uid_outer_ext_ent_filename = f'{self.dataset_name}_aug_outer_ext_ent_mapping.pkl'
        
        if itemset_type =='outer':
            
            if not os.path.exists(uid_outer_ext_ent_filename):        
                for uid in self.train_user_dict:
                    for pid in items:
                        if pid in self.train_user_dict[uid]:
                            continue
                        if pid not in self.kg:
                            continue
                        for rel in self.kg[pid]:
                            for ent_id in self.kg[pid][rel]:
                                key = uid,rel,ent_id
                                if key not in d_outer:
                                    d_outer[key] = []
                                d_outer[uid,rel,ent_id].append(pid)   
                with open(uid_outer_ext_ent_filename, 'wb') as f:
                    pickle.dump(d_outer, f)
            else:
                with open(uid_outer_ext_ent_filename, 'rb') as f:
                    d_outer = pickle.load(f)
        '''
        #print('Created cache data structures')
        #import math
        #N_HOP_BITS = 6
        #ENT_ID_BITS = int(math.log2(self.n_entities)) + 1
        #REL_ID_BITS = int(math.log2(self.n_relations)) + 1
        #pathIO = PathFileIO(N_HOP_BITS, ENT_ID_BITS, REL_ID_BITS) 

        #nproc = 16
        '''
        for uid in self.user_dict:
            print('uid: ', uid)
            random_walk_sample_paths_recursive(uid, 
                kg=kg_copy, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, PROD_ENT=PROD_ENT, U2P_REL=U2P_REL,
                                pathIO = pathIO,
                                logdir=os.path.join(self.logdir, logdir),
                                user_dict=self.train_user_dict,
                                uid_ext_ent_mapping=d,
                                uid_outer_ext_ent_mapping=d_outer,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                )
        '''
        #print('BBBBBBBBBBBB: ', itemset_type)
        '''
        old dataset code
        with mp.Pool(nproc) as pool:
            pool.starmap( functools.partial(random_walk_sample_paths_recursive, 
                kg=kg_copy, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, PROD_ENT=PROD_ENT, U2P_REL=U2P_REL,
                                pathIO = pathIO,
                                logdir=os.path.join(self.logdir, logdir),
                                user_dict=self.train_user_dict,
                                uid_ext_ent_mapping=d,
                                uid_outer_ext_ent_mapping=d_outer,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                ),
                                tqdm([[uid] for uid in self.user_dict ])
                                #tqdm([[pid] for pid in self.items ])  
                                )
        '''

        '''
        for uid in self.user_dict:
                random_walk_typed_collaborative(uid,
                kg=self.aug_kg, 
                items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, 
                                USER_ENT=USER, PROD_ENT=PROD_ENT, EXT_ENT=ENTITY, 
                                U2P_REL=U2P_REL,
                                pathIO = pathIO,
                                logdir=os.path.join(self.logdir, logdir),
                                user_dict=self.train_user_dict,
                                uid_ext_ent_mapping=d,
                                uid_outer_ext_ent_mapping=d_outer,
                                ignore_rels=ignore_rels,
                                max_paths=max_paths,
                                itemset_type=itemset_type,
                                REL_TYPE2ID=self.rel_type2id
                                )

        
        # typed collaborative dataset code
        '''
   

        #'''
    '''
    def path_sampler(self, ignore_rels=set(), max_hop=None, p=0.001, max_paths=4000, logdir='paths'):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        # undirected knowledge graph hypotesis (for each relation, there exists its inverse)
        nproc = 6
        with mp.Pool(nproc) as pool:
            #ans = 
            pool.starmap( functools.partial(dfs_sample_paths, kg=kg, items=self.items, n_hop=max_hop, KG2T=self.kg2t, R2T=self.rel_id2type, PROD_ENT=PROD_ENT,
                                logdir=os.path.join(self.logdir, logdir),
                                ignore_rels=ignore_rels,
                                p=p,
                                max_paths=max_paths),
                                 tqdm([[pid] for pid in self.items ])  )
        

        #with open(os.path.join(self.logdir, 'paths.txt' ), 'w') as f:
        #        stats_dict = dict()
        #        #stats_dict['setup'] = {  'ignore_rels': list(ignore_rels), 'n_hop' : n_hop}
        #        for path in ans:
        #            f.write(str(path) )

          
        #        #json.dump(stats_dict, f)
        #return reachable_by_user  
    '''      

    
    def item_stats(self, n_hop=3):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        trie = self.ptrie.trie
        
        R2T = self.rel_id2type
        KG2T = KG_RELATION[self.dataset_name]
        
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        
        pid_rel_stats, pid_ent_stats = dict(), dict()
        
        for pid in tqdm(self.items):
            if pid not in self.aug_kg[PROD_ENT]:
                continue
            seen = set()
            q = [((pid, PROD_ENT), 0) ]
            pid_rel_stats[pid] = dict()
            pid_ent_stats[pid] = dict()
            while len(q):
                (h_id, h_type), dist = q.pop()
                if dist >= n_hop:
                    continue
                
                for rel, tails in self.aug_kg[h_type][h_id].items():
                    rel_type = rel
                    for (t_id, t_type) in tails:
                        if t_id in seen:
                            continue
                        seen.add(t_id)
                        
                        if rel_type not in pid_rel_stats[pid]:
                            pid_rel_stats[pid][rel_type] = defaultdict(int)                      
                        if t_type not in pid_ent_stats[pid]:
                            pid_ent_stats[pid][t_type] = defaultdict(set)
                            
                        #if (dist+1) not in pid_rel_stats[rel_type]:
                        #    pid_rel_stats[rel_type][dist+1] = defaultdict(int)
                        #if (dist+1) not in pid_ent_stats[t_type]:
                        #    pid_ent_stats[t_type][dist+1] = defaultdict(set)
                        
                        pid_rel_stats[pid][rel_type][dist+1] += 1
                        pid_ent_stats[pid][t_type][dist+1].add(t_id)
                        
                        q.append( ((t_id, t_type),dist+1) )
        return pid_rel_stats, pid_ent_stats
    def reachable_items_at_hop(self, ignore_rels=set(), n_hop=None):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        l = []
        items_in_reach = defaultdict(set)

                     
        # undirected knowledge graph hypotesis (for each relation, there exists its inverse)
        nproc = 4 
        with mp.Pool(nproc) as pool:
            ans = pool.starmap( functools.partial(bfs_base_kg, kg=kg, items=self.items, n_hop=n_hop, ignore_rels=ignore_rels),
                                 tqdm([[pid] for pid in self.items ])  )
        for pid, pid_reachable in ans:
            items_in_reach[pid] = pid_reachable

            
        reachable_by_user = defaultdict(set)
        reachable_by_user_count = defaultdict(int)
        with mp.Pool(nproc) as pool:
            ans = pool.starmap( functools.partial(update_users, user_dict=user_dict, pid_to_reachable=items_in_reach),
                                 tqdm([[uid] for uid in self.user_dict ])  )
        for uid, reachable_pids in ans:
            reachable_by_user[uid] = reachable_pids
            reachable_by_user_count[uid] = len(reachable_by_user)

        print('Avg reachable items: ', np.mean( [reachable_by_user_count[uid]  for uid in user_dict ] ))
        with open(os.path.join(self.logdir, 'reachable_items_at_hop.json' ), 'w') as f:
                stats_dict = dict()
                stats_dict['setup'] = {  'ignore_rels': list(ignore_rels), 'n_hop' : n_hop}


                stats_dict['queries'] = dict()
                stats_dict['queries']['reachable_by_user'] = { int(k) : [int(x) for x in v]  for k,v in   reachable_by_user.items() }
                stats_dict['queries']['reachable_by_user_count'] = reachable_by_user_count
                stats_dict['queries']['reachable_items_from_pid'] = { int(k) : [int(x) for x in v]  for k,v in   items_in_reach.items() }
          
                json.dump(stats_dict, f)
        return reachable_by_user  
                
    
    
    
    def reachable_items(self, ignore_rels=set()):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        l = []
        component_ids = [-1 for _ in range(len(self.items))]
        items_in_component = defaultdict(set)
        component_count = defaultdict(int)
        def bfs(pid, seen, comp_id, ignore_rels=set() ):
            from collections import deque
            q = deque()
            q.append((pid,0) )
            seen.add(pid)
            while len(q):
                ent, dist = q.popleft()
                if ent in items:
                    component_ids[ent] = comp_id
                    
                    items_in_component[comp_id].add(ent)
                    component_count[comp_id] += 1
                if ent not in kg:
                    continue

                for r, tails in kg[ent].items():
                    if r in ignore_rels:
                        continue
                    for t in tails:
                        if t not in seen:
                            q.append( (t,dist+1))
                            seen.add(t) 
            #items_in_component[comp_id] = N
        # undirected knowledge graph hypotesis (for each relation, there exists its inverse)
        id = 0
        seen = set()
        for pid in self.items:
            if pid not in seen:
                bfs(pid, seen, id, ignore_rels=ignore_rels)
                id += 1
        reachable_by_user = defaultdict(set)
        reachable_by_user_count = defaultdict(int)
        for uid in user_dict:
            seen_components = set()
            for pid in user_dict[uid]:
                if component_ids[pid] in seen_components:
                    continue
                reachable_by_user[uid].update(items_in_component[component_ids[pid]])
                reachable_by_user_count[uid] += component_count[component_ids[pid]]
                seen_components.add(component_ids[pid])
        #for comp_id, n_pids in component_count.items():
        #    print(comp_id, n_pids)

        print('Avg reachable items (% total): ', 100/len(self.items) * np.mean( [reachable_by_user_count[uid]  for uid in user_dict ] ))
        
        with open(os.path.join(self.logdir, 'reachable_items.json' ), 'w') as f:
                stats_dict = dict()
                stats_dict['setup'] = {  'ignore_rels': list(ignore_rels), 'n_hop' : None}


                stats_dict['queries'] = dict()
                stats_dict['queries']['reachable_by_user'] = { int(k) : [int(x) for x in v] for k,v in   reachable_by_user.items() }
                stats_dict['queries']['reachable_by_user_count'] = reachable_by_user_count
                stats_dict['queries']['reachable_items_from_pid'] = { int(pid) : [int(x) for x in items_in_component[component_ids[pid]] ] for pid in   self.items }
          
                json.dump(stats_dict, f)


        return reachable_by_user if store_item_ids else reachable_by_user_count  
    

    def load_augmented_kg(self):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        trie = self.ptrie.trie
        
        R2T = self.rel_id2type
        KG2T = KG_RELATION[self.dataset_name]
        print(R2T)
        
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        
        self.aug_kg = dict()
        self.aug_kg[USER] = dict()
        print('Creating augmented kg')
        for uid in user_dict:
            pids = user_dict[uid]
            self.aug_kg[USER][uid] = dict()
            self.aug_kg[USER][uid][U2P_REL] = list()
            
            if PROD_ENT not in self.aug_kg:
                self.aug_kg[PROD_ENT] = dict()
                
            for pid in pids:
                self.aug_kg[USER][uid][U2P_REL].append((pid, PROD_ENT))
                
                if pid not in self.aug_kg[PROD_ENT]:
                    self.aug_kg[PROD_ENT][pid] = dict()
                if U2P_REL not in self.aug_kg[PROD_ENT][pid]:
                    self.aug_kg[PROD_ENT][pid][U2P_REL] = list()
                    
                self.aug_kg[PROD_ENT][pid][U2P_REL].append((uid, USER))
        
        for h in self.kg:
            for rel, tails in self.kg[h].items():
                for t in tails:
                    # get tail entity type, uniquely determined by head_ent + rel_type
                    # kg is composed only of (h, REL, t)  where either of (h,t) can be PROD or EXTERNAL_ENT
                    TAIL_ENT = KG2T[PROD_ENT][R2T[rel]]
                    
                    h1,t1 = h,t
                    # to simplify the code, assume h is PROD, if it is not, swap it with the tail
                    if t in self.aug_kg[PROD_ENT]:
                        # swap them , to have product as head, just to reduce amount of code below
                        h1,t1 = t,h
                    #print(rel, h1, t1, '::::',h,t)
                    if h1 not in self.aug_kg[PROD_ENT]:
                        self.aug_kg[PROD_ENT][h1] = dict()
                    if R2T[rel] not in self.aug_kg[PROD_ENT][h1]:
                        self.aug_kg[PROD_ENT][h1][R2T[rel]] = list()
                    
                    if TAIL_ENT not in self.aug_kg:
                        self.aug_kg[TAIL_ENT] = dict()
                    if t1 not in self.aug_kg[TAIL_ENT]:
                        self.aug_kg[TAIL_ENT][t1] = dict()
                        
                    if R2T[rel] not in self.aug_kg[TAIL_ENT][t1]:
                        self.aug_kg[TAIL_ENT][t1][R2T[rel]] = list()  
                        
                    self.aug_kg[PROD_ENT][h1][R2T[rel]].append( (t1, TAIL_ENT )   )
                    self.aug_kg[TAIL_ENT][t1][R2T[rel]].append( (h1, PROD_ENT )   )
        print('Created augmented kg')      
    def load_augmented_kg_V2(self):
        kg, user_dict, items = self.kg, self.user_dict, self.items
        trie = self.ptrie.trie
        
        R2T = self.rel_id2type
        KG2T = KG_RELATION[self.dataset_name]
        print(R2T)
        
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        
        self.aug_kg = dict()
        self.aug_kg[USER] = dict()
        print('Creating augmented kg')
        for uid in user_dict:
            pids = user_dict[uid]
            self.aug_kg[USER][uid] = dict()
            self.aug_kg[USER][uid][U2P_REL] = defaultdict(list)#list()
            
            if PROD_ENT not in self.aug_kg:
                self.aug_kg[PROD_ENT] = dict()
                
            for pid in pids:
                self.aug_kg[USER][uid][U2P_REL][PROD_ENT].append(pid)
                
                if pid not in self.aug_kg[PROD_ENT]:
                    self.aug_kg[PROD_ENT][pid] = dict()
                if U2P_REL not in self.aug_kg[PROD_ENT][pid]:
                    self.aug_kg[PROD_ENT][pid][U2P_REL] = defaultdict(list)
                    
                self.aug_kg[PROD_ENT][pid][U2P_REL][USER].append(uid)
        
        for h in self.kg:
            for rel, tails in self.kg[h].items():
                for t in tails:
                    # get tail entity type, uniquely determined by head_ent + rel_type
                    # kg is composed only of (h, REL, t)  where either of (h,t) can be PROD or EXTERNAL_ENT
                    TAIL_ENT = KG2T[PROD_ENT][R2T[rel]]
                    
                    h1,t1 = h,t
                    # to simplify the code, assume h is PROD, if it is not, swap it with the tail
                    if t in self.aug_kg[PROD_ENT]:
                        # swap them , to have product as head, just to reduce amount of code below
                        h1,t1 = t,h
                    #print(rel, h1, t1, '::::',h,t)
                    if h1 not in self.aug_kg[PROD_ENT]:
                        self.aug_kg[PROD_ENT][h1] = dict()
                    if R2T[rel] not in self.aug_kg[PROD_ENT][h1]:
                        self.aug_kg[PROD_ENT][h1][R2T[rel]] = defaultdict(list)
                    
                    if TAIL_ENT not in self.aug_kg:
                        self.aug_kg[TAIL_ENT] = dict()
                    if t1 not in self.aug_kg[TAIL_ENT]:
                        self.aug_kg[TAIL_ENT][t1] = dict()
                        
                    if R2T[rel] not in self.aug_kg[TAIL_ENT][t1]:
                        self.aug_kg[TAIL_ENT][t1][R2T[rel]] = defaultdict(list) 
                        
                    self.aug_kg[PROD_ENT][h1][R2T[rel]][TAIL_ENT].append( t1   )
                    self.aug_kg[TAIL_ENT][t1][R2T[rel]][PROD_ENT].append( h1   )
        print('Created augmented kg')              
'''

    def reachable_items_constrained(self, n_hop=None ):
        if n_hop is None:
            n_hop = -2 + max([len(v) for v in PATH_PATTERN[self.dataset_name].values()])
        kg, user_dict, items = self.kg, self.user_dict, self.items
        
        PROD_ENT, U2P_REL =  MAIN_PRODUCT_INTERACTION[self.dataset_name] 
        
        pid_to_reachable = defaultdict(set)
        pid_reaches_unique_metapaths =defaultdict(set) 


        nproc = 6
        with mp.Pool(nproc) as pool:

            ans = pool.starmap( functools.partial(bfs, aug_kg=self.aug_kg, ptrie=self.ptrie, n_hop=n_hop, product_entity_name=PROD_ENT, user_entity_name=USER, u2p_rel_name=U2P_REL),
                                 tqdm([[pid] for pid in self.items ])  )
        for pid, reachable_items, valid_metapaths, ptrie_terminal_counts in tqdm(ans):
            pid_to_reachable[pid] = reachable_items
            pid_reaches_unique_metapaths[pid] = valid_metapaths

            for terminal_id, counts in ptrie_terminal_counts.items():
                self.ptrie.terminal_nodes[terminal_id].count += counts

        import copy
        computed_ptrie = copy.deepcopy(self.ptrie)
        self.ptrie.reset_counts()
        
        reachable_by_user = defaultdict(set)
        with mp.Pool(nproc) as pool:
            ans = pool.starmap( functools.partial(update_users, user_dict=self.user_dict, pid_to_reachable=pid_to_reachable),
                                 tqdm([[uid ] for uid in self.aug_kg[USER] ])  )
        for uid, reachable_pids  in ans:
            reachable_by_user[uid] = reachable_pids

        
        with open(os.path.join(self.logdir,f'reachable_items_constrained.json' ), 'w') as f:
                stats_dict = dict()
                stats_dict['setup'] = { n_hop : n_hop }


                stats_dict['queries'] = dict()
                stats_dict['queries']['reachable_by_user'] = { int(k) : [int(x) for x in v] for k,v in   reachable_by_user.items() }
                stats_dict['queries']['reachable_items_from_pid'] = { int(k) : [int(x) for x in v] for k,v in   pid_to_reachable.items() }
                stats_dict['queries']['valid_metapaths_from_pid'] = { int(k): [int(x) for x in v] for k,v in  pid_reaches_unique_metapaths.items() }
          
                json.dump(stats_dict, f)
            
        return reachable_by_user, computed_ptrie, pid_to_reachable, pid_reaches_unique_metapaths    




    def compute_metapath_frequencies(self):
        def path_frequencies(ptrie):
            trie = ptrie.trie
            ans = []
            q = [ (trie, []) ]
            while len(q):
                trie, path = q.pop()
                for k in trie:
                    if k == ptrie.TERMINATION:
                        ans.append((path, trie[k].count ) )
                    else:
                        q.append( (trie[k], [item for item in path] + [k])    )
            return ans
        def pid_item_reach_count(pid_reach):
            pid_list = []
            n_items = []
            for pid, items in pid_reach.items():
                pid_list.append(pid)
                # note: the fact that an item is reached from "pid" is equivalent to the existence of one metapath
                #       that allows to reach it
                n_items.append(len(items))
            pd.DataFrame.from_dict({ 'pid': pid_list, 'reached items': n_items })\
            .to_csv(os.path.join(self.logdir,'pid_reach.csv'))  
                
        
        reachable, freq_trie, pid_reach, pid_reaches_unique_metapaths = self.reachable_items_constrained()
        pid_item_reach_count(pid_reach)
        path_names = []
        path_freqs = []
        for path_list, freq in sorted(path_frequencies(freq_trie), key=lambda x: x[-1], reverse=True):
            path_name_list = [ elem.upper() if i%2==1 else elem for i, elem in enumerate(path_list) ]
            path_str = '->'.join( path_name_list)
            #print( path_str , freq )
            path_names.append(path_str)
            path_freqs.append(freq)
        pd.DataFrame.from_dict({ 'path': path_names, 'frequency': path_freqs })\
        .to_csv(os.path.join(self.logdir,'path_frequency.csv') )   

    def reachability_stats(self, max_hops):
        n_hops = list(range(1, max_hops+1))
        rel_types = list(self.rel_id2type.items())
        reachable_kg_stats = dict()
        for n_hop, (rel_id,rel_type) in itertools.product(n_hops, rel_types):
            #print('N_hop: ', n_hop, ' - rel:', rel_type)
            reach = self.reachable_items_at_hop(n_hop=n_hop, ignore_rels={rel_id})
            reachable_kg_stats[n_hop,rel_type] = reach   
            
        l_hop = []
        l_rel_name = []
        l_reach_value = [] 

        for (hop, rel_name), freq in reachable_kg_stats.items():
            avg_items_per_user = np.mean( [ len(reachable_items) for uid, reachable_items in freq.items()] )
            reach_value = (avg_items_per_user/ 
                           len(self.items) *100)

            l_hop.append(hop)
            l_rel_name.append(rel_name)
            l_reach_value.append(reach_value)
        import pandas as pd
        pd.set_option('display.precision', 3)
        df = pd.DataFrame.from_dict(  {  'Number of hops' : l_hop, 
                                       'Removed relation': l_rel_name,
                                       '% reachable items': l_reach_value  } )
        pd.pivot_table(df, values='% reachable items', index='Number of hops', columns='Removed relation').to_csv(os.path.join(self.logdir,'remove_rel_by_hop.csv') )       
'''



if __name__ == '__main__':
    MODEL = 'kgat'
    ML1M = 'ml1m'
    LFM1M ='lfm1m'
    CELL='cellphones'
    ROOT_DIR = os.environ('TREX_DATA_ROOT') if 'TREX_DATA_ROOT' in os.environ else '../..'
    # Dataset directories.
    DATA_DIR = {
        ML1M: f'{ROOT_DIR}/data/{ML1M}/preprocessed/{MODEL}',
        LFM1M: f'{ROOT_DIR}/data/{LFM1M}/preprocessed/{MODEL}',
        CELL: f'{ROOT_DIR}/data/{CELL}/preprocessed/{MODEL}'
    }
    dataset_name = 'ml1m'
    dirpath = DATA_DIR[dataset_name]#.replace('ripple', 'kgat')
    ml1m_kg = KGstats(dataset_name, dirpath)
    dataset_name = 'lfm1m'
    dirpath = DATA_DIR[dataset_name]#.replace('ripple', 'kgat')
    lfm1m_kg = KGstats(dataset_name, dirpath)




