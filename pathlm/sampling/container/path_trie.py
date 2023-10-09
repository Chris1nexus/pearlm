from pathlm.knowledge_graphs.kg_macros import FEATURED_BY_ARTIST, ARTIST
from pathlm.knowledge_graphs.kg_utils import PATH_PATTERN
from pathlm.models.rl.PGPR.pgpr_utils import *



class TerminationNode:
    def __init__(self, id):
        self.count = 0
        self.id = id
        

class PathTrie:
    TERMINATION = 'end_entity'
    
    def __init__(self, valid_path_patterns):

        trie = dict()
        self.trie = trie
        self.id = 0
        self.terminal_nodes = dict()
        for path_id, path in valid_path_patterns.items():
            d = trie
            for hop, (rel, ent_type) in enumerate(path):
                if hop==0:
                    if ent_type not in d:
                        d[ent_type] = dict()
                    d = d[ent_type]
                    continue
                if rel not in d:
                    d[rel] = dict()
                if ent_type not in d[rel]:
                    d[rel][ent_type] = dict()
                d = d[rel][ent_type]

            # d[PathTrie.TERMINATION] = None
            d[PathTrie.TERMINATION] = TerminationNode(self.id)
            self.terminal_nodes[self.id] = d[PathTrie.TERMINATION] 
            self.id += 1
            
    def reset_counts(self):
        trie = self.trie
        q = [trie]
        
        while len(q):
            trie = q.pop()

            for k, v in trie.items():
                if k == PathTrie.TERMINATION:
                    trie[PathTrie.TERMINATION].count = 0
                else:
                    q.append(v )
    
    def match(self, path):
        cur = self.trie
        for hop, (rel, ent_type) in enumerate(path):
            if hop == 0:
                if ent_type not in cur:
                    return False
                cur = cur[ent_type]
                continue
            if rel not in cur:
                return False
            cur = cur[rel]
            if ent_type not in cur:
                return False
            cur = cur[ent_type]
        if PathTrie.TERMINATION in cur:
            return True
        return False        
    
    def k_hop_match(self, path, k=1):
        cur = self.trie
        for hop, (rel, ent_type) in enumerate(path):
            if hop == 0:
                if ent_type not in cur:
                    return False
                cur = cur[ent_type]
                continue
            if rel not in cur:
                return False
            cur = cur[rel]
            if ent_type not in cur:
                return False
            cur = cur[ent_type]
            
            if PathTrie.TERMINATION in cur and hop == k:
                return True
            elif hop >= k:
                break
        return False
if __name__ == '__main__':
    ptrie = PathTrie(PATH_PATTERN['ml1m'])
    query_path = list(PATH_PATTERN['ml1m'][3])
    query_path.append((FEATURED_BY_ARTIST, ARTIST))
    ptrie.match(query_path)