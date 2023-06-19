
class Trie:
    TERMINATION = ''
    class Item:
        def __init__(self, key, trie, counter):
            self.key = key
            self.trie = trie
            self.counter = counter

    def __init__(self, strings):
        
        self.trie = dict()
        self.strings = strings
        for string in strings:
            self.insert(string)
    def compute_unique_prefixes(self):
        prefix_map = dict()
        for word in self.strings:
            cur_trie = self.trie
            i = 0
            while i < len(word):
                cur_ch = word[i]
                i += 1
                if len(cur_trie[cur_ch].trie) == 1 and cur_trie[cur_ch].counter == 1:
                    break
                cur_trie = cur_trie[cur_ch].trie
            prefix_map[word] = word[:i]  
        return prefix_map


    def insert(self, word):
        cur_trie = self.trie
        for ch in word:
            if ch not in cur_trie:
                cur_trie[ch] = Trie.Item(ch, dict(), 0)
            cur_item = cur_trie[ch]
            cur_item.counter += 1
            cur_trie = cur_item.trie
        cur_trie[Trie.TERMINATION] = None   

if __name__ == '__main__':
    s = """CINEMATOGRAPHER = 'cinematographer'
    PRODCOMPANY = 'prodcompany'
    COMPOSER = 'composer'
    CATEGORY = 'category'
    ACTOR = 'actor'
    COUNTRY = 'country'
    WIKIPAGE = 'wikipage'
    EDITOR = 'editor'
    WRITTER = 'writter'
    DIRECTOR = 'director'"""
    entities = []
    for line in s.split('\n'):
        print(line.rstrip().replace(' ', '').split('=')[1].replace('\'', ''))
        ent_name = line.rstrip().replace(' ', '').split('=')[1].replace('\'', '')
        entities.append(ent_name)

    trie = Trie(entities)

    trie.compute_unique_prefixes()