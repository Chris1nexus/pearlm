import logging
import logging.handlers
import os
import pickle
import sys


#Implemented datasets
ML1M = 'ml1m'
LFM1M = 'lfm1m'
CELL = 'cellphones'

ROOT_DIR = os.environ['DATA_ROOT'] if 'DATA_ROOT' in os.environ else '.'
LOG_DIR = f'{ROOT_DIR}/logs'
TRANSE = 'transe'
IMPLEMENTED_KGE = [TRANSE]

def get_log_dir(dataset_name: str, embedding_name: str) -> str:
    ans = os.path.join(LOG_DIR, dataset_name, 'embeddings', embedding_name)
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans

def get_dataset_info_dir(dataset_name: str) -> str:
    ans = os.path.join(ROOT_DIR, 'data', dataset_name, 'preprocessed/mapping')
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans

def get_embedding_rootdir(dataset_name: str) -> str:
    ans = os.path.join(ROOT_DIR, 'weights', dataset_name, 'embeddings')
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans

def get_embedding_ckpt_rootdir(dataset_name: str) -> str:
    ans =  os.path.join(ROOT_DIR, 'weights', dataset_name, 'embeddings/ckpt')
    if not os.path.isdir(ans):
        os.makedirs(ans)
    return ans
def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# ENTITIES/RELATIONS SHARED BY ALL DATASETS
USER = 'user'
PRODUCT = 'product'
ENTITY = 'entity'
RELATION = 'relation'
INTERACTION = {
    ML1M: "watched",
    LFM1M: "listened",
    CELL: "purchase",
}
SELF_LOOP = 'self_loop'
PRODUCED_BY_PRODUCER = 'produced_by_producer'
PRODUCER = 'producer'

# ML1M ENTITIES
CINEMATOGRAPHER = 'cinematographer'
PRODCOMPANY = 'prodcompany'
COMPOSER = 'composer'
CATEGORY = 'category'
ACTOR = 'actor'
COUNTRY = 'country'
WIKIPAGE = 'wikipage'
EDITOR = 'editor'
WRITTER = 'writter'
DIRECTOR = 'director'

# LASTFM ENTITIES
ARTIST = 'artist'
ENGINEER = 'engineer'
GENRE = 'genre'

# CELL ENTITIES
BRAND = 'brand'
RPRODUCT = 'rproduct'

# ML1M RELATIONS
DIRECTED_BY_DIRECTOR = 'directed_by_director'
PRODUCED_BY_COMPANY = 'produced_by_prodcompany'
STARRED_BY_ACTOR = 'starred_by_actor'
RELATED_TO_WIKIPAGE = 'related_to_wikipage'
EDITED_BY_EDITOR = 'edited_by_editor'
WROTE_BY_WRITTER = 'wrote_by_writter'
CINEMATOGRAPHY_BY_CINEMATOGRAPHER = 'cinematography_by_cinematographer'
COMPOSED_BY_COMPOSER = 'composed_by_composer'
PRODUCED_IN_COUNTRY = 'produced_in_country'
BELONG_TO_CATEGORY = 'belong_to_category'

# LASTFM RELATIONS
MIXED_BY_ENGINEER = 'mixed_by_engineer'
FEATURED_BY_ARTIST = 'featured_by_artist'
BELONG_TO_GENRE = 'belong_to_genre'

# CELL RELATIONS
PURCHASE = 'purchase'
ALSO_BOUGHT_RP = 'also_bought_related_product'
ALSO_VIEWED_RP = 'also_viewed_related_product'
ALSO_BOUGHT_P = 'also_bought_product'
ALSO_VIEWED_P = 'also_viewed_product'

MAIN_PRODUCT_INTERACTION = {
    ML1M: (PRODUCT, INTERACTION[ML1M]),
    LFM1M: (PRODUCT, INTERACTION[LFM1M]),
    CELL: (PRODUCT, INTERACTION[CELL])
}

# Define KG structure for each dataset TODO Should be in a config file
KG_RELATION = {
    ML1M: {
        USER: {
            INTERACTION[ML1M]: PRODUCT,
        },
        ACTOR: {
            STARRED_BY_ACTOR: PRODUCT,
        },
        DIRECTOR: {
            DIRECTED_BY_DIRECTOR: PRODUCT,
        },
        PRODUCT: {
            INTERACTION[ML1M]: USER,
            PRODUCED_BY_COMPANY: PRODCOMPANY,
            PRODUCED_BY_PRODUCER: PRODUCER,
            EDITED_BY_EDITOR: EDITOR,
            WROTE_BY_WRITTER: WRITTER,
            CINEMATOGRAPHY_BY_CINEMATOGRAPHER: CINEMATOGRAPHER,
            BELONG_TO_CATEGORY: CATEGORY,
            DIRECTED_BY_DIRECTOR: DIRECTOR,
            STARRED_BY_ACTOR: ACTOR,
            COMPOSED_BY_COMPOSER: COMPOSER,
            PRODUCED_IN_COUNTRY: COUNTRY,
            RELATED_TO_WIKIPAGE: WIKIPAGE,
        },
        PRODCOMPANY: {
            PRODUCED_BY_COMPANY: PRODUCT,
        },
        COMPOSER: {
            COMPOSED_BY_COMPOSER: PRODUCT,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: PRODUCT,
        },
        WRITTER: {
            WROTE_BY_WRITTER: PRODUCT,
        },
        EDITOR: {
            EDITED_BY_EDITOR: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO_CATEGORY: PRODUCT,
        },
        CINEMATOGRAPHER: {
            CINEMATOGRAPHY_BY_CINEMATOGRAPHER: PRODUCT,
        },
        COUNTRY: {
            PRODUCED_IN_COUNTRY: PRODUCT,
        },
        WIKIPAGE: {
            RELATED_TO_WIKIPAGE: PRODUCT,
        }
    },
    LFM1M: {
        USER: {
            INTERACTION[LFM1M]: PRODUCT,
        },
        ARTIST: {
            FEATURED_BY_ARTIST: PRODUCT,
        },
        ENGINEER: {
            MIXED_BY_ENGINEER: PRODUCT,
        },
        PRODUCT: {
            INTERACTION[LFM1M]: USER,
            PRODUCED_BY_PRODUCER: PRODUCER,
            FEATURED_BY_ARTIST: ARTIST,
            MIXED_BY_ENGINEER: ENGINEER,
            BELONG_TO_GENRE: GENRE,
        },
        PRODUCER: {
            PRODUCED_BY_PRODUCER: PRODUCT,
        },
        GENRE: {
            BELONG_TO_GENRE: PRODUCT,
        },
    },
    CELL: {
        USER: {
            PURCHASE: PRODUCT,
        },
        PRODUCT: {
            PURCHASE: USER,
            PRODUCED_BY_COMPANY: BRAND,
            BELONG_TO_CATEGORY: CATEGORY,
            ALSO_BOUGHT_RP: RPRODUCT,
            ALSO_VIEWED_RP: RPRODUCT,
            ALSO_BOUGHT_P: PRODUCT,
            ALSO_VIEWED_P: PRODUCT,
        },
        BRAND: {
            PRODUCED_BY_COMPANY: PRODUCT,
        },
        CATEGORY: {
            BELONG_TO_CATEGORY: PRODUCT,
        },
        RPRODUCT: {
            ALSO_BOUGHT_RP: PRODUCT,
            ALSO_VIEWED_RP: PRODUCT,
        }
    },
}

def save_embed(dataset_name: str, embed_name: str, state_dict: dict):
    EMBEDDING_DIR = get_embedding_rootdir(dataset_name)
    embed_file = os.path.join(EMBEDDING_DIR, f'{embed_name}_embed.pkl')
    pickle.dump(state_dict, open(embed_file, 'wb'))


def load_embed(dataset_name: str, embed_name: str=None):
    EMBEDDING_DIR = get_embedding_rootdir(dataset_name)
    embed_file = os.path.join(EMBEDDING_DIR, dataset_name, f'{embed_name}_embed.pkl')
    print(f'Load {embed_name} embedding:', embed_file)
    if not os.path.exists(embed_file):
        # Except for file not found, raise error
        raise FileNotFoundError(f'Embedding file {embed_file} not found.')
    embed = pickle.load(open(embed_file, 'rb'))
    return embed

def get_knowledge_derived_relations(dataset_name):
    main_entity, main_relation = MAIN_PRODUCT_INTERACTION[dataset_name]
    ans = list(KG_RELATION[dataset_name][main_entity].keys())
    ans.remove(main_relation)
    return ans

