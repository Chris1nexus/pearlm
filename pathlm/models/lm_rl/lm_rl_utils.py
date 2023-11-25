
from pathlm.models.rl.PGPR.pgpr_utils import get_knowledge_derived_relations, MODEL_DATASET_DIR,INTERACTION, DATASET_INFO_DIR,\
        PRODUCT, USER, ENTITY, RELATION

from pathlm.sampling.samplers.constants import LiteralPath, TypeMapper

def tokenize_augmented_kg(kg, tokenizer, use_token_ids=False):
    type_id_to_subtype_mapping = kg.dataset_info.groupwise_global_eid_to_subtype.copy()
    rel_id2type = kg.rel_id2type.copy()
    type_id_to_subtype_mapping[RELATION] = {int(k): v for k, v in rel_id2type.items()}

    aug_kg = kg.aug_kg

    token_id_to_token = dict()
    kg_to_vocab_mapping = dict()
    tokenized_kg = dict()

    for token, token_id in tokenizer.get_vocab().items():
        if not token[0].isalpha():
            continue

        cur_type = token[0]
        cur_id = int(token[1:])

        type = TypeMapper.mapping[cur_type]
        subtype = type_id_to_subtype_mapping[type][cur_id]
        if cur_type == LiteralPath.rel_type:
            cur_id = None
        value = token
        if use_token_ids:
            value = token_id
        kg_to_vocab_mapping[(subtype, cur_id)] = value

    for head_type in aug_kg:
        for head_id in aug_kg[head_type]:
            head_key = head_type, head_id
            if head_key not in kg_to_vocab_mapping:
                continue
            head_ent_token = kg_to_vocab_mapping[head_key]
            tokenized_kg[head_ent_token] = dict()

            for rel in aug_kg[head_type][head_id]:
                rel_token = kg_to_vocab_mapping[rel, None]
                tokenized_kg[head_ent_token][rel_token] = set()

                for tail_type in aug_kg[head_type][head_id][rel]:
                    for tail_id in aug_kg[head_type][head_id][rel][tail_type]:
                        tail_key = tail_type, tail_id
                        if tail_key not in kg_to_vocab_mapping:
                            continue
                        tail_token = kg_to_vocab_mapping[tail_key]
                        tokenized_kg[head_ent_token][rel_token].add(tail_token)
    return tokenized_kg, kg_to_vocab_mapping