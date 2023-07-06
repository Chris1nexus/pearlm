import argparse
import os
import pickle
from collections import defaultdict
from typing import List, Dict
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from datasets import load_from_disk, DatasetDict, Dataset
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,\
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, LogitsProcessorList,\
    set_seed, GPT2LMHeadModel, GPT2Model, EarlyStoppingCallback
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import is_torch_tpu_available
import pandas as pd
from os.path import join

from pathlm.models.lm.generation_constraints import ConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
#from pathlm.models.lm.path_dataset import PathDataset
from pathlm.models.rl.PGPR.pgpr_utils import RELATION
from pathlm.sampling.container.constants import LiteralPath, TypeMapper
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.tools.mapper import EmbeddingMapper
from pathlm.utils import SEED, get_pid_to_eid, get_set, check_dir


from multiprocessing import Pool
from datasets import Dataset
from os import listdir
from os.path import isfile, join
import pandas as pd
import math
from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map

class PathDataset:
    def __init__(self, dataset_name: str, base_data_dir: str="", task: str=None, sample_size: str=None, n_hop: str=None, plain_text_path=False):
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.data_dir = join(self.base_data_dir, "paths_random_walk")
        self.task = task
        self.sample_size = sample_size
        self.n_hop = n_hop

        self.read_single_csv_to_hf_dataset()
        # Get eid2name and rid2name
        self.eid2name = get_eid_to_name_map(self.base_data_dir)
        self.rid2name = get_rid_to_name_map(self.base_data_dir)
        self.plain_text_path = plain_text_path


    # Based on the path struct, for now it is p to p
    def convert_numeric_path_to_textual_path(self, path: str) -> str:
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            # Handle user and watched relation
            if pos == 0:
                ans.append(f"U{token}")
            elif pos == 1:
                ans.append(token)
            # Handle recommendation
            elif pos == 2 or pos == 6 or pos == 10:
                #ans.append("<recommendation>")
                ans.append(self.eid2name[token])
            # Handle entity
            elif pos % 2 == 0:
                ans.append(self.eid2name[token])
            # Handle relation
            else:
                ans.append(self.rid2name[token])
            ans.append("<word_end>")
        return " ".join(ans)

    def keep_numeric(self, path: str) -> str:
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            # Handle user
            if pos == 0:
                ans.append(f"U{token}")
            elif pos == 1:
                ans.append(token)
            # Handle recommendation
            elif pos == 2 or pos == 6 or pos == 10:
                ans.append(f"P{token}")
            # Handle entity
            elif pos % 2 == 0:
                ans.append(f"E{token}")
            # Handle relation
            else:
                ans.append(f"R{token}")
        return " ".join(ans)

    def read_csv_as_dataframe(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(join(self.data_dir, filename), header=None, names=["path"], index_col=None)

    def read_multiple_csv_to_hf_dataset(self):
        file_list = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]

        # set up your pool
        #with Pool(processes=8) as pool:  # orc whatever your hardware can support
        #    df_list = pool.map(self.read_csv_as_dataframe, file_list)#
        #
        #    # reduce the list of dataframes to a single dataframe
        df_list = []
        for filename in file_list:
            df_list.append(self.read_csv_as_dataframe(filename))

        combined_df = pd.concat(df_list, ignore_index=True)


        # Convert to HuggingFace Dataset
        self.dataset = Dataset.from_pandas(combined_df)

    def read_single_csv_to_hf_dataset(self):
        #file_list = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        #filename = f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt'
        filename = f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt'
        #filepath = join(self.data_dir, filename)


        df = self.read_csv_as_dataframe(filename)
        self.dataset = Dataset.from_pandas(df)
        
        #for filename in file_list:
        #    print(filename)
        #    if filename == f'paths_{self.task}_{self.sample_size}_{self.n_hop}.txt':#
        #
        #        df = self.read_csv_as_dataframe(filename)
        #        self.dataset = Dataset.from_pandas(df)
        #        continue



    def show_random_examples(self):
        print(self.dataset["path"][:10])
def _initialise_type_masks(tokenizer, allow_special=False):
    ent_mask = []
    rel_mask = []
    class_weight = 1
    token_id_to_token = dict()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
        if token[0] == LiteralPath.rel_type: #or (not token[0].isalpha() and allow_special) :
            rel_mask.append(class_weight)
        else:
            rel_mask.append(0)
        if token[0] == LiteralPath.user_type or token[0] == LiteralPath.prod_type or token[0] == LiteralPath.ent_type:# or (not token[0].isalpha() and allow_special):
            ent_mask.append(class_weight)
        else:
            ent_mask.append(0)

        token_id_to_token[token_id] = token
    #print(ent_mask)
    #print(rel_mask)
    return ent_mask, rel_mask, token_id_to_token



def _initialise_weights_from_kge(embeds, tokenizer, kg, model_config, model, args):
    print('Using embeddings: ', args.emb_filename)
    model_config.update({
        'hidden_size': args.emb_size,
        'num_attention_heads': args.emb_size // 10
    })
    print('ORIGINAL EMBEDDING OVERWRITTED BY TRANSE')
    mapper = EmbeddingMapper(tokenizer, kg, embeds)
    mapper.init_with_embedding(model.transformer.wte.weight)
    return model, model_config


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
        kg_to_vocab_mapping[(subtype, cur_id)] = token_id

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


def get_user_negatives(dataset_name: str) -> Dict[str, List[str]]:
    NOT_IN_TOKENIZER = {'1871', '831', '1950', '478', '2285'}
    data_dir = f"data/{dataset_name}"
    ikg_ids = set(get_pid_to_eid(data_dir).values())
    uid_negatives = {}
    # Generate paths for the test set
    train_set = get_set(dataset_name, set_str='train')
    valid_set = get_set(dataset_name, set_str='valid')
    for uid in tqdm(train_set.keys(), desc="Calculating user negatives", colour="green"):
        uid_negatives[uid] = list(set(ikg_ids - set(train_set[uid]) - set(valid_set[uid])) - NOT_IN_TOKENIZER)
    return uid_negatives






#TODO: TEST FORWARD + LOSS
class DistilGPT2TwoHeadModel(GPT2LMHeadModel):
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID] 

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        
        self.num_labels = config.vocab_size

        self.ent_mask = config.ent_mask
        self.rel_mask = config.rel_mask
        self.ent_mask_weight = torch.FloatTensor(self.ent_mask)
        self.rel_mask_weight = torch.FloatTensor(self.rel_mask)      
        self.ent_mask = torch.LongTensor(self.ent_mask)
        self.rel_mask = torch.LongTensor(self.rel_mask)      
        self.context_length = config.n_ctx

        idxs = [i%2 == 0 for i in range(self.context_length)  ]
        #idx_mask = torch.stack( [torch.BoolTensor(idxs) for _ in range(self.context_length)] )
        self.even_idx_mask = torch.BoolTensor(idxs)
        
        self.num_kg_types = len(DistilGPT2TwoHeadModel.kg_categories)

        self.id_to_str = config.token_id_to_token

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=self.num_kg_types, embedding_dim=config.n_embd) # for entities, relations, and special tokens

        #self.train_type_ids, self.train_type_embeds = self.__init_type_embeddings(self.config.train_batch_size, seq_len)
        #self.test_type_ids, self.test_type_embeds = self.__init_type_embeddings(self.config.test_batch_size, seq_len)
        self.type_ids_cache = dict()
        self.type_embeds_cache = dict()
        self.idx_mask_cache = dict()

        self.type_ids_row, self.type_embeds_row = self.__init_type_embeddings(1, self.context_length)

        # Create an additional linear layer for the second prediction head
        self.entity_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.relation_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def __init_type_embeddings(self,  batch_size, num_hops):
        #num_hops = self.config.num_hops
        n_tokens = num_hops#num_hops + 1 + num_hops + 2
        type_ids = torch.ones((batch_size,n_tokens) , dtype=torch.long)


        for i in range(n_tokens):
            if i == 0 or i == n_tokens-1:
                type_ids[:,i] = DistilGPT2TwoHeadModel.SPECIAL_ID
            elif i % 2 == 1:
                type_ids[:,i] = DistilGPT2TwoHeadModel.ENTITY_ID
            elif i % 2 == 0:
                type_ids[:,i] = DistilGPT2TwoHeadModel.RELATION_ID
        type_ids = type_ids.to(self.type_embeddings.weight.device)
        type_embeds = self.type_embeddings(type_ids)
        return type_ids, type_embeds

    def __get_type_embeds(self, n_rows, n_cols):
        row = self.type_ids_row[:,:(n_cols-1)]
        row = torch.hstack( [row, torch.ones((1,1))*  DistilGPT2TwoHeadModel.SPECIAL_ID  ] )
        type_ids = torch.vstack([row for _ in range(n_rows)]) 
        return type_ids, self.type_embeddings(type_ids.to(self.type_embeddings.weight.device))
    def __get_even_idx_mask(self,n_rows, n_cols):
        mask_key = (n_rows,n_cols)
        cur_mask = self.even_idx_mask[:(n_cols) ]            
        return torch.vstack([cur_mask for _ in range(n_rows) ] )
    '''
    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))

        self.transformer.parallelize(self.device_map)
        self.type_embeddings = self.type_embeddings.to(self.transformer.first_device)
        self.lm_entity_head = self.lm_entity_head.to(self.transformer.first_device)
        self.lm_relation_head = self.lm_relation_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.transformer.deparallelize()
        self.type_embeddings = self.type_embeddings.to("cpu")
        self.transformer = self.transformer.to("cpu")
        self.lm_entity_head = self.lm_entity_head.to("cpu")
        self.lm_relation_head = self.lm_relation_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    #def get_output_embeddings(self):
    #    return self.lm_entity_head

    #def set_output_embeddings(self, new_embeddings):
    #    self.lm_entity_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs
    '''



    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        #s = []
        #for token_id in input_ids[0]:
        #    s.append( self.id_to_str[token_id.item()] )
        #print(' '.join(s))
        #print()

        batch_size, seq_len = input_ids.shape
        k = (batch_size, seq_len+1) 
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len+1)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids[:,:-1], type_embeds[:,:-1,:]
        
        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]
        #print(type_ids)
        #if self.model_parallel:
        #    torch.cuda.set_device(inputs_embeds.device)
        #    type_embeds = type_embeds.to(inputs_embeds.device)
        
        
        if inputs_embeds is not None:
            inputs_embeds += type_embeds

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        """
        sequence_output = transformer_outputs[0]
        
        entity_mask = type_ids[:, :] == self.ENTITY_ID
        relation_mask = type_ids[:, :] == self.RELATION_ID


        #print(entity_mask.shape)
        entity_sequence_output = sequence_output * entity_mask.unsqueeze(-1)
        relation_sequence_output = sequence_output * relation_mask.unsqueeze(-1)
        
        entity_scores = self.entity_head(entity_sequence_output)
        relation_scores = self.relation_head(relation_sequence_output)

        entity_loss = None
        relation_loss = None
        if labels is not None:
            # Shift scores and labels by one
            entity_scores_shifted = entity_scores[:, :-1, :].contiguous()
            relation_scores_shifted = relation_scores[:, :-1, :].contiguous()
            
            labels_shifted = labels[:, 1:].contiguous()
            entity_labels = labels_shifted * entity_mask[:, 1:]
            relation_labels = labels_shifted * relation_mask[:, 1:]

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            
            entity_loss = loss_fct(entity_scores_shifted.view(-1, self.config.vocab_size), entity_labels.view(-1))
            relation_loss = loss_fct(relation_scores_shifted.view(-1, self.config.vocab_size), relation_labels.view(-1))

        total_loss = (entity_loss + relation_loss) if entity_loss is not None and relation_loss is not None else None

        # Combine entity and relation scores back into original sequence order
        combined_scores = entity_scores * entity_mask.unsqueeze(-1) + relation_scores * relation_mask.unsqueeze(-1)

        if not return_dict:
            output = (combined_scores,) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=total_loss,
            logits=combined_scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

        """

        hidden_states = transformer_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_entity_head.weight.device)


        # Get indexes of type embeddings
        #entity_token_ids = types_ids == DistilGPT2TwoHeadModel.ENTITY_ID
        #relation_token_ids = types_ids == DistilGPT2TwoHeadModel.RELATION_ID

        # Get logits from the two heads, first based on entity tokens, then on relation tokens
        lm_entity_logits = self.entity_head(hidden_states)#[entity_token_ids])
        lm_relation_logits = self.relation_head(hidden_states)#[relation_token_ids])
        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]




        '''
        
        start,E,R,E,R,E,R,E,end

        start,E,R,E,R,E,end

        start,E,R,E,R,E
        E,R,E,R,E,end

        # ent
        start,R,R,end
        start,E,E,E,end  #(slice)
        start,R,R,end     # [:(-1)]
        E,E,E,end  #(slice) [1:-1]   labels       

        # rel
        start,E,E,E,end   #(slice) [1:-1]  preds 
        start,R,R,end      #(slice)[1:]   labels
        '''

        def compute_loss(logits, labels, class_mask=None):
            if class_mask is not None:
                class_mask = class_mask.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=class_mask)
            logits = logits.contiguous()
            labels = labels.contiguous()
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)) 
            return lm_loss  
        #'''         
        loss = 0.
        # entity pred mask
        logits_mask = (type_ids != DistilGPT2TwoHeadModel.ENTITY_ID)[:, :(-1)].to(lm_entity_logits.device)#.unsqueeze(-1).expand(self.type_embeds.size())  
        label_mask = (type_ids != DistilGPT2TwoHeadModel.RELATION_ID)[:, 1:(-1)].to(lm_entity_logits.device)#.unsqueeze(-1)
        


        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]
        

        loss += compute_loss(lm_entity_logits[:,:(-1),:][logits_mask], input_ids[:,1:(-1)][label_mask],#lm_entity_logits[:,:-1,:][idx_mask,:],input_ids[:,1:][idx_mask],
                        #lm_entity_logits[:,:-1,:][:,idxs,:],input_ids[:,1:][:,idxs], #lm_entity_logits[:,:(-1),:][logits_mask], input_ids[:,1:(-1)][label_mask],
                    self.ent_mask_weight)


        # relation pred mask
        logits_mask = (type_ids != DistilGPT2TwoHeadModel.RELATION_ID)[:,1:(-1)].to(lm_entity_logits.device)#.unsqueeze(-1).expand(self.type_embeds.size())  
        label_mask = (type_ids != DistilGPT2TwoHeadModel.ENTITY_ID)[:,1:].to(lm_entity_logits.device)#.unsqueeze(-1)
        

        loss += compute_loss(lm_relation_logits[:,1:(-1),:][logits_mask], input_ids[:,1:][label_mask],#lm_relation_logits[:,:-1,:][idx_mask,:], input_ids[:,1:][idx_mask],
                    #lm_relation_logits[:,:-1,:][:,idxs,:], input_ids[:,1:][:,idxs],  #lm_relation_logits[:,1:(-1),:][logits_mask], input_ids[:,1:][label_mask],
                    self.rel_mask_weight)

        for i in range(sequence_len ):
            if i % 2 == 1:
                lm_entity_logits[:,i, :] = lm_relation_logits[:,i,:]  
        #'''
        #for i in range(sequence_len ):
        #    if i % 2 == 0:
        #        lm_entity_logits[:,i, :] = lm_relation_logits[:,i,:]  
        #loss = 0.
        #if labels is not None:
        #    loss = compute_loss(lm_entity_logits[:,:(-1),:], labels[:,1:])
        '''
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            entity_logits_shifted = lm_entity_logits[..., :-1:2, :].contiguous()
            relation_logits_shifted = lm_relation_logits[..., 1:-1:2, :].contiguous()
     
            labels_shifted = labels[..., 1:].contiguous()
            entity_labels_shifted = labels_shifted[..., ::2].contiguous()
            relation_labels_shifted = labels_shifted[..., 1::2].contiguous()
     
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            entity_loss = loss_fct(entity_logits_shifted.view(-1, entity_logits_shifted.size(-1)), entity_labels_shifted.view(-1))
            relation_loss = loss_fct(relation_logits_shifted.view(-1, relation_logits_shifted.size(-1)), relation_labels_shifted.view(-1))
            loss = entity_loss + relation_loss  

        #print(entity_logits[0,:3, :5], entity_logits.device)
        #print(relation_logits[0,:3, :5], relation_logits.device)
        #print(entity_logits[0,:3, :5], entity_logits.device)
        #print()
        
        for i in range(sequence_len ):
            if i % 2 == 0:
                lm_entity_logits[:,i, :] = lm_relation_logits[:,i,:]  
        '''
        if not return_dict:
            output = (lm_entity_logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output                

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_entity_logits,#logits,#lm_entity_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        #"""



class DistilGPT2Pos(GPT2LMHeadModel):
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.num_kg_types = len(DistilGPT2Pos.kg_categories)

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=self.num_kg_types,
                                                  embedding_dim=config.hidden_size)  # for entities, relations, and special tokens
        self.type_ids_cache = dict()
        self.type_embeds_cache = dict()
        self.idx_mask_cache = dict()

    def __init_type_embeddings(self, batch_size, num_hops):
        # num_hops = self.config.num_hops
        n_tokens = num_hops  # num_hops + 1 + num_hops + 2
        type_ids = torch.ones((batch_size, n_tokens), dtype=torch.long)

        for i in range(n_tokens):
            if i == 0 or i == n_tokens - 1:
                type_ids[:, i] = DistilGPT2Pos.SPECIAL_ID
            elif i % 2 == 1:
                type_ids[:, i] = DistilGPT2Pos.ENTITY_ID
            elif i % 2 == 0:
                type_ids[:, i] = DistilGPT2Pos.RELATION_ID
        type_ids = type_ids.to(self.type_embeddings.weight.device)
        type_embeds = self.type_embeddings(type_ids)
        return type_ids, type_embeds

    def __get_type_embeds(self, n_rows, n_cols):
        row = self.type_ids_row[:, :(n_cols - 1)]
        row = torch.hstack([row, torch.ones((1, 1)) * DistilGPT2Pos.SPECIAL_ID])
        type_ids = torch.vstack([row for _ in range(n_rows)])
        return type_ids, self.type_embeddings(type_ids.to(self.type_embeddings.weight.device))

    def __get_even_idx_mask(self, n_rows, n_cols):
        mask_key = (n_rows, n_cols)
        cur_mask = self.even_idx_mask[:(n_cols)]
        return torch.vstack([cur_mask for _ in range(n_rows)])

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        batch_size, seq_len = input_ids.shape
        k = (batch_size, seq_len)
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids, type_embeds

        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]

        if inputs_embeds is not None:
            inputs_embeds += type_embeds

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = transformer_outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + transformer_outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )




class CustomTrainer(Trainer):

    def __init__(
            self,
            dataset_name=None,
            n_hop=3,
            infer_batch_size=1,
            n_sequences_per_user=10,
            n_beams=30,
            tokenizer=None,
            eval_device='cpu',
            tokenized_kg=None,
            experiment_name=None,
            logit_processor_type='gcd',
            **kwargs
    ):
        super().__init__(**kwargs)

        data_dir = f"data/{dataset_name}"
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.n_hop = n_hop
        self.eval_device = eval_device

        self.SEQUENCE_LEN = 2 * n_hop + 2  # Special tokens [BOS] included

        self.N_RET_SEQ = n_sequences_per_user
        self.N_BEAMS = n_beams
        self.INFERENCE_BATCH_SIZE = infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user
        print('Sequence length: ', self.SEQUENCE_LEN)

        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_pid_to_eid(data_dir).values()])
        self.user_negatives_token_ids = get_user_negatives_tokens_ids(dataset_name, tokenizer)
        self.user_negatives = get_user_negatives(dataset_name)
        self.id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in uids}
        init_condition_fn = lambda uid: f"[BOS] U{uid} R-1"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in uids]}

        logit_processor = None
        logit_proc_kwargs = {}
        if logit_processor_type == 'gcd':
            logit_processor_cls = ConstrainedLogitsProcessorWordLevel 
        elif logit_processor_type == 'pgcd':
            logit_processor_cls = PrefixConstrainedLogitsProcessorWordLevel
        else:
            logit_processor_cls = PLMLogitsProcessorWordLevel 
            ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
            logit_proc_kwargs['ent_mask'] = ent_mask
            logit_proc_kwargs['rel_mask'] = rel_mask
            logit_proc_kwargs['token_id_to_token'] = token_id_to_token
        print('Using: ', logit_processor_cls)


        self.logits_processor = LogitsProcessorList([
            logit_processor_cls(tokenized_kg=tokenized_kg,
                                force_token_map=self.user_negatives_token_ids,
                                tokenizer=tokenizer,
                                total_length=self.SEQUENCE_LEN,  # LAST_TOKEN_POS,
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                id_to_uid_token_map=self.id_to_uid_token_map,
                                eos_token_ids=[
                                self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)],
                                **logit_proc_kwargs
                            )
        ])

        self.test_dataset = Dataset.from_dict(self.inference_paths)

    def __generate_topks_withWordLevel(self, model):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        batch_size = self.INFERENCE_BATCH_SIZE
        topk = defaultdict(list)
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)) as pbar:
            for i in range(0, len(self.test_dataset), batch_size):
                batch = self.test_dataset[i:i + batch_size]
                inputs = tokenizer(batch["uid"], return_tensors='pt', add_special_tokens=False, ).to(self.eval_device)
                outputs = model.generate(
                    **inputs,
                    max_length=self.SEQUENCE_LEN,
                    min_length=self.SEQUENCE_LEN,
                    num_return_sequences=self.N_RET_SEQ,
                    num_beams=self.N_BEAMS,
                    length_penalty=0.,
                    num_beam_groups=5,
                    diversity_penalty=0.3,
                    do_sample=False,
                    # top_p=0.4,
                    logits_processor=self.logits_processor,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                def normalize_tuple(logits_tuple):
                    # Normalize each tensor in the tuple
                    normalized_tuple = tuple(torch.softmax(logits, dim=-1) for logits in logits_tuple)
                    return normalized_tuple

                def calculate_sequence_scores(normalized_tuple, sequences):
                    # Get the last 5 tokens from each sequence
                    last_5_tokens = sequences[:, -5:]
                    sequence_scores = []
                    # Iterate over each tensor in the normalized tuple
                    for i in range(5):
                        # Get the probabilities corresponding to the ith token in last_5_tokens
                        probs = normalized_tuple[i].gather(1, last_5_tokens[:, i].unsqueeze(1))
                        sequence_scores.append(probs)
                    # Convert the list of tensors into a single tensor
                    sequence_scores = torch.cat(sequence_scores, dim=-1)
                    # Calculate the average score over the last 5 positions for each sequence
                    sequence_scores = sequence_scores.mean(dim=-1)
                    return sequence_scores

                outputs.scores = normalize_tuple(outputs.scores)
                outputs.sequences_scores = calculate_sequence_scores(outputs.scores, outputs.sequences)
                sorted_indices = outputs.sequences_scores.argsort(descending=True)
                sorted_sequences = outputs.sequences[sorted_indices]
                K = 10
                
                for sequence in sorted_sequences:
                    sequence = tokenizer.decode(sequence).split(' ')
                    #print(sequence)
                    uid = sequence[1][1:]
                    if len(topk[uid]) >= K:
                        continue
                    recommended_token = sequence[-1]
                    recommended_item = recommended_token[1:]
                    if not recommended_token.startswith("P"):
                        continue
                    if recommended_item not in self.user_negatives[uid]:
                        continue
                    if recommended_item in topk[uid]:
                        continue
                    topk[uid].append(recommended_item)
                pbar.update(batch_size)
        print("Average topk length:", sum(len(v) for v in topk.values()) / len(topk))
        # print("Percentage of sequence that contain invalid item:", count/len(sorted_sequences))
        return topk

    def evaluate(self, model):
        # Generate paths for the test users
        # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok

        topks = self.__generate_topks_withWordLevel(model)
        #check_dir(f"./results/{self.dataset_name}/{self.custom_model_name}")
        #pickle.dump(topks, open(f"./results/{self.dataset_name}/{self.custom_model_name}/topks.pkl", "wb"))
        check_dir(f"./results/{self.dataset_name}/{self.experiment_name}")
        pickle.dump(topks, open(f"./results/{self.dataset_name}/{self.experiment_name}/topks.pkl", "wb"))        
        metrics = {"ndcg": [], "mmr": [], }
        for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
            hits = []
            for recommended_item in topk:
                if recommended_item in self.test_set[uid]:
                    hits.append(1)
                else:
                    hits.append(0)
            while len(hits) < 10:
                hits.append(0) 
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
            metrics["ndcg"].append(ndcg)
            metrics["mmr"].append(mmr)

        print(
            f"no of users: {len(self.test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
        metrics_ = dict()
        for k in metrics:
            metrics_[f'eval_{k}'] = np.mean(metrics[k])
        return metrics_

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):

        logs: Dict[str, float] = {}
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

        metrics = None
        if self.control.should_evaluate and self.control.should_save:
            metrics = self.evaluate(model)
            logs.update(metrics)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # finish logging results
        if self.control.should_log:
            self.log(logs)
# Read an example and return the tokenized version
def tokenize_function(examples: str, context_length: int = 200):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)


def fine_tune(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
        # 'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    pretrain_model = f'./clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}/checkpoint-{args.pretrain_ckpt}'
    # Initializing the selected model style configuration
    config = AutoConfig.from_pretrained(
        f'clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}/checkpoint-{args.pretrain_ckpt}',
        **config_kwargs
    )

    print('Model config: ', config)
    config.update({'num_hops': args.n_hop,
                   'sample_size_pretrain': args.sample_size_pretrain,
                   'sample_size_finetune': args.sample_size_finetune,
                   'sample_size_hop': args.sample_size_hop,
                   'task': args.task,
                   'train_batch_size': args.batch_size,
                   'test_batch_size': args.infer_batch_size,
                   'ent_mask': ent_mask,
                   'rel_mask': rel_mask,
                   'token_id_to_token': token_id_to_token,
                   })

    # Initializing a model from the configuration
    if args.task == 'finetune' and args.pretrain_ckpt is not None:
        print('Loading from checkpoint for finetuning: ', args.pretrain_ckpt)
        model = DistilGPT2Pos.from_pretrained(pretrain_model,
                                              config=config)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)

    # Training arguments
    custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.n_beams}-{args.n_seq_infer}-{args.logit_processor_type}"

    STEP_INTERVAL = 50
    EVAL_STEP_INTERVAL = 500

    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=EVAL_STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=3e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,  # number of warmup steps for learning rate
        save_steps=EVAL_STEP_INTERVAL,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        #no_cuda=True,
        seed=SEED,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomTrainer(
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        n_beams=args.n_beams,        
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        experiment_name=custom_name,
        data_collator=data_collator)

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


def train_end_to_end(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)
    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
        # 'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    embeds=None

    # Initializing a model from the configuration
    if args.continue_training and args.pretrain_ckpt is not None:
        pretrain_model = f'./clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.logit_processor_type}/checkpoint-{args.pretrain_ckpt}'
        config = AutoConfig.from_pretrained(
            pretrain_model,
            **config_kwargs
        )
        print('Loading from checkpoint for continuing traning: ', args.pretrain_ckpt)
        model = DistilGPT2Pos.from_pretrained(pretrain_model,
                                              config=config)
        print(model.config)
    else:
        print('TRAINING NEW MODEL')
        if 'plm-rec' in model_name:
            model_class, model_subname = model_name.split('@')
            if len(args.emb_filename) > 0:
                    embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
                    try:
                        embeds = pickle.load(open(embed_filepath, 'rb'))
                    except:
                        embeds = None
                    if embeds:
                        print('Using embeddings: ',args.emb_filename)
                        config_kwargs.update({
                        'hidden_size':int(args.emb_size),
                        'num_attention_heads':int(args.emb_size)//10
                        })

            config = AutoConfig.from_pretrained(
                model_class,
                **config_kwargs
            )

            model_cls = DistilGPT2TwoHeadModel#(config)
        else:
            config = AutoConfig.from_pretrained(
                model_name,
                **config_kwargs
            )

            model_cls = DistilGPT2Pos#(config)
    print('Model config: ', config)
    config.update({'num_hops': args.n_hop,
                   'sample_size_pretrain': args.sample_size_pretrain,
                   'sample_size_finetune': args.sample_size_finetune,
                   'sample_size_hop': args.sample_size_hop,
                   'task': args.task,
                   'train_batch_size': args.batch_size,
                   'test_batch_size': args.infer_batch_size,
                   'ent_mask': ent_mask,
                   'rel_mask': rel_mask,
                   'token_id_to_token': token_id_to_token})

    model = model_cls(config)


    # model = DistilGPT2TwoHeadModel(config)
    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)

    if embeds:
        mapper = EmbeddingMapper(tokenizer, kg, embeds)      
        mapper.init_with_embedding(model.transformer.wte.weight)
        print(f'Model {model_name} initialized with custom embeddings of size: ', model.transformer.wte.weight.shape)

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)

    # Training arguments
    #custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.n_beams}-{args.n_seq_infer}-{args.logit_processor_type}"
    custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_finetune}-{args.sample_size_hop}-{args.logit_processor_type}"


    STEP_INTERVAL = 100
    EVAL_STEP_INTERVAL = 1000
    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=EVAL_STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=3e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=True,#True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,  # number of warmup steps for learning rate
        save_steps=EVAL_STEP_INTERVAL,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,
        #no_cuda=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomTrainer(
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        n_beams=args.n_beams,         
        tokenizer=tokenizer,
        eval_device=args.eval_device,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        experiment_name=custom_name,
        logit_processor_type=args.logit_processor_type,
        data_collator=data_collator)

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


def train_pretraining(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)

    config_kwargs = {
        'vocab_size': len(tokenizer),
        'n_ctx': context_length,
        # 'n_positions': context_length,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    pretrain_model = f'./clm-pretrain-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}-{args.logit_processor_type}/checkpoint-{args.pretrain_ckpt}'
    config = AutoConfig.from_pretrained(
        pretrain_model,
        **config_kwargs
    )

    # Initializing a model from the configuration
    # model = DistilGPT2TwoHeadModel(config)
    if args.continue_training and args.pretrain_ckpt is not None:
        print('Loading from checkpoint for continuing traning: ', args.pretrain_ckpt)
        model = DistilGPT2Pos.from_pretrained(pretrain_model,
                                              config=config)
        print(model.config)
    else:
        print('TRAINING NEW MODEL')
        config = AutoConfig.from_pretrained(
            model_name,
            **config_kwargs
        )
        model = DistilGPT2Pos(config)

    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'

    data_dir_mapping = f'{dirpath}/mapping/'
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)
    try:
        embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
        embeds = pickle.load(open(embed_filepath, 'rb'))
        print('TRANSE LOADED')
        model, config = _initialise_weights_from_kge(embeds, tokenizer, kg, config_kwargs, model, args)
    except:
        pass

    print('Model config: ', config)
    config.update({'num_hops': args.n_hop,
                   'sample_size_pretrain': args.sample_size_pretrain,
                   'sample_size_finetune': args.sample_size_finetune,
                   'sample_size_hop': args.sample_size_hop,
                   'task': args.task,
                   'train_batch_size': args.batch_size,
                   'test_batch_size': args.infer_batch_size,
                   'ent_mask': ent_mask,
                   'rel_mask': rel_mask,
                   'token_id_to_token': token_id_to_token})

    tokenized_kg, _ = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True)

    # Training arguments
    custom_name = f"{args.task}-{args.dataset}-{args.model}-{args.sample_size_pretrain}-{args.sample_size_hop}"

    STEP_INTERVAL = 100
    EVAL_STEP_INTERVAL = 1000
    training_args = TrainingArguments(
        f"clm-{custom_name}",
        evaluation_strategy="steps",
        save_strategy='steps',
        eval_steps=EVAL_STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=2e-4,#3e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=True,
        logging_first_step=True,
        # use_mps_device=True,
        num_train_epochs=20,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=250,  # number of warmup steps for learning rate
        save_steps=EVAL_STEP_INTERVAL,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        load_best_model_at_end=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train model
    trainer.train()

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--task", type=str, default="end-to-end", help="{pretrain, finetune, end-to-end}")
    parser.add_argument("--sample_size_pretrain", type=str, default="1500",
                        help="Which sample size dataset to use for pretraining")
    parser.add_argument("--sample_size_finetune", type=str, default="500",
                        help="Which sample size dataset to use for fine-tuning/end-to-end")
    parser.add_argument("--sample_size_hop", type=str, default="3",
                        help="Which number of hops dataset to use for fine-tuning/end-to-end")
    parser.add_argument("--logit_processor_type", type=str, default="gcd",
                        help="Path sequence deconding method: default to Graph Constrained Decoding")    


    # Model arguments
    parser.add_argument("--model", type=str, default="gpt2-large",
                        help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--nproc", type=int, default=8, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=256, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="Test batch size")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")
    parser.add_argument("--load_data", type=bool, default=False, help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default='1', help="")
    parser.add_argument("--infer_batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=30,
                        help="Number of sequences generated for each user")
    parser.add_argument("--n_beams", type=int, default=30,
                        help="Number of sequences generated for each user")    

    # Parameter relative to resume training
    parser.add_argument("--continue_training", type=bool, default=False,
                        help="Whether to continue training from a checkpoint or not")
    parser.add_argument("--pretrain_ckpt", type=none_or_int, default='8500',
                        help="Checkpoint from which to resume training of the model (default to starting from scratch)")
    # Parameter relative to weight initialization
    parser.add_argument("--embedding_root_dir", type=str, default="./embedding-weights",
                        help="default: ./embedding-weights")
    parser.add_argument("--emb_filename", type=str, default='', help="default: 'transe_embed.pkl'")
    parser.add_argument("--emb_size", type=int, default=100,
                        help="Transformer Embedding size (must match external embedding size, if chosen)")

    args = parser.parse_args()

    set_seed(SEED)

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    sample_size = args.sample_size_pretrain if args.task == 'pretrain' else args.sample_size_finetune
    dataset_hop_size = args.sample_size_hop

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if args.load_data:
        task = args.task
        if task == 'finetune':  # They share the same dataset
            task = 'end-to-end'
        tokenized_dataset = load_from_disk(
            f"data/{dataset_name}/{TOKENIZER_TYPE}/{task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)
    else:
        # Load the dataset
        data_dir = f"data/{dataset_name}"
        plain_text_path = True

        print("Loading and processing path sequences...")
        dataset = PathDataset(dataset_name, data_dir, task=args.task, sample_size=sample_size, n_hop=dataset_hop_size,
                              plain_text_path=plain_text_path)

        dataset.show_random_examples()
        dataset = dataset.dataset
        print(type(dataset))
        if not os.path.exists(tokenizer_file):
            # Word level tokenizer
            print("Training tokenizer...")
            tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
            special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
            tokenizer.train_from_iterator(dataset["path"], trainer=trainer)
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[BOS]:0 $A:0 [EOS]:0",
                special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]"))]
            )

            tokenizer.save(tokenizer_file)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
                                                eos_token="[EOS]", bos_token="[BOS]",
                                                pad_token="[PAD]", unk_token="[UNK]",
                                                mask_token="[MASK]", use_fast=True)
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, max_len=args.context_length,
                                                eos_token="[EOS]", bos_token="[BOS]",
                                                pad_token="[PAD]", unk_token="[UNK]",
                                                mask_token="[MASK]", use_fast=True)

        if args.task == "pretrain":
            # Load the specified tokenizer
            print("Tokenizing dataset...")

            tokenized_dataset = dataset.map(tokenize_function,
                                            batched=True,
                                            num_proc=args.nproc,
                                            remove_columns=["path"]
                                            )
            print("Train/Validation split...")
            tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.10)
            print(tokenized_dataset['train'][0])
        elif args.task == "finetune" or args.task == "end-to-end":
            print("Tokenizing dataset...")
            tokenized_dataset = dataset.map(tokenize_function,
                                            batched=True,
                                            num_proc=args.nproc,
                                            remove_columns=["path"]
                                            )
            tokenized_dataset = DatasetDict({
                "train": tokenized_dataset,
            })

        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(
            f"data/{dataset_name}/{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")

    # Train the model
    if args.load_model: #ENSURE IS WORKING
        # Training arguments
        curr_sample_size = args.sample_size_pretrain if args.task == 'pretrain' else args.sample_size_finetune
        custom_name = f'clm-{args.task}-{args.dataset}-{args.model}-{curr_sample_size}-{args.sample_size_hop}-{args.logit_processor_type}/checkpoint-{args.eval_ckpt_iter}'  # f"clm-from_scratch-{args.dataset}-{args.model}"
        model = AutoModelForCausalLM.from_pretrained(
            custom_name)  # f'models-weights/{dataset_name}/{model_name}/{custom_name}')
    else:
        if args.task == "pretrain":
            model = train_pretraining(model_name, tokenizer, tokenized_dataset, args.context_length, args)
        elif args.task == "finetune":
            model = fine_tune(model_name, tokenizer, tokenized_dataset, args.context_length, args)
        elif args.task == "end-to-end":
            model = train_end_to_end(model_name, tokenizer, tokenized_dataset, args.context_length, args)

    #evaluate(model, args)
