import argparse
import math
import os
from typing import List
import pickle
import random
from collections import defaultdict
from typing import List, Dict

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import multiprocessing as mp
import itertools
import functools
from transformers.utils import is_torch_tpu_available
import torch

import numpy as np
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, PhrasalConstraint, LogitsProcessorList, \
    set_seed, pipeline, GPT2LMHeadModel



import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config





from pathlm.models.lm.generation_constraints import TypifiedForceLastTokenLogitsProcessorWordLevel, ConstrainedLogitsProcessorWordLevel
from pathlm.models.lm.evaluate import evaluate
from pathlm.models.lm.lm_utils import MLM_MODELS
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import SEED, get_pid_to_eid, get_eid_to_name_map, get_data_dir, get_set, check_dir
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k 
from pathlm.tools.mapper import EmbeddingMapper
from pathlm.sampling.container.kg_analyzer import KGstats
from pathlm.sampling.container.constants import LiteralPath, TypeMapper
from pathlm.models.rl.PGPR.pgpr_utils import PRODUCT, USER, ENTITY, RELATION
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)








#TODO: TEST FORWARD + LOSS
class DistilGPT2TwoHeadModel(GPT2LMHeadModel):
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID] 

    def __init__(self, config):
        super().__init__(config)
        self.config = config
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
        self.lm_entity_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.lm_relation_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=True)


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
        k = (batch_size, seq_len) 
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids, type_embeds
        
        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]
        #if self.model_parallel:
        #    torch.cuda.set_device(inputs_embeds.device)
        #    type_embeds = type_embeds.to(inputs_embeds.device)
        
        
        if inputs_embeds is not None:
            input_embeds += type_embeds

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


        hidden_states = transformer_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_entity_head.weight.device)


        # Get indexes of type embeddings
        #entity_token_ids = types_ids == DistilGPT2TwoHeadModel.ENTITY_ID
        #relation_token_ids = types_ids == DistilGPT2TwoHeadModel.RELATION_ID

        # Get logits from the two heads, first based on entity tokens, then on relation tokens
        lm_entity_logits = self.lm_entity_head(hidden_states)#[entity_token_ids])
        lm_relation_logits = self.lm_relation_head(hidden_states)#[relation_token_ids])





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
        logits_mask = (type_ids != DistilGPT2TwoHeadModel.ENTITY_ID)[:, :(-1)]#.unsqueeze(-1).expand(self.type_embeds.size())  
        label_mask = (type_ids != DistilGPT2TwoHeadModel.RELATION_ID)[:, 1:(-1)]#.unsqueeze(-1)
        
        #print(logits_mask.shape)
        #print(label_mask.shape)
        #print(lm_entity_logits[:,:(-1),:].shape)
        #print( input_ids[:,1:(-1)].shape)
        #print()

        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]
        
        #idxs = [i%2 == 0 for i in range(sequence_len-1) ]
        #idx_mask = torch.stack( [torch.BoolTensor(idxs) for _ in range(batch_size)] )


        #mask_key = (batch_size,sequence_len-1)
        #if mask_key not in self.idx_mask_cache:        
        #    self.idx_mask_cache[mask_key] = self.__get_even_idx_mask(batch_size, sequence_len-1)
        #idx_mask = self.idx_mask_cache[mask_key].to(lm_entity_logits.device)

        loss += compute_loss(lm_entity_logits[:,:(-1),:][logits_mask], input_ids[:,1:(-1)][label_mask],#lm_entity_logits[:,:-1,:][idx_mask,:],input_ids[:,1:][idx_mask],
                        #lm_entity_logits[:,:-1,:][:,idxs,:],input_ids[:,1:][:,idxs], #lm_entity_logits[:,:(-1),:][logits_mask], input_ids[:,1:(-1)][label_mask],
                    self.ent_mask_weight)


        # relation pred mask
        logits_mask = (type_ids != DistilGPT2TwoHeadModel.RELATION_ID)[:,1:(-1)]#.unsqueeze(-1).expand(self.type_embeds.size())  
        label_mask = (type_ids != DistilGPT2TwoHeadModel.ENTITY_ID)[:,1:]#.unsqueeze(-1)
        
        #print(logits_mask.shape)
        #print(label_mask.shape)
        #print(lm_relation_logits[:,1:(-1),:].shape)
        #print( input_ids[:,1:].shape)        
        #print()
        #print()
        
        #idxs = [i%2 == 1 for i in range(sequence_len-1)  ]
        #idx_mask = torch.stack( [torch.BoolTensor(idxs) for _ in range(batch_size)] )
        #print(idx_mask.device)
        #idx_mask = torch.logical_not(self.idx_mask_cache[mask_key].to(lm_entity_logits.device))


        LAMBDA = 200.
        loss += LAMBDA * compute_loss(lm_relation_logits[:,1:(-1),:][logits_mask], input_ids[:,1:][label_mask],#lm_relation_logits[:,:-1,:][idx_mask,:], input_ids[:,1:][idx_mask],
                    #lm_relation_logits[:,:-1,:][:,idxs,:], input_ids[:,1:][:,idxs],  #lm_relation_logits[:,1:(-1),:][logits_mask], input_ids[:,1:][label_mask],
                    self.rel_mask_weight)
        #print(loss)
        #idxs = [[i%2 == 0 for _ in range(lm_entity_logits.shape[-1]) ]     for i in range(sequence_len) ]
        #idx_mask = torch.stack( [torch.BoolTensor(idxs) for _ in range(batch_size)] ).to(lm_entity_logits.device)
        #'''
        #out_logits = torch.zeros(lm_entity_logits.shape)
        #print(lm_entity_logits)
        for i in range(sequence_len ):
            if i % 2 == 1:
                lm_entity_logits[:,i, :] = lm_relation_logits[:,i,:]  
            #else:
            #    out_logits[:,i, :] = lm_relation_logits[:,i,:]
        #print(lm_entity_logits)
        '''

        batch_size = input_ids.shape[0]
        sequence_len = input_ids.shape[-1]        
        mask_key = (batch_size,sequence_len)
        if mask_key not in self.idx_mask_cache:         
            self.idx_mask_cache[mask_key] =  self.__get_even_idx_mask(batch_size, sequence_len) 
        idx_mask = self.idx_mask_cache[mask_key]
        mask_key = (batch_size,sequence_len,lm_entity_logits.shape[-1])
        if mask_key not in self.idx_mask_cache:
            idx_mask = torch.stack([idx_mask for _ in range(lm_entity_logits.shape[-1])], dim=-1)#.to(lm_entity_logits.device)
            self.idx_mask_cache[mask_key] =  idx_mask
        idx_mask = self.idx_mask_cache[mask_key]
        logits =  torch.where(idx_mask.to(lm_entity_logits.device),lm_entity_logits, lm_relation_logits) 
        loss = compute_loss(logits[:,:-1,:], input_ids[:,1:])
        '''

        #if not return_dict:
        #    output = (logits,) + transformer_outputs[1:]
        #    return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_entity_logits,#logits,#lm_entity_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )




#TODO: TEST WITHOUT GROUP TEXTS
def generate_type_ids(sequence):
    # Define type ids for special tokens, entities, and relations
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2

    type_ids = []
    for idx, token_id in enumerate(sequence['input_ids']):
        if idx == 0 or idx == len(sequence['input_ids']) - 1:
            type_ids.append(SPECIAL_ID)
        elif idx % 2 == 1:
            type_ids.append(ENTITY_ID)
        elif idx % 2 == 0:
            type_ids.append(RELATION_ID)
    sequence['type_ids'] = ' '.join(type_ids)
    return sequence


class CustomTrainer(Trainer):

    def __init__(
        self,
        dataset_name=None,
        n_hop=3,
        infer_batch_size=1,
        n_sequences_per_user=10,
        tokenizer=None,
        eval_device='cpu',
        tokenized_kg=None,
        **kwargs
    ):   
        super().__init__(**kwargs)


        data_dir = f"data/{dataset_name}"
        model = kwargs['model']
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.custom_model_name = model.name_or_path.split("/")[-1]
        self.test_set = get_set(dataset_name, set_str='test')
        uids = list(self.test_set.keys())
        self.n_hop = n_hop
        self.eval_device = eval_device

        #self.SEQUENCE_LEN =  2 + 2 + n_hop*2 + (n_hop-1)*2  # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
        self.SEQUENCE_LEN =  2*n_hop+1 +1  # 14#22#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1

        self.INFERENCE_BATCH_SIZE = args.infer_batch_size
        self.N_SEQUENCES_PER_USER = n_sequences_per_user
        print('Sequence length: ',self.SEQUENCE_LEN)


        # Load user negatives
        self.last_item_idx = max([int(id) for id in get_pid_to_eid(data_dir).values()])
        self.user_negatives = get_user_negatives_tokens_ids(dataset_name, tokenizer)

        #self.generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=eval_device)
        
        topk = defaultdict(list)
        non_product_count = 0

        self.id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in uids}

        #'''
        #tokenizer
        #init_condition_fn = lambda uid: f"Us U{uid} Rf R-1 Ps"
        #init_condition_fn = lambda uid: f"{self.tokenizer.eos_token} U{uid} R-1"
        init_condition_fn = lambda uid: f"U{uid} R-1"
        self.inference_paths = {'uid': [init_condition_fn(uid) for uid in uids] }
        

        
        self.logits_processor = LogitsProcessorList([
            
                ConstrainedLogitsProcessorWordLevel(tokenized_kg=tokenized_kg,
                        force_token_map=self.user_negatives, 
                        tokenizer=tokenizer, 
                        total_length=self.SEQUENCE_LEN,#LAST_TOKEN_POS,
                        num_return_sequences=self.N_SEQUENCES_PER_USER,
                        id_to_uid_token_map=self.id_to_uid_token_map,
                        eos_token_ids=[self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)])#6)
            #ForceLastTokenLogitsProcessorWordLevel(user_negatives, tokenizer=tokenizer, total_length=LAST_TOKEN_POS)
            # 7 = 2 input token + 5 generated tokens
        ])

        self.test_dataset = Dataset.from_dict(self.inference_paths)

    def __lazy_load_data(dataset):
            for row in dataset:
                    yield row["uid"]

    def __generate_topks_withWordLevel(self, model):
        self.generator = pipeline('text-generation', model=model, tokenizer=self.tokenizer, device=self.eval_device)

        outputs = self.generator(CustomTrainer.__lazy_load_data(self.test_dataset),#f"Us U{uid} Rf R-1",
                                max_length=self.SEQUENCE_LEN,#22#15  # 2 + 2 + 5*2 + 4*2       7 = 2 * 2 input token + 5 * 2 generated tokens + 1
                                num_return_sequences=self.N_SEQUENCES_PER_USER,
                                #num_beams=self.N_SEQUENCES_PER_USER,
                                #do_sample=True,
                                #top_p=0.4,
                                logits_processor=self.logits_processor,
                                batch_size=self.INFERENCE_BATCH_SIZE,
        )  
        topk = defaultdict(list)
        non_product_count = 0
        with tqdm(initial=0, desc="Generating topks", colour="green", total=len(self.user_negatives)  ) as pbar:
            for output_batch in outputs:
                    for output in output_batch:
                        output = output['generated_text'].split(" ")
                        #uid = output[1][1:]
                        print(output)
                        uid = output[0][1:]
                        recommended_token = output[-1]
                        recommended_item = recommended_token[1:]
                        if len(recommended_token) < 2  or not recommended_token.startswith("P"):

                            non_product_count += 1
                            pass
                        #print(output)
                        topk[uid].append(recommended_item)
                    pbar.update(1)
        print(f"Non product count: {non_product_count}")        
        return topk

    def evaluate(self, model):
        # Generate paths for the test users
        # This euristic assume that our scratch models use wordlevel and ft models use BPE, not ideal but for now is ok

        topks = self.__generate_topks_withWordLevel(model)
        check_dir(f"./results/{self.dataset_name}/{self.custom_model_name}")
        pickle.dump(topks, open(f"./results/{self.dataset_name}/{self.custom_model_name}/topks.pkl", "wb"))
        metrics = {"ndcg": [], "mmr": [],  }
        for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
            hits = []
            for recommended_item in topk:
                if recommended_item in self.test_set[uid]:
                    hits.append(1)
                else:
                    hits.append(0)
            ndcg = ndcg_at_k(hits, len(hits))
            mmr = mmr_at_k(hits, len(hits))
            metrics["ndcg"].append(ndcg)
            metrics["mmr"].append(mmr)

        print(f"no of users: {len(self.test_set.keys())}, ndcg: {np.mean(metrics['ndcg'])}, mmr: {np.mean(metrics['mmr'])}")
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
            '''
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            '''
            metrics = self.evaluate(model.module)
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
def tokenize_function(examples: str, context_length: int=200):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length)

def train_from_scratch(model_name: str, tokenizer, tokenized_dataset, context_length, args: argparse.Namespace):
    embed_filepath = os.path.join(args.embedding_root_dir, args.dataset, args.emb_filename)
    try:
        embeds = pickle.load(open(embed_filepath, 'rb'))
    except:
        embeds = None

    ent_mask = []
    rel_mask = []
    class_weight = 1.
    token_id_to_token = dict()
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
        if token[0] == LiteralPath.rel_type or not token[0].isalpha():
            rel_mask.append(class_weight)
        else:
            rel_mask.append(0)
        if token[0] != LiteralPath.rel_type or not token[0].isalpha():
            ent_mask.append(class_weight)
        else:
            ent_mask.append(0)

        token_id_to_token[token_id] = token
    #print(ent_mask)
    #print(rel_mask)
    config_kwargs ={        
        'vocab_size':len(tokenizer),
        'n_ctx':context_length,
        #'n_positions': context_length,
        'pad_token_id':tokenizer.pad_token_id,
        'bos_token_id':tokenizer.bos_token_id,
        'eos_token_id':tokenizer.eos_token_id,
    }

    if embeds:
        print('Using embeddings: ',args.emb_filename)
    config_kwargs.update({
    'hidden_size':args.emb_size,
    'num_attention_heads':args.emb_size//10
    })

    # Initializing the selected model style configuration
    config = AutoConfig.from_pretrained(
        model_name,
        **config_kwargs
    )
    print('Model config: ', config)
    config.update({        'num_hops': args.n_hop,
        'train_batch_size':args.batch_size,
        'test_batch_size':args.infer_batch_size,
        'ent_mask': ent_mask,
        'rel_mask': rel_mask,
        'token_id_to_token': token_id_to_token})

    # Initializing a model from the configuration
    #model = DistilGPT2TwoHeadModel(config)
    if args.pretrain_ckpt is not None:
        custom_name = f'clm-from_scratch-{args.dataset}-{args.model}-ckpt/checkpoint-{args.pretrain_ckpt}'
        model = AutoModelForCausalLM.from_pretrained(custom_name)
    else:
        model = AutoModelForCausalLM.from_config(config)
    #model = DistilGPT2TwoHeadModel(config)
    ROOT_DIR = os.environ('DATA_ROOT') if 'DATA_ROOT' in os.environ else '.'
    # Dataset directories.
    dirpath = f'{ROOT_DIR}/data/{args.dataset}/preprocessed'
    
    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{args.dataset}/preprocessed/mapping/')
    kg = KGstats(args, args.dataset, dirpath, data_dir=data_dir_mapping)  
    if embeds:
        mapper = EmbeddingMapper(tokenizer, kg, embeds)      
        mapper.init_with_embedding(model.transformer.wte.weight)
    
    def tokenize_augmented_kg(kg,  tokenizer, use_token_ids=False):
        type_id_to_subtype_mapping = kg.dataset_info.groupwise_global_eid_to_subtype.copy()
        rel_id2type = kg.rel_id2type.copy()
        type_id_to_subtype_mapping[RELATION] = {int(k):v for k,v in rel_id2type.items()}        

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
            kg_to_vocab_mapping[(subtype,cur_id)] = token_id
        
        for head_type in aug_kg:
            for head_id in aug_kg[head_type]:
                head_key = head_type,head_id
                if head_key not in kg_to_vocab_mapping:
                    continue            
                head_ent_token = kg_to_vocab_mapping[head_key]
                tokenized_kg[head_ent_token] = dict()
                
                for rel in aug_kg[head_type][head_id]:
                    rel_token = kg_to_vocab_mapping[rel,None]
                    tokenized_kg[head_ent_token][rel_token] = set()
                    
                    for tail_type in aug_kg[head_type][head_id][rel]:
                        for tail_id in aug_kg[head_type][head_id][rel][tail_type]:
                            tail_key = tail_type,tail_id
                            if tail_key not in kg_to_vocab_mapping:
                                continue
                            tail_token = kg_to_vocab_mapping[tail_key]
                            tokenized_kg[head_ent_token][rel_token].add(tail_token)
                    
        return tokenized_kg, kg_to_vocab_mapping

    tokenized_kg, _ = tokenize_augmented_kg(kg,  tokenizer, use_token_ids=True)
    #
    '''TypifiedForceLastTokenLogitsProcessorWordLevel(force_token_map=self.user_negatives, 
                        tokenizer=tokenizer, 
                        total_length=self.SEQUENCE_LEN,#LAST_TOKEN_POS,
                        num_return_sequences=self.N_SEQUENCES_PER_USER,
                        id_to_uid_token_map=self.id_to_uid_token_map,
                        eos_token_ids=[self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)])
    '''

    # Training arguments
    custom_name = f"from_scratch-{args.dataset}-{args.model}"

    #TODO: Add to the dataset the type_ids
    #tokenized_dataset.map(generate_type_ids, batched=False, num_proc=args.nproc)
    STEP_INTERVAL=100
    # Training arguments for Causal Language Model task
    training_args = TrainingArguments(
        f"clm-{custom_name}",
            evaluation_strategy="steps",
            save_strategy='steps',
        eval_steps=STEP_INTERVAL,
        logging_steps=STEP_INTERVAL,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=False,
        fp16=False,
        logging_first_step=True,
        #use_mps_device=True,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        warmup_steps=1000,  # number of warmup steps for learning rate
        save_steps=STEP_INTERVAL,
        save_total_limit=2,
        #load_best_model_at_end=True,
        metric_for_best_model='ndcg',
        greater_is_better=True,
        seed=SEED,
    )


    if model_name in MLM_MODELS:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomTrainer(
        dataset_name=args.dataset,
        tokenized_kg=tokenized_kg,
        n_hop=args.n_hop,
        infer_batch_size=args.infer_batch_size,
        n_sequences_per_user=args.n_seq_infer,
        tokenizer=tokenizer,
        eval_device=args.eval_device,
                model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator)


    # Train model
    trainer.train()
    # Evaluate model
    #eval_results = trainer.evaluate()
    #print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Save model
    weight_path = f"./models-weights/{args.dataset}/{args.model}/{custom_name}"
    check_dir(f"./models-weights/{args.dataset}/{args.model}/{custom_name}")
    trainer.save_model(weight_path)
    return model

def stratified_sampling(dataset, valid_size: float=0.05):
    # Extract user_ids
    uid_to_idxs = {}
    for idx, path in enumerate(dataset['path']):
        uid = path.split(' ')[0]
        if uid not in uid_to_idxs:
            uid_to_idxs[uid] = []
        uid_to_idxs[uid].append(idx)

    # Create indices for stratified split
    train_indices, test_indices = [], []

    for uid, idxs in uid_to_idxs.items():
        np.random.shuffle(idxs)  # randomize user specific indices

        split_point = int(len(idxs) * valid_size)  # calculate split point

        # Append to the respective lists
        test_indices.extend(idxs[:split_point])
        train_indices.extend(idxs[split_point:])

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset.select(train_indices),
        'test': dataset.select(test_indices),
    })
    return dataset_dict


def add_user_id(example):
    example["user_id"] = example['path'].split(' ')[0]
    return example
def none_or_str(value):
    if value == 'None':
        return None
    return value
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--nproc", type=int, default=2, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=24, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=24, help="Test batch size")
    parser.add_argument("--context_length", type=int, default=200,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")    
    parser.add_argument("--load_data", type=bool, default=False, help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default='10000', help="")
    parser.add_argument("--infer_batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=10, help="Number of sequences generated for each user at inference time")

    parser.add_argument("--embedding_root_dir", type=str, default="./embedding-weights", help="default: ./embedding-weights")
    parser.add_argument("--emb_filename", type=str, default='transe_embed.pkl', help="default: 'transe_embed.pkl'")
    parser.add_argument("--emb_size", type=int, default=100, help="Transformer Embedding size (must match external embedding size, if chosen)")
    parser.add_argument("--pretrain_ckpt", type=none_or_int, default=None, help="Checkpoint from which to resume training of the model (default to starting from scratch)")



    args = parser.parse_args()


    set_seed(SEED)

    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model
    dataset_name = args.dataset

    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    if args.load_data:
        tokenized_dataset = load_from_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=args.context_length,
                                            eos_token="[EOS]", bos_token="[BOS]",
                                            pad_token="[PAD]", unk_token="[UNK]",
                                            mask_token="[MASK]", use_fast=True)        
    else:
        # Load the dataset
        data_dir = f"data/{dataset_name}"
        plain_text_path=True

        print("Loading and processing path sequences...")
        dataset = PathDataset(dataset_name, data_dir, plain_text_path=plain_text_path)
        def convert_and_add_uid(paths_dict, convert_fn):
            batch_dict = {"path":[], "user_id":[]}

            paths_list = batch_dict['path']
            user_list = batch_dict['user_id'] 

            for elem in paths_dict["path"]:
                paths_list.append(convert_fn(elem))
                user_list.append(elem.split(' ')[0])
            return batch_dict     
        def convert_path_and_add_uid(paths_dict, convert_fn):
            batch_dict = {"path":[], "user_id":[]}

            paths_list = batch_dict['path']
            user_list = batch_dict['user_id'] 

            for elem in paths_dict["path"]:
                paths_list.append(convert_fn(elem))
                user_list.append(elem.split(' ')[0]  )
            return batch_dict                     
        def convert_typed_path_and_add_uid(paths_dict, convert_fn):
            batch_dict = {"path":[], "user_id":[]}

            paths_list = batch_dict['path']
            user_list = batch_dict['user_id'] 

            for elem in paths_dict["path"]:
                paths_list.append(convert_fn(elem))
                user_list.append(elem.split(' ')[1][1:]  )
            return batch_dict                    
        if plain_text_path:
            convert_fn = dataset.identity_op#dataset.convert_numeric_path_to_textual_path
            #dataset.dataset = dataset.dataset.map(lambda x: {"path": [dataset.convert_numeric_path_to_textual_path(elem) for elem in x["path"] ] },
            #                batched=True, num_proc=args.nproc)
        else:
            convert_fn = dataset.keep_numeric_typed#keep_numeric
            #dataset.dataset = dataset.dataset.map(lambda x: {"path": [dataset.keep_numeric(elem) for elem in x["path"]]  },
            #                            batched=True, num_proc=args.nproc)     
        
        dataset.dataset = dataset.dataset.map(lambda x: convert_path_and_add_uid(x,convert_fn),#convert_and_add_uid(x, convert_fn),
                                        batched=True, num_proc=args.nproc)    
        dataset.dataset = dataset.dataset.class_encode_column('user_id')                                            
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
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=args.context_length,
                                                eos_token="[EOS]", bos_token="[BOS]",
                                                pad_token="[PAD]", unk_token="[UNK]",
                                                mask_token="[MASK]", use_fast=True)                

        #Check correctness of the encoding
        #print(dataset["path"][0], tokenizer.encode(dataset["path"][0]).tokens)

        # Load the specified tokenizer
        print("Train/Validation split...")
        dataset_split = dataset.train_test_split(test_size=0.05, stratify_by_column='user_id')
        # Convert DatasetDict to desired format
        dataset_split = DatasetDict({
            'train': dataset_split['train'].remove_columns('user_id'),
            'test': dataset_split['test'].remove_columns('user_id'),
        })

        print("Tokenizing dataset...")
        pre1 = f"data/{dataset_name}/{TOKENIZER_TYPE}/pre1_from_scratch_tokenized_dataset.hf"

        if not os.path.exists(pre1):
            tokenized_dataset = dataset_split.map(tokenize_function, 
                batched=True, 
                num_proc=args.nproc,
                remove_columns=["path"]
            )
            check_dir(pre1)
            tokenized_dataset.save_to_disk(pre1)
        else:
            tokenized_dataset = load_from_disk(pre1)

        # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
        check_dir(f"{data_dir}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")
        tokenized_dataset.save_to_disk(f"data/{dataset_name}/{TOKENIZER_TYPE}/from_scratch_tokenized_dataset.hf")


    # Train the model
    if args.load_model:
        # Training arguments
        custom_name = f'clm-from_scratch-{args.dataset}-{args.model}/checkpoint-{args.eval_ckpt_iter}'#f"clm-from_scratch-{args.dataset}-{args.model}"
        #custom_name = 'distilgpt2-checkpoint-10000'
        model = AutoModelForCausalLM.from_pretrained(custom_name)#f'models-weights/{dataset_name}/{model_name}/{custom_name}')
    else:
        model = train_from_scratch(model_name, tokenizer, tokenized_dataset, args.context_length, args)
    evaluate(model, args)




