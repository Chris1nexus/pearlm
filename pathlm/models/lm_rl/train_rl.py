import os
from typing import List, Any
import torch
import argparse
from datetime import datetime

from tqdm import tqdm
from textrl import train_agent_with_evaluation
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedTokenizerFast, PhrasalConstraint, LogitsProcessorList, \
    set_seed, pipeline, GPT2LMHeadModel


import numpy as np
from collections import defaultdict
import math
from transformers import LogitsProcessor
import random
import pfrl


from pathlm.models.lm.perlm import PERLM
from pathlm.models.lm.decoding_constraints import ConstrainedLogitsProcessorWordLevel, PrefixConstrainedLogitsProcessorWordLevel, PLMLogitsProcessorWordLevel
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.utils import SEED, get_pid_to_eid, get_eid_to_name_map, get_data_dir, get_set, check_dir
from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k 
from pathlm.sampling import KGsampler
from pathlm.sampling.samplers.constants import LiteralPath, TypeMapper

from pathlm.models.lm.lm_utils import get_user_negatives_tokens_ids, \
    _initialise_type_masks, \
    get_user_negatives, get_user_positives
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k
from pathlm.utils import get_pid_to_eid, get_set

from pathlm.models.rl.PGPR.pgpr_utils import get_knowledge_derived_relations, MODEL_DATASET_DIR,INTERACTION, DATASET_INFO_DIR,\
        PRODUCT, USER, ENTITY, RELATION

from pathlm.models.lm_rl.env import PathRLenv
from pathlm.models.lm_rl.model import TextRLActor
from pathlm.models.lm_rl.eval_rl import evaluate
from pathlm.models.lm_rl.decoding_constraints import RLConstrainedLogitsProcessorWordLevel
from pathlm.models.lm_rl.lm_rl_utils import tokenize_augmented_kg



import functools

import wandb


def make_env(idx, test, process_seeds, model,tokenizer,dataset_name,n_hop,kg):
    # Use different random seeds for train and test envs
    process_seed = int(process_seeds[idx])
    env_seed = 2**32 - 1 - process_seed if test else process_seed
    eval_epsilon = 0.001
    TOPK_CANDIDATES_SIZE=10
    env = PathRLenv(model, 
                    tokenizer, 
                    dataset_name, 
                    n_hop,
                    kg=kg,    
                    n_sequences_per_user=TOPK_CANDIDATES_SIZE,
                    max_sequence_len=None,
                   n_special_tokens=1)
    if test:
        # Randomize actions like epsilon-greedy in evaluation as well
        env = pfrl.wrappers.RandomizeAction(env, eval_epsilon)
    set_seed(env_seed)
    #env.seed(env_seed)
    #if args.monitor:
    #    env = pfrl.wrappers.Monitor(
    #        env, args.outdir, mode="evaluation" if test else "training"
    #    )
    #if args.render:
    #    env = pfrl.wrappers.Render(env)
    return env
def identity_f(tokens : List[str]) -> str:
    return ' '.join(tokens)


def main(args):


    #import multiprocessing as mp
    #mp.set_start_method('spawn')
    #mp.set_start_method('forkserver', force=True)

    



    TOKENIZER_TYPE = "WordLevel"
    model_name = args.model #'gpt2-large'#"#" # args.model
    dataset_name = args.dataset#'ml1m'#"lfm1m"   # args.dataset
    context_length = 200     # args.context_length
    #eval_ckpt_iter = 368000
    prompt_gen_n_hop=3
    tokenizer_dir = f'./tokenizers/{dataset_name}'
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")
    #custom_name = f'pathlm/clm-end-to-end-{dataset_name}-{model_name}-2000-{n_hop}-gcd/checkpoint-{eval_ckpt_iter}'#'clm-end-to-end-ml1m-distilgpt2-2000-3-gcd/checkpoint-368000' #f'clm-from_scratch-{dataset_name}-{model_name}-ckpt/checkpoint-{eval_ckpt_iter}'
    #custom_name = 'pathlm/clm-pretrain-ml1m-gpt2-large-2500-7/checkpoint-11000'
    #/home/xrh1/experiments/RECSYS2023/pathlm/pathlm/clm-pretrain-ml1m-gpt2-large-2500-7

    custom_name = f'weights/{args.dataset}/{args.model}/pretrain-{args.dataset}-{args.model}-{args.sample_size}-{args.n_hop}'
    ROOT_DIR = '.'
    # Dataset directories.
    data_dir = f'{ROOT_DIR}/data/{dataset_name}'
    dirpath = f'{data_dir}/preprocessed'
    tokenizer_dir = f'tokenizers/{dataset_name}'
    tokenizer_file = os.path.join(ROOT_DIR, tokenizer_dir, f"{TOKENIZER_TYPE}.json")


    data_dir_mapping = os.path.join(ROOT_DIR, f'data/{dataset_name}/preprocessed/mapping/')
    kg = KGsampler(args, dataset_name, dirpath, data_dir=data_dir_mapping)  

    from tokenizers import decoders
    #tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file , max_len=context_length,
    #                                    eos_token="[EOS]", bos_token="[BOS]",
    #                                    pad_token="[PAD]", unk_token="[UNK]",
    #                                    mask_token="[MASK]", use_fast=True)   
    tokenizer = PreTrainedTokenizerFast.from_pretrained(custom_name)  
    class WordLevelDecoder:
        def __init__(self):
            pass
            
        def decode(self, tokens: List[str]) -> str:
            return tokens
        
    #tokenizer.decoder = WordLevelDecoder()
    tokenizer.convert_tokens_to_string = identity_f  


    TOPK_CANDIDATES_SIZE = 10
    TOPK_SIZE = 10
    set_seed(SEED)
    model = PERLM.from_pretrained(os.path.join(custom_name) )
    model.cuda()


    env = PathRLenv(model, 
                    tokenizer, 
                    dataset_name, 
                    prompt_gen_n_hop,
                    kg=kg,    
                    n_sequences_per_user=TOPK_CANDIDATES_SIZE,
                    max_sequence_len=None,
                   n_special_tokens=1,
                   pred_positive_item_reward=args.pos_item_rw,
                   pred_negative_item_reward=args.neg_item_rw,
                   wrong_path_reward=args.wrong_path_rw
                    )



    '''
    def make_batch_env(test):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test, process_seeds, model,tokenizer,dataset_name,n_hop,kg)
                for idx, env in enumerate(range(num_envs))
            ]
        )
        vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        return vec_env
    eval_epsilon=0.001
    num_envs = 2
    '''
    pfrl.utils.set_random_seed(SEED)



    test_set = get_set(dataset_name, set_str=PathRLenv.TEST_SET)
    uids = list(test_set.keys())
    last_item_idx = max([int(id) for id in get_pid_to_eid(data_dir).values()])
    user_negatives_token_ids = get_user_negatives_tokens_ids(dataset_name, tokenizer)
    user_negatives = get_user_negatives(dataset_name)
    id_to_uid_token_map = {tokenizer.convert_tokens_to_ids(f'U{uid}'): f'{uid}' for uid in uids}
    ent_mask, rel_mask, token_id_to_token = _initialise_type_masks(tokenizer)    





    logit_proc_kwargs = {}

    print('Using: ', RLConstrainedLogitsProcessorWordLevel)

    N_SEQUENCES_PER_USER=env.N_SEQUENCES_PER_USER
    SEQUENCE_LEN = env.SEQUENCE_LEN
    tokenized_kg, kg_to_vocab_mapping = tokenize_augmented_kg(kg, tokenizer, use_token_ids=True) 
    logits_processor = RLConstrainedLogitsProcessorWordLevel(tokenized_kg=tokenized_kg,
                            force_token_map=user_negatives_token_ids,
                            tokenizer=tokenizer,
                            total_length=SEQUENCE_LEN,
                            num_return_sequences=N_SEQUENCES_PER_USER,
                            id_to_uid_token_map=id_to_uid_token_map,
                            eos_token_ids=[
                            tokenizer.convert_tokens_to_ids(tokenizer.eos_token)],
                            vocab_size=max(model.config.vocab_size, len(tokenizer.get_vocab()) ),
                            **logit_proc_kwargs
                            )


    observaton_list = env.inference_paths
    actor = TextRLActor(env, model, tokenizer, logits_processor,
                        top_k=0, 
                        top_p=args.top_p, 
                        optimizer='adamW',
                        act_deterministically=args.act_deterministic,
                        
                        temperature=1)
    agent = actor.agent_ppo(update_interval=args.update_interval, minibatch_size=args.minibatch_size, epochs=args.n_epochs, lr=args.lr, ent_coef=args.ent_coef)
    n_episodes = args.n_ep
    max_episode_len = env.SEQUENCE_LEN  # max sentence length

    for i in range(1, n_episodes + 1):
        input_item=random.choice(env.inference_paths)
        obs, input_ids = env.reset(input_item)
        R = 0
        t = 0
        while True:
            action, _ = agent.act(obs, input_ids)
            
            (obs, input_ids), reward, done, pred = env.step(action)
            R += sum(reward)/len(reward)
            t += 1
            reset = t == max_episode_len
            agent.observe(obs, reward, done, reset)
            #print(agent.training)
            if done or reset:
                break
        if i % 5 == 0:
            print('episode:', i, 'R:', R)
        if i % 50  == 0:
            
            
            #evaluation = evaluate(actor, env.inference_paths, test_set, user_negatives, topk_size=10)
                                
            metrics = evaluate(actor, model, env, tokenizer, test_set, user_negatives, topk_size=10, BATCH_SIZE = 16, DEVICE='cuda')
            rl_model_path = os.path.dirname(custom_name) + f'-rl/ckpt-{i}'
            os.makedirs(rl_model_path, exist_ok=True)
            agent.save(rl_model_path)              
            print(metrics)
            if args.wandb:
                wandb.log(metrics)            
        if i % 50 == 0:
            print('statistics:', agent.get_statistics())
            if args.wandb:
                stat_dict = {}
                for k,v in agent.get_statistics():
                    stat_dict[k] = v

                wandb.log(stat_dict)
    
    ckpt_dir = None


    rl_model_path = custom_name + f'-rl/ckpt-{i}'
    log_path = os.path.join(args.output_dir, rl_model_path)
    os.makedirs(log_path, exist_ok=True)
    agent.save(log_path)              

    #'''








def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--dataset", type=str, default="ml1m", help="{ml1m, lfm1m}")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use from HuggingFace pretrained models")
    parser.add_argument("--nproc", type=int, default=2, help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=24, help="Train batch size")
    parser.add_argument("--test_batch_size", type=int, default=24, help="Test batch size")
    parser.add_argument("--context_length", type=int, default=200,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")  
    parser.add_argument("--sample_size", type=str, default="1000",
                        help="Which sample size dataset to use for pretraining")

    parser.add_argument("--load_data", type=bool, default=False, help="")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--eval_device", type=str, default='cuda:0', help="")
    parser.add_argument("--eval_ckpt_iter", type=int, default='10000', help="")
    parser.add_argument("--infer_batch_size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--n_seq_infer", type=int, default=10, help="Number of sequences generated for each user at inference time")


    parser.add_argument("--embedding_root_dir", type=str, default="./embedding-weights", help="default: ./embedding-weights")
    parser.add_argument("--emb_filename", type=str, default='transe_embed.pkl', help="default: 'transe_embed.pkl'")
    parser.add_argument("--emb_size", type=int, default=100, help="Transformer Embedding size (must match external embedding size, if chosen)")

    # Hyperparameters
    parser.add_argument("--pos_item_rw", type=float, default=1., help="Positive item reward")
    parser.add_argument("--neg_item_rw", type=float, default=0., help="Negative item reward")
    parser.add_argument("--wrong_path_rw", type=float, default=0., help="Wrong path reward")
    parser.add_argument("--n_ep", type=int, default=10000,
                        help="Number of episodes")
    parser.add_argument("--update_interval", type=int, default=100, help="Update interval RL agent")
    parser.add_argument("--minibatch_size", type=int, default=400, help="Batch size RL agent")
    parser.add_argument("--n_epochs", type=int, default=20, help="Epochs RL agent")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--ent_coef", type=float, default=0,
                        help="Entropy coefficient for PPO")    
    parser.add_argument("--top_p", type=float, default=0,
                        help="Top p sampling (default is ignore this technique)")
    parser.add_argument('--act_deterministic', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')

    args = parser.parse_args()
    return args





if __name__ == '__main__':
    args = parse_args()
    project_name = f'rl_llm@{args.dataset}'
    run_name=f"{args.exp_name}@{args.model}@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join(project_name, run_name)
    os.makedirs(log_dir, exist_ok=True)
    args.output_dir = log_dir 

    
    if args.wandb:

        run_name=f"{args.exp_name}@{args.model}@{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            name=run_name,
            # track hyperparameters and run metadata
            config=vars(args)
        )
    print(args)
    main(args)
