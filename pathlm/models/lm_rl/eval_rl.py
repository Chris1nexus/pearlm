from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from pathlm.models.lm.metrics import ndcg_at_k, mmr_at_k


'''
def evaluate(actor, model, env, tokenizer, test_set, user_negatives, topk_size=10, BATCH_SIZE = 16, DEVICE='cuda'):
    # Generate paths for the test users
    env = actor.env
    agent = actor.agent
    #topks, topk_sequences = self.__generate_topks_withWordLevel(model)
    #custom_model_name = self.custom_model_name.split('/')[:-1]
    #check_dir(f"./results/{self.dataset_name}/{custom_model_name}")
    topks = {uid:[] for uid in user_negatives}
    topk_sequences = {uid:[] for uid in user_negatives}
    with tqdm(initial=0, desc="Generating topks", colour="green", total=len(user_negatives), leave=True) as pbar:
        N_SEQUENCES_PER_PROMPT = env.n_sequences_per_user
        for i in range(0, len(env.inference_paths), BATCH_SIZE):

            batch_prompts = []
            for prompt in env.inference_paths[i:(i+BATCH_SIZE)]:
                batch_prompts.extend([prompt['input']]*N_SEQUENCES_PER_PROMPT)

            with torch.inference_mode(), agent.eval_mode():
                while True:
                    feature_dict = tokenizer(batch_prompts,
                                              return_tensors='pt',
                                              add_special_tokens=False).to(DEVICE)
                    prediction = model(**feature_dict, output_hidden_states=True)

                    outputs = prediction.hidden_states[-0].squeeze(0) 
                    outputs = outputs[...,-1,:]
                    #print(outputs.shape)
                    obs = outputs#.unsqueeze(0)
                    input_ids = feature_dict['input_ids']#.unsqueeze(0)
                    #print(model.device)
                    #print(obs.device)
                    #print(agent.device)
                    #print(input_ids.shape)
                    #print(obs.shape)            
                    CUR_SEQ_LEN = input_ids.shape[-1]
                    action = agent.act(obs, input_ids)
                    #print(action.shape)
                    action_tensor = torch.LongTensor(action).unsqueeze(-1).to(DEVICE)#.cuda()
                    input_ids = torch.cat([input_ids, action_tensor], dim=-1 )
                    CUR_SEQ_LEN = input_ids.shape[-1]
                    batch_prompts = tokenizer.batch_decode(input_ids.reshape(-1,CUR_SEQ_LEN ))
                    if CUR_SEQ_LEN >= env.SEQUENCE_LEN+1:
                        break            
            
            for seq in batch_prompts:
                #print(uid, seq)
                tokens = seq.split(' ')
                
                uid = tokens[1][1:]

                recommended_token = tokens[-2]
                recommended_item = recommended_token[1:]
                
                if not recommended_token.startswith("P"):
                    continue
                if recommended_item not in user_negatives[uid]:
                    continue
                if recommended_item in topks[uid]:
                    continue            

                topks[uid].append(recommended_item)
                topk_sequences[uid].append(seq)
            pbar.update(min(BATCH_SIZE, len(user_negatives)-i ))
    #pickle.dump(topks, open(f"./results/{self.dataset_name}/{custom_model_name}/topk_items.pkl", "wb"))
    #pickle.dump(topk_sequences, open(f"./results/{self.dataset_name}/{custom_model_name}/pred_paths.pkl", "wb"))   
    evaluation = defaultdict(list)
    for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
        hits = []
        
        for recommended_item in topk:
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
        while len(hits) < topk_size:
            hits.append(0)
        #print(uid, topk, hits)
        ndcg = ndcg_at_k(hits, len(hits))
        mmr = mmr_at_k(hits, len(hits))
        evaluation[f"ndcg@{topk_size}"].append(ndcg)
        evaluation[f"mmr@{topk_size}"].append(mmr)

    print(
        f"no of users: {len(test_set.keys())}, ndcg: {np.mean(evaluation[f'ndcg@{topk_size}'])}, mmr: {np.mean(evaluation[f'mmr@{topk_size}'])}")
    metrics_ = dict()
    for k in evaluation:
        metrics_[f'eval_{k}'] = np.mean(evaluation[k])
    return metrics_
'''
'''
def evaluate(model, dataset, test_set, user_negatives, topk_size=10):
    # Generate paths for the test users

    #topks, topk_sequences = self.__generate_topks_withWordLevel(model)
    #custom_model_name = self.custom_model_name.split('/')[:-1]
    #check_dir(f"./results/{self.dataset_name}/{custom_model_name}")
    topks = {uid:[] for uid in user_negatives}
    topk_sequences = {uid:[] for uid in user_negatives}
    with tqdm(initial=0, desc="Generating topks", colour="green", total=len(user_negatives), leave=True) as pbar:
        for input_prompt in tqdm(dataset):
            input_seq = input_prompt['input']
            input_tokens = input_seq.split(' ')
            uid = input_tokens[1][1:]
            
            sequences = model.predict(input_prompt)
            for seq in sequences:
                #print(uid, seq)
                tokens = seq.split(' ')


                recommended_token = tokens[-2]
                recommended_item = recommended_token[1:]
                
                if not recommended_token.startswith("P"):
                    continue
                if recommended_item not in user_negatives[uid]:
                    continue
                if recommended_item in topks[uid]:
                    continue            

                topks[uid].append(recommended_item)
                topk_sequences[uid].append(seq)
            pbar.update(1)
    #pickle.dump(topks, open(f"./results/{self.dataset_name}/{custom_model_name}/topk_items.pkl", "wb"))
    #pickle.dump(topk_sequences, open(f"./results/{self.dataset_name}/{custom_model_name}/pred_paths.pkl", "wb"))   
    evaluation = defaultdict(list)
    for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
        hits = []
        
        for recommended_item in topk:
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
        while len(hits) < topk_size:
            hits.append(0)
        #print(uid, topk, hits)
        ndcg = ndcg_at_k(hits, len(hits))
        mmr = mmr_at_k(hits, len(hits))
        evaluation[f"ndcg@{topk_size}"].append(ndcg)
        evaluation[f"mmr@{topk_size}"].append(mmr)

    print(
        f"no of users: {len(test_set.keys())}, ndcg: {np.mean(evaluation[f'ndcg@{topk_size}'])}, mmr: {np.mean(evaluation[f'mmr@{topk_size}'])}")
    metrics_ = dict()
    for k in evaluation:
        metrics_[f'eval_{k}'] = np.mean(evaluation[k])
    return metrics_
'''



def evaluate(actor, model, env, tokenizer, test_set, user_negatives, topk_size=10, BATCH_SIZE = 16, DEVICE='cuda'):
    # Generate paths for the test users
    env = actor.env
    agent = actor.agent
    #topks, topk_sequences = self.__generate_topks_withWordLevel(model)
    #custom_model_name = self.custom_model_name.split('/')[:-1]
    #check_dir(f"./results/{self.dataset_name}/{custom_model_name}")
    topks = {uid:[] for uid in user_negatives}
    topk_sequences = {uid:[] for uid in user_negatives}
    with tqdm(initial=0, desc="Generating topks", colour="green", total=len(user_negatives), leave=True) as pbar,\
        torch.inference_mode(),\
        agent.eval_mode():
        N_SEQUENCES_PER_PROMPT = env.n_sequences_per_user #30 #env.n_sequences_per_user
        
        for i in range(0, len(env.inference_paths), BATCH_SIZE):

            batch_prompts = []
            for prompt in env.inference_paths[i:(i+BATCH_SIZE)]:
                batch_prompts.extend([prompt['input']]*N_SEQUENCES_PER_PROMPT)
            
            CUR_BATCH_SIZE = min(BATCH_SIZE, len(env.inference_paths)-i )
            prev_scores = None
            topk_prompts = None
            while True:
                feature_dict = tokenizer(batch_prompts,
                                          return_tensors='pt',
                                          add_special_tokens=False).to(DEVICE)
                prediction = model(**feature_dict, output_hidden_states=True)
                #print(prediction)
                #for elem in prediction:
                #        print(elem)
                #print(len(prediction.hidden_states))
                #for elem in prediction.hidden_states:
                #    print(elem.shape)
                # get hidden states corresponding to last layer (env.unfreeze_layer_from_past is automatically set to 1 by the env superclass)
                outputs = prediction.hidden_states[-env.unfreeze_layer_from_past].squeeze(0)
                #print(prediction.hidden_states[-env.unfreeze_layer_from_past].shape)
                #print(prediction.hidden_states[-env.unfreeze_layer_from_past].squeeze(0).shape)
                #print(prediction.hidden_states)
                #print(outputs[:5,-1,:])
                #print(outputs.data[:5,-1,:])
                #print()
                outputs = outputs.data[...,-1,:]
                #print(outputs.shape)
                obs = outputs#.unsqueeze(0)
                input_ids = feature_dict['input_ids']#.unsqueeze(0)
                #print(model.device)
                #print(obs.device)
                #print(agent.device)
                #print(input_ids.shape)
                #print(obs.shape)            
                CUR_SEQ_LEN = input_ids.shape[-1]
                action, scores = agent.act(obs, input_ids)
                #print(scores.shape)
                #print(scores.shape, scores[action[...,np.newaxis] ])
                
                '''
                action_index = torch.LongTensor(action).view(-1,1).to(scores.device)
                token_scores = torch.gather(scores, -1, action_index)
                if prev_scores is None:
                    prev_scores = token_scores
                else:
                    prev_scores = prev_scores * token_scores
                    
                '''
                
                #print(action.shape)
                action_tensor = torch.LongTensor(action).unsqueeze(-1).to(DEVICE)#.cuda()
                input_ids = torch.cat([input_ids, action_tensor], dim=-1 )
                CUR_SEQ_LEN = input_ids.shape[-1]
                #print('AAAAAAAAAAAAAA')
                #print(input_ids.shape)
                #print(input_ids.dtype)
                batch_prompts = tokenizer.batch_decode(input_ids.reshape(-1,CUR_SEQ_LEN ))
                if CUR_SEQ_LEN >= env.SEQUENCE_LEN+1:
                    topk_prompts=batch_prompts
                    '''
                    #topk_prompts
                    #print(prev_scores.shape)
                    prev_scores = prev_scores.reshape(CUR_BATCH_SIZE,N_SEQUENCES_PER_PROMPT)
                    _, indices = torch.topk(prev_scores, topk_size, dim=-1)
                    offset = torch.LongTensor([[i* N_SEQUENCES_PER_PROMPT] for i in range(CUR_BATCH_SIZE)] ).to(scores.device)
                    indices = (indices + offset).reshape(-1)
                    #print(indices.shape)
                    #print(indices)
                    #indices = indices.reshape(-1,1)
                    
                    #input_ids = input_ids.reshape(CUR_BATCH_SIZE,N_SEQUENCES_PER_PROMPT,CUR_SEQ_LEN )
                    #print(indices.shape)
                    #print(input_ids.shape)
                    #print(indices.unsqueeze(-1).shape)
                    best_input_ids = input_ids[indices,...]#torch.gather(input_ids,1,indices.unsqueeze(-1))
                    #print(best_input_ids.shape)
                    #print(best_input_ids)
                    #print(best_input_ids.shape)
                    #print(best_input_ids.dtype)   
                    #print(best_input_ids.tolist())
                    topk_prompts = tokenizer.batch_decode(best_input_ids)#.reshape(-1, CUR_SEQ_LEN))
                    '''
                    break            
            
            for seq in topk_prompts:#batch_prompts:
                
                tokens = seq.split(' ')
                
                uid = tokens[1][1:]
                #print(uid, seq)
                recommended_token = tokens[-2]
                recommended_item = recommended_token[1:]
                
                if not recommended_token.startswith("P"):
                    continue
                if recommended_item not in user_negatives[uid]:
                    continue
                if recommended_item in topks[uid]:
                    continue            

                topks[uid].append(recommended_item)
                topk_sequences[uid].append(seq)
            pbar.update(min(BATCH_SIZE, len(user_negatives)-i ))
    #pickle.dump(topks, open(f"./results/{self.dataset_name}/{custom_model_name}/topk_items.pkl", "wb"))
    #pickle.dump(topk_sequences, open(f"./results/{self.dataset_name}/{custom_model_name}/pred_paths.pkl", "wb"))   
    metrics = defaultdict(list)
    for uid, topk in tqdm(topks.items(), desc="Evaluating", colour="green"):
        hits = []
        #print(uid, len(topk))
        for recommended_item in topk:
            if recommended_item in test_set[uid]:
                hits.append(1)
            else:
                hits.append(0)
        while len(hits) < topk_size:
            hits.append(0)
        #print(uid, topk, hits)
        ndcg = ndcg_at_k(hits, len(hits))
        mmr = mmr_at_k(hits, len(hits))
        metrics[f"ndcg@{topk_size}"].append(ndcg)
        metrics[f"mmr@{topk_size}"].append(mmr)

    print(
        f"no of users: {len(test_set.keys())}, ndcg: {np.mean(metrics[f'ndcg@{topk_size}'])}, mmr: {np.mean(metrics[f'mmr@{topk_size}'])}")
    metrics_ = dict()
    for k in metrics:
        metrics_[f'eval_{k}'] = np.mean(metrics[k])
    return metrics_
