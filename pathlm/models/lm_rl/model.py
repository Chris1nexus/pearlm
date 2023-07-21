

import itertools

import numpy as np
import pfrl
import torch
import torch.nn.functional as F
from typing import List, Any

from pfrl.agents.ppo import _elementwise_clip
from pfrl.utils.mode_of_distribution import mode_of_distribution
from torch import autocast
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)

from textrl.actor import top_k_top_p_filtering



def get_modulelist_pos(model):
    module_list_pos = 0
    for ids, i in enumerate(list(model.children())):
        if isinstance(i, torch.nn.ModuleList):
            module_list_pos = ids
    return module_list_pos


class HFModelListModule(torch.nn.Module):
    def __init__(self, module_list):
        super(HFModelListModule, self).__init__()
        self.module_list = module_list

    def forward(self, hidden):
        for module in self.module_list:
            hidden = module(hidden)[0]
        return hidden


class SoftmaxCategoricalHead(torch.nn.Module):
    def __init__(self, env, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        super().__init__()
        self.env = env
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def forward(self, logits):
        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        # repetition penalty from https://github.com/huggingface/transformers/pull/2303/files#diff-6b72b98c4c2dcfc6cc606843917733f5d858374fbc22a735ff483bbc0c1e63ea
        if self.repetition_penalty != 1.0:
            for seq_num, predicted in enumerate(self.env.predicted):
                for previous_tokens in set(predicted):
                    prev_token_id = self.env.tokenizer.convert_tokens_to_ids(previous_tokens)
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if torch.all(logits[:, seq_num, prev_token_id] < 0):
                        logits[:, seq_num, prev_token_id] *= self.repetition_penalty
                    else:
                        logits[:, seq_num, prev_token_id] /= self.repetition_penalty
        logits = logits / self.temperature
        logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
        return torch.distributions.Categorical(logits=logits)
    
    
class ConstrainedTextPPO(pfrl.agents.PPO):
    
    def __init__(self, logits_processor, model, opt, **kwargs):
        super().__init__(model, opt, **kwargs)
        self.logits_processor = logits_processor
    
    def _update_if_dataset_is_ready(self):
        dataset_size = (
                sum(len(episode) for episode in self.memory)
                + len(self.last_episode)
                + (
                    0
                    if self.batch_last_episode is None
                    else sum(len(episode) for episode in self.batch_last_episode)
                )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = pfrl.agents.ppo._make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = pfrl.agents.ppo._make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = self._compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []

    def _compute_explained_variance(self, transitions):
        """Compute 1 - Var[return - v]/Var[return].

        This function computes the fraction of variance that value predictions can
        explain about returns.
        """
        t = np.array([tr["v_teacher"] for tr in transitions])
        y = np.array([tr["v_pred"] for tr in transitions])
        vart = np.var(t)
        if vart == 0:
            return np.nan
        else:
            return float(1 - np.var(np.average(t) - y) / vart)

    def act(self, obs: Any, input_ids: Any) -> Any:
        return self.batch_act([obs], [input_ids])#[0]        
        
    def batch_act(self, batch_obs, batch_input_ids):
        
        if self.training:
            return self._batch_act_train(batch_obs, batch_input_ids)
        else:
            return self._batch_act_eval(batch_obs, batch_input_ids)

    def _batch_act_train(self, batch_obs, batch_input_ids):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                print('Recurrent')
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_distrib, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
            
                action_distrib, batch_value = self.model(b_state)
            logits_list = []
            for input_ids, logits in zip(batch_input_ids, action_distrib.logits):
                filtered_logits = self.logits_processor(input_ids, logits, is_inference=False)
                logits_list.append(filtered_logits)
            filtered_logits = torch.stack(logits_list)
            #print( ((filtered_logits > -torch.inf) & (filtered_logits < torch.inf)).sum()  )
            action_distrib = torch.distributions.categorical.Categorical(logits=filtered_logits)
            
            #print('XXXX: ', action_distrib)
            #print('YYYY: ', batch_value)
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action[0], action_distrib.probs[0]      
        
    @autocast('cuda')
    def _batch_act_eval(self, batch_obs, batch_input_ids):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            # TODO
            
            action_distrib, _ = self.model(b_state)

            logits_list = []
            for input_ids, logits in zip(batch_input_ids, action_distrib.logits):
                
                #print(input_ids.shape, logits.shape)
                filtered_logits = self.logits_processor(input_ids, logits, is_inference=True)
                logits_list.append(filtered_logits)
            filtered_logits = torch.stack(logits_list)
            action_distrib = torch.distributions.categorical.Categorical(logits=filtered_logits)

            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action[0], action_distrib.probs[0]

    def _lossfun(
            self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):
        prob_ratio = torch.exp(log_probs - log_probs_old)
        loss_policy = -torch.mean(
            torch.min(
                (prob_ratio * advs),
                (torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs),
            ),
        )
        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred.squeeze(), vs_teacher.squeeze())
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred.squeeze(), vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred.squeeze(), vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))
        loss = (
                loss_policy
                + self.value_func_coef * loss_value_func
                + self.entropy_coef * loss_entropy
        )
        return loss    

class TextRLActor:
    def __init__(self, env, model, tokenizer, logits_processor, optimizer='sgd', gpu_id=0,
                 unfreeze_layer_from_past=0,
                 act_deterministically=True,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 repetition_penalty=1.0):
        self.agent = None
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = model
        
        self.logits_processor = logits_processor
        
        if hasattr(model.config, 'word_embed_proj_dim'):
            self.obs_size = model.config.word_embed_proj_dim
        else:
            self.obs_size = model.config.hidden_size
        self.converter = self.model.lm_head
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer = optimizer
        self.repetition_penalty = repetition_penalty
        self.unfreeze_layer_from_past = unfreeze_layer_from_past

        parents = [parent[0] for parent in model.named_children()]
        if 'transformer' in parents:  # gpt2/bloom:
            transformers_model = model.transformer
        elif 'model' in parents:  # bart
            transformers_model = model.model
        elif 'decoder' in parents:  # t5
            transformers_model = model.decoder
        else:
            raise ValueError('model not supported')

        if unfreeze_layer_from_past > 0:
            self.middle_model = HFModelListModule(list(transformers_model.children())
                                                  [get_modulelist_pos(transformers_model)]
                                                  [-self.unfreeze_layer_from_past:])
            self.remaining_model = torch.nn.Sequential(
                *list(transformers_model.children())[get_modulelist_pos(transformers_model) + 1:])
        else:
            self.middle_model = torch.nn.Sequential()
            self.remaining_model = torch.nn.Sequential()

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6, ent_coef=0):
        policy = torch.nn.Sequential(
            self.middle_model,
            self.remaining_model,
            self.converter,
            SoftmaxCategoricalHead(self.env,
                                   temperature=self.temperature,
                                   top_k=self.top_k,
                                   top_p=self.top_p,
                                   repetition_penalty=self.repetition_penalty)
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.obs_size // 2),
            torch.nn.Linear(self.obs_size // 2, self.obs_size // 4),
            torch.nn.Linear(self.obs_size // 4, 1)
        )
        model = pfrl.nn.Branched(policy, vf)
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == 'adamw':
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            opt = self.optimizer
        model = model.cuda()
        agent = ConstrainedTextPPO(
            self.logits_processor,
            model,
            opt,
            gpu=self.gpu_id,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=ent_coef,
            gamma=0.95,  # https://arxiv.org/abs/2210.01241
            lambd=1,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically
        )
        self.agent = agent
        return agent

    @autocast('cuda')
    def predict(self, input_item):
        t = 0
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs, input_ids = self.env.reset(input_item)
                while True:
                    action, _ = self.agent.act(obs, input_ids)
                    (obs, input_ids), reward, done, pred = self.env.step(action)
                    t += 1
                    reset = t >= self.env.env_max_length
                    self.agent.observe(obs, reward, done, reset)
                    if done or reset:
                        return pred.get('predicted_str')   

'''
import itertools

import numpy as np
import pfrl
import torch
import torch.nn.functional as F
from typing import List, Any

from pfrl.agents.ppo import _elementwise_clip
from pfrl.utils.mode_of_distribution import mode_of_distribution
from torch import autocast
from pfrl.utils.mode_of_distribution import mode_of_distribution
from pfrl.utils.recurrent import (
    concatenate_recurrent_states,
    flatten_sequences_time_first,
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
)

from textrl.actor import top_k_top_p_filtering



def get_modulelist_pos(model):
    module_list_pos = 0
    for ids, i in enumerate(list(model.children())):
        if isinstance(i, torch.nn.ModuleList):
            module_list_pos = ids
    return module_list_pos


class HFModelListModule(torch.nn.Module):
    def __init__(self, module_list):
        super(HFModelListModule, self).__init__()
        self.module_list = module_list

    def forward(self, hidden):
        for module in self.module_list:
            hidden = module(hidden)[0]
        return hidden


class SoftmaxCategoricalHead(torch.nn.Module):
    def __init__(self, env, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        super().__init__()
        self.env = env
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def forward(self, logits):
        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        # repetition penalty from https://github.com/huggingface/transformers/pull/2303/files#diff-6b72b98c4c2dcfc6cc606843917733f5d858374fbc22a735ff483bbc0c1e63ea
        if self.repetition_penalty != 1.0:
            for seq_num, predicted in enumerate(self.env.predicted):
                for previous_tokens in set(predicted):
                    prev_token_id = self.env.tokenizer.convert_tokens_to_ids(previous_tokens)
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if torch.all(logits[:, seq_num, prev_token_id] < 0):
                        logits[:, seq_num, prev_token_id] *= self.repetition_penalty
                    else:
                        logits[:, seq_num, prev_token_id] /= self.repetition_penalty
        logits = logits / self.temperature
        logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)
        return torch.distributions.Categorical(logits=logits)
    
    
class ConstrainedTextPPO(pfrl.agents.PPO):
    
    def __init__(self, logits_processor, model, opt, **kwargs):
        super().__init__(model, opt, **kwargs)
        self.logits_processor = logits_processor
    
    def _update_if_dataset_is_ready(self):
        dataset_size = (
                sum(len(episode) for episode in self.memory)
                + len(self.last_episode)
                + (
                    0
                    if self.batch_last_episode is None
                    else sum(len(episode) for episode in self.batch_last_episode)
                )
        )
        if dataset_size >= self.update_interval:
            self._flush_last_episode()
            if self.recurrent:
                dataset = pfrl.agents.ppo._make_dataset_recurrent(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    max_recurrent_sequence_len=self.max_recurrent_sequence_len,
                    device=self.device,
                )
                self._update_recurrent(dataset)
            else:
                dataset = pfrl.agents.ppo._make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                    device=self.device,
                )
                assert len(dataset) == dataset_size
                self._update(dataset)
            self.explained_variance = self._compute_explained_variance(
                list(itertools.chain.from_iterable(self.memory))
            )
            self.memory = []

    def _compute_explained_variance(self, transitions):
        """Compute 1 - Var[return - v]/Var[return].

        This function computes the fraction of variance that value predictions can
        explain about returns.
        """
        t = np.array([tr["v_teacher"] for tr in transitions])
        y = np.array([tr["v_pred"] for tr in transitions])
        vart = np.var(t)
        if vart == 0:
            return np.nan
        else:
            return float(1 - np.var(np.average(t) - y) / vart)

    def act(self, obs: Any, input_ids: Any) -> Any:
        return self.batch_act([obs], [input_ids])[0]        
        
    def batch_act(self, batch_obs, batch_input_ids):
        
        if self.training:
            return self._batch_act_train(batch_obs, batch_input_ids)
        else:
            return self._batch_act_eval(batch_obs, batch_input_ids)

    def _batch_act_train(self, batch_obs, batch_input_ids):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        num_envs = len(batch_obs)
        if self.batch_last_episode is None:
            self._initialize_batch_variables(num_envs)
        assert len(self.batch_last_episode) == num_envs
        assert len(self.batch_last_state) == num_envs
        assert len(self.batch_last_action) == num_envs

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            if self.recurrent:
                print('Recurrent')
                assert self.train_prev_recurrent_states is None
                self.train_prev_recurrent_states = self.train_recurrent_states
                (
                    (action_distrib, batch_value),
                    self.train_recurrent_states,
                ) = one_step_forward(
                    self.model, b_state, self.train_prev_recurrent_states
                )
            else:
            
                action_distrib, batch_value = self.model(b_state)
            logits_list = []
            for input_ids, logits in zip(batch_input_ids, action_distrib.logits):
                filtered_logits = self.logits_processor(input_ids, logits, is_inference=False)
                logits_list.append(filtered_logits)
            filtered_logits = torch.stack(logits_list)
            #print( ((filtered_logits > -torch.inf) & (filtered_logits < torch.inf)).sum()  )
            action_distrib = torch.distributions.categorical.Categorical(logits=filtered_logits)
            
            #print('XXXX: ', action_distrib)
            #print('YYYY: ', batch_value)
            batch_action = action_distrib.sample().cpu().numpy()
            self.entropy_record.extend(action_distrib.entropy().cpu().numpy())
            self.value_record.extend(batch_value.cpu().numpy())

        self.batch_last_state = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action        
        
    @autocast('cuda')
    def _batch_act_eval(self, batch_obs, batch_input_ids):
        assert not self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), pfrl.utils.evaluating(self.model):
            # TODO
            
            action_distrib, _ = self.model(b_state)

            logits_list = []
            for input_ids, logits in zip(batch_input_ids, action_distrib.logits):
                
                #print(input_ids.shape, logits.shape)
                filtered_logits = self.logits_processor(input_ids, logits, is_inference=True)
                logits_list.append(filtered_logits)
            filtered_logits = torch.stack(logits_list)
            action_distrib = torch.distributions.categorical.Categorical(logits=filtered_logits)

            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

    def _lossfun(
            self, entropy, vs_pred, log_probs, vs_pred_old, log_probs_old, advs, vs_teacher
    ):
        prob_ratio = torch.exp(log_probs - log_probs_old)
        loss_policy = -torch.mean(
            torch.min(
                (prob_ratio * advs),
                (torch.clamp(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs),
            ),
        )
        if self.clip_eps_vf is None:
            loss_value_func = F.mse_loss(vs_pred.squeeze(), vs_teacher.squeeze())
        else:
            clipped_vs_pred = _elementwise_clip(
                vs_pred,
                vs_pred_old - self.clip_eps_vf,
                vs_pred_old + self.clip_eps_vf,
            )
            loss_value_func = torch.mean(
                torch.max(
                    F.mse_loss(vs_pred.squeeze(), vs_teacher, reduction="none"),
                    F.mse_loss(clipped_vs_pred.squeeze(), vs_teacher, reduction="none"),
                )
            )
        loss_entropy = -torch.mean(entropy)

        self.value_loss_record.append(float(loss_value_func))
        self.policy_loss_record.append(float(loss_policy))
        loss = (
                loss_policy
                + self.value_func_coef * loss_value_func
                + self.entropy_coef * loss_entropy
        )
        return loss    

class TextRLActor:
    def __init__(self, env, model, tokenizer, logits_processor, optimizer='sgd', gpu_id=0,
                 unfreeze_layer_from_past=0,
                 act_deterministically=True,
                 temperature=1.0,
                 top_k=0,
                 top_p=1.0,
                 repetition_penalty=1.0):
        self.agent = None
        self.n_actions = max(model.config.vocab_size, tokenizer.vocab_size)
        self.env = env
        self.gpu_id = gpu_id
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.model = model
        
        self.logits_processor = logits_processor
        
        if hasattr(model.config, 'word_embed_proj_dim'):
            self.obs_size = model.config.word_embed_proj_dim
        else:
            self.obs_size = model.config.hidden_size
        self.converter = self.model.lm_head
        self.act_deterministically = act_deterministically
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer = optimizer
        self.repetition_penalty = repetition_penalty
        self.unfreeze_layer_from_past = unfreeze_layer_from_past

        parents = [parent[0] for parent in model.named_children()]
        if 'transformer' in parents:  # gpt2/bloom:
            transformers_model = model.transformer
        elif 'model' in parents:  # bart
            transformers_model = model.model
        elif 'decoder' in parents:  # t5
            transformers_model = model.decoder
        else:
            raise ValueError('model not supported')

        if unfreeze_layer_from_past > 0:
            self.middle_model = HFModelListModule(list(transformers_model.children())
                                                  [get_modulelist_pos(transformers_model)]
                                                  [-self.unfreeze_layer_from_past:])
            self.remaining_model = torch.nn.Sequential(
                *list(transformers_model.children())[get_modulelist_pos(transformers_model) + 1:])
        else:
            self.middle_model = torch.nn.Sequential()
            self.remaining_model = torch.nn.Sequential()

    def agent_ppo(self, update_interval=10, minibatch_size=3000, epochs=20, lr=3e-6, ent_coef=0):
        policy = torch.nn.Sequential(
            self.middle_model,
            self.remaining_model,
            self.converter,
            SoftmaxCategoricalHead(self.env,
                                   temperature=self.temperature,
                                   top_k=self.top_k,
                                   top_p=self.top_p,
                                   repetition_penalty=self.repetition_penalty)
        )
        vf = torch.nn.Sequential(
            torch.nn.Linear(self.obs_size, self.obs_size // 2),
            torch.nn.Linear(self.obs_size // 2, self.obs_size // 4),
            torch.nn.Linear(self.obs_size // 4, 1)
        )
        model = pfrl.nn.Branched(policy, vf)
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == 'adamw':
                opt = torch.optim.AdamW(model.parameters(), lr=lr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            opt = self.optimizer
        model = model.cuda()
        agent = ConstrainedTextPPO(
            self.logits_processor,
            model,
            opt,
            gpu=self.gpu_id,
            update_interval=update_interval,
            minibatch_size=minibatch_size,
            epochs=epochs,
            clip_eps_vf=None,
            entropy_coef=ent_coef,
            gamma=0.95,  # https://arxiv.org/abs/2210.01241
            lambd=1,
            standardize_advantages=True,
            act_deterministically=self.act_deterministically
        )
        self.agent = agent
        return agent

    @autocast('cuda')
    def predict(self, input_item):
        t = 0
        with torch.inference_mode():
            with self.agent.eval_mode():
                obs, input_ids = self.env.reset(input_item)
                while True:
                    action = self.agent.act(obs, input_ids)
                    (obs, input_ids), reward, done, pred = self.env.step(action)
                    t += 1
                    reset = t >= self.env.env_max_length
                    self.agent.observe(obs, reward, done, reset)
                    if done or reset:
                        return pred.get('predicted_str')    

'''