#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:12:20 2023

@author: oscar
"""

import sys
sys.path.append('/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts')

import os
import numpy as np
from cpprb import PrioritizedReplayBuffer

import torch
from torch.optim import Adam
import torch.nn.functional as F

from SAC.utils import soft_update, hard_update

###### GoT-SAC #######
from SAC.got_sac_network import GaussianPolicy, QNetwork
from SAC.got_sac_network import DeterministicPolicy, set_seed
from SAC.got_sac_network import GoTPolicy as GaussianTransformerPolicy
from SAC.got_sac_network import GoTQNetwork as TransformerQNetwork
from SAC.got_sac_network import DeterministicGoTPolicy as DeterministicTransformerPolicy

###### ViT-SAC ######
# from SAC.vit_sac_network import GaussianTransformerPolicy, GaussianPolicy, TransformerQNetwork, QNetwork
# from SAC.vit_sac_network import  DeterministicTransformerPolicy, DeterministicPolicy, set_seed

class SAC(object):
    def __init__(self, action_dim, pstate_dim, policy_type, critic_type,
                 policy_attention_fix, critic_attention_fix, pre_buffer, seed,
                 LR_C = 1e-3, LR_A = 1e-3, LR_ALPHA=1e-4, BUFFER_SIZE=int(2e5), 
                 TAU=5e-3, POLICY_FREQ = 2, GAMMA = 0.99, ALPHA=0.05,
                 block = 2, head = 4, automatic_entropy_tuning=True):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = ALPHA

        self.pstate_dim = pstate_dim
        self.action_dim = action_dim
        
        self.itera = 0
        self.guidence_weight = 1.0
        self.engage_weight = 1.0
        self.buffer_size_expert = 5e3
        self.batch_expert = 0

        self.policy_type = policy_type
        self.critic_type = critic_type
        self.policy_freq = POLICY_FREQ
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.pre_buffer = pre_buffer # expert priors buffer
        self.seed = int(seed)

        self.block = block
        self.head = head

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        set_seed(self.seed)

        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE,
                                          {"obs": {"shape": (128,160,4)},
                                           "act": {"shape":action_dim},
                                           "pobs": {"shape":pstate_dim},
                                           "next_pobs": {"shape":pstate_dim},
                                           "rew": {},
                                           "next_obs": {"shape": (128,160,4)},
                                           "engage": {},
                                           "done": {}},
                                          next_of=("obs"))

        if self.pre_buffer:
            self.replay_buffer_expert = PrioritizedReplayBuffer(self.buffer_size_expert,
                                                                {"obs": {"shape": (128,160,4)},
                                                                 "act_exp": {"shape":action_dim},
                                                                 "pobs": {"shape":pstate_dim},
                                                                 "next_pobs": {"shape":pstate_dim},
                                                                 "rew": {},
                                                                 "next_obs": {"shape": (128,160,4)},
                                                                 "done": {}},
                                                                next_of=("obs"))

        ################# Initialize Critic Network ##############
        if self.critic_type == "Transformer":
            self.critic = TransformerQNetwork(self.action_dim, self.pstate_dim).to(device=self.device)

            if critic_attention_fix:
                params = list(self.critic.fc1.parameters()) + list(self.critic.fc2.parameters()) +\
                         list(self.critic.fc3.parameters()) + list(self.critic.fc11.parameters()) +\
                         list(self.critic.fc21.parameters()) + list(self.critic.fc31.parameters())
                self.critic_optim = Adam(params, LR_C)
            else:
                self.critic_optim = Adam(self.critic.parameters(), LR_C)

            self.critic_target = TransformerQNetwork(self.action_dim, self.pstate_dim,
                                                     self.block, self.head).to(self.device)
            hard_update(self.critic_target, self.critic)
        else:
            self.critic = QNetwork(self.action_dim, self.pstate_dim).to(device=self.device)
            self.critic_optim = Adam(self.critic.parameters(), LR_C)
            self.critic_target = QNetwork(self.action_dim, self.pstate_dim).to(self.device)

        hard_update(self.critic_target, self.critic)

        ############## Initialize Policy Network ################
        if self.policy_type == "GaussianConvNet":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - self.action_dim
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=LR_ALPHA)

            self.policy = GaussianPolicy(self.action_dim, self.pstate_dim).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

        elif self.policy_type == "GaussianTransformer":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - self.action_dim
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=LR_ALPHA)

            ######### Initializing Transformer based Actor ##########
            self.policy = GaussianTransformerPolicy(self.action_dim, self.pstate_dim,
                                                    self.block, self.head).to(self.device)
            
            if policy_attention_fix:
                params = list(self.policy.fc1.parameters()) + list(self.policy.fc2.parameters()) +\
                         list(self.policy.mean_linear.parameters()) + list(self.policy.log_std_linear.parameters()) #+ 
                self.policy_optim = Adam(params, LR_A)
            else:
                self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

        elif self.policy_type == 'DeterministicTransformer':
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicTransformerPolicy(self.action_dim, self.pstate_dim,
                                                         self.block, self.head).to(self.device)
            
            if policy_attention_fix:
                params = list(self.policy.fc1.parameters()) + list(self.policy.fc2.parameters()) +\
                         list(self.policy.mean_linear.parameters()) + list(self.policy.log_std_linear.parameters())
                self.policy_optim = Adam(params, LR_A)
            else:
                self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.action_dim, self.pstate_dim).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=LR_A)

    def choose_action(self, istate, pstate, evaluate=False):
        if istate.ndim < 4:
            istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0,3,1,2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)
        
        if evaluate is False:
            action, _, _ = self.policy.sample([istate, pstate])
        else:
            _, _, action = self.policy.sample([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()

    def learn_guidence(self, engage, batch_size=64):

        agent_buffer_size = self.replay_buffer.get_stored_size()

        if self.pre_buffer:
            exp_buffer_size = self.replay_buffer_expert.get_stored_size()
            scale_factor = 1
            
            self.batch_expert = min(np.floor(exp_buffer_size/agent_buffer_size * batch_size / scale_factor), batch_size)

            batch_agent = batch_size
        
        if self.batch_expert > 0:
            expert_flag = True
            data_agent = self.replay_buffer.sample(batch_agent)
            data_expert = self.replay_buffer_expert.sample(self.batch_expert)

            istates_agent, pstates_agent, actions_agent, engages = \
                data_agent['obs'], data_agent['pobs'], data_agent['act'], data_agent['engage']
            rewards_agent, next_istates_agent, next_pstates_agent, dones_agent = \
                data_agent['rew'], data_agent['next_obs'], data_agent['next_pobs'], data_agent['done']

            istates_expert, pstates_expert, actions_expert = \
                data_expert['obs'], data_expert['pobs'], data_expert['act_exp']
            rewards_expert, next_istates_expert, next_pstates_expert, dones_expert = \
                data_expert['rew'], data_expert['next_obs'], data_expert['next_pobs'], data_expert['done']

            istates = np.concatenate((istates_agent, istates_expert), axis=0)
            pstates = np.concatenate([pstates_agent, pstates_expert], axis=0)
            actions = np.concatenate([actions_agent, actions_expert], axis=0)
            rewards = np.concatenate([rewards_agent, rewards_expert], axis=0)
            next_istates = np.concatenate([next_istates_agent, next_istates_expert], axis=0)
            next_pstates = np.concatenate([next_pstates_agent, next_pstates_expert], axis=0)
            dones = np.concatenate([dones_agent, dones_expert], axis=0)

        else:
            expert_flag = False
            data = self.replay_buffer.sample(batch_size)
            istates, pstates, actions, engages = data['obs'], data['pobs'], data['act'], data['engage']
            rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']
            
        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        engages = torch.FloatTensor(engages).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])  
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        ##### Pre buffer (expert) guidence loss, Optional #####
        if expert_flag:
            istates_expert = torch.FloatTensor(istates_expert).permute(0,3,1,2).to(self.device)
            pstates_expert = torch.FloatTensor(pstates_expert).to(self.device)
            actions_expert = torch.FloatTensor(actions_expert).to(self.device)
            _, _, predicted_actions = self.policy.sample([istates_expert, pstates_expert]) 
            guidence_loss = self.guidence_weight * F.mse_loss(predicted_actions, actions_expert).mean()
        else:
            guidence_loss = 0.0

        ##### Real-time engage loss, Optional ######
        engage_index = (engages == 1).nonzero(as_tuple=True)[0]
        if engage_index.numel() > 0:
            istates_expert = istates[engage_index]
            pstates_expert = pstates[engage_index]
            actions_expert = actions[engage_index]
            _, _, predicted_actions = self.policy.sample([istates_expert, pstates_expert]) 
            engage_loss = self.engage_weight * F.mse_loss(predicted_actions, actions_expert).mean()
        else:
            engage_loss = 0.0

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() + guidence_loss + engage_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Automatic Entropy Adjustment #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        ##### update priorities #####
        # priorities = td_errors
        # priorities = priorities.cpu().numpy()
        # if expert_flag:
        #     self.replay_buffer.update_priorities(indexes_agent, priorities[0:batch_size])
        #     self.replay_buffer_expert.update_priorities(indexes_expert, priorities[-int(self.batch_expert):])
        # else:
        #     self.replay_buffer.update_priorities(indexes, priorities)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        return qf1_loss.item(), policy_loss.item()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0,3,1,2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0,3,1,2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.sample([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
                
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ##### Automatic Entropy Adjustment #####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.itera % self.policy_freq == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.itera += 1

        ##### update priorities #####
        # priorities = td_errors
        # priorities = priorities.cpu().numpy()
        # self.replay_buffer.update_priorities(indexes, priorities)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
        return qf1_loss.item(), policy_loss.item()
    
    def store_transition(self,  s, ps, a, ae, i, r, s_, ps_, d=0):
        self.replay_buffer.add(obs=s,
                               pobs=ps,
                               act=a,
                               acte=ae,
                               intervene=i,
                               rew=r,
                               next_obs=s_,
                               next_pobs=ps_,
                               done=d)

    def store_transition(self, s, a, ps, ps_, r, s_, engage, a_exp, d=0):
        if a is not None:
            self.replay_buffer.add(obs=s,
                    act=a,
                    pobs=ps,
                    next_pobs=ps_,
                    rew=r,
                    next_obs=s_,
                    engage = engage,
                    done=d)
        else:
            self.replay_buffer.add(obs=s,
                    act=a_exp,
                    pobs=ps,
                    next_pobs=ps_,
                    rew=r,
                    next_obs=s_,
                    engage = engage,
                    done=d)

    def initialize_expert_buffer(self, s, a_exp, ps, ps_, r, s_, d=0):

        self.replay_buffer_expert.add(obs=s,
                act_exp=a_exp,
                pobs=ps,
                next_pobs=ps_,
                rew=r,
                next_obs=s_,
                done=d)

    # Save and load model parameters
    def load_model(self, output):
        if output is None: return
        self.policy.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.policy.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def save(self, filename, directory, reward, seed):
        torch.save(self.policy.state_dict(), '%s/%s_reward%s_seed%s_actor.pth' % (directory, filename, reward, seed))
        torch.save(self.critic.state_dict(), '%s/%s_reward%s_seed%s_critic.pth' % (directory, filename, reward, seed))

    def load(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))    

    def load_target(self):
        hard_update(self.critic_target, self.critic)

    def load_actor(self, filename, directory):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

    def save_transition(self, output, timeend=0):
        self.replay_buffer.save_transitions(file='{}/{}'.format(output, timeend))

    def load_transition(self, output):
        if output is None: return
        self.replay_buffer.load_transitions('{}.npz'.format(output))
