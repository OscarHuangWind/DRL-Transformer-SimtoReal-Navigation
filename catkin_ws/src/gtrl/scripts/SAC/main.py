#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 15:10:17 2023

@author: oscar
"""

#!/usr/bin/env python

import sys
sys.path.append('/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts')

import os
import time
import glob
import yaml
import statistics
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from collections import deque
import matplotlib.pyplot as plt
from cpprb import PrioritizedReplayBuffer
# from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F

from SAC.DRL import SAC
from Environments.env_lab import GazeboEnv

import rospy
from geometry_msgs.msg import Twist

def evaluate(network, eval_episodes=10, epoch=0):
    obs_list = deque(maxlen=frame_stack)
    env.collision = 0
    ep = 0
    avg_reward_list = []
    while ep < eval_episodes:
        count = 0
        obs, goal = env.reset()
        done = False
        avg_reward = 0.0

        for i in range(4):
            obs_list.append(obs)

        observation = np.concatenate((obs_list[-4], obs_list[-3], obs_list[-2], obs_list[-1]), axis=-1)

        while not done and count < max_steps:
            
            if count == 0:
                action = network.choose_action(np.array(state), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
    
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                last_goal = goal
                obs_, _, _, _, _, _ , _, done, goal, target = env.step(a_in, timestep)        
                observation = np.concatenate((obs_, obs_, obs_, obs_), axis=-1)
                
                for i in range(4):
                    obs_list.append(obs_)           

                if done:
                    print("\n..............................................")
                    print("Bad Initialization, skip this episode.")
                    print("..............................................")
                    ep -= 1
                    env.collision -= 1
                    break

                count += 1
                continue
            
            act = network.choose_action(np.array(observation), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
            a_in = [(act[0] + 1) * linear_cmd_scale, act[1]*angular_cmd_scale]
            obs_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, count)        
            avg_reward += reward
            observation = np.concatenate((obs_list[-3], obs_list[-2], obs_list[-1], obs_), axis=-1)
            obs_list.append(obs_)
            count += 1
        
        ep += 1
        avg_reward_list.append(avg_reward)
        print("\n..............................................")
        print("%i Loop, Steps: %i, Avg Reward: %f, Collision No. : %i " % (ep, count, avg_reward, env.collision))
        print("..............................................")
    reward = statistics.mean(avg_reward_list)
    col = env.collision
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward: %f, Collision No.: %i" % (eval_episodes, epoch, reward, col))
    print("..............................................")
    return reward

def plot_animation_figure():

    plt.figure()
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC") + ' Lr_a: ' + str(lr_a) + ' Lr_c: ' + str(lr_c) +' Target Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_target_list)

    plt.subplot(2, 2, 2)
    plt.title('Collision Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_collision_list)


    plt.subplot(2, 2, 3)
    plt.title('Pedal ' + str(ep_real))
    plt.scatter(np.arange(len(pedal_list)), pedal_list, s=6, c='coral')
    
    plt.subplot(2, 2, 4)
    plt.title('Steering')
    plt.scatter(np.arange(len(steering_list)), steering_list, s=6, c='coral')
    
    plt.tight_layout()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC") + ' Lr_a: ' + str(lr_a) + ' Lr_c: ' + str(lr_c))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep_real), reward_list)
    plt.plot(np.arange(ep_real), reward_mean_list)

    plt.subplot(2, 2, 2)
    plt.title('Heuristic Reward')
    plt.xlabel('Episode')
    plt.ylabel('Heuristic Reward')
    plt.plot(np.arange(ep_real), reward_heuristic_list)

    plt.subplot(2, 2, 3)
    plt.title('Action Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_action_list)

    plt.subplot(2, 2, 4)
    plt.title('Freeze Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep_real), reward_freeze_list)

    plt.tight_layout()

    plt.pause(0.001)  # pause a bit so that plots are updated

def key_callback(cmd):
    key_cmd.linear.x = cmd.linear.x
    key_cmd.angular.z = cmd.angular.z
    global intervention 
    intervention = cmd.angular.x

if __name__ == "__main__":

    # Set the parameters for the implementation
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")  # cuda or cpu

    path = os.getcwd()
    yaml_path = os.path.join(path, 'config.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ##### Individual parameters for each model ######
    model = 'GoT-SAC'
    mode_param = config[model]
    model_name = mode_param['name']
    policy_type = mode_param['actor_type']
    critic_type = mode_param['critic_type']
    transformer_block = mode_param['block']
    transformer_head = mode_param['head']

    ###### Default parameters for DRL ######
    max_steps = config['MAX_STEPS']
    max_episodes = config['MAX_EPISODES']
    batch_size = config['BATCH_SIZE']
    lr_a = config['LR_A']
    lr_c = config['LR_C']
    gamma = config['GAMMA']
    tau = config['TAU']
    policy_freq = config['ACTOR_FREQ']
    buffer_size = config['BUFFER_SIZE']
    frame_stack = config['FRAME_STACK']
    plot_interval = config['PLOT_INTERVAL']
    
    ##### Evaluation #####
    save_interval = config['SAVE_INTERVAL']
    save_threshold = config['SAVE_THRESHOLD']
    reward_threshold = config['REWARD_THRESHOLD']
    eval_threshold = config['EVAL_THRESHOLD'] 
    eval_ep = config['EVAL_EPOCH']
    save_models = config['SAVE']

    ##### Attention #####
    pre_train = config['PRE_TRAIN'] # whether intialize with pre-trained parameter
    attention_only = config['ATTENTION_ONLY'] # whether load the attention only from the pretrained GoT
    policy_attention_fix = config['P_ATTENTION_FIX'] # whether fix the weights and bias of policy attention
    critic_attention_fix = config['C_ATTENTION_FIX'] #whether fix the weights and bias of value attention

    ##### Human Intervention #####
    pre_buffer = config['PRE_BUFFER'] # Human expert buffer
    human_guidence = config['HUMAN_INTERVENTION'] # whether need guidance from human driver

    ##### Entropy ######
    auto_tune = config['AUTO_TUNE']
    alpha = config['ALPHA']
    lr_alpha = config['LR_ALPHA']

    ##### Environment ######
    seed = config['SEED']
    env_name = config['ENV_NAME']
    driver = config['DRIVER']
    robot = config['ROBOT']
    linear_cmd_scale = config['L_SCALE']
    angular_cmd_scale = config['A_SCALE']

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    folder_name = "./final_curves"
    if save_models and not os.path.exists(folder_name):
        os.makedirs(folder_name)
    folder_name = "./final_models"
    if save_models and not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # Create the training environment
    # master_uri = rosgraph.get_master_uri()
    # print(master_uri)
    
    master_uri = '11311'
    env = GazeboEnv('main.launch', master_uri, 1, 1, 1)

    cmd = rospy.Subscriber('/scout/telekey', Twist, key_callback, queue_size=1)
    key_cmd = Twist()
    intervention = 0
    time.sleep(5)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env.seed(seed)
    
    state, _ = env.reset()
    state_dim = state.shape
    action_dim = 2
    physical_state_dim = 2 # Polar coordinate
    max_action = 1
    
    # Initialize the agent
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, policy_attention_fix,
              critic_attention_fix, pre_buffer, seed, lr_c, lr_a, lr_alpha,
              buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block,
              head=transformer_head, automatic_entropy_tuning=auto_tune)

    ###### Initializing pretrained network if possible ######
    if pre_train:
        if attention_only:
            # policy_type = "DeterministicTransformer"
            name = "SAC_IL_scout_image_rrc_fisheye_GoT_normalize_Oscar_seed1_64patches_2depth_8heads_2048mlp"
            il_ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, 
                         policy_attention_fix, critic_attention_fix, human_guidence, 
                         seed, lr_c, lr_a, lr_alpha, buffer_size, tau, policy_freq,
                         gamma, alpha, block=transformer_block, head=transformer_head,
                         automatic_entropy_tuning=auto_tune)
            il_ego.load_actor(name, directory="./final_models")
    
            ###### Assign the attention only ########
            ego.policy.trans = il_ego.policy.trans
            ego.policy.fc_embed = il_ego.policy.fc_embed

        else:
            name = 'SAC_IL_scout_image_rrc_fisheye_GoT_normalize_Oscar_seed1_64patches_2depth_8heads_2048mlp'
            ego.load_actor(name, directory="./final_models")

    ###### Pre intialiaze corner replay buffer, Optional #######
    if pre_buffer:
        data_dir = '/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts'
        files = natsorted(glob.glob(os.path.join(data_dir) + '/IL/Data/' + env_name + '/' + driver + '/*.npz'))
        obs_list = []
        act_list = []
        goal_list = []
        r_list = []
        next_obs_list = []
        next_goal_list = []
        done_list = []
        
        for idx, file in enumerate(files):
            
            obs = np.load(file)['obs']
            act = np.load(file)['act']
            goal = np.load(file)['goal']
            r = np.load(file)['reward']
            next_obs = np.load(file)['next_obs']
            next_goal = np.load(file)['next_goal']
            done = np.load(file)['done']
            
            obs_list.append(np.array(obs))
            act_list.append(np.array(act))
            goal_list.append(np.array(goal))
            r_list.append(np.array(r))
            next_obs_list.append(np.array(next_obs))
            next_goal_list.append(np.array(next_goal))
            done_list.append(np.array(done))
        
        obs_dataset = np.concatenate(obs_list, axis=0)
        act_dataset = np.concatenate(act_list, axis=0)
        goal_dataset = np.concatenate(goal_list, axis=0)
        reward_dataset = np.concatenate(r_list, axis=0)
        next_obs_dataset = np.concatenate(next_obs_list, axis=0)
        next_goal_dataset = np.concatenate(next_goal_list, axis=0)
        done_dataset = np.concatenate(done_list, axis=0)
    
        ego.initialize_expert_buffer(obs_dataset, act_dataset, goal_dataset[:,:2], 
                                     next_goal_dataset[:,:2], reward_dataset,
                                     next_obs_dataset, done_dataset)

    # Create evaluation data store
    evaluations = []
    
    ep_real = 0
    done = False
    reward_list = []
    reward_heuristic_list = []
    reward_action_list = []
    reward_freeze_list = []
    reward_target_list = []
    reward_collision_list = []
    reward_mean_list = []
    
    pedal_list = []
    steering_list = []

    plt.ion()

    total_timestep = 0

    # Begin the training loop
    for ep in tqdm(range(0, max_episodes), ascii=True):
        episode_reward = 0
        episode_heu_reward = 0.0
        episode_act_reward = 0.0
        episode_tar_reward = 0.0
        episode_col_reward = 0.0
        episode_fr_reward = 0.0
        s_list = deque(maxlen=frame_stack)
        s, goal = env.reset()

        for i in range(4):
            s_list.append(s)

        state = np.concatenate((s_list[-4], s_list[-3], s_list[-2], s_list[-1]), axis=-1)

        for timestep in range(max_steps):
            # On termination of episode
            if timestep == 0:
                action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                last_goal = goal
                s_, _, _, _, _, _ , reward, done, goal, target = env.step(a_in, timestep)        
                state = np.concatenate((s_, s_, s_, s_), axis=-1)
                
                for i in range(4):
                    s_list.append(s_)           

                if done:
                    print("Bad Initialization, skip this episode.")
                    break

                continue
            
            if done or timestep == max_steps-1:
                ep_real += 1
    
                done = False

                reward_list.append(episode_reward)
                reward_mean_list.append(np.mean(reward_list[-20:]))
                reward_heuristic_list.append(episode_heu_reward)
                reward_action_list.append(episode_act_reward)
                reward_target_list.append(episode_tar_reward)
                reward_collision_list.append(episode_col_reward)
                reward_freeze_list.append(episode_fr_reward)

                # if reward_mean_list[-1] >= reward_threshold and ep_real > eval_threshold:
                #     reward_threshold = reward_mean_list[-1]
                #     print("Evaluating the Performance.")
                #     avg_reward = evaluate(ego, eval_ep, ep_real)
                #     evaluations.append(avg_reward)
                #     if avg_reward > save_threshold:
                #         ego.save(file_name, directory=folder_name, reward=int(np.floor(avg_reward)), seed=seed)
                #         save_threshold = avg_reward

                pedal_list.clear()
                steering_list.clear()
                total_timestep += timestep 
                print('\n',
                      '\n',
                      'Robot: ', 'Scout',
                      'Episode:', ep_real,
                      'Step:', timestep,
                      'Tottal Steps:', total_timestep,
                      'R:', episode_reward,
                      'Overak R:', reward_mean_list[-1],
                      'Expert Batch:', np.int8(ego.batch_expert),
                      'Temperature:', ego.alpha.detach().cpu().numpy().item(),
                      'Lr_a:', lr_a,
                      'Lr_c', lr_c,
                      'seed:', seed,
                      'Env:', env_name,
                      "Filename:", model_name,
                      '\n')

                if (ep_real % save_interval == 0):
                    np.save(os.path.join('final_curves', 'reward_seed' + str(seed) + '_' + model_name),
                            reward_mean_list, allow_pickle=True, fix_imports=True)

                if ep_real % plot_interval == 0:
                    plot_animation_figure()
                    plt.ioff()
                    plt.show()

                break

            if intervention:
                action_exp = [key_cmd.linear.x, key_cmd.angular.z]
                action = None
                a_in = [(action_exp[0] + 1) * linear_cmd_scale, action_exp[1]*angular_cmd_scale]
                pedal_list.append(round((action_exp[0] + 1)/2,2))
                steering_list.append(round(action_exp[1],2))
            else:
                action = ego.choose_action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                action_exp = None
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1]*angular_cmd_scale]
                pedal_list.append(round((action[0] + 1)/2,2))
                steering_list.append(round(action[1],2))

            last_goal = goal
            s_, r_h, r_a, r_f, r_c, r_t, reward, done, goal, target = env.step(a_in, timestep)

            episode_reward += reward
            episode_heu_reward += r_h
            episode_act_reward += r_a
            episode_fr_reward += r_f
            episode_col_reward += r_c
            episode_tar_reward += r_t

            next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)

            # Save the tuple in replay buffer
            ego.store_transition(state, action, last_goal[:2], goal[:2], reward, next_state, intervention, action_exp, done)

            # Train the SAC model
            if human_guidence or pre_buffer:
                ego.learn_guidence(intervention, batch_size)
            else:
                ego.learn(batch_size)

            # Update the counters
            state = next_state
            s_list.append(s_)

    # After the training is done, evaluate the network and save it
    avg_reward = evaluate(ego, eval_ep, ep_real)
    evaluations.append(avg_reward)
    if avg_reward > save_threshold:
        ego.save(model_name, directory=folder_name, reward=int(np.floor(avg_reward)), seed=seed)

    np.save(os.path.join('final_curves', 'reward_seed' + str(seed) + '_' + model_name), reward_mean_list, allow_pickle=True, fix_imports=True)