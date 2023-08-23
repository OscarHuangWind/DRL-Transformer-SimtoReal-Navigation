#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:33:39 2022

@author: oscar
"""

import sys
sys.path.append('/home/oscar/ws_oscar/DRL-Transformer-SimtoReal-Navigation/catkin_ws/src/gtrl/scripts')

import os
import time
import statistics
import numpy as np
from tqdm import tqdm 
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from Environments.env_lab import GazeboEnv
# from Environments.env_image_scout_hospital import GazeboEnv

import rospy
from geometry_msgs.msg import Twist


def plot_animation_figure(ep):

    ep -= 200
    plt.figure()
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC") +' Target Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep), reward_target_list)

    plt.subplot(2, 2, 2)
    plt.title('Collision Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep), reward_collision_list)


    plt.subplot(2, 2, 3)
    plt.title('Pedal ' + str(ep_real))
    plt.scatter(np.arange(len(pedal_list)), pedal_list, s=6, c='coral')
    
    plt.subplot(2, 2, 4)
    plt.title('Steering')
    plt.scatter(np.arange(len(steering_list)), steering_list, s=6, c='coral')
    
    plt.tight_layout()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title(env_name + ' ' + str("SAC"))
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.plot(np.arange(ep), reward_list)
    plt.plot(np.arange(ep), reward_mean_list)

    plt.subplot(2, 2, 2)
    plt.title('Heuristic Reward')
    plt.xlabel('Episode')
    plt.ylabel('Heuristic Reward')
    plt.plot(np.arange(ep), reward_heuristic_list)

    plt.subplot(2, 2, 3)
    plt.title('Action Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep), reward_action_list)

    plt.subplot(2, 2, 4)
    plt.title('Freeze Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(np.arange(ep), reward_freeze_list)

    plt.tight_layout()

    plt.pause(0.001)  # pause a bit so that plots are updated

def key_callback(cmd):
    key_cmd.linear.x = cmd.linear.x
    key_cmd.angular.z = cmd.angular.z

if __name__ == "__main__":

    # Set the parameters for the implementation
    device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")  # cuda or cpu
    env_name = "RRC"
    driver = "Oscar_GoT_augmentend"
    robot = 'scout'
    seed = 0 # Random seed number
    max_steps = 300
    max_episodes = int(200)  # Maximum number of steps to perform
    save_models = True  # Weather to save the model or not
    batch_size = 32  # Size of the mini-batch
    frame_stack = 4
    file_name = "SAC_scout_image_rrc_fisheye_smooth_nofreeze_oneshot_transformer"  # name of the file to store the policy
    plot_interval = int(1)

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if not os.path.exists("./Data/" + str(env_name) + '/' + str(driver)):
        os.makedirs("./Data/" + str(env_name) + '/' + str(driver))

    master_uri = '11311'
    env = GazeboEnv('main.launch', master_uri, 1, 1, 1)
    
    cmd = rospy.Subscriber('/scout/telekey', Twist, key_callback, queue_size=1)
    key_cmd = Twist()
    engage = 0
    
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
    
    # Create evaluation data store
    evaluations = []
    
    ep_real = 200
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

        ######## Demonstration List #########
        obs_list = []
        act_list = []
        goal_list= []

        for timestep in range(max_steps):
            # On termination of episode
            if timestep < 3:
                a_in = [0.0, 0.0]
                last_goal = goal
                s_, _, _, _, _, _ ,_, done, goal, target = env.step(a_in, timestep)        
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
                
                np.savez('Data/{}/{}/demo_{}_{}.npz'.format(env_name, driver, robot, ep_real),
                          obs=np.array(obs_list, dtype=np.float32), 
                          act=np.array(act_list, dtype=np.float32),
                          goal=np.array(goal_list, dtype=np.float32))

                reward_list.append(episode_reward)
                reward_mean_list.append(np.mean(reward_list[-20:]))
                reward_heuristic_list.append(episode_heu_reward)
                reward_action_list.append(episode_act_reward)
                reward_target_list.append(episode_tar_reward)
                reward_collision_list.append(episode_col_reward)
                reward_freeze_list.append(episode_fr_reward)
                
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
                      'seed:', seed,
                      'Env:', env_name,
                      "Filename:", file_name,
                      '\n')
                
                if ep_real % plot_interval == 0:
                    plot_animation_figure(ep_real)
                    plt.ioff()
                    plt.show()

                break

            action = [key_cmd.linear.x, key_cmd.angular.z]
            a_in = [(action[0] + 1) * 0.5, action[1]*np.pi*2]
            last_goal = goal
            s_, r_h, r_a, r_f, r_c, r_t, reward, done, goal, target = env.step(a_in, timestep)

            episode_reward += reward
            episode_heu_reward += r_h
            episode_act_reward += r_a
            episode_fr_reward += r_f
            episode_col_reward += r_c
            episode_tar_reward += r_t
            pedal_list.append(round((action[0] + 1)/2,2))
            steering_list.append(round(action[1],2))

            next_state = np.concatenate((s_list[-3], s_list[-2], s_list[-1], s_), axis=-1)

            obs_list.append(state)
            act_list.append(action)
            goal_list.append(goal)

            # Update the counters
            state = next_state
            s_list.append(s_)