import math
import os
import sys
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt 
import numpy as np 
from helpers import *


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = WrapEnv(args.env_name)
    model = ActorCritic(4, env.num_actions, args.num_skips)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = np.concatenate([state] * 4, axis=0)
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    sum_reward = 0
    for ep_counter in itertools.count(1):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        
        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit = model(Variable( state.unsqueeze(0) ))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            action_np = action.numpy()[0][0]
            if action_np < model.n_real_acts:
                state_new, reward, done, info = env.step(action_np)
                state = np.append(state.numpy()[1:,:,:], state_new, axis=0)
                done = done or episode_length >= args.max_episode_length
                
                reward = max(min(reward, 1), -1)
                episode_length += 1
            else:
                state = state.numpy()
                reward = 0.
                for _ in range(action_np - model.n_real_acts + 2):
                    state_new, rew, done, info = env.step(0)  # instead of random perform NOOP=0
                    state = np.append(state[1:,:,:], state_new, axis=0) 
                    done = done or episode_length >= args.max_episode_length

                    reward += rew
                    episode_length += 1
                    if done:
                        break
                reward = max(min(reward, 1), -1)

            sum_reward += reward
            if done:
                state = env.reset()
                state = np.concatenate([state] * 4, axis=0)
                
                print('ep len {}, sum rew {}'.format(episode_length, sum_reward))
                episode_length = 0
                sum_reward = 0

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _ = model(Variable( state.unsqueeze(0) ))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            policy_loss = policy_loss - \
                log_probs[i] * Variable(advantage.data) - 0.01 * entropies[i]

        optimizer.zero_grad()

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40.)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
