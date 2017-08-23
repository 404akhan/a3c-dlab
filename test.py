import math
import os
import sys
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
from collections import deque

import numpy as np 
from helpers import *


def test(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = WrapEnv(args.env_name)
    model = ActorCritic(4, env.num_actions, args.num_skips)

    model.eval()

    state = env.reset()
    state = np.concatenate([state] * 4, axis=0)
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    action_stat = [0] * (model.n_real_acts + model.n_aux_acts)

    start_time = time.time()
    episode_length = 0

    for ep_counter in itertools.count(1):
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            
            if not os.path.exists('model-a3c-aux'):
                os.makedirs('model-a3c-aux')
            torch.save(shared_model.state_dict(), 'model-a3c-aux/model-{}.pth'.format(args.model_name))
            print('saved model')

        value, logit = model(Variable(state.unsqueeze(0), volatile=True))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()

        action_np = action[0, 0]
        action_stat[action_np] += 1

        if action_np < model.n_real_acts:
            state_new, reward, done, info = env.step(action_np)
            
            if args.testing: 
                print('episode', episode_length, 'normal action', action_np, 'lives', info['ale.lives'])
                env.render()
            state = np.append(state.numpy()[1:,:,:], state_new, axis=0)
            done = done or episode_length >= args.max_episode_length

            reward_sum += reward
            episode_length += 1
        else:
            state = state.numpy()

            for _ in range(action_np - model.n_real_acts + 2):
                state_new, rew, done, info = env.step(0) # instead of random perform NOOP=0

                if args.testing: 
                    print('episode', episode_length, 'no_op action', action_np, 'lives', info['ale.lives'])
                    # env.render()
                state = np.append(state[1:,:,:], state_new, axis=0) 
                done = done or episode_length >= args.max_episode_length

                reward_sum += rew
                episode_length += 1
                if done:
                    break

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            print("actions stats real {}, aux {}".format(action_stat[:model.n_real_acts], action_stat[model.n_real_acts:]))

            reward_sum = 0
            episode_length = 0
            state = env.reset()
            state = np.concatenate([state] * 4, axis=0)
            action_stat = [0] * (model.n_real_acts + model.n_aux_acts)
            if not args.testing: time.sleep(60)

        state = torch.from_numpy(state)
