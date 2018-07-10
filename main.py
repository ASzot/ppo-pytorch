import gym

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from memory import RolloutStorage
from model import Policy

from multiprocessing_env import SubprocVecEnv
from tensorboardX import SummaryWriter

import os
import shutil
import copy

CLIP_PARAM = 0.2
N_EPOCH = 4
MINI_BATCH_SIZE = 32
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01
LR = 7e-4
EPS = 1e-5
MAX_GRAD_NORM = 0.5
N_STEPS = 5
N_FRAMES = 10e6
CUDA = True
GAMMA = 0.99
TAU = 0.95
N_ENVS = 16
ENV_NAME = 'Reacher-v1'
SAVE_INTERVAL = 10000
MODEL_DIR = 'weights'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def update(rollouts, policy, optimizer):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    for epoch_i in range(N_EPOCH):
        data_generator = rollouts.feed_forward_generator(advantages,
                MINI_BATCH_SIZE)

        value_losses = []
        action_losses = []
        entropy_losses = []
        losses = []
        for obs, actions, returns, masks, old_action_log_probs, adv_targ in data_generator:
            values, action_log_probs, dist_entropy, = policy.evaluate_actions(obs, actions)

            ratio = torch.exp(action_log_probs - old_action_log_probs)

            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * adv_targ

            action_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(returns, values)
            optimizer.zero_grad()

            loss = (value_loss * VALUE_LOSS_COEFF + action_loss - dist_entropy *
                    ENTROPY_COEFF)
            loss.backward()

            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            value_losses.append(value_loss.item())
            action_losses.append(action_loss.item())
            entropy_losses.append(dist_entropy.item())
            losses.append(loss.item())

    return np.mean(value_losses), np.mean(action_losses), np.mean(entropy_losses), np.mean(losses)


if os.path.exists('runs'):
    shutil.rmtree('runs')
writer = SummaryWriter()

def make_env():
    return gym.make(ENV_NAME)

envs = [make_env for i in range(N_ENVS)]
envs = SubprocVecEnv(envs)

obs_shape = envs.observation_space.shape
print('Obs shape', obs_shape)

policy = Policy(obs_shape, envs.action_space)
if CUDA:
    policy.cuda()

optimizer = optim.Adam(policy.parameters(), lr=LR, eps=EPS)

rollouts = RolloutStorage(N_STEPS, N_ENVS, obs_shape, envs.action_space)

current_obs = torch.zeros(N_ENVS, *obs_shape)
obs = envs.reset()

def update_current_obs(obs):
    obs = torch.from_numpy(obs).float()
    current_obs[:, :] = obs

update_current_obs(obs)

if CUDA:
    rollouts.cuda()
    current_obs.cuda()

rollouts.observations[0].copy_(current_obs)

episode_rewards = torch.zeros([N_ENVS, 1])
final_rewards = torch.zeros([N_ENVS, 1])

n_updates = int(N_FRAMES // N_STEPS // N_ENVS)
for update_i in tqdm(range(n_updates)):
    for step in range(N_STEPS):
        with torch.no_grad():
            value, action, action_log_prob = policy.act(rollouts.observations[step])

        take_actions = action.squeeze(1).cpu().numpy()

        obs, reward, done, info = envs.step(take_actions)

        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward

        masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        current_obs *= masks

        update_current_obs(obs)

        rollouts.insert(current_obs, action, action_log_prob, value, reward, masks)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.observations[-1]).detach()

    rollouts.compute_returns(next_value, GAMMA, TAU)

    value_loss, action_loss, entropy_loss, overall_loss = update(rollouts, policy,
            optimizer)

    rollouts.after_update()

    writer.add_scalar('data/action_loss', action_loss, update_i)
    writer.add_scalar('data/value_loss', value_loss, update_i)
    writer.add_scalar('data/entropy_loss', entropy_loss, update_i)
    writer.add_scalar('data/overall_loss', overall_loss, update_i)
    writer.add_scalar('data/avg_reward', final_rewards.mean(), update_i)

    if update_i % SAVE_INTERVAL == 0:
        save_model = policy
        if CUDA:
            save_model = copy.deepcopy(policy).cpu()

        torch.save(save_model, os.path.join(MODEL_DIR, 'model_%i.pt' % update_i))


writer.close()
