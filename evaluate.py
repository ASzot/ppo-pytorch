import torch
import gym
import numpy as np
import types
from multiprocessing_env import DummyVecEnv, VecNormalize

ENV_NAME = 'Reacher-v1'
load_path = 'weights/model_1000.pt'
policy = torch.load(load_path)


def make_env():
    return gym.make(ENV_NAME)
env = make_env

env = DummyVecEnv([env])
obs_shape = env.observation_space.shape
current_obs = torch.zeros(1, *obs_shape)
def update_current_obs(obs):
    obs = torch.from_numpy(obs).float()
    current_obs[:, :] = obs


for i in range(1000):
    obs = env.reset()
    update_current_obs(obs)
    done = False
    episode_reward = 0.0
    while not done:
        with torch.no_grad():
            _, action, _ = policy.act(current_obs, deterministic=True)
        action = action.squeeze(1).cpu().numpy()
        obs, reward, done, _ = env.step(action)

        episode_reward += reward

        update_current_obs(obs)
        env.render()

    print(reward)

