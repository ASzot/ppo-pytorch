import torch
import gym
import numpy as np


ENV_NAME = 'Humanoid-v1'
load_path = 'weights/model_80000.pt'
policy = torch.load(load_path)

env = gym.make(ENV_NAME)

for i in range(1000):
    obs = env.reset()
    done = False
    while not done:
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).float()

        with torch.no_grad():
            _, action, _ = policy.act(obs, deterministic=True)
        action = action.squeeze(1).cpu().numpy()
        obs, reward, done, _ = env.step(action)
        print(reward)
        env.render()

