import gym
from torch.autograd import Variable
import torch
from memory import Memory

# What does volatile in Variable do?

env = gym.make("Reacher-v1")

print('observation space', env.observation_space)

BATCH_SIZE = 2048
ITER_COUNT = 500
GAMMA = 0.99
TAU = 0.0
LEARN_BATCH_SIZE = 4096

def policy(state_var):
    pass

def value(state_var):
    pass

def estimate_advantages(rewards, masks, gamma, tau):
    pass

# Must accept a variable number of args
def shuffle_tensors():
    pass


for i_iter in range(ITER_COUNT):

    memory = Memory()

    for i_step in range(BATCH_SIZE):
        state = env.reset()

        episode_reward = 0.0

        for timestep in range(1e5):
            state_var = Variable(torch.DoubleTensor(state).unsqueeze(0), volatile=True)

            action = policy(state_var)

            next_state, reward, done, _ = env.step(action)

            mask = 0 if done else 1

            episode_reward += reward
            memory.append(state, action, mask, next_state, reward)

            state = next_state

    batch = memory.sample()

    states = torch.from_numpy(batch.state)
    actions = torch.from_numpy(np.stack(batch.action))
    rewards = torch.from_numpy(np.stack(batch.reward))
    masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))

    values = value(Variable(states, volatile=True))

    log_probs = policy_net.get_log_prob(Variable(states, volatile=True),
            Variable(actions))

    advantages, returns = estimate_advantages(rewards, masks, values,
            GAMMA, TAU)

    lr_mult = max(1.0 - float(i_iter) / ITER_COUNT, 0)

    optim_iter_num = int(math.ceil(states.shape[0] / LEARN_BATCHS_SIZE))
    for _ in range(optim_iter_num):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).cuda()

        states, actions, returns, advantages, fixed_log_probs = shuffle_tensors(states,
                actions, returns, advantages, fixed_log_probs)



