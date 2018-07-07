import gym
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from memory import Memory
import torch.optim as optim

def init_weights(m):
    w = m.weight.data
    w.normal_(0, 1)
    w *= 1 / torch.sqrt(w.pow(2).sum(1, keepdim=True))

    b = m.bias.data
    nn.init.constant_(b, 0)
    return m

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.fc_mean = init_weights(nn.Linear(num_inputs, num_outputs))
        self.log_std = self.fc_mean + torch.zeros(num_outputs)


    def forward(self, x):
        action_mean = self.fc_mean(x)
        return FixedNormal(action_mean, torch.ones(num_outputs))



class Mlp(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.actor_hidden = nn.Sequential(
                get_linear(num_inputs, 64),
                nn.Tanh(),
                get_linear(64, 64),
                nn.Tanh()
            )

        self.critic = nn.Sequential(
                get_linear(num_inputs, 64),
                nn.Tanh(),
                get_linear(64, 64),
                nn.Tanh()
            )

    def forward(self, inputs):
        return self.actor_hidden(inputs), self.critic(inputs)

    def get_linear(self, inputs, outputs):
        return init_weights(nn.Linear(inputs, outputs))



class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()

        self.actor_critic = Mlp(obs_shape[0])

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(64, num_outputs)


    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mean()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, ation, action_log_probs

    def get_value(self, inputs, states, masks):
        value, _ = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states





CLIP_PARAM = 0.2
EPOCH = 4
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

def update(rollouts, policy, optimizer):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    for epoch_i in range(EPOCH):
        data_generator = rollouts.feed_forward_generator(advantages,
                MINI_BATCH_SIZE)

        value_losses = []
        action_losses = []
        entropy_losses = []
        for obs, actions, returns, masks, old_action_log_probs, adv_targ in data_generator:
            values, action_log_probs, dist_entropy = policy.evaluate_actions(obs, actions)

            ratio = torch.exp(action_log_probs - old_action_log_probs)

            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM)

            action_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(return_batch, values)
            optimizer.zero_grad()

            (value_loss * VALUE_LOSS_COEFF + action_loss - dist_entropy *
                    ENTROPY_COEFF).backward()

            nn.utils.clip_grad_norm(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            value_losses.append(value_loss)
            action_losses.append(action_loss)
            entropy_losses.append(dist_entropy)

    return np.mean(value_losses), np.mean(action_losses), np.mean(entropy_losses)


env = gym.make("Reacher-v1")

obs_shape = env.observation_space.shape

policy = Policy(obs_shape, env.action_space)

action_shape = env.action_space.shape[0]

optimizer = optim.Adam(policy.parameters(), lr=LR, eps=EPS)

rollouts = RolloutStorage(N_STEPS, 1, obs_shape, env.action_space)


obs = envs.reset()

rollouts.observations[0].copy_(obs)

episode_rewards = torch.zeros([1, 1])
final_rewards = torch.zeros([1, 1])

n_updates = N_FRAMES / N_STEPS
for update_i in range(n_updates):
    for step in range(N_STEPS):
        with torch.no_grad():
            value, action, action_log_prob = policy.act(rollouts.observations[step])

        take_actions = action.squeeze(1).cpu().numpy()

        obs, reward, done, info = env.step(take_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward

        masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        rollouts.insert(obs, action, action_log_prob, value, reward, masks)

    with torch.no_grad():
        next_value = policy.get_value(rollouts.observations[-1],
                rollouts.states[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, GAMMA, TAU)

        value_loss, action_loss, entropy_loss = update(rollouts, policy,
                optimizer)

        rollouts.after_update()
