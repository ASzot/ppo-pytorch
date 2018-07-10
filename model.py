import torch.nn as nn
import torch.nn.functional as F
import torch

class Mlp(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.actor_hidden = nn.Sequential(
                init_(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_(nn.Linear(64, 64)),
                nn.Tanh(),
            )

        self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_(nn.Linear(64, 64)),
                nn.Tanh(),
                init_(nn.Linear(64, 1)),
            )

        self.train()

    def forward(self, inputs):
        return self.actor_hidden(inputs), self.critic(inputs)


def init_(m):
    w = m.weight.data
    w.normal_(0, 1)
    w *= 1 / torch.sqrt(w.pow(2).sum(1, keepdim=True))

    b = m.bias.data
    nn.init.constant_(b, 0)
    return m


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()

        self.actor_critic = Mlp(obs_shape[0])

        num_outputs = action_space.shape[0]

        self.fc_mean = init_(nn.Linear(64, num_outputs))
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def __get_dist(self, actor_features):
        action_mean = self.fc_mean(actor_features)
        action_log_std = self.action_log_std.expand_as(action_mean)

        return torch.distributions.Normal(action_mean, action_log_std.exp())


    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy


