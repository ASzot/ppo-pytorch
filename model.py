import torch.nn as nn
import torch.nn.functional as F
import torch

class Mlp(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.actor_hidden = nn.Sequential(
                init_weights(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_weights(nn.Linear(64, 64)),
                nn.Tanh(),
            )

        self.critic = nn.Sequential(
                init_weights(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_weights(nn.Linear(64, 64)),
                nn.Tanh(),
                init_weights(nn.Linear(64, 1)),
            )

    def forward(self, inputs):
        return self.actor_hidden(inputs), self.critic(inputs)

    def get_linear(self, inputs, outputs):
        return init_weights(nn.Linear(inputs, outputs))


def init_weights(m):
    w = m.weight.data
    w.normal_(0, 1)
    w *= 1 / torch.sqrt(w.pow(2).sum(1, keepdim=True))

    b = m.bias.data
    b *= 0.0
    return m


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, log_std=0.0):
        super().__init__()

        self.actor_critic = Mlp(obs_shape[0])

        num_outputs = action_space.shape[0]

        self.fc_mean = init_weights(nn.Linear(64, num_outputs))
        self.logstd = nn.Parameter(torch.ones(1, num_outputs) * log_std)

    def __get_dist(self, actor_features):
        action_mean = self.fc_mean(actor_features)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        return torch.distributions.Normal(action_mean, action_std)


    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)

        if deterministic:
            action = dist.mean()
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy


