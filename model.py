import torch.nn as nn
import torch.nn.functional as F
import torch

# Init layer to have the proper weight initializations.
def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m


# Standard feed forward network for actor and critic with tanh activations
class Mlp(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        # We do not want to select action yet as that will be probablistic.
        self.actor_hidden = nn.Sequential(
                init_layer(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
            )

        self.critic = nn.Sequential(
                init_layer(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 1)),
            )

        self.train()

    def forward(self, inputs):
        return self.actor_hidden(inputs), self.critic(inputs)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()

        self.actor_critic = Mlp(obs_shape[0])

        num_outputs = action_space.shape[0]

        # How we will define our normal distribution to sample action from
        self.action_mean = init_layer(nn.Linear(64, num_outputs))
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def __get_dist(self, actor_features):
        action_mean = self.action_mean(actor_features)
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


