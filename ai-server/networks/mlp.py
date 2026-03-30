"""
MLP networks for Breakout RL agents.

Provides:
  - QNetwork: DQN Q-value network (obs -> Q-values for each action)
  - PolicyNetwork: Discrete policy (obs -> action logits)
  - ActorCriticNetwork: PPO shared-backbone actor-critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class QNetwork(nn.Module):
    """DQN Q-value network: obs -> Q(s,a) for each discrete action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Discrete policy network for GRPO: obs -> action probabilities."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.net(x)

    def get_dist(self, obs: torch.Tensor) -> Categorical:
        """Return categorical distribution over actions."""
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and return (action, log_prob)."""
        dist = self.get_dist(obs)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions: return (log_probs, entropy)."""
        dist = self.get_dist(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ActorCriticNetwork(nn.Module):
    """
    PPO Actor-Critic with shared backbone.
    Actor outputs continuous action (tanh-squashed Gaussian).
    Critic outputs scalar state value.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head: outputs mean of Gaussian for continuous action [-1, 1]
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic head: outputs scalar value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        value = self.critic(features).squeeze(-1)
        action_mean = self.actor_mean(features)
        return action_mean, value

    def get_dist(self, obs: torch.Tensor) -> Normal:
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and return (action, log_prob, value)."""
        features = self.shared(obs)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        # Tanh squash to [-1, 1]
        action_squashed = torch.tanh(action)
        # Log prob with tanh correction
        log_prob = dist.log_prob(action).sum(-1)
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6).sum(-1)

        return action_squashed.squeeze(-1), log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions: return (log_probs, entropy, values)."""
        features = self.shared(obs)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        # Inverse tanh to get pre-squash action
        actions_unsqueezed = actions.unsqueeze(-1) if actions.dim() == 1 else actions
        eps = 1e-6
        pre_tanh = 0.5 * torch.log((1 + actions_unsqueezed + eps) / (1 - actions_unsqueezed + eps))

        log_prob = dist.log_prob(pre_tanh).sum(-1)
        log_prob -= torch.log(1 - actions_unsqueezed.pow(2) + eps).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, entropy, value
