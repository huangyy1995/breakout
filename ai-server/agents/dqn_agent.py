"""
dqn_agent.py - Deep Q-Network agent with experience replay and target network.
"""

import numpy as np
import torch
import torch.nn.functional as F

from agents.base import BaseAgent
from agents.replay_buffer import ReplayBuffer
from networks.mlp import QNetwork


class DQNAgent(BaseAgent):
    """DQN agent with epsilon-greedy exploration, replay buffer, and target network."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        grad_clip: float = 10.0,
        device: str = "cpu",
    ):
        super().__init__(obs_dim, device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start

        # Networks
        self.q_net = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.train_steps = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            obs_t = self._to_tensor(obs).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return q_values.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, **kwargs) -> dict:
        """Perform one gradient step on a batch from the replay buffer."""
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Current Q values
        q_values = self.q_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (no grad)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1

        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self._update_epsilon()

        return {
            "train/loss": loss.item(),
            "train/q_mean": q_values.mean().item(),
            "train/epsilon": self.epsilon,
        }

    def _update_epsilon(self):
        """Linear epsilon decay."""
        fraction = min(1.0, self.train_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _get_save_dict(self) -> dict:
        return {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
        }

    def _load_save_dict(self, checkpoint: dict):
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
        self.epsilon = checkpoint["epsilon"]
