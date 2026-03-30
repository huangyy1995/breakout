"""
ppo_agent.py - Proximal Policy Optimization with actor-critic and GAE.
"""

import numpy as np
import torch
import torch.nn.functional as F

from agents.base import BaseAgent
from networks.mlp import ActorCriticNetwork


class RolloutBuffer:
    """Stores rollout data for PPO on-policy updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get_batches(self, advantages, returns, mini_batch_size: int, device):
        """Yield mini-batches for PPO update."""
        n = len(self.states)
        states = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, mini_batch_size):
            end = start + mini_batch_size
            idx = indices[start:end]
            yield {
                "states": states[idx],
                "actions": actions[idx],
                "old_log_probs": old_log_probs[idx],
                "advantages": adv[idx],
                "returns": ret[idx],
            }


class PPOAgent(BaseAgent):
    """PPO agent with continuous action space and GAE."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        rollout_length: int = 2048,
        mini_batch_size: int = 64,
        epochs_per_rollout: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(obs_dim, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.rollout_length = rollout_length
        self.mini_batch_size = mini_batch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCriticNetwork(obs_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.train_steps = 0

    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Select action. Returns (action_value, log_prob, value)."""
        with torch.no_grad():
            obs_t = self._to_tensor(obs).unsqueeze(0)
            action, log_prob, value = self.network.get_action(obs_t, deterministic)
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, value, done):
        """Store transition in rollout buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)

    def ready_to_train(self) -> bool:
        """Check if we have enough data for a training update."""
        return len(self.buffer) >= self.rollout_length

    def train_step(self, last_obs: np.ndarray = None, **kwargs) -> dict:
        """Perform PPO update on the collected rollout."""
        if not self.ready_to_train():
            return {}

        # Compute last value for GAE
        last_value = 0.0
        if last_obs is not None:
            with torch.no_grad():
                obs_t = self._to_tensor(last_obs).unsqueeze(0)
                _, last_value = self.network(obs_t)
                last_value = last_value.item()

        advantages, returns = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.epochs_per_rollout):
            for batch in self.buffer.get_batches(advantages, returns, self.mini_batch_size, self.device):
                new_log_probs, entropy, values = self.network.evaluate_actions(
                    batch["states"], batch["actions"]
                )

                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.train_steps += 1
        self.buffer.clear()

        n = max(num_updates, 1)
        return {
            "train/policy_loss": total_policy_loss / n,
            "train/value_loss": total_value_loss / n,
            "train/entropy": total_entropy / n,
        }

    def _get_save_dict(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }

    def _load_save_dict(self, checkpoint: dict):
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
