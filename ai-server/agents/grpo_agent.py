"""
grpo_agent.py - Group Relative Policy Optimization.

GRPO (DeepSeek-R1, 2025) eliminates the critic network. Instead of estimating
a value baseline, it samples G complete trajectories from the current policy
for each start state, computes each trajectory's total return, then uses the
group mean/std as the baseline and normalization.
"""

import numpy as np
import torch

from agents.base import BaseAgent
from networks.mlp import PolicyNetwork


class Trajectory:
    """Stores a single trajectory (episode) of (state, action, log_prob, reward)."""

    __slots__ = ("states", "actions", "log_probs", "rewards", "advantage")

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.advantage = 0.0  # scalar, same for all steps

    def add(self, state, action, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def compute_return(self, gamma: float) -> float:
        """Compute discounted cumulative return."""
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
        return G

    def __len__(self):
        return len(self.states)


class GRPOAgent(BaseAgent):
    """
    GRPO agent with discrete action space.

    Per iteration:
      1. Sample B start states
      2. For each, roll out G trajectories using current policy
      3. Compute group-relative advantages (trajectory-level)
      4. PPO-style clipped update over all collected transitions
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        group_size: int = 8,
        batch_states: int = 4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.02,
        update_epochs: int = 5,
        mini_batch_size: int = 128,
        max_episode_length: int = 2000,
        device: str = "cpu",
    ):
        super().__init__(obs_dim, device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.group_size = group_size
        self.batch_states = batch_states
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.max_episode_length = max_episode_length

        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.train_steps = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action from policy."""
        with torch.no_grad():
            obs_t = self._to_tensor(obs).unsqueeze(0)
            action, _ = self.policy.get_action(obs_t, deterministic)
        return action.item()

    def act_with_log_prob(self, obs: np.ndarray):
        """Select action and return (action, log_prob) for trajectory collection."""
        with torch.no_grad():
            obs_t = self._to_tensor(obs).unsqueeze(0)
            action, log_prob = self.policy.get_action(obs_t, deterministic=False)
        return action.item(), log_prob.item()

    def collect_trajectories(self, env) -> tuple:
        """
        Collect B * G trajectories for one GRPO iteration.

        Returns:
            all_trajectories: list of Trajectory objects with advantages set
            metrics: dict of collection statistics
        """
        all_trajectories = []
        all_returns = []

        for b in range(self.batch_states):
            group_trajectories = []
            group_returns = []

            for g in range(self.group_size):
                # Reset to a new start state (game always resets to level 1)
                obs, _ = env.reset()
                traj = Trajectory()

                for step in range(self.max_episode_length):
                    action, log_prob = self.act_with_log_prob(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    traj.add(obs, action, log_prob, reward)
                    obs = next_obs

                    if terminated or truncated:
                        break

                ret = traj.compute_return(self.gamma)
                group_trajectories.append(traj)
                group_returns.append(ret)

            # Compute group-relative advantages
            returns_arr = np.array(group_returns, dtype=np.float32)
            mean_r = returns_arr.mean()
            std_r = returns_arr.std() + 1e-8

            for traj, ret in zip(group_trajectories, group_returns):
                traj.advantage = (ret - mean_r) / std_r

            all_trajectories.extend(group_trajectories)
            all_returns.extend(group_returns)

        metrics = {
            "train/group_return_mean": float(np.mean(all_returns)),
            "train/group_return_std": float(np.std(all_returns)),
            "collection/total_steps": sum(len(t) for t in all_trajectories),
            "collection/mean_length": float(np.mean([len(t) for t in all_trajectories])),
        }
        return all_trajectories, metrics

    def train_step(self, trajectories: list = None, **kwargs) -> dict:
        """
        Perform PPO-style clipped update over collected trajectories.

        Args:
            trajectories: list of Trajectory objects with advantages set
        """
        if trajectories is None or len(trajectories) == 0:
            return {}

        # Flatten all transitions
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for traj in trajectories:
            all_states.extend(traj.states)
            all_actions.extend(traj.actions)
            all_old_log_probs.extend(traj.log_probs)
            # Same advantage for all steps in a trajectory
            all_advantages.extend([traj.advantage] * len(traj))

        states = torch.as_tensor(np.array(all_states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(all_actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.array(all_old_log_probs), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.array(all_advantages), dtype=torch.float32, device=self.device)

        # Normalize advantages globally
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(all_states)
        total_policy_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = advantages[idx]

                # Evaluate current policy
                new_log_probs, entropy = self.policy.evaluate_actions(batch_states, batch_actions)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.train_steps += 1

        k = max(num_updates, 1)
        return {
            "train/policy_loss": total_policy_loss / k,
            "train/entropy": total_entropy / k,
        }

    def _get_save_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }

    def _load_save_dict(self, checkpoint: dict):
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint["train_steps"]
