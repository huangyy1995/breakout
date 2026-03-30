"""
replay_buffer.py - Experience replay buffer for DQN.
"""

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size ring buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> dict:
        """Sample a random batch and return as tensors dict."""
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))

        return {
            "states": torch.as_tensor(np.array(batch.state), dtype=torch.float32),
            "actions": torch.as_tensor(np.array(batch.action), dtype=torch.long),
            "rewards": torch.as_tensor(np.array(batch.reward), dtype=torch.float32),
            "next_states": torch.as_tensor(np.array(batch.next_state), dtype=torch.float32),
            "dones": torch.as_tensor(np.array(batch.done), dtype=torch.float32),
        }

    def __len__(self):
        return len(self.buffer)
