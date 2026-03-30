"""
base.py - Abstract base class for all RL agents.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract RL agent interface."""

    def __init__(self, obs_dim: int, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.device = torch.device(device)

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Any:
        """Select an action given an observation."""
        ...

    @abstractmethod
    def train_step(self, **kwargs) -> dict:
        """Perform one training update. Returns dict of metrics."""
        ...

    def save(self, path: str):
        """Save agent state to file."""
        torch.save(self._get_save_dict(), path)

    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._load_save_dict(checkpoint)

    @abstractmethod
    def _get_save_dict(self) -> dict:
        """Return dict of state to save."""
        ...

    @abstractmethod
    def _load_save_dict(self, checkpoint: dict):
        """Restore state from checkpoint dict."""
        ...

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on agent's device."""
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
