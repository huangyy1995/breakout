"""
config.py - Hyperparameter configuration dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class BaseConfig:
    """Shared training configuration."""
    agent: str = "dqn"
    seed: int = 42
    ws_url: str = "ws://localhost:8765"
    headless: bool = False
    reward_shaping: bool = False
    frame_skip: int = 4
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 500
    eval_freq: int = 100
    eval_episodes: int = 10
    lr: float = 1e-4
    gamma: float = 0.99
    hidden_dim: int = 256
    log_interval: int = 10

    # Logging
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "breakout-rl"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: list = field(default_factory=list)

    # Resume
    resume: Optional[str] = None

    @property
    def run_name(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{self.agent}_{self.seed}_{ts}"


@dataclass
class DQNConfig(BaseConfig):
    """DQN-specific hyperparameters."""
    agent: str = "dqn"
    episodes: int = 3000
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    grad_clip: float = 10.0
    learning_starts: int = 1000


@dataclass
class PPOConfig(BaseConfig):
    """PPO-specific hyperparameters."""
    agent: str = "ppo"
    total_steps: int = 500_000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    rollout_length: int = 2048
    mini_batch_size: int = 64
    epochs_per_rollout: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class GRPOConfig(BaseConfig):
    """GRPO-specific hyperparameters."""
    agent: str = "grpo"
    iterations: int = 2000
    lr: float = 1e-4
    gamma: float = 0.99
    group_size: int = 8
    batch_states: int = 4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02
    update_epochs: int = 5
    mini_batch_size: int = 128
    max_episode_length: int = 2000


def get_config(agent: str, **overrides) -> BaseConfig:
    """Create config for the given agent type with optional overrides."""
    configs = {
        "dqn": DQNConfig,
        "ppo": PPOConfig,
        "grpo": GRPOConfig,
    }
    if agent not in configs:
        raise ValueError(f"Unknown agent: {agent}. Choose from {list(configs.keys())}")

    config = configs[agent]()
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config
