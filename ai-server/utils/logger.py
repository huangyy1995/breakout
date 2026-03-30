"""
logger.py - Unified logging to TensorBoard + W&B + console.
"""

import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class ConsoleWriter:
    """Logs metrics to console at regular intervals."""

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self._episode_count = 0
        self._start_time = time.time()

    def log(self, metrics: dict, step: int):
        self._episode_count += 1
        if self._episode_count % self.log_interval == 0:
            elapsed = time.time() - self._start_time
            parts = [f"step={step}"]
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            log.info(f"[{elapsed:.0f}s] {' | '.join(parts)}")

    def finish(self):
        pass


class TensorBoardWriter:
    """Wraps torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, writer):
        self.writer = writer

    def log(self, metrics: dict, step: int):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class WandbWriter:
    """Logs to Weights & Biases."""

    def log(self, metrics: dict, step: int):
        import wandb
        wandb.log(metrics, step=step)

    def log_video(self, frames, step: int, fps: int = 30):
        import wandb
        import numpy as np
        video = wandb.Video(np.array(frames), fps=fps, format="mp4")
        wandb.log({"eval/gameplay": video}, step=step)

    def log_model(self, path: str, metadata: dict = None):
        import wandb
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata=metadata,
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self):
        import wandb
        wandb.finish()


class Logger:
    """Unified logging to TensorBoard + W&B + console."""

    def __init__(self, config):
        """
        Args:
            config: Object with attributes:
                - run_name (str)
                - log_interval (int)
                - tensorboard (bool)
                - wandb (bool)
                - wandb_project (str)
                - wandb_entity (str or None)
                - wandb_name (str or None)
                - wandb_tags (list[str])
        """
        self.writers = [ConsoleWriter(getattr(config, "log_interval", 10))]

        if getattr(config, "tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_dir = Path("runs") / config.run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                self.writers.append(TensorBoardWriter(
                    SummaryWriter(log_dir=str(run_dir))
                ))
                log.info(f"TensorBoard logging to runs/{config.run_name}")
            except ImportError:
                log.warning("tensorboard not installed, skipping TensorBoard logging")

        if getattr(config, "wandb", False):
            try:
                import wandb
                wandb.init(
                    project=getattr(config, "wandb_project", "breakout-rl"),
                    entity=getattr(config, "wandb_entity", None),
                    name=getattr(config, "wandb_name", None) or config.run_name,
                    tags=getattr(config, "wandb_tags", []),
                    config=self._config_to_dict(config),
                    save_code=True,
                )
                self.writers.append(WandbWriter())
                log.info(f"W&B logging to project={config.wandb_project}")
            except ImportError:
                log.warning("wandb not installed, skipping W&B logging")

    def log(self, metrics: dict, step: int):
        """Log scalar metrics to all active writers."""
        for w in self.writers:
            w.log(metrics, step)

    def log_video(self, frames: list, step: int, fps: int = 30):
        """Log gameplay video (W&B only)."""
        for w in self.writers:
            if hasattr(w, "log_video"):
                w.log_video(frames, step, fps)

    def log_model(self, path: str, metadata: dict = None):
        """Log model checkpoint as artifact (W&B only)."""
        for w in self.writers:
            if hasattr(w, "log_model"):
                w.log_model(path, metadata)

    def finish(self):
        """Close all writers."""
        for w in self.writers:
            w.finish()

    @staticmethod
    def _config_to_dict(config) -> dict:
        """Convert config object to dict for W&B."""
        if hasattr(config, "__dict__"):
            return {k: v for k, v in vars(config).items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, type(None)))}
        return {}
