#!/usr/bin/env python3
"""
train.py - Main training entry point for Breakout RL agents.

Usage:
    python train.py --agent dqn --episodes 5000 --headless
    python train.py --agent ppo --total-steps 1000000 --reward-shaping
    python train.py --agent grpo --iterations 2000 --group-size 8 --headless
    python train.py --agent dqn --resume checkpoints/dqn_ep2000.pt
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from utils.config import get_config, DQNConfig, PPOConfig, GRPOConfig
from utils.logger import Logger
from ws_env import BreakoutWebSocketEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_agent(config):
    """Create the appropriate agent based on config."""
    from state_processor import OBS_DIM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    if config.agent == "dqn":
        from agents.dqn_agent import DQNAgent
        return DQNAgent(
            obs_dim=OBS_DIM,
            action_dim=4,
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_steps,
            grad_clip=config.grad_clip,
            device=device,
        )
    elif config.agent == "ppo":
        from agents.ppo_agent import PPOAgent
        return PPOAgent(
            obs_dim=OBS_DIM,
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_ratio=config.clip_ratio,
            rollout_length=config.rollout_length,
            mini_batch_size=config.mini_batch_size,
            epochs_per_rollout=config.epochs_per_rollout,
            entropy_coef=config.entropy_coef,
            value_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
            device=device,
        )
    elif config.agent == "grpo":
        from agents.grpo_agent import GRPOAgent
        return GRPOAgent(
            obs_dim=OBS_DIM,
            action_dim=4,
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            gamma=config.gamma,
            group_size=config.group_size,
            batch_states=config.batch_states,
            clip_ratio=config.clip_ratio,
            entropy_coef=config.entropy_coef,
            update_epochs=config.update_epochs,
            mini_batch_size=config.mini_batch_size,
            max_episode_length=config.max_episode_length,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent: {config.agent}")


def save_checkpoint(agent, config, episode, metrics, logger):
    """Save model checkpoint."""
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"{config.agent}_ep{episode}.pt"
    agent.save(str(path))
    log.info(f"Saved checkpoint: {path}")

    logger.log_model(str(path), metadata={"episode": episode, **metrics})


def evaluate(agent, env, num_episodes: int = 10) -> dict:
    """Run evaluation episodes and return metrics."""
    rewards = []
    scores = []
    lengths = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.act(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]  # PPO returns (action, log_prob, value)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        scores.append(info.get("score", 0))
        lengths.append(steps)

    return {
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/std_reward": float(np.std(rewards)),
        "eval/mean_score": float(np.mean(scores)),
        "eval/mean_length": float(np.mean(lengths)),
    }


# ── Training loops ──────────────────────────────────────────────────


def train_dqn(agent, env, config, logger):
    """DQN training loop."""
    total_steps = 0
    best_eval_reward = -float("inf")

    for episode in range(1, config.episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            # Train after warmup
            if total_steps >= config.learning_starts:
                metrics = agent.train_step()
                if metrics:
                    logger.log(metrics, total_steps)

        # Episode metrics
        ep_metrics = {
            "train/episode_reward": episode_reward,
            "train/episode_length": episode_steps,
            "train/score": info.get("score", 0),
            "train/level": info.get("level", 1),
            "train/epsilon": agent.epsilon,
        }
        logger.log(ep_metrics, total_steps)

        # Evaluation
        if episode % config.eval_freq == 0:
            eval_metrics = evaluate(agent, env, config.eval_episodes)
            logger.log(eval_metrics, total_steps)
            log.info(f"Episode {episode}: eval_reward={eval_metrics['eval/mean_reward']:.2f}, "
                     f"eval_score={eval_metrics['eval/mean_score']:.1f}")

            if eval_metrics["eval/mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval/mean_reward"]
                best_path = Path(config.checkpoint_dir) / f"{config.agent}_best.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(best_path))
                log.info(f"New best model: reward={best_eval_reward:.2f}")

        # Checkpoint
        if episode % config.save_freq == 0:
            save_checkpoint(agent, config, episode, ep_metrics, logger)

    log.info(f"DQN training complete. {config.episodes} episodes, {total_steps} total steps.")


def train_ppo(agent, env, config, logger):
    """PPO training loop."""
    obs, info = env.reset()
    total_steps = 0
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0
    best_eval_reward = -float("inf")

    while total_steps < config.total_steps:
        action, log_prob, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, log_prob, reward, value, done)
        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        total_steps += 1

        if done:
            episode_count += 1
            ep_metrics = {
                "train/episode_reward": episode_reward,
                "train/episode_length": episode_steps,
                "train/score": info.get("score", 0),
                "train/level": info.get("level", 1),
            }
            logger.log(ep_metrics, total_steps)

            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0

            # Evaluation
            if episode_count % config.eval_freq == 0:
                eval_metrics = evaluate(agent, env, config.eval_episodes)
                logger.log(eval_metrics, total_steps)

                if eval_metrics["eval/mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval/mean_reward"]
                    best_path = Path(config.checkpoint_dir) / f"{config.agent}_best.pt"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    agent.save(str(best_path))

            # Checkpoint
            if episode_count % config.save_freq == 0:
                save_checkpoint(agent, config, episode_count, ep_metrics, logger)

        # PPO update when rollout buffer is full
        if agent.ready_to_train():
            train_metrics = agent.train_step(last_obs=obs)
            if train_metrics:
                logger.log(train_metrics, total_steps)

    log.info(f"PPO training complete. {episode_count} episodes, {total_steps} total steps.")


def train_grpo(agent, env, config, logger):
    """GRPO training loop."""
    best_eval_reward = -float("inf")

    for iteration in range(1, config.iterations + 1):
        # Collect G trajectories per start state, B start states
        trajectories, collect_metrics = agent.collect_trajectories(env)
        logger.log(collect_metrics, iteration)

        # PPO-style update
        train_metrics = agent.train_step(trajectories=trajectories)
        if train_metrics:
            logger.log(train_metrics, iteration)

        log.info(
            f"Iter {iteration}/{config.iterations}: "
            f"return_mean={collect_metrics['train/group_return_mean']:.2f}, "
            f"return_std={collect_metrics['train/group_return_std']:.2f}, "
            f"steps={collect_metrics['collection/total_steps']}"
        )

        # Evaluation
        if iteration % config.eval_freq == 0:
            eval_metrics = evaluate(agent, env, config.eval_episodes)
            logger.log(eval_metrics, iteration)

            if eval_metrics["eval/mean_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval/mean_reward"]
                best_path = Path(config.checkpoint_dir) / f"{config.agent}_best.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save(str(best_path))

        # Checkpoint
        if iteration % config.save_freq == 0:
            save_checkpoint(agent, config, iteration, collect_metrics, logger)

    log.info(f"GRPO training complete. {config.iterations} iterations.")


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train a Breakout RL agent")

    # General
    p.add_argument("--agent", choices=["dqn", "ppo", "grpo"], default="dqn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ws-url", default="ws://localhost:8765")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--reward-shaping", action="store_true")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--save-freq", type=int, default=500)
    p.add_argument("--eval-freq", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--resume", default=None)

    # DQN
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--target-update-freq", type=int, default=None)
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)

    # PPO
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--rollout-length", type=int, default=None)
    p.add_argument("--epochs-per-rollout", type=int, default=None)
    p.add_argument("--clip-ratio", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--value-coef", type=float, default=None)

    # GRPO
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--group-size", type=int, default=None)
    p.add_argument("--batch-states", type=int, default=None)
    p.add_argument("--update-epochs", type=int, default=None)
    p.add_argument("--max-episode-length", type=int, default=None)

    # Logging
    p.add_argument("--tensorboard", action="store_true", default=True)
    p.add_argument("--no-tensorboard", action="store_false", dest="tensorboard")
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="breakout-rl")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--log-interval", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    # Build config with CLI overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    # Convert hyphenated keys to underscore
    overrides = {k.replace("-", "_"): v for k, v in overrides.items()}
    config = get_config(args.agent, **overrides)

    log.info(f"Training {config.agent.upper()} agent")
    log.info(f"Config: {config}")

    set_seed(config.seed)

    # Create logger
    logger = Logger(config)

    # Create environment
    env = BreakoutWebSocketEnv(
        ws_url=config.ws_url,
        render_mode="none" if config.headless else "human",
        reward_shaping=config.reward_shaping,
        frame_skip=config.frame_skip,
        continuous=(config.agent == "ppo"),
    )

    # Create agent
    agent = create_agent(config)

    # Resume from checkpoint
    if config.resume:
        log.info(f"Resuming from {config.resume}")
        agent.load(config.resume)

    try:
        if config.agent == "dqn":
            train_dqn(agent, env, config, logger)
        elif config.agent == "ppo":
            train_ppo(agent, env, config, logger)
        elif config.agent == "grpo":
            train_grpo(agent, env, config, logger)
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        # Save emergency checkpoint
        ckpt_path = Path(config.checkpoint_dir) / f"{config.agent}_interrupted.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(ckpt_path))
        log.info(f"Emergency checkpoint saved: {ckpt_path}")
    finally:
        env.close()
        logger.finish()


if __name__ == "__main__":
    main()
