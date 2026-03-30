#!/usr/bin/env python3
"""
play.py - Watch a trained agent play Breakout in the browser at real-time speed.

Usage:
    python play.py --agent dqn --checkpoint checkpoints/dqn_best.pt
"""

import argparse
import logging
import time

import numpy as np
import torch

from state_processor import OBS_DIM
from ws_env import BreakoutWebSocketEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [play] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("play")


def load_agent(agent_type: str, checkpoint_path: str, hidden_dim: int = 256):
    """Load a trained agent from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent_type == "dqn":
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(obs_dim=OBS_DIM, hidden_dim=hidden_dim, device=device)
    elif agent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(obs_dim=OBS_DIM, hidden_dim=hidden_dim, device=device)
    elif agent_type == "grpo":
        from agents.grpo_agent import GRPOAgent
        agent = GRPOAgent(obs_dim=OBS_DIM, hidden_dim=hidden_dim, device=device)
    else:
        raise ValueError(f"Unknown agent: {agent_type}")

    agent.load(checkpoint_path)
    log.info(f"Loaded {agent_type.upper()} agent from {checkpoint_path}")
    return agent


def play(agent, env, num_episodes: int = 5, fps: float = 15.0):
    """Play episodes at real-time speed with rendering enabled."""
    frame_time = 1.0 / fps

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        log.info(f"Episode {ep} starting...")

        while not done:
            start = time.time()

            action = agent.act(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Throttle to target FPS for watchable playback
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        log.info(f"Episode {ep}: reward={total_reward:.2f}, "
                 f"score={info.get('score', 0)}, "
                 f"level={info.get('level', 1)}, "
                 f"steps={steps}")

    log.info("Play session complete.")


def main():
    p = argparse.ArgumentParser(description="Watch a trained agent play Breakout")
    p.add_argument("--agent", choices=["dqn", "ppo", "grpo"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--ws-url", default="ws://localhost:8765")
    p.add_argument("--fps", type=float, default=15.0, help="Playback speed (actions per second)")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    args = p.parse_args()

    agent = load_agent(args.agent, args.checkpoint, args.hidden_dim)

    env = BreakoutWebSocketEnv(
        ws_url=args.ws_url,
        render_mode="human",  # Enable rendering
        reward_shaping=False,
        frame_skip=args.frame_skip,
        continuous=(args.agent == "ppo"),
    )

    try:
        play(agent, env, args.episodes, args.fps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
