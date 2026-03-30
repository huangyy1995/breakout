#!/usr/bin/env python3
"""
evaluate.py - Evaluate a trained Breakout RL agent.

Usage:
    python evaluate.py --agent dqn --checkpoint checkpoints/dqn_best.pt --episodes 100
"""

import argparse
import logging
import sys

import numpy as np
import torch

from state_processor import OBS_DIM
from ws_env import BreakoutWebSocketEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [eval] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")


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


def evaluate(agent, env, num_episodes: int, verbose: bool = False):
    """Run evaluation and print results."""
    rewards = []
    scores = []
    levels = []
    lengths = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.act(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        scores.append(info.get("score", 0))
        levels.append(info.get("level", 1))
        lengths.append(steps)

        if verbose:
            log.info(f"  Episode {ep}: reward={total_reward:.2f}, "
                     f"score={info.get('score', 0)}, "
                     f"level={info.get('level', 1)}, "
                     f"steps={steps}")

    # Summary
    print("\n" + "=" * 50)
    print(f"Evaluation Results ({num_episodes} episodes)")
    print("=" * 50)
    print(f"  Reward:  {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Score:   {np.mean(scores):.1f} +/- {np.std(scores):.1f}")
    print(f"  Level:   {np.mean(levels):.1f} +/- {np.std(levels):.1f}")
    print(f"  Length:  {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}")
    print(f"  Max score: {max(scores)}")
    print(f"  Max reward: {max(rewards):.2f}")
    print("=" * 50)


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Breakout RL agent")
    p.add_argument("--agent", choices=["dqn", "ppo", "grpo"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--ws-url", default="ws://localhost:8765")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--reward-shaping", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    agent = load_agent(args.agent, args.checkpoint, args.hidden_dim)

    env = BreakoutWebSocketEnv(
        ws_url=args.ws_url,
        render_mode="none",
        reward_shaping=args.reward_shaping,
        frame_skip=args.frame_skip,
        continuous=(args.agent == "ppo"),
    )

    try:
        evaluate(agent, env, args.episodes, args.verbose)
    finally:
        env.close()


if __name__ == "__main__":
    main()
