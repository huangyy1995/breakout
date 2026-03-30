"""
ws_env.py - Gymnasium-compatible environment that communicates with the
Breakout game via WebSocket.

Usage:
    env = BreakoutWebSocketEnv(ws_url="ws://localhost:8765")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import asyncio
import json
import logging
import time
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from state_processor import process_state, OBS_DIM

log = logging.getLogger(__name__)


class BreakoutWebSocketEnv(gym.Env):
    """
    Gymnasium wrapper around the browser Breakout game via WebSocket.

    The browser runs Game.stepForAI() on each step() call.
    Communication is synchronous: step() sends an action and blocks
    until the browser returns the result.
    """

    metadata = {"render_modes": ["human", "none"]}

    # Action mapping: discrete action ID -> game action JSON
    ACTION_MAP = {
        0: {"type": "move", "direction": -1.0},   # left
        1: {"type": "move", "direction": 0.0},     # stay
        2: {"type": "move", "direction": 1.0},     # right
        3: {"type": "launch"},                      # launch ball
    }

    def __init__(
        self,
        ws_url: str = "ws://localhost:8765",
        render_mode: str = "none",
        reward_shaping: bool = False,
        frame_skip: int = 4,
        max_steps: int = 10_000,
        timeout_steps: int = 600,
        continuous: bool = False,
    ):
        super().__init__()
        self.ws_url = ws_url
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.timeout_steps = timeout_steps  # Steps without brick hit -> truncate
        self.continuous = continuous

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        if continuous:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(4)

        self._ws = None
        self._loop = None
        self._prev_state = None
        self._steps_since_hit = 0
        self._total_steps = 0
        self._connected = False

    def _ensure_connection(self):
        """Ensure WebSocket connection is established."""
        if self._ws is not None and self._connected:
            return

        import websockets.sync.client as ws_sync
        log.info(f"Connecting to {self.ws_url}...")
        self._ws = ws_sync.connect(self.ws_url)
        self._connected = True
        log.info("Connected to WebSocket relay")

    def _send(self, msg: dict) -> dict:
        """Send a JSON message and wait for response."""
        self._ws.send(json.dumps(msg))
        raw = self._ws.recv()
        return json.loads(raw)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_connection()

        # Configure headless mode
        if self.render_mode == "none":
            self._send({"type": "config", "data": {"renderEnabled": False}})

        # Reset the game
        response = self._send({"type": "reset"})
        raw_state = response.get("data", response)

        obs = process_state(raw_state)
        self._prev_state = raw_state
        self._steps_since_hit = 0
        self._total_steps = 0

        info = {
            "score": raw_state.get("score", 0),
            "lives": raw_state.get("lives", 3),
            "level": raw_state.get("level", 1),
        }
        return obs, info

    def step(self, action):
        self._ensure_connection()

        # Convert action
        if self.continuous:
            game_action = {"type": "move", "direction": float(np.clip(action, -1, 1))}
        else:
            action_int = int(action)
            game_action = self.ACTION_MAP.get(action_int, self.ACTION_MAP[1])

        # Auto-launch: if ball not launched, send launch first
        if self._prev_state and not self._prev_state.get("ball", {}).get("launched", True):
            self._send({"type": "step", "action": {"type": "launch"}})

        # Frame skip: repeat action N times, accumulate reward
        total_reward = 0.0
        terminated = False
        truncated = False
        raw_state = self._prev_state
        info = {}

        for _ in range(self.frame_skip):
            response = self._send({"type": "step", "action": game_action})
            result = response.get("data", response)

            raw_state = result.get("state", result)
            step_reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})

            total_reward += step_reward
            self._total_steps += 1

            # Track steps since last brick hit
            if step_reward > 0:
                self._steps_since_hit = 0
            else:
                self._steps_since_hit += 1

            if done:
                game_state = raw_state.get("gameState", "")
                terminated = game_state in ("GAME_OVER",)
                # LEVEL_COMPLETE is also terminal for the episode
                if game_state == "LEVEL_COMPLETE":
                    terminated = True
                break

        # Reward shaping
        if self.reward_shaping and raw_state:
            total_reward += self._shape_reward(raw_state)

        # Timeout truncation
        if self._steps_since_hit >= self.timeout_steps:
            total_reward -= 1.0
            truncated = True

        # Max steps truncation
        if self._total_steps >= self.max_steps:
            truncated = True

        obs = process_state(raw_state)
        self._prev_state = raw_state

        return obs, total_reward, terminated, truncated, info

    def _shape_reward(self, state: dict) -> float:
        """Apply optional reward shaping."""
        reward = 0.0

        paddle_x = state.get("paddle", {}).get("x", 0.5)
        ball_x = state.get("ball", {}).get("x", 0.5)
        ball_vy = state.get("ball", {}).get("vy", 0.0)

        # Encourage paddle to track ball X position
        distance = abs(paddle_x - ball_x)
        reward += 0.01 * (1.0 - distance)

        # Small bonus for ball moving upward
        if ball_vy < 0:
            reward += 0.001

        return reward

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False

    def __del__(self):
        self.close()
