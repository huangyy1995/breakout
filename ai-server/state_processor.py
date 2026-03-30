"""
state_processor.py - Convert raw JSON game state to fixed-size observation vector.

Input:  Raw JSON from Game.getState() via WebSocket
Output: numpy float32 array of shape (233,)

Layout:
  [0:2]     Paddle (x, width)
  [2:7]     Ball (x, y, vx, vy, launched)
  [7:231]   Brick grid (14 cols x 16 rows = 224 HP values)
  [231:233] Meta (lives_ratio, bricks_remaining_ratio)
"""

import numpy as np

OBS_DIM = 233
GRID_COLS = 14
GRID_ROWS = 16


def process_state(raw_state: dict) -> np.ndarray:
    """Convert raw JSON game state to fixed-size float32 vector (233,)."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Paddle: [0:2]
    paddle = raw_state.get("paddle", {})
    obs[0] = paddle.get("x", 0.5)
    obs[1] = paddle.get("width", 0.18)

    # Ball: [2:7]
    ball = raw_state.get("ball", {})
    obs[2] = ball.get("x", 0.5)
    obs[3] = ball.get("y", 0.5)
    obs[4] = (ball.get("vx", 0.0) + 1.0) / 2.0  # normalize [-1,1] -> [0,1]
    obs[5] = (ball.get("vy", 0.0) + 1.0) / 2.0
    obs[6] = 1.0 if ball.get("launched", False) else 0.0

    # Brick grid: [7:231] — 14 cols x 16 rows = 224
    bricks = raw_state.get("bricks", [])
    for brick in bricks:
        col = int(brick.get("x", 0) * GRID_COLS)
        row = int(brick.get("y", 0) * GRID_ROWS)
        col = min(col, GRID_COLS - 1)
        row = min(row, GRID_ROWS - 1)
        hp = brick.get("hp", 1)
        obs[7 + row * GRID_COLS + col] = hp / 10.0  # normalize HP

    # Meta: [231:233]
    lives = raw_state.get("lives", 3)
    obs[231] = lives / 3.0

    bricks_remaining = raw_state.get("bricksRemaining", 0)
    total_bricks = bricks_remaining + len(bricks)
    if total_bricks > 0:
        obs[232] = bricks_remaining / total_bricks
    else:
        obs[232] = 0.0

    return obs
