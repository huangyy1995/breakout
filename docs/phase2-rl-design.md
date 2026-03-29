# Phase 2: Python RL Integration — Design Document

## 1. Overview

### Goal
Build a Python-side reinforcement learning training system that connects to the Breakout game via WebSocket, runs DQN, PPO, and GRPO agents, and trains them to play the game from raw game state observations.

### Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────────┐
│ Browser (localhost:3000)                                        │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐   │
│  │ Game.js  │──▶│ AIController │──▶│ WebSocketBridge       │   │
│  │ stepForAI│   │ reset/step   │   │ ws://localhost:8765   │   │
│  └──────────┘   └──────────────┘   └───────────┬───────────┘   │
└────────────────────────────────────────────────┼───────────────┘
                                                  │ JSON over WS
┌────────────────────────────────────────────────┼───────────────┐
│ Python (ai-server)                              │               │
│  ┌──────────────────────────────────────────────▼─────────┐     │
│  │ ws_env.py — BreakoutWebSocketEnv (Gymnasium wrapper)   │     │
│  │   reset() → obs    step(action) → obs, reward, done    │     │
│  └───────────────────┬────────────────────────────────────┘     │
│                      │                                          │
│  ┌───────────────────▼───────────────────────────────────┐      │
│  │ agents/                                               │      │
│  │   dqn_agent.py  — DQN with replay buffer + target net │      │
│  │   ppo_agent.py  — PPO with actor-critic               │      │
│  │   grpo_agent.py — GRPO group relative policy opt      │      │
│  └───────────────────┬───────────────────────────────────┘      │
│                      │                                          │
│  ┌───────────────────▼─────────────────────┐                    │
│  │ train.py — main training loop           │                    │
│  │   --agent dqn|ppo                       │                    │
│  │   --episodes 5000                       │                    │
│  │   --headless (disable browser render)   │                    │
│  │   --checkpoint-dir ./checkpoints        │                    │
│  └───────────────────┬─────────────────────┘                    │
│                      │                                          │
│  ┌───────────────────▼─────┐   ┌──────────────────────┐        │
│  │ evaluate.py             │   │ play.py              │        │
│  │ Load checkpoint, run    │   │ Load checkpoint,     │        │
│  │ 100 episodes, report    │   │ render in browser    │        │
│  │ mean/std reward         │   │ at 60 FPS real-time  │        │
│  └─────────────────────────┘   └──────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Communication Flow (one training step)
```
Python                              Browser
  │                                    │
  │─── {"type":"step","action":{..}} ──▶│
  │                                    │── Game.stepForAI(action)
  │◀── {"type":"step_result","data":   │
  │      {state,reward,done,info}}     │
  │                                    │
```

---

## 2. Observation Space

### Raw State (from `Game.getState()`)

```jsonc
{
  "paddle": { "x": 0.42, "y": 0.85, "width": 0.18 },     // all [0,1]
  "ball":   { "x": 0.5, "y": 0.6, "vx": 0.3, "vy": -0.4, "launched": true },
  "bricks": [                                                // variable length
    { "x": 0.02, "y": 0.12, "w": 0.06, "h": 0.03, "hp": 2 },
    ...
  ],
  "score": 120,
  "lives": 3,
  "level": 1,
  "gameState": "PLAYING",
  "bricksRemaining": 24
}
```

### Flattened Observation Vector (for neural network input)

The Python `BreakoutWebSocketEnv` will convert the raw JSON into a fixed-size float32 vector:

| Segment | Fields | Size | Notes |
|---------|--------|------|-------|
| Paddle  | `x, width` | 2 | `y` is constant, omit |
| Ball    | `x, y, vx, vy, launched` | 5 | `launched` as 0/1 |
| Brick grid | Fixed 14x16 grid of HP values | 224 | `0` = empty/destroyed, `>0` = HP. Max grid is 14 cols x 16 rows. Sparse levels have 0-padding. |
| Meta    | `lives/maxLives, bricksRemaining/totalBricks` | 2 | Normalized |
| **Total** | | **233** | Fixed size for all levels |

**Brick grid mapping:** The game positions bricks on a logical grid (`cols x rows`). We map each brick's `(col, row)` back from its `(x, y)` pixel position into a fixed 14x16 matrix. This gives a spatial representation the CNN can learn from.

### Alternative: Image-based Observation

For CNN-based approaches, we can also provide a downscaled grayscale screenshot:
- Capture the canvas at 84x84 resolution (standard Atari DQN size)
- Stack 4 consecutive frames -> shape `(4, 84, 84)`
- This is optional and slower (requires `canvas.toDataURL()` over WebSocket)
- **Recommendation:** Start with the vector observation. Add image obs later if needed.

---

## 3. Action Space

### Discrete (for DQN)

| Action ID | Meaning |
|-----------|---------|
| 0 | Move left (`direction: -1.0`) |
| 1 | Stay (`direction: 0.0`) |
| 2 | Move right (`direction: 1.0`) |
| 3 | Launch ball (only meaningful before first launch) |

**Size:** `Discrete(4)`

The `position` action type is not used for DQN -- directional control is simpler and sufficient.

### Continuous (for PPO)

| Dimension | Range | Meaning |
|-----------|-------|---------|
| 0 | `[-1, 1]` | Paddle direction (mapped to `move` action) |

**Size:** `Box(low=-1, high=1, shape=(1,))`

Ball launch is handled automatically: if the ball is not launched, the env auto-sends a launch action before the move.

### Discrete (for GRPO)

GRPO uses the same discrete action space as DQN:

| Action ID | Meaning |
|-----------|---------|
| 0 | Move left (`direction: -1.0`) |
| 1 | Stay (`direction: 0.0`) |
| 2 | Move right (`direction: 1.0`) |
| 3 | Launch ball |

**Size:** `Discrete(4)`

GRPO samples G actions per state from a policy network (categorical distribution), rolls out complete trajectories for each, then uses group-relative advantages to update the policy — no value network needed.

---

## 4. Reward Design

### Base Rewards (from `Game.stepForAI`)

| Event | Reward | Source |
|-------|--------|--------|
| Brick destroyed | +1 | `_pendingReward` |
| Level complete | +50 | `_pendingReward` |
| Life lost | -5 | `_pendingReward` |

### Shaped Rewards (added in Python env wrapper)

To accelerate learning, the Python env adds small shaping bonuses:

| Signal | Reward | Rationale |
|--------|--------|-----------|
| Paddle tracks ball X | +0.01 * (1 - \|paddle_x - ball_x\|) | Encourages positioning under the ball |
| Ball moving upward | +0.001 | Discourages letting ball fall |
| Episode timeout (no brick hit for 600 steps) | -1, truncate | Prevents the agent from bouncing forever without hitting bricks |

Shaping rewards are **opt-in** via `--reward-shaping` flag, disabled by default for pure RL comparison.

---

## 5. File Structure

```
ai-server/
├── Dockerfile                   # Python 3.11 slim image
├── requirements.txt             # torch, gymnasium, websockets, numpy, tensorboard, wandb
├── README.md                    # Quick start guide
│
├── ws_server.py                 # WebSocket relay server
├── ws_env.py                    # BreakoutWebSocketEnv — Gymnasium-compatible
├── state_processor.py           # Raw JSON -> flat vector conversion
├── train.py                     # Main training entry point (CLI)
├── evaluate.py                  # Load checkpoint, run N episodes, report stats
├── play.py                      # Load checkpoint, play in browser at real-time speed
│
├── agents/
│   ├── __init__.py
│   ├── base.py                  # Abstract Agent interface
│   ├── dqn_agent.py             # DQN + target network + replay buffer
│   ├── ppo_agent.py             # PPO actor-critic
│   ├── grpo_agent.py            # GRPO group relative policy optimization
│   └── replay_buffer.py         # Experience replay for DQN
│
├── networks/
│   ├── __init__.py
│   ├── mlp.py                   # Simple MLP Q-network / policy network
│   └── cnn.py                   # (Future) CNN for image observations
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # TensorBoard + W&B + console logging
│   └── config.py                # Hyperparameter dataclasses
│
├── checkpoints/                 # Saved model weights (gitignored)
├── runs/                        # TensorBoard logs (gitignored)
└── wandb/                       # W&B local cache (gitignored)
```

---

## 6. Component Design

### 6.1 `ws_env.py` — BreakoutWebSocketEnv

```python
class BreakoutWebSocketEnv(gymnasium.Env):
    """
    Gymnasium-compatible wrapper around the browser game via WebSocket.
    
    The browser runs Game.stepForAI() on each step() call.
    Communication is synchronous: step() sends an action and blocks
    until the browser returns the result.
    """
    
    metadata = {"render_modes": ["human", "none"]}
    
    def __init__(self, ws_url="ws://localhost:8765", render_mode="none",
                 reward_shaping=False, frame_skip=4):
        self.ws_url = ws_url
        self.frame_skip = frame_skip        # Repeat action N times per step
        self.reward_shaping = reward_shaping
        
        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(233,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # left/stay/right/launch
        
        self._ws = None
        self._prev_state = None
        self._steps_since_hit = 0
    
    def reset(self, seed=None, options=None):
        # Send {"type": "reset"} -> receive initial state
        # Convert to flat obs vector
        # Configure headless if render_mode == "none"
        ...
        return obs, info
    
    def step(self, action):
        # Map discrete action -> game action JSON
        # Send {"type": "step", "action": {...}} x frame_skip
        # Accumulate reward over skipped frames
        # Convert final state to flat obs vector
        # Apply reward shaping if enabled
        # Check timeout (truncation)
        ...
        return obs, reward, terminated, truncated, info
    
    def close(self):
        # Close WebSocket connection
        ...
```

**Key decisions:**
- **Synchronous WebSocket**: `step()` blocks until the browser responds. This is fine because training is not real-time -- we want deterministic lockstep.
- **Frame skip = 4**: Each `step()` call repeats the action 4 times in the game (4x `stepForAI` calls). This matches Atari DQN conventions and speeds up training.
- **Auto-launch**: If the ball is not launched, the env sends `{type:'launch'}` before the move action.

### 6.2 `ws_server.py` — WebSocket Relay Server

A thin relay server that sits between Python and the browser:

```
Python (ws client) ──▶ ws_server.py (port 8765) ◀── Browser (ws client)
```

**Why a relay?** The browser's `WebSocketBridge` connects *out* as a client. Python also connects *out* as a client. The relay server accepts both and forwards messages between them.

```python
class RelayServer:
    """
    Accepts exactly two clients:
      1. Browser game (identified by first message containing "connected")
      2. Python trainer (everything else)
    
    Messages from Python -> forwarded to Browser
    Messages from Browser -> forwarded to Python
    """
```

**Alternative (simpler for dev):** Modify `WebSocketBridge.js` to run a WebSocket *server* in the browser using a service worker, and have Python connect directly. But browser WS servers are non-standard. The relay approach is more portable.

### 6.3 `agents/dqn_agent.py` — Deep Q-Network

**Architecture:**

```
obs (233,) -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU -> FC(4) -> Q-values
```

**Key hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | Adam optimizer |
| Discount gamma | 0.99 | |
| Replay buffer size | 100,000 | |
| Batch size | 64 | |
| Target net update | Every 1,000 steps | Hard copy |
| epsilon start -> end | 1.0 -> 0.05 | Linear decay over 50k steps |
| Frame skip | 4 | Applied in env |
| Gradient clipping | max_norm=10 | |

**Training loop (pseudocode):**
```python
for episode in range(num_episodes):
    obs = env.reset()
    while not done:
        action = agent.select_action(obs, epsilon)  # epsilon-greedy
        next_obs, reward, done, truncated, info = env.step(action)
        buffer.push(obs, action, reward, next_obs, done)
        
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            loss = agent.train_step(batch)
        
        if steps % target_update_freq == 0:
            agent.update_target_net()
        
        obs = next_obs
```

### 6.4 `agents/ppo_agent.py` — Proximal Policy Optimization

**Architecture:**

```
           obs (233,)
               │
        ┌──────┴──────┐
        │  Shared MLP  │
        │ FC(256)->ReLU │
        │ FC(256)->ReLU │
        └──────┬──────┘
          ┌────┴────┐
    ┌─────┴───┐ ┌───┴─────┐
    │ Actor   │ │ Critic  │
    │ FC(1)   │ │ FC(1)   │
    │ -> tanh │ │ -> value │
    └─────────┘ └─────────┘
```

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip ratio epsilon | 0.2 |
| Rollout length | 2048 steps |
| Mini-batch size | 64 |
| Epochs per rollout | 10 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |

**PPO uses continuous actions** (`Box(-1, 1)`), which maps to paddle direction. This is more natural for PPO than discrete actions.

### 6.5 `agents/grpo_agent.py` — Group Relative Policy Optimization

GRPO (DeepSeek-R1, 2025) eliminates the critic network entirely. Instead of estimating a value baseline, it samples **G complete trajectories** from the current policy for each episode start state, computes each trajectory's total return, then uses the **group mean/std** as the baseline and normalization.

**Why GRPO for Breakout:**
- Breakout episodes are short (typically 200-2000 steps). Rolling out G full trajectories is feasible.
- No value network means fewer parameters, simpler code, and no value estimation bias.
- The group-relative baseline naturally adapts to the reward scale — no reward normalization needed.
- Particularly strong when reward signal is sparse (early training when agent rarely hits bricks).

**Architecture (policy only — no critic):**

```
obs (233,) -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU -> FC(4) -> softmax -> action probs
```

**Algorithm per iteration:**

```
For each iteration:
  1. Sample a batch of B start states (via env.reset())
  2. For each start state s_i, roll out G complete trajectories:
       τ_i^1, τ_i^2, ..., τ_i^G  using current policy π_θ
  3. Compute total return for each trajectory:
       R_i^j = Σ_t γ^t · r_t   for trajectory j from state i
  4. Compute group-relative advantage for each trajectory:
       Â_i^j = (R_i^j - mean(R_i^1..G)) / (std(R_i^1..G) + ε)
  5. For each (state, action) pair in each trajectory, compute:
       ratio = π_θ(a|s) / π_θ_old(a|s)
       L_clip = min(ratio · Â, clip(ratio, 1-ε, 1+ε) · Â)
  6. Update θ to maximize L_clip (+ entropy bonus)
```

**Key difference from PPO:** Steps 2-4 replace the GAE advantage estimation that requires a learned value function. The advantage comes purely from comparing sibling trajectories.

**Key hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Group size G | 8 | Trajectories per start state |
| Batch size B | 4 | Start states per iteration |
| Learning rate | 1e-4 | Adam optimizer |
| Discount gamma | 0.99 | |
| Clip ratio epsilon | 0.2 | Same as PPO |
| Entropy coefficient | 0.02 | Slightly higher to encourage exploration |
| Update epochs | 5 | PPO-style multi-epoch over collected data |
| Mini-batch size | 128 | Over all G*B trajectories' transitions |
| Max episode length | 2000 | Truncate long episodes |

**Training loop (pseudocode):**

```python
for iteration in range(num_iterations):
    all_trajectories = []
    
    for b in range(B):                        # B start states
        obs_init = env.reset()
        group_trajectories = []
        
        for g in range(G):                    # G rollouts per start
            env.reset_to(obs_init)            # replay from same state
            trajectory = rollout(policy, env) # collect full episode
            group_trajectories.append(trajectory)
        
        # Group-relative advantage
        returns = [sum(γ^t * r for t,r in τ) for τ in group_trajectories]
        mean_R, std_R = mean(returns), std(returns)
        
        for τ, R in zip(group_trajectories, returns):
            advantage = (R - mean_R) / (std_R + 1e-8)
            # Assign this scalar advantage to ALL (s,a) pairs in τ
            for transition in τ:
                transition.advantage = advantage
            all_trajectories.append(τ)
    
    # PPO-style clipped update over all collected transitions
    for epoch in range(update_epochs):
        for mini_batch in shuffle_and_batch(all_trajectories):
            ratio = π_θ(a|s) / π_θ_old(a|s)
            loss = -min(ratio * Â, clip(ratio) * Â) - β * entropy
            optimizer.step(loss)
```

**Implementation notes:**

- **Replaying from same start state**: `stepForAI` is deterministic given the same initial state and action sequence. We call `resetForAI()` before each of the G rollouts in a group. Since the game resets to level 1 with deterministic brick layout, all G trajectories start from the same state. (Ball launch has randomness, which provides natural trajectory diversity.)
- **Per-step vs per-trajectory advantage**: We assign the same group-relative advantage to every step within a trajectory. This is the standard GRPO formulation. An optional variant uses **per-step discounted returns** within each trajectory, still normalized at the group level.
- **Memory**: With G=8, B=4, and max 2000 steps each, worst case is 64k transitions per iteration — easily fits in memory.
- **Comparison with PPO**: GRPO trades sample efficiency (needs G rollouts per state) for simplicity and stability (no value function approximation error). For Breakout where episodes are fast via WebSocket, this tradeoff is favorable.

### 6.6 `state_processor.py`

```python
def process_state(raw_state: dict) -> np.ndarray:
    """Convert raw JSON game state to fixed-size float32 vector (233,)."""
    
    obs = np.zeros(233, dtype=np.float32)
    
    # Paddle: [0:2]
    obs[0] = raw_state["paddle"]["x"]
    obs[1] = raw_state["paddle"]["width"]
    
    # Ball: [2:7]
    ball = raw_state["ball"]
    obs[2] = ball["x"]
    obs[3] = ball["y"]
    obs[4] = (ball["vx"] + 1) / 2   # normalize from [-1,1] to [0,1]
    obs[5] = (ball["vy"] + 1) / 2
    obs[6] = 1.0 if ball["launched"] else 0.0
    
    # Brick grid: [7:231] — 14 cols x 16 rows = 224
    for brick in raw_state["bricks"]:
        col = int(brick["x"] * 14)   # approximate grid column
        row = int(brick["y"] * 16)   # approximate grid row
        col = min(col, 13)
        row = min(row, 15)
        obs[7 + row * 14 + col] = brick["hp"] / 10.0  # normalize HP
    
    # Meta: [231:233]
    obs[231] = raw_state["lives"] / 3.0
    total = max(1, raw_state["bricksRemaining"] + len(raw_state["bricks"]))
    obs[232] = raw_state["bricksRemaining"] / total
    
    return obs
```

### 6.7 `train.py` — CLI Entry Point

```bash
# Train DQN for 5000 episodes, headless
python train.py --agent dqn --episodes 5000 --headless

# Train PPO for 1M steps with reward shaping
python train.py --agent ppo --total-steps 1000000 --reward-shaping

# Train GRPO with group size 8, 2000 iterations
python train.py --agent grpo --iterations 2000 --group-size 8 --headless

# Resume from checkpoint
python train.py --agent dqn --resume checkpoints/dqn_ep2000.pt

# Evaluate a trained model
python evaluate.py --agent dqn --checkpoint checkpoints/dqn_best.pt --episodes 100

# Watch a trained agent play (renders in browser)
python play.py --agent dqn --checkpoint checkpoints/dqn_best.pt
```

**CLI arguments:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--agent` | `dqn\|ppo\|grpo` | `dqn` | Agent algorithm |
| `--episodes` | int | 3000 | Number of episodes (DQN) |
| `--total-steps` | int | 500000 | Total env steps (PPO) |
| `--iterations` | int | 2000 | Number of iterations (GRPO) |
| `--group-size` | int | 8 | Trajectories per start state (GRPO) |
| `--batch-states` | int | 4 | Start states per iteration (GRPO) |
| `--headless` | flag | false | Disable browser rendering |
| `--reward-shaping` | flag | false | Enable shaped rewards |
| `--frame-skip` | int | 4 | Frames per agent step |
| `--checkpoint-dir` | path | `./checkpoints` | Save location |
| `--save-freq` | int | 500 | Save checkpoint every N episodes |
| `--ws-url` | str | `ws://localhost:8765` | WebSocket server URL |
| `--resume` | path | null | Resume from checkpoint |
| `--seed` | int | 42 | Random seed |
| `--lr` | float | 1e-4/3e-4 | Learning rate |
| `--tensorboard` | flag | true | Enable TensorBoard logging |
| `--wandb` | flag | false | Enable W&B logging |
| `--wandb-project` | str | `breakout-rl` | W&B project name |
| `--wandb-entity` | str | null | W&B team/user name |
| `--wandb-name` | str | auto | W&B run name (default: `{agent}_{seed}_{timestamp}`) |
| `--wandb-tags` | str[] | [] | W&B tags for filtering (e.g., `--wandb-tags baseline v2`) |

---

## 7. Training Pipeline

### Step-by-step Setup

```bash
# Terminal 1: Start game dev server
cd breakout/
npm run dev
# -> http://localhost:3000

# Terminal 2: Start WebSocket relay server
cd ai-server/
pip install -r requirements.txt
python ws_server.py
# -> ws://localhost:8765

# Terminal 3: Open browser, connect to WS
# Visit http://localhost:3000
# In console: window.__BREAKOUT_AI.connect("ws://localhost:8765")

# Terminal 4: Start training
cd ai-server/
python train.py --agent dqn --episodes 5000 --headless
```

### Docker Compose (Production)

```yaml
services:
  breakout:
    build: .
    ports: ["8080:80"]

  ai-server:
    build: ./ai-server
    ports: ["8765:8765"]
    volumes:
      - ./ai-server/checkpoints:/app/checkpoints
      - ./ai-server/runs:/app/runs
    command: python ws_server.py

  trainer:
    build: ./ai-server
    depends_on: [ai-server, breakout]
    volumes:
      - ./ai-server/checkpoints:/app/checkpoints
      - ./ai-server/runs:/app/runs
    command: >
      python train.py
        --agent dqn
        --episodes 10000
        --headless
        --ws-url ws://ai-server:8765
```

### Headless / Fast-Forward Mode

When `--headless` is passed:
1. Python sends `{"type": "config", "data": {"renderEnabled": false}}` to disable canvas drawing
2. The game loop in `main.js` skips all `renderer.*` calls
3. `stepForAI()` runs at native JS speed (no `requestAnimationFrame` throttle)
4. Training speed: estimated ~500-2000 steps/sec depending on hardware

For even faster training, we can set `gameSpeed` multiplier to run multiple game ticks per WebSocket roundtrip. This is configured via:
```python
env.configure({"renderEnabled": False, "gameSpeed": 10})
```

---

## 8. Monitoring & Evaluation

### 8.1 Dual Logging Architecture: TensorBoard + W&B

We use **TensorBoard for local real-time monitoring** and **W&B for experiment management & comparison**. Both are optional and can be enabled independently.

```
train.py
  │
  ├── logger.py (unified interface)
  │     ├── TensorBoardWriter   (local, --tensorboard, default ON)
  │     ├── WandbWriter         (cloud, --wandb, default OFF)
  │     └── ConsoleWriter       (always on, summary every N episodes)
  │
  ├── runs/                     (TensorBoard logs, gitignored)
  └── wandb/                    (W&B local cache, gitignored)
```

### 8.2 Metrics

| Metric | Logged every | Description |
|--------|-------------|-------------|
| `train/episode_reward` | episode | Total reward per episode |
| `train/episode_length` | episode | Steps per episode |
| `train/score` | episode | Game score at episode end |
| `train/level` | episode | Highest level reached |
| `train/epsilon` | step | Exploration rate (DQN) |
| `train/loss` | step | Training loss |
| `train/q_mean` | step | Mean Q-value (DQN) |
| `train/policy_loss` | step | Policy loss (PPO/GRPO) |
| `train/value_loss` | step | Value loss (PPO only) |
| `train/entropy` | step | Policy entropy (PPO/GRPO) |
| `train/group_return_mean` | iteration | Mean group return (GRPO) |
| `train/group_return_std` | iteration | Return spread within groups (GRPO) |
| `eval/mean_reward` | eval | Mean reward over N episodes |
| `eval/mean_score` | eval | Mean game score |

### 8.3 TensorBoard (Local)

```bash
# Launch TensorBoard
tensorboard --logdir ai-server/runs/
# -> http://localhost:6006

# Compare multiple runs
tensorboard --logdir ai-server/runs/ --reload_interval 10
```

**Best for:** Real-time loss curves during training, quick local debugging, no account needed.

### 8.4 Weights & Biases (Cloud)

#### Setup

```bash
pip install wandb
wandb login   # one-time, paste API key from https://wandb.ai/authorize
```

#### Usage

```bash
# Enable W&B logging
python train.py --agent dqn --episodes 5000 --wandb

# With custom project/tags
python train.py --agent ppo --total-steps 1M \
  --wandb --wandb-project breakout-rl \
  --wandb-tags baseline ppo-v1

# Compare GRPO vs DQN
python train.py --agent grpo --iterations 2000 --wandb --wandb-tags grpo-v1
python train.py --agent dqn --episodes 5000 --wandb --wandb-tags dqn-v1
# -> Compare at https://wandb.ai/<entity>/breakout-rl
```

#### W&B Features Used

| Feature | Purpose |
|---------|---------|
| **Run Dashboard** | Real-time training curves (reward, loss, score) with smoothing |
| **Run Comparison** | Side-by-side DQN vs PPO vs GRPO on same chart |
| **Hyperparameter Tracking** | Auto-logs all CLI args + config; filterable in table view |
| **System Metrics** | GPU util, CPU, memory — auto-collected |
| **Media Logging** | Record agent gameplay videos every N eval episodes |
| **Sweeps** | Hyperparameter search (grid/random/bayesian) |
| **Artifacts** | Version model checkpoints, link to producing run |
| **Alerts** | Notify when reward plateaus or training diverges |

#### `logger.py` Implementation

```python
class Logger:
    """Unified logging to TensorBoard + W&B + console."""

    def __init__(self, config):
        self.writers = [ConsoleWriter(config.log_interval)]

        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writers.append(TensorBoardWriter(
                SummaryWriter(log_dir=f"runs/{config.run_name}")
            ))

        if config.wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_name or config.run_name,
                tags=config.wandb_tags,
                config=vars(config),          # auto-log all hyperparams
                save_code=True,               # snapshot source code
            )
            self.writers.append(WandbWriter())

    def log(self, metrics: dict, step: int):
        """Log scalar metrics to all active writers."""
        for w in self.writers:
            w.log(metrics, step)

    def log_video(self, frames: list, step: int, fps: int = 30):
        """Log gameplay video (W&B only)."""
        for w in self.writers:
            if hasattr(w, 'log_video'):
                w.log_video(frames, step, fps)

    def log_model(self, path: str, metadata: dict = None):
        """Log model checkpoint as artifact (W&B only)."""
        for w in self.writers:
            if hasattr(w, 'log_model'):
                w.log_model(path, metadata)

    def finish(self):
        for w in self.writers:
            w.finish()


class WandbWriter:
    def log(self, metrics, step):
        import wandb
        wandb.log(metrics, step=step)

    def log_video(self, frames, step, fps=30):
        import wandb
        import numpy as np
        video = wandb.Video(np.array(frames), fps=fps, format="mp4")
        wandb.log({"eval/gameplay": video}, step=step)

    def log_model(self, path, metadata=None):
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
```

#### W&B Sweeps (Hyperparameter Search)

```yaml
# sweep.yaml — example for DQN
program: train.py
method: bayes
metric:
  name: eval/mean_score
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  gamma:
    values: [0.95, 0.99, 0.995]
  batch_size:
    values: [32, 64, 128]
  hidden_dim:
    values: [128, 256, 512]
  epsilon_decay:
    distribution: uniform
    min: 0.99
    max: 0.9999
command:
  - ${env}
  - python
  - ${program}
  - --agent=dqn
  - --episodes=3000
  - --wandb
  - ${args}
```

```bash
# Create and run sweep
wandb sweep sweep.yaml          # returns sweep_id
wandb agent <sweep_id>          # starts training with sampled hyperparams
# Launch multiple agents in parallel for faster search
```

### 8.5 Gameplay Video Recording

During evaluation, record agent gameplay for visual inspection:

```python
# In evaluate.py
def record_episode(env, agent, logger, step):
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frame = env.render()        # get canvas frame via WS
        frames.append(frame)
        action = agent.act(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
    logger.log_video(frames, step)  # uploads to W&B
```

Videos are logged at:
- Every `--eval-freq` episodes during training
- All episodes during `evaluate.py` and `play.py`
- Viewable in W&B Media panel with scrubbing and frame-by-frame

### 8.6 Checkpoint Saving

- Auto-save every `--save-freq` episodes: `checkpoints/{agent}_ep{N}.pt`
- Best model (highest mean eval reward): `checkpoints/{agent}_best.pt`
- Checkpoint contains: model weights, optimizer state, training step, epsilon (DQN), hyperparameters
- When `--wandb` is enabled, checkpoints are also logged as **W&B Artifacts** with full lineage (linked to producing run, hyperparams, metrics at save time)

### 8.7 Evaluation Protocol

`evaluate.py` runs N episodes with no exploration (epsilon=0 for DQN, deterministic mode for PPO/GRPO) and reports:
- Mean/std episode reward
- Mean/std game score
- Mean/std level reached
- Max score achieved

### 8.8 Recommended Workflow

```
1. Local development:    --tensorboard (default)
   └─ Quick iteration, real-time curves at localhost:6006

2. Serious training:     --tensorboard --wandb
   └─ Full tracking, cloud dashboard, experiment comparison

3. Hyperparameter tuning: wandb sweep sweep.yaml
   └─ Bayesian search across lr, gamma, hidden_dim, etc.

4. Final comparison:     W&B Run Comparison view
   └─ Side-by-side DQN vs PPO vs GRPO with same eval protocol
```

---

## 9. Expected Training Timeline

| Milestone | DQN (episodes) | PPO (env steps) | GRPO (iterations x G) | Expected Behavior |
|-----------|----------------|-----------------|----------------------|-------------------|
| Random | ~200 | ~10k | ~50 x 8 | Agent launches ball, moves randomly |
| Tracking | ~500 | ~30k | ~100 x 8 | Paddle begins following ball X position |
| First bricks | ~1500 | ~80k | ~250 x 8 | Intentionally breaks bricks |
| 30% clear | ~3000 | ~200k | ~400 x 8 | Consistent paddle-ball contact |
| Level 1 clear | ~5000 | ~400k | ~600 x 8 | Clears level 1 reliably |
| Multi-level | ~10000+ | ~1M+ | ~1500+ x 8 | Strategic play, multiple levels |

These are rough estimates. Actual convergence depends heavily on hyperparameters, reward shaping, and frame skip.

---

## 10. Algorithm Comparison

| Property | DQN | PPO | GRPO |
|----------|-----|-----|------|
| **Action space** | Discrete(4) | Continuous [-1,1] | Discrete(4) |
| **Value function** | Q-network | Learned critic V(s) | None (group baseline) |
| **Experience** | Off-policy (replay buffer) | On-policy (rollout buffer) | On-policy (group rollouts) |
| **Sample efficiency** | High (replay reuse) | Medium | Low (G rollouts per state) |
| **Wall-clock per iteration** | Fast (single step + batch) | Medium (rollout + epochs) | Slow (G full episodes + epochs) |
| **Implementation complexity** | Medium | High (GAE, dual loss) | Low (no value net, simpler code) |
| **Stability** | Can diverge (overestimation) | Stable (clipping + GAE) | Very stable (relative baseline) |
| **Hyperparameter sensitivity** | High (lr, buffer, epsilon) | Medium | Low (mainly G and lr) |
| **Sparse reward handling** | Poor (needs shaped rewards) | Fair | Good (group comparison) |
| **Best for** | Fast iteration, baseline | Continuous control, production | Sparse reward, simplicity |

**Recommended starting point:** DQN for quick iteration, then GRPO for cleaner training signal. PPO as the production-quality middle ground once hyperparameters are tuned.

---

## 11. Future Extensions (Not in Scope Now)

- **Image-based observations**: Canvas screenshot via CNN feature extractor. Requires `canvas.toDataURL()` + base64 transfer over WS. Slower but more general.
- **Self-play / curriculum**: Start on later levels, train level-specific policies, then combine.
- **Prioritized experience replay** (PER): Weight samples by TD error for DQN.
- **Recurrent policy (LSTM)**: Handle partial observability if using image obs.
- **Multi-agent**: Multiple browser tabs training in parallel, sharing a replay buffer.
- **Model export to ONNX**: Run trained model directly in the browser via ONNX.js, no Python needed.
- **GRPO with per-step advantages**: Instead of assigning the same advantage to all steps in a trajectory, use per-step discounted returns normalized at the group level — hybrid between GRPO and GAE.
