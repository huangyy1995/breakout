# Breakout RL — AI Training Server

Python-based reinforcement learning training system for the Breakout game.

## Quick Start

### Prerequisites

- Python 3.11+
- The Breakout game running (see parent README)
- A browser window open to the game

### Setup

```bash
cd ai-server/

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Terminal 1: Start game dev server
cd breakout/
npm run dev

# Terminal 2: Start WebSocket relay server
cd ai-server/
python ws_server.py

# Terminal 3: Open browser, connect to WS
# Visit http://localhost:3000
# In console: window.__BREAKOUT_AI.connect("ws://localhost:8765")

# Terminal 4: Start training
cd ai-server/
python train.py --agent dqn --episodes 5000 --headless
```

### Agents

| Agent | Command |
|-------|---------|
| DQN | `python train.py --agent dqn --episodes 5000` |
| PPO | `python train.py --agent ppo --total-steps 1000000` |
| GRPO | `python train.py --agent grpo --iterations 2000 --group-size 8` |

### Evaluation

```bash
python evaluate.py --agent dqn --checkpoint checkpoints/dqn_best.pt --episodes 100
```

### Watch Agent Play

```bash
python play.py --agent dqn --checkpoint checkpoints/dqn_best.pt
```

### Monitoring

```bash
# TensorBoard (local)
tensorboard --logdir runs/

# W&B (cloud) — add --wandb flag during training
python train.py --agent dqn --episodes 5000 --wandb
```

## Project Structure

```
ai-server/
├── ws_server.py          # WebSocket relay server
├── ws_env.py             # Gymnasium-compatible environment
├── state_processor.py    # Game state → observation vector
├── train.py              # Main training CLI
├── evaluate.py           # Model evaluation
├── play.py               # Watch trained agent play
├── agents/
│   ├── base.py           # Abstract agent interface
│   ├── dqn_agent.py      # DQN with replay buffer
│   ├── ppo_agent.py      # PPO actor-critic
│   ├── grpo_agent.py     # GRPO group relative policy
│   └── replay_buffer.py  # Experience replay
├── networks/
│   └── mlp.py            # MLP architectures
├── utils/
│   ├── logger.py         # TensorBoard + W&B logging
│   └── config.py         # Hyperparameter configs
├── checkpoints/          # Saved models (gitignored)
└── runs/                 # TensorBoard logs (gitignored)
```

## CLI Reference

See `python train.py --help` for full argument list.

Key flags:
- `--agent {dqn,ppo,grpo}` — Algorithm to use
- `--headless` — Disable browser rendering for faster training
- `--reward-shaping` — Enable shaped rewards
- `--resume PATH` — Resume from checkpoint
- `--wandb` — Enable W&B logging
- `--tensorboard` / `--no-tensorboard` — Toggle TensorBoard
