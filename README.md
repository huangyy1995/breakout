# Breakout Game - Neon Bricks

A modern, mobile-friendly HTML5 Canvas Breakout game featuring neon-glow visuals, 50 dynamic levels, power-up system, procedural audio, and built-in Gym-like AI interfaces for Reinforcement Learning agents.

🎮 **Live Demo:** [https://breakout-game-282666957438.asia-east1.run.app](https://breakout-game-282666957438.asia-east1.run.app)

---

## 🌟 Key Features

- **Premium Visuals:** Neon-glow aesthetics, animated grid background, gradient particles, multi-hit brick HP indicators, and indestructible metallic obstacles.
- **Mobile First:** Relative touch control (delta-based dragging), raised paddle position for unobstructed view, touch zone guide bar. Desktop supports keyboard and mouse.
- **50-Level Progression:** 10 unique brick patterns (Diamonds, Checkerboards, Pyramids, U/V shapes, Boss blocks) with scaling difficulty — more columns, more rows, higher HP, smaller paddle, faster ball.
- **Power-Up System:** Three collectible drops from destroyed bricks:
  - 🔵 **+BALL** — launches a new ball from the paddle
  - 🟣 **SPLIT** — every active ball clones itself (mirrored direction)
  - 🔴 **+LIFE** — gain one extra life
- **Multi-Ball:** Up to 8 balls on screen simultaneously. When the primary ball falls but extras exist, an extra is promoted — no life lost.
- **Procedural Audio:** All sound effects and background music generated via Web Audio API (zero audio files). Pentatonic ambient BGM with independent music/SFX toggle buttons.
- **Mobile Cheat:** Rapidly tap the upper-left corner 5 times to skip the current level (hidden cheat code).
- **Gym-Style AI Interface:** Normalized game state, `reset()` / `step(action)` API, WebSocket bridge for Python RL agents.
- **Containerized:** Multi-stage Docker build (Vite + Nginx Alpine), ready for Google Cloud Run.

---

## 🚀 Local Development

Vanilla JS + Vite — no frameworks.

```bash
# Install dependencies
npm install

# Start development server (HMR)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

Dev server runs at `http://localhost:3000`.

### Cheat Codes
- **Press `N`** (desktop) — instantly clear all bricks and skip to next level.
- **Tap upper-left corner 5× rapidly** (mobile) — same effect.

---

## 📁 Project Structure

```
breakout/
├── index.html                     # Single-page entry point
├── vite.config.js                 # Vite configuration
├── package.json                   # Dependencies (only vite)
├── Dockerfile / docker-compose.yml / nginx.conf / deploy.sh
│
└── src/
    ├── main.js                    # Bootstrap, game loop, event wiring
    ├── engine/
    │   ├── Game.js                # Core game state machine, update loop
    │   ├── Physics.js             # Ball↔Paddle and Ball↔Brick collision
    │   ├── levels/
    │   │   └── LevelManager.js    # Level progression & difficulty curves
    │   └── entities/
    │       ├── Ball.js            # Ball movement, wall bounce, trail
    │       ├── Brick.js           # Brick + BrickGrid (10 patterns)
    │       ├── Paddle.js          # Paddle (keyboard/touch/AI control)
    │       └── PowerUp.js         # Falling power-up items
    ├── input/
    │   └── InputManager.js        # Unified keyboard/touch/mouse input
    ├── ui/
    │   ├── Renderer.js            # Canvas rendering (neon aesthetic)
    │   ├── Menu.js                # UI screen/HUD management
    │   ├── ParticleSystem.js      # Brick break & paddle hit particles
    │   ├── SoundManager.js        # Web Audio API SFX + BGM
    │   └── FullscreenManager.js   # Fullscreen/orientation handling
    ├── ai/
    │   ├── AIController.js        # AI interface wrapper
    │   └── WebSocketBridge.js     # Remote AI via WebSocket
    └── styles/
        └── index.css              # All styling (screens, HUD, animations)
```

---

## 🐋 Docker & Deployment

```bash
# Run locally via Docker Compose
docker-compose up -d
# → http://localhost:8080

# Deploy to Google Cloud Run
gcloud run deploy breakout-game \
  --source . \
  --region asia-east1 \
  --allow-unauthenticated \
  --port 80
```

---

## 🧠 AI / Reinforcement Learning Interface

Open the browser console:

```javascript
const ai = window.__BREAKOUT_AI;

// Gym-like Reset → returns initial state
const s0 = ai.reset();

// Gym-like Step → returns { state, reward, done, info }
const result = ai.step({ type: 'position', position: 0.5 });

// Configure for headless training
ai.configure({ renderEnabled: false, gameSpeed: 10 });
```

**Action types:**
- `{ type: 'move', direction: -1..1 }` — relative paddle movement
- `{ type: 'position', position: 0..1 }` — absolute paddle position
- `{ type: 'launch' }` — launch ball

**State:** All positions/velocities normalized to `[0, 1]`. Includes paddle, ball, brick grid, score, lives, level.

**Rewards:** +1 per brick destroyed, +50 level complete, −5 life lost.

---

## 🔮 Roadmap

### Phase 1: Game Polish ✅
- [x] Sound effects (paddle, brick, wall, power-up, life lost, level complete, game over)
- [x] Background music (procedural pentatonic ambient)
- [x] Power-up system (+BALL, SPLIT, +LIFE)
- [x] Multi-ball support
- [x] Mobile touch optimization (delta control, raised paddle, touch bar)
- [x] Music/SFX toggle buttons
- [x] Hidden mobile cheat code

### Phase 2: Python RL Integration
- [ ] Python WebSocket server for training loop
- [ ] DQN / PPO / GRPO agent implementation (PyTorch)
- [ ] Headless high-speed training mode

### Phase 3: Leaderboards & Cloud
- [ ] Firebase / Cloud SQL high-score tracking
- [ ] CI/CD via Cloud Build
