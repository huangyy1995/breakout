# Breakout Game - Neon Bricks

A modern, mobile-friendly HTML5 Canvas Breakout game featuring a neon-glow visual design, 50 dynamic levels, and built-in Gym-like AI interfaces for Deep Learning and Reinforcement Learning (RL) agents.

🎮 **Live Demo:** [https://breakout-game-282666957438.asia-east1.run.app](https://breakout-game-282666957438.asia-east1.run.app)

---

## 🌟 Key Features

- **Premium Visuals:** Neon-glow aesthetics, animated backgrounds, multi-hit brick HP, gradient particles, and indestructible metallic obstacles.
- **Mobile First:** Unified `InputManager` supports desktop keyboard tracking, mouse dragging, and mobile touch. Includes a native Fullscreen API wrapper & portrait orientation lock.
- **50-Level Progression:** Dynamic difficulty spanning 50 levels. Featuring 10 unique brick patterns (Diamonds, Checkerboards, Pyramids, U/V shapes) and challenging multi-hit/invincible boss levels.
- **Gym-Style AI Interface:** Completely decoupled game state logic. Exposes normalized properties (paddle coords, ball coords/velocities, brick matrix) entirely between `[0, 1]` for pristine tensor ingestion.
- **Containerized:** Multi-stage Dockerfile packaging Vite + Node build into a lightweight Nginx Alpine container. Ready for Google Cloud Run (Serverless).

---

## 🚀 Local Development

The project uses Vanilla JS + Vite, so there's no bulky frameworks.

```bash
# Install dependencies
npm install

# Start development server with Hot Module Replacement (HMR)
npm run dev

# Build for production
npm run build
```
Once the dev server is running, visit `http://localhost:3000`.

### Debugging Cheat Codes
- **Press `N`** during gameplay to instantly clear all bricks and skip to the next level.

---

## 🐋 Docker & Deployment

The codebase is pre-configured to be deployed via Docker or directly mapped to Google Cloud Run.

**Run via Docker Compose:**
```bash
docker-compose up -d
```
Then visit `http://localhost:8080`.

**Deploy to Google GCP Cloud Run (Source-based):**
```bash
gcloud run deploy breakout-game \
  --source . \
  --region asia-east1 \
  --allow-unauthenticated \
  --port 80
```

---

## 🧠 AI / Reinforcement Learning Interface

This game engine is built specifically with future AI integration in mind. 
It supports both direct browser console injection and a `WebSocketBridge` for Python scripts.

Open your browser console and type:
```javascript
const ai = window.__BREAKOUT_AI;

// 1. Get current state (Returns normalized JSON dict)
console.log(ai.getState());

// 2. Gym-like Reset
const initialState = ai.reset();

// 3. Gym-like Step
// Action type: 'move' (direction: -1 to 1) or 'position' (position: 0 to 1) 
// Returns { state, reward, done, info }
const result = ai.step({ type: 'move', direction: 0.8 });
console.log(result.reward, result.done);
```

---

## 🔮 Future Plan & Roadmap

As this repository evolves, the following features are planned for sequential implementation:

### Phase 1: Game Polish
- [ ] Add sound effects for paddle hits, brick breaks, and level complete.
- [ ] Add glowing temporary power-ups (e.g., elongated paddle, multiple balls).

### Phase 2: Python RL Integration (Deep Learning)
- [ ] Establish a standalone Python WebSocket server (`ai-server` in docker-compose).
- [ ] Write a PyTorch-based Deep Q-Network (DQN) or Proximal Policy Optimization (PPO) agent.
- [ ] Train the AI to play Breakout by piping `window.__BREAKOUT_AI` steps directly into the Python training loop.
- [ ] Allow tweaking rendering speed (`ai.configure({ renderEnabled: false, gameSpeed: 10 })`) for headless high-speed training.

### Phase 3: Leaderboards & Cloud
- [ ] Integrate Firebase or Google Cloud SQL to track player high scores versus the AI model's high score.
- [ ] Setup CI/CD pipelines in Google Cloud Build upon Git push.
