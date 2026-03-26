/**
 * Main entry point - bootstraps the Breakout game.
 */
import { Game, GameState } from './engine/Game.js';
import { Renderer } from './ui/Renderer.js';
import { InputManager } from './input/InputManager.js';
import { ParticleSystem } from './ui/ParticleSystem.js';
import { Menu } from './ui/Menu.js';
import { FullscreenManager } from './ui/FullscreenManager.js';
import { AIController } from './ai/AIController.js';
import { WebSocketBridge } from './ai/WebSocketBridge.js';
import './styles/index.css';

// =========================================
// Bootstrap
// =========================================

const canvas = document.getElementById('game-canvas');
const renderer = new Renderer(canvas);
const { width, height } = renderer.getDimensions();

const game = new Game(width, height);
const input = new InputManager(canvas);
const particles = new ParticleSystem();
const menu = new Menu();
const fullscreen = new FullscreenManager();
const aiController = new AIController(game);
const wsBridge = new WebSocketBridge(aiController);

// =========================================
// Expose AI interface globally for testing
// =========================================

window.__BREAKOUT_AI = {
  getState: () => aiController.getState(),
  applyAction: (action) => aiController.applyAction(action),
  reset: () => aiController.reset(),
  step: (action, dt) => aiController.step(action, dt),
  configure: (config) => aiController.configure(config),
  connect: (url) => wsBridge.connect(url),
  disconnect: () => wsBridge.disconnect(),
  getConfig: () => aiController.getConfig(),
};

// =========================================
// UI Button handlers
// =========================================

document.getElementById('btn-start').addEventListener('click', () => {
  game.start();
  // Try fullscreen on mobile
  if ('ontouchstart' in window) {
    fullscreen.request();
  }
});

document.getElementById('btn-fullscreen').addEventListener('click', () => {
  fullscreen.toggle();
});

document.getElementById('btn-resume').addEventListener('click', () => {
  game.resume();
});

document.getElementById('btn-quit').addEventListener('click', () => {
  game.quit();
});

document.getElementById('btn-restart').addEventListener('click', () => {
  game.start();
});

document.getElementById('btn-next-level').addEventListener('click', () => {
  game.nextLevel();
});

document.getElementById('btn-pause').addEventListener('click', () => {
  game.pause();
});

// =========================================
// Game callbacks
// =========================================

game.onStateChange = (newState) => {
  switch (newState) {
    case GameState.MENU:
      menu.showStart();
      break;
    case GameState.PLAYING:
      menu.showPlaying();
      menu.updateHUD(game.score, game.levelManager.getLevel(), game.lives);
      break;
    case GameState.PAUSED:
      menu.showPause();
      break;
    case GameState.GAME_OVER:
      menu.showGameOver(game.score);
      break;
    case GameState.LEVEL_COMPLETE:
      menu.showLevelComplete(game.score);
      break;
  }
};

game.onBrickDestroyed = (brick) => {
  particles.spawnBrickBreak(brick);
};

game.onPaddleHit = (x, y) => {
  particles.spawnPaddleHit(x, y);
};

game.onLifeLost = () => {
  menu.shakeScreen();
};

game.onScoreChange = (score, level, lives) => {
  menu.updateHUD(score, level, lives);
};

game.onPowerUpCollected = (powerUp) => {
  // Re-use paddle-hit particles as a quick visual burst at the paddle
  particles.spawnPaddleHit(powerUp.x, game.paddle.y);
};

// Mobile cheat: rapidly tap upper-right corner 5× to skip the current level.
// Uses window-level listener (see InputManager) so it fires even when the HUD
// overlay sits on top of the canvas in that area.
input.onCheatActivated = () => {
  if (game.state === GameState.PLAYING) {
    game.brickGrid.bricks.forEach(b => { b.alive = false; });
  } else if (game.state === GameState.PAUSED) {
    // Resume first so the next update() sees PLAYING and processes level-complete.
    game.brickGrid.bricks.forEach(b => { b.alive = false; });
    game.resume();
  }
};

// =========================================
// Handle resize
// =========================================

let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    renderer.resize();
    const { width: newW, height: newH } = renderer.getDimensions();
    game.resize(newW, newH);
  }, 100);
});

// =========================================
// Debug / Cheat keys
// =========================================

window.addEventListener('keydown', (e) => {
  // Press 'N' to clear all bricks and instantly win the level
  if (e.key === 'n' || e.key === 'N') {
    if (game.state === GameState.PLAYING) {
      game.brickGrid.bricks.forEach(b => b.alive = false);
    }
  }
});

// =========================================
// Game Loop
// =========================================

let lastTime = 0;

function gameLoop(timestamp) {
  const dt = lastTime ? (timestamp - lastTime) / 1000 : 1 / 60;
  lastTime = timestamp;

  // Update
  game.update(dt, input);
  particles.update(dt);

  // Render (skip if AI disables rendering)
  if (!aiController.enabled || aiController.renderEnabled) {
    renderer.clear(dt);
    renderer.drawBricks(game.brickGrid);
    renderer.drawPaddle(game.paddle);

    if (game.state === GameState.PLAYING || game.ball.launched) {
      renderer.drawBall(game.ball);
      for (const eb of game.extraBalls) {
        renderer.drawBall(eb);
      }
    }

    renderer.drawPowerUps(game.powerUps);
    renderer.drawCheatIndicator(input.cornerTapCount, input.cornerTapRequired);

    particles.draw(renderer.ctx);
  }

  requestAnimationFrame(gameLoop);
}

// Start
menu.showStart();
requestAnimationFrame(gameLoop);
