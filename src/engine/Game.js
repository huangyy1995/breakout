/**
 * Game - Main game class orchestrating the breakout game loop.
 */
import { Paddle } from './entities/Paddle.js';
import { Ball } from './entities/Ball.js';
import { BrickGrid } from './entities/Brick.js';
import { Physics } from './Physics.js';
import { LevelManager } from './levels/LevelManager.js';
import { PowerUp, PowerUpType } from './entities/PowerUp.js';

/** Drop chance per destroyed brick (0–1). */
const POWERUP_DROP_CHANCE = 0.15;

/** Cap on how many extra balls can exist at once. */
const MAX_EXTRA_BALLS = 7;

/** Game states */
export const GameState = {
  MENU: 'MENU',
  PLAYING: 'PLAYING',
  PAUSED: 'PAUSED',
  GAME_OVER: 'GAME_OVER',
  LEVEL_COMPLETE: 'LEVEL_COMPLETE',
};

export class Game {
  /**
   * @param {number} canvasWidth - logical canvas width
   * @param {number} canvasHeight - logical canvas height
   */
  constructor(canvasWidth, canvasHeight) {
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;

    // State
    this.state = GameState.MENU;
    this.score = 0;
    this.lives = 3;
    this.maxLives = 3;
    this.combo = 0;

    // Entities
    this.paddle = new Paddle({ canvasWidth, canvasHeight });
    this.ball = new Ball({ canvasWidth, canvasHeight });
    this.brickGrid = new BrickGrid();
    this.levelManager = new LevelManager();

    /** @type {Ball[]} Additional balls spawned by power-ups. */
    this.extraBalls = [];
    /** @type {PowerUp[]} Active falling power-ups. */
    this.powerUps = [];

    // Callbacks
    this.onBrickDestroyed = null; // (brick) => {}
    this.onBrickHit = null;       // (brick) => {} — hit but not destroyed
    this.onPaddleHit = null;      // (x, y) => {}
    this.onWallHit = null;        // () => {}
    this.onLifeLost = null;       // () => {}
    this.onStateChange = null;    // (newState) => {}
    this.onScoreChange = null;    // (score, level, lives) => {}
    this.onPowerUpCollected = null; // (powerUp) => {}

    // AI mode
    this.aiControlled = false;
    this._pendingReward = 0;
  }

  /** Start a new game from level 1 */
  start() {
    this.score = 0;
    this.lives = this.maxLives;
    this.combo = 0;
    this.levelManager.reset();
    this._setupLevel();
    this._setState(GameState.PLAYING);
  }

  /** Resume from pause */
  resume() {
    if (this.state === GameState.PAUSED) {
      this._setState(GameState.PLAYING);
    }
  }

  /** Pause the game */
  pause() {
    if (this.state === GameState.PLAYING) {
      this._setState(GameState.PAUSED);
    }
  }

  /** Quit to menu */
  quit() {
    this._setState(GameState.MENU);
  }

  /** Advance to next level */
  nextLevel() {
    this.levelManager.nextLevel();
    this._setupLevel();
    this._setState(GameState.PLAYING);
  }

  /** Setup current level */
  _setupLevel() {
    const level = this.levelManager.getLevel();
    this.brickGrid.generate(level, this.canvasWidth, this.canvasHeight);

    // Apply difficulty modifiers
    this.ball.reset();
    this.ball.baseSpeed = this.canvasHeight * 0.55 * this.levelManager.getSpeedMultiplier();
    this.ball.speed = this.ball.baseSpeed;

    this.paddle.reset();
    this.paddle.width = this.canvasWidth * 0.18 * this.levelManager.getPaddleWidthMultiplier();

    this.ball.attachToPaddle(this.paddle);
    this.combo = 0;

    // Clear any leftover multi-ball state
    this.extraBalls = [];
    this.powerUps = [];
  }

  /**
   * Update game logic.
   * @param {number} dt - delta time in seconds
   * @param {import('../input/InputManager.js').InputManager|null} input - input manager
   */
  update(dt, input) {
    if (this.state !== GameState.PLAYING) return;

    // Clamp dt to prevent tunneling on lag
    dt = Math.min(dt, 1 / 30);

    // Handle input
    if (input && !this.aiControlled) {
      if (input.isTouchDelta) {
        // Relative/delta touch control: move paddle by accumulated drag distance
        const dx = input.consumeDelta();
        this.paddle.x = Math.max(0, Math.min(
          this.canvasWidth - this.paddle.width,
          this.paddle.x + dx
        ));
        this.paddle.targetX = this.paddle.x;
        this.paddle.update(dt, false); // clamp + glow only
      } else if (input.isPositionBased) {
        this.paddle.setTargetX(input.targetX);
        this.paddle.update(dt, true);
      } else {
        this.paddle.setDirection(input.direction);
        this.paddle.update(dt, false);
      }

      if (input.consumeLaunch()) {
        this.ball.launch();
      }
      if (input.consumePause()) {
        this.pause();
        return;
      }
    } else if (this.aiControlled) {
      // AI updates paddle directly
      this.paddle.update(dt, true);
    }

    // --- Primary ball ---
    const ballWallHit = this.ball.update(dt, this.paddle);
    if (ballWallHit && this.onWallHit) this.onWallHit();

    if (this.ball.launched && Physics.ballPaddleCollision(this.ball, this.paddle)) {
      this.combo = 0;
      if (this.onPaddleHit) this.onPaddleHit(this.ball.x, this.paddle.y);
    }

    const { destroyed, hit } = Physics.ballBrickCollisions(this.ball, this.brickGrid);
    if (hit && this.onBrickHit) this.onBrickHit(hit);
    for (const brick of destroyed) {
      this.combo++;
      this.score += 10 * this.combo;
      this._pendingReward += 1;
      this._maybeDropPowerUp(brick);
      if (this.onBrickDestroyed) this.onBrickDestroyed(brick);
    }

    // --- Extra balls ---
    for (let i = this.extraBalls.length - 1; i >= 0; i--) {
      const eb = this.extraBalls[i];
      const ebWallHit = eb.update(dt, this.paddle);
      if (ebWallHit && this.onWallHit) this.onWallHit();

      if (eb.launched && Physics.ballPaddleCollision(eb, this.paddle)) {
        this.combo = 0;
        if (this.onPaddleHit) this.onPaddleHit(eb.x, this.paddle.y);
      }

      const { destroyed: extraDestroyed, hit: extraHit } = Physics.ballBrickCollisions(eb, this.brickGrid);
      if (extraHit && this.onBrickHit) this.onBrickHit(extraHit);
      for (const brick of extraDestroyed) {
        this.combo++;
        this.score += 10 * this.combo;
        this._pendingReward += 1;
        this._maybeDropPowerUp(brick);
        if (this.onBrickDestroyed) this.onBrickDestroyed(brick);
      }

      if (eb.isBelowScreen()) {
        this.extraBalls.splice(i, 1);
      }
    }

    // --- Power-ups ---
    for (let i = this.powerUps.length - 1; i >= 0; i--) {
      const pu = this.powerUps[i];
      pu.update(dt);

      if (pu.isBelowScreen(this.canvasHeight)) {
        this.powerUps.splice(i, 1);
      } else if (pu.collidesPaddle(this.paddle)) {
        this._collectPowerUp(pu.type);
        this.powerUps.splice(i, 1);
        if (this.onPowerUpCollected) this.onPowerUpCollected(pu);
      }
    }

    // Update bricks animation
    this.brickGrid.update(dt);

    // Notify score change
    if (destroyed.length > 0 && this.onScoreChange) {
      this.onScoreChange(this.score, this.levelManager.getLevel(), this.lives);
    }

    // Check primary ball fell
    if (this.ball.isBelowScreen()) {
      if (this.extraBalls.length > 0) {
        // Promote first extra ball — no life lost
        const next = this.extraBalls.shift();
        this.ball.x = next.x;
        this.ball.y = next.y;
        this.ball.vx = next.vx;
        this.ball.vy = next.vy;
        this.ball.speed = next.speed;
        this.ball.launched = true;
        this.ball.trail = [];
      } else {
        this.lives--;
        this._pendingReward -= 5;
        this.combo = 0;
        this.powerUps = [];

        if (this.onLifeLost) this.onLifeLost();
        if (this.onScoreChange) {
          this.onScoreChange(this.score, this.levelManager.getLevel(), this.lives);
        }

        if (this.lives <= 0) {
          this._setState(GameState.GAME_OVER);
        } else {
          this.ball.reset();
          this.ball.attachToPaddle(this.paddle);
        }
      }
    }

    // Check level complete
    if (this.brickGrid.allDestroyed()) {
      this._pendingReward += 50;
      this._setState(GameState.LEVEL_COMPLETE);
    }
  }

  /**
   * Maybe drop a power-up when a brick is destroyed.
   * @param {import('./entities/Brick.js').Brick} brick
   */
  _maybeDropPowerUp(brick) {
    if (Math.random() > POWERUP_DROP_CHANCE) return;
    const r = Math.random();
    const type = r < 0.34
      ? PowerUpType.MULTI_BALL
      : r < 0.67
        ? PowerUpType.SPLIT_BALL
        : PowerUpType.EXTRA_LIFE;
    this.powerUps.push(new PowerUp(brick.x + brick.width / 2, brick.y + brick.height / 2, type));
  }

  /**
   * Activate a collected power-up.
   * @param {string} type - PowerUpType value
   */
  _collectPowerUp(type) {
    const activeBalls = [this.ball, ...this.extraBalls].filter(b => b.launched);

    if (type === PowerUpType.MULTI_BALL) {
      // Spawn a new ball from the paddle (not from current ball position)
      if (this.extraBalls.length >= MAX_EXTRA_BALLS) return;
      const nb = new Ball({ canvasWidth: this.canvasWidth, canvasHeight: this.canvasHeight });
      nb.baseSpeed = this.ball.baseSpeed;
      nb.maxSpeed  = this.ball.maxSpeed;
      nb.speed     = this.ball.speed || this.ball.baseSpeed;
      nb.x = this.paddle.centerX;
      nb.y = this.paddle.y - nb.radius - 2;
      nb.glowColor = '#00f0ff';
      nb.midColor  = '#00c8ff';
      nb.launched  = true;
      const angle  = -Math.PI / 2 + (Math.random() - 0.5) * Math.PI * 0.7;
      nb.vx = Math.cos(angle) * nb.speed;
      nb.vy = -Math.abs(Math.sin(angle) * nb.speed);
      this.extraBalls.push(nb);

    } else if (type === PowerUpType.SPLIT_BALL) {
      if (activeBalls.length === 0) {
        this._collectPowerUp(PowerUpType.MULTI_BALL);
        return;
      }
      const slots = MAX_EXTRA_BALLS - this.extraBalls.length;
      const toSplit = activeBalls.slice(0, slots);
      for (const b of toSplit) {
        const nb = this._spawnExtraBall(b, 'green');
        nb.vx = -b.vx; // mirror horizontally
        nb.vy = b.vy;
        this.extraBalls.push(nb);
      }

    } else if (type === PowerUpType.EXTRA_LIFE) {
      if (this.lives < this.maxLives) {
        this.lives++;
        if (this.onScoreChange) {
          this.onScoreChange(this.score, this.levelManager.getLevel(), this.lives);
        }
      }
    }
  }

  /**
   * Create a new Ball cloned from a source ball.
   * @param {Ball} source
   * @param {'cyan'|'green'} tint
   * @returns {Ball}
   */
  _spawnExtraBall(source, tint) {
    const nb = new Ball({ canvasWidth: this.canvasWidth, canvasHeight: this.canvasHeight });
    nb.baseSpeed = source.baseSpeed;
    nb.maxSpeed = source.maxSpeed;
    nb.speed = source.speed || source.baseSpeed;
    nb.x = source.x;
    nb.y = source.y;
    nb.launched = true;
    if (tint === 'cyan') {
      nb.glowColor = '#00f0ff';
      nb.midColor = '#00c8ff';
    } else {
      nb.glowColor = '#00ff88';
      nb.midColor = '#00d870';
    }
    return nb;
  }

  /** Set game state and fire callback */
  _setState(newState) {
    this.state = newState;
    if (this.onStateChange) {
      this.onStateChange(newState);
    }
  }

  /**
   * Resize game entities when canvas changes.
   * @param {number} newWidth
   * @param {number} newHeight
   */
  resize(newWidth, newHeight) {
    const scaleX = newWidth / this.canvasWidth;
    const scaleY = newHeight / this.canvasHeight;

    this.canvasWidth = newWidth;
    this.canvasHeight = newHeight;

    // Scale paddle
    this.paddle.canvasWidth = newWidth;
    this.paddle.canvasHeight = newHeight;
    this.paddle.x *= scaleX;
    this.paddle.y = newHeight - 110;
    this.paddle.speed = newWidth * 0.8;

    // Scale primary ball
    this.ball.canvasWidth = newWidth;
    this.ball.canvasHeight = newHeight;
    this.ball.x *= scaleX;
    this.ball.y *= scaleY;
    this.ball.baseSpeed = newHeight * 0.55 * this.levelManager.getSpeedMultiplier();
    this.ball.maxSpeed = newHeight * 0.9;

    // Scale extra balls
    for (const eb of this.extraBalls) {
      eb.canvasWidth = newWidth;
      eb.canvasHeight = newHeight;
      eb.x *= scaleX;
      eb.y *= scaleY;
      eb.baseSpeed = newHeight * 0.55 * this.levelManager.getSpeedMultiplier();
      eb.maxSpeed = newHeight * 0.9;
    }

    // Regenerate bricks at new size
    if (this.state === GameState.PLAYING) {
      this.brickGrid.generate(this.levelManager.getLevel(), newWidth, newHeight);
    }
  }

  // =========================================
  // AI Control Interface
  // =========================================

  /**
   * Get complete game state as JSON (normalized for AI).
   * All positions in [0, 1] range.
   */
  getState() {
    return {
      paddle: this.paddle.getState(),
      ball: this.ball.getState(),
      bricks: this.brickGrid.getState(this.canvasWidth, this.canvasHeight),
      score: this.score,
      lives: this.lives,
      level: this.levelManager.getLevel(),
      gameState: this.state,
      bricksRemaining: this.brickGrid.aliveCount(),
    };
  }

  /**
   * Apply an action from AI.
   * @param {object} action
   * @param {string} action.type - 'move' | 'launch' | 'position'
   * @param {number} [action.direction] - for 'move': -1 to 1
   * @param {number} [action.position] - for 'position': 0 to 1 (normalized)
   */
  applyAction(action) {
    if (!action) return;

    switch (action.type) {
      case 'move':
        this.paddle.setNormalizedDirection(action.direction || 0);
        break;
      case 'position':
        this.paddle.setNormalizedPosition(action.position || 0.5);
        break;
      case 'launch':
        this.ball.launch();
        break;
    }
  }

  /**
   * Gym-like reset: restart game and return initial state.
   * @returns {object} initial state
   */
  resetForAI() {
    this.aiControlled = true;
    this.start();
    this._pendingReward = 0;
    return this.getState();
  }

  /**
   * Gym-like step: apply action, advance one frame, return results.
   * @param {object} action
   * @param {number} [dt=1/60] - time step
   * @returns {{ state: object, reward: number, done: boolean, info: object }}
   */
  stepForAI(action, dt = 1 / 60) {
    this._pendingReward = 0;

    this.applyAction(action);
    this.paddle.update(dt, action.type === 'position');
    this.ball.update(dt, this.paddle);

    // Run physics
    if (this.ball.launched && Physics.ballPaddleCollision(this.ball, this.paddle)) {
      this.combo = 0;
    }

    const { destroyed } = Physics.ballBrickCollisions(this.ball, this.brickGrid);
    for (const brick of destroyed) {
      this.combo++;
      this.score += 10 * this.combo;
      this._pendingReward += 1;
    }

    this.brickGrid.update(dt);

    if (this.ball.isBelowScreen()) {
      this.lives--;
      this._pendingReward -= 5;
      this.combo = 0;
      if (this.lives <= 0) {
        this._setState(GameState.GAME_OVER);
      } else {
        this.ball.reset();
        this.ball.attachToPaddle(this.paddle);
      }
    }

    if (this.brickGrid.allDestroyed()) {
      this._pendingReward += 50;
      this._setState(GameState.LEVEL_COMPLETE);
    }

    const done = this.state === GameState.GAME_OVER || this.state === GameState.LEVEL_COMPLETE;

    return {
      state: this.getState(),
      reward: this._pendingReward,
      done,
      info: {
        score: this.score,
        lives: this.lives,
        level: this.levelManager.getLevel(),
        bricksRemaining: this.brickGrid.aliveCount(),
      },
    };
  }
}
