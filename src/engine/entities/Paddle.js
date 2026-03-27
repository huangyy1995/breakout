/**
 * Paddle entity for the Breakout game.
 * Represents the player-controlled paddle at the bottom of the screen.
 */
export class Paddle {
  /**
   * @param {object} config
   * @param {number} config.canvasWidth - logical canvas width
   * @param {number} config.canvasHeight - logical canvas height
   */
  constructor({ canvasWidth, canvasHeight }) {
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;

    // Dimensions (relative to canvas)
    this.width = canvasWidth * 0.18;
    this.height = 14;
    this.cornerRadius = 4;

    // Position — raised higher so the finger rests below the paddle on mobile
    this.x = (canvasWidth - this.width) / 2;
    this.y = canvasHeight - 110;

    // Movement
    this.speed = canvasWidth * 0.8; // pixels per second
    this.targetX = this.x;
    this.direction = 0; // -1 left, 0 none, 1 right

    // Visual
    this.glowIntensity = 0;
  }

  /**
   * Set direct position (for touch/mouse input).
   * @param {number} x - target center x position
   */
  setTargetX(x) {
    this.targetX = x - this.width / 2;
  }

  /**
   * Set keyboard direction.
   * @param {number} dir - -1 (left), 0 (stop), 1 (right)
   */
  setDirection(dir) {
    this.direction = dir;
  }

  /**
   * Set normalized direction from AI.
   * @param {number} normalized - value in [-1, 1]
   */
  setNormalizedDirection(normalized) {
    this.direction = Math.max(-1, Math.min(1, normalized));
  }

  /**
   * Set normalized target position from AI.
   * @param {number} normalized - value in [0, 1]
   */
  setNormalizedPosition(normalized) {
    this.targetX = normalized * (this.canvasWidth - this.width);
  }

  /**
   * Update paddle position.
   * @param {number} dt - delta time in seconds
   * @param {boolean} useTarget - whether to use targetX (touch mode) or direction (keyboard mode)
   */
  update(dt, useTarget = false) {
    if (useTarget) {
      // Smooth interpolation toward target
      const diff = this.targetX - this.x;
      const step = this.speed * 1.5 * dt;
      if (Math.abs(diff) < step) {
        this.x = this.targetX;
      } else {
        this.x += Math.sign(diff) * step;
      }
    } else if (this.direction !== 0) {
      this.x += this.direction * this.speed * dt;
    }

    // Clamp to bounds
    this.x = Math.max(0, Math.min(this.canvasWidth - this.width, this.x));

    // Update glow
    this.glowIntensity = Math.abs(this.direction) > 0 ? 1 : Math.max(0, this.glowIntensity - dt * 3);
  }

  /** Get center X */
  get centerX() {
    return this.x + this.width / 2;
  }

  /** Get center Y */
  get centerY() {
    return this.y + this.height / 2;
  }

  /** Get normalized state for AI */
  getState() {
    return {
      x: this.x / this.canvasWidth,
      y: this.y / this.canvasHeight,
      width: this.width / this.canvasWidth,
    };
  }

  /** Reset to center */
  reset() {
    this.x = (this.canvasWidth - this.width) / 2;
    this.targetX = this.x;
    this.direction = 0;
  }
}
