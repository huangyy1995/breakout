/**
 * Ball entity for the Breakout game.
 */
export class Ball {
  /**
   * @param {object} config
   * @param {number} config.canvasWidth
   * @param {number} config.canvasHeight
   */
  constructor({ canvasWidth, canvasHeight }) {
    this.canvasWidth = canvasWidth;
    this.canvasHeight = canvasHeight;

    this.radius = 7;
    this.baseSpeed = canvasHeight * 0.55;
    this.maxSpeed = canvasHeight * 0.9;
    this.speed = this.baseSpeed;

    // Position & velocity
    this.x = canvasWidth / 2;
    this.y = canvasHeight - 60;
    this.vx = 0;
    this.vy = 0;

    // State
    this.launched = false;
    this.trail = []; // for visual trail effect

    // Visual tinting — override for extra balls
    this.glowColor = '#ff2d95';
    this.midColor = '#ff6eb4';
  }

  /**
   * Attach ball to paddle before launch.
   * @param {import('./Paddle.js').Paddle} paddle
   */
  attachToPaddle(paddle) {
    this.x = paddle.centerX;
    this.y = paddle.y - this.radius - 2;
    this.vx = 0;
    this.vy = 0;
    this.launched = false;
  }

  /** Launch the ball upward with slight random angle */
  launch() {
    if (this.launched) return;
    this.launched = true;
    const angle = -Math.PI / 2 + (Math.random() - 0.5) * 0.6;
    this.vx = Math.cos(angle) * this.speed;
    this.vy = Math.sin(angle) * this.speed;
  }

  /**
   * Update ball position.
   * @param {number} dt - delta time in seconds
   * @param {import('./Paddle.js').Paddle} paddle - for attachment when not launched
   */
  update(dt, paddle) {
    if (!this.launched) {
      this.attachToPaddle(paddle);
      return;
    }

    // Store trail position
    this.trail.push({ x: this.x, y: this.y, alpha: 1 });
    if (this.trail.length > 12) this.trail.shift();

    // Update trail alpha
    this.trail.forEach((t, i) => {
      t.alpha = (i + 1) / this.trail.length * 0.5;
    });

    // Move
    this.x += this.vx * dt;
    this.y += this.vy * dt;

    // Wall collisions (left, right, top)
    if (this.x - this.radius <= 0) {
      this.x = this.radius;
      this.vx = Math.abs(this.vx);
    }
    if (this.x + this.radius >= this.canvasWidth) {
      this.x = this.canvasWidth - this.radius;
      this.vx = -Math.abs(this.vx);
    }
    if (this.y - this.radius <= 0) {
      this.y = this.radius;
      this.vy = Math.abs(this.vy);
    }
  }

  /** Check if ball fell below the screen */
  isBelowScreen() {
    return this.y - this.radius > this.canvasHeight;
  }

  /** Increase speed slightly (for difficulty) */
  increaseSpeed(factor = 1.02) {
    this.speed = Math.min(this.maxSpeed, this.speed * factor);
    const currentSpeed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    if (currentSpeed > 0) {
      const ratio = this.speed / currentSpeed;
      this.vx *= ratio;
      this.vy *= ratio;
    }
  }

  /** Get normalized state for AI */
  getState() {
    return {
      x: this.x / this.canvasWidth,
      y: this.y / this.canvasHeight,
      vx: this.vx / this.maxSpeed,
      vy: this.vy / this.maxSpeed,
      launched: this.launched,
    };
  }

  /** Reset ball */
  reset() {
    this.speed = this.baseSpeed;
    this.launched = false;
    this.trail = [];
    this.x = this.canvasWidth / 2;
    this.y = this.canvasHeight - 60;
    this.vx = 0;
    this.vy = 0;
    this.glowColor = '#ff2d95';
    this.midColor = '#ff6eb4';
  }
}
