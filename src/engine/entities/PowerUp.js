/**
 * Power-up entity — drops from destroyed bricks and falls toward the paddle.
 */

export const PowerUpType = {
  MULTI_BALL: 'multi_ball',  // Spawn one additional ball
  SPLIT_BALL: 'split_ball',  // Every active ball splits into two
};

export const POWERUP_STYLES = {
  [PowerUpType.MULTI_BALL]: { fill: '#00ff88', glow: 'rgba(0, 255, 136, 0.9)', label: '+BALL' },
  [PowerUpType.SPLIT_BALL]: { fill: '#a855f7', glow: 'rgba(168, 85, 247, 0.9)', label: 'SPLIT' },
};

export class PowerUp {
  /**
   * @param {number} x - center X
   * @param {number} y - center Y
   * @param {string} type - PowerUpType value
   */
  constructor(x, y, type) {
    this.x = x;
    this.y = y;
    this.type = type;
    this.width = 46;
    this.height = 18;
    this.speed = 115; // pixels per second (downward)
    this.glowTime = 0; // for pulsing animation
  }

  /** @param {number} dt */
  update(dt) {
    this.y += this.speed * dt;
    this.glowTime += dt;
  }

  /** @param {number} canvasHeight */
  isBelowScreen(canvasHeight) {
    return this.y - this.height / 2 > canvasHeight;
  }

  /**
   * @param {import('./Paddle.js').Paddle} paddle
   * @returns {boolean}
   */
  collidesPaddle(paddle) {
    return (
      this.y + this.height / 2 >= paddle.y &&
      this.y - this.height / 2 <= paddle.y + paddle.height &&
      this.x + this.width / 2 >= paddle.x &&
      this.x - this.width / 2 <= paddle.x + paddle.width
    );
  }
}
