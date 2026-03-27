/**
 * Power-up entity — drops from destroyed bricks and falls toward the paddle.
 */

export const PowerUpType = {
  MULTI_BALL:  'multi_ball',   // New ball launched from paddle
  SPLIT_BALL:  'split_ball',   // Each active ball clones itself
  EXTRA_LIFE:  'extra_life',   // Gain one life (up to maxLives)
};

export const POWERUP_STYLES = {
  [PowerUpType.MULTI_BALL]: {
    fill: '#00f0ff', glow: 'rgba(0, 240, 255, 0.9)', label: '+BALL',
  },
  [PowerUpType.SPLIT_BALL]: {
    fill: '#a855f7', glow: 'rgba(168, 85, 247, 0.9)', label: 'SPLIT',
  },
  [PowerUpType.EXTRA_LIFE]: {
    fill: '#ff2d95', glow: 'rgba(255, 45, 149, 0.9)', label: '+LIFE',
  },
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
    this.width = 48;
    this.height = 20;
    this.speed = 110; // pixels per second downward
    this.glowTime = 0;
  }

  update(dt) {
    this.y += this.speed * dt;
    this.glowTime += dt;
  }

  isBelowScreen(canvasHeight) {
    return this.y - this.height / 2 > canvasHeight;
  }

  collidesPaddle(paddle) {
    return (
      this.y + this.height / 2 >= paddle.y &&
      this.y - this.height / 2 <= paddle.y + paddle.height &&
      this.x + this.width / 2 >= paddle.x &&
      this.x - this.width / 2 <= paddle.x + paddle.width
    );
  }
}
