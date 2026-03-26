/**
 * Renderer - Canvas rendering with neon-glow aesthetic.
 */
import { BRICK_COLORS } from '../engine/entities/Brick.js';
import { POWERUP_STYLES } from '../engine/entities/PowerUp.js';

export class Renderer {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.width = 0;
    this.height = 0;

    // Background animation
    this.bgTime = 0;
    this.gridOffset = 0;

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  /** Resize canvas to fill container */
  resize() {
    const dpr = window.devicePixelRatio || 1;
    const container = this.canvas.parentElement;
    const w = container.clientWidth;
    const h = container.clientHeight;

    // Use a logical resolution that works well on mobile
    // Keep aspect ratio ~9:16 for mobile, but adapt
    this.width = 420;
    this.height = this.width * (h / w);

    // Multiply actual backing store pixel count by DPR for high-DPI sharpness
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    
    // Scale context so logical drawing commands match the backing store
    this.ctx.scale(dpr, dpr);

    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
  }

  /** Get logical dimensions */
  getDimensions() {
    return { width: this.width, height: this.height };
  }

  /** Clear canvas with background */
  clear(dt) {
    const ctx = this.ctx;

    // Dark gradient background
    const grad = ctx.createLinearGradient(0, 0, 0, this.height);
    grad.addColorStop(0, '#0a0a1a');
    grad.addColorStop(0.5, '#0d0d2b');
    grad.addColorStop(1, '#0a0a1a');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, this.width, this.height);

    // Animated grid
    this.bgTime += dt;
    this.gridOffset = (this.gridOffset + dt * 15) % 40;

    ctx.strokeStyle = 'rgba(0, 240, 255, 0.04)';
    ctx.lineWidth = 0.5;

    // Vertical grid lines
    for (let x = 0; x < this.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
      ctx.stroke();
    }

    // Horizontal grid lines (scrolling)
    for (let y = -40 + this.gridOffset; y < this.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
      ctx.stroke();
    }
  }

  /**
   * Draw the paddle.
   * @param {import('../engine/entities/Paddle.js').Paddle} paddle
   */
  drawPaddle(paddle) {
    const ctx = this.ctx;

    // Glow
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 15 + paddle.glowIntensity * 10;

    // Gradient fill
    const grad = ctx.createLinearGradient(paddle.x, paddle.y, paddle.x + paddle.width, paddle.y);
    grad.addColorStop(0, '#00c8ff');
    grad.addColorStop(0.5, '#00f0ff');
    grad.addColorStop(1, '#00c8ff');

    ctx.fillStyle = grad;

    // Rounded rect
    const r = paddle.cornerRadius;
    ctx.beginPath();
    ctx.roundRect(paddle.x, paddle.y, paddle.width, paddle.height, r);
    ctx.fill();

    // Top highlight
    ctx.shadowBlur = 0;
    const highlightGrad = ctx.createLinearGradient(paddle.x, paddle.y, paddle.x, paddle.y + paddle.height);
    highlightGrad.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
    highlightGrad.addColorStop(0.5, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = highlightGrad;
    ctx.beginPath();
    ctx.roundRect(paddle.x, paddle.y, paddle.width, paddle.height / 2, [r, r, 0, 0]);
    ctx.fill();

    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  /**
   * Draw the ball with trail.
   * @param {import('../engine/entities/Ball.js').Ball} ball
   */
  drawBall(ball) {
    const ctx = this.ctx;
    const glow = ball.glowColor || '#ff2d95';
    const mid = ball.midColor || '#ff6eb4';

    // Trail
    for (const t of ball.trail) {
      ctx.globalAlpha = t.alpha * 0.6;
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(t.x, t.y, ball.radius * t.alpha, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Ball glow
    ctx.shadowColor = glow;
    ctx.shadowBlur = 20;

    // Ball gradient
    const grad = ctx.createRadialGradient(
      ball.x - 2, ball.y - 2, 0,
      ball.x, ball.y, ball.radius
    );
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(0.4, mid);
    grad.addColorStop(1, glow);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }

  /**
   * Draw falling power-ups.
   * @param {import('../engine/entities/PowerUp.js').PowerUp[]} powerUps
   */
  drawPowerUps(powerUps) {
    const ctx = this.ctx;
    for (const pu of powerUps) {
      const style = POWERUP_STYLES[pu.type];
      const pulse = Math.sin(pu.glowTime * 7) * 0.25 + 0.75;

      ctx.save();
      ctx.shadowColor = style.glow;
      ctx.shadowBlur = 14 * pulse;

      const rx = pu.width / 2;
      const ry = pu.height / 2;

      ctx.globalAlpha = 0.93;
      ctx.fillStyle = style.fill;
      ctx.beginPath();
      ctx.roundRect(pu.x - rx, pu.y - ry, pu.width, pu.height, ry);
      ctx.fill();

      // Inner label
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 9px Orbitron, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(style.label, pu.x, pu.y);

      ctx.restore();
    }
  }

  /**
   * Draw upper-right cheat-tap progress indicator.
   * @param {number} count - current tap count
   * @param {number} [required=5]
   */
  drawCheatIndicator(count, required = 5) {
    if (count <= 0) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = '#ffe048';
    ctx.font = 'bold 10px Orbitron, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.shadowColor = 'rgba(255, 224, 72, 0.8)';
    ctx.shadowBlur = 8;
    ctx.fillText(`⚡ ${count}/${required}`, this.width - 6, 6);
    ctx.restore();
  }

  /**
   * Draw bricks.
   * @param {import('../engine/entities/Brick.js').BrickGrid} brickGrid
   */
  drawBricks(brickGrid) {
    const ctx = this.ctx;

    for (const brick of brickGrid.bricks) {
      if (!brick.alive && brick.scale <= 0) continue;

      ctx.save();

      if (brick.scale < 1) {
        const cx = brick.x + brick.width / 2;
        const cy = brick.y + brick.height / 2;
        ctx.translate(cx, cy);
        ctx.scale(brick.scale, brick.scale);
        ctx.translate(-cx, -cy);
      }

      // Glow
      ctx.shadowColor = brick.color.glow;
      ctx.shadowBlur = 8 + brick.hitFlash * 15;

      // Fill
      const flashMix = brick.hitFlash;
      if (flashMix > 0) {
        ctx.fillStyle = '#ffffff';
        ctx.globalAlpha = flashMix;
        ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
        ctx.globalAlpha = 1 - flashMix;
      }

      // Gradient fill
      if (brick.isIndestructible) {
        ctx.shadowColor = 'rgba(200, 200, 200, 0.3)';
        const grad = ctx.createLinearGradient(brick.x, brick.y, brick.x, brick.y + brick.height);
        grad.addColorStop(0, '#b0b0b0');
        grad.addColorStop(1, '#505050');
        ctx.fillStyle = grad;
        ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
        
        // Add a metallic cross pattern
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(brick.x, brick.y);
        ctx.lineTo(brick.x + brick.width, brick.y + brick.height);
        ctx.moveTo(brick.x + brick.width, brick.y);
        ctx.lineTo(brick.x, brick.y + brick.height);
        ctx.stroke();
      } else {
        const grad = ctx.createLinearGradient(brick.x, brick.y, brick.x, brick.y + brick.height);
        grad.addColorStop(0, brick.color.fill);
        grad.addColorStop(1, this._darken(brick.color.fill, 0.3));
        ctx.fillStyle = grad;
        ctx.fillRect(brick.x, brick.y, brick.width, brick.height);
      }
      ctx.globalAlpha = 1;

      // Border
      ctx.shadowBlur = 0;
      ctx.strokeStyle = brick.isIndestructible ? 'rgba(0,0,0,0.8)' : 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = brick.isIndestructible ? 1.5 : 0.5;
      ctx.strokeRect(brick.x, brick.y, brick.width, brick.height);

      // HP indicator for multi-hit bricks
      if (!brick.isIndestructible && brick.hitPoints > 1) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '10px Orbitron, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          brick.hitPoints.toString(),
          brick.x + brick.width / 2,
          brick.y + brick.height / 2
        );
      }

      ctx.shadowColor = 'transparent';
      ctx.restore();
    }
  }

  /**
   * Darken a hex color.
   * @param {string} hex
   * @param {number} amount - 0..1
   * @returns {string}
   */
  _darken(hex, amount) {
    const num = parseInt(hex.replace('#', ''), 16);
    const r = Math.max(0, ((num >> 16) & 0xff) * (1 - amount)) | 0;
    const g = Math.max(0, ((num >> 8) & 0xff) * (1 - amount)) | 0;
    const b = Math.max(0, (num & 0xff) * (1 - amount)) | 0;
    return `rgb(${r},${g},${b})`;
  }
}
