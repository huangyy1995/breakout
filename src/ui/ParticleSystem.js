/**
 * ParticleSystem - visual effects for brick destruction, ball trail, etc.
 */
export class ParticleSystem {
  constructor() {
    this.particles = [];
  }

  /**
   * Spawn particles from a destroyed brick.
   * @param {import('../engine/entities/Brick.js').Brick} brick
   */
  spawnBrickBreak(brick) {
    const cx = brick.x + brick.width / 2;
    const cy = brick.y + brick.height / 2;
    const count = 12;

    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 / count) * i + Math.random() * 0.5;
      const speed = 80 + Math.random() * 160;
      this.particles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0.6 + Math.random() * 0.4,
        maxLife: 0.6 + Math.random() * 0.4,
        size: 2 + Math.random() * 4,
        color: brick.color.fill,
        type: 'square',
      });
    }
  }

  /**
   * Spawn sparkle on paddle hit.
   * @param {number} x
   * @param {number} y
   */
  spawnPaddleHit(x, y) {
    for (let i = 0; i < 6; i++) {
      const angle = -Math.PI / 2 + (Math.random() - 0.5) * Math.PI;
      const speed = 60 + Math.random() * 100;
      this.particles.push({
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0.3 + Math.random() * 0.2,
        maxLife: 0.3 + Math.random() * 0.2,
        size: 1.5 + Math.random() * 2,
        color: '#00f0ff',
        type: 'circle',
      });
    }
  }

  /**
   * Update all particles.
   * @param {number} dt
   */
  update(dt) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vy += 200 * dt; // gravity
      p.life -= dt;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }
  }

  /**
   * Draw all particles.
   * @param {CanvasRenderingContext2D} ctx
   */
  draw(ctx) {
    for (const p of this.particles) {
      const alpha = Math.max(0, p.life / p.maxLife);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = p.color;

      if (p.type === 'square') {
        const s = p.size * alpha;
        ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
      } else {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * alpha, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  /** Clear all particles */
  clear() {
    this.particles = [];
  }
}
