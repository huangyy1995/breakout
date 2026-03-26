/**
 * Physics module - collision detection for ball ↔ paddle and ball ↔ bricks.
 */
export class Physics {
  /**
   * Check and resolve ball-paddle collision.
   * Returns true if collision occurred.
   * @param {import('./entities/Ball.js').Ball} ball
   * @param {import('./entities/Paddle.js').Paddle} paddle
   * @returns {boolean}
   */
  static ballPaddleCollision(ball, paddle) {
    if (ball.vy < 0) return false; // Ball moving upward

    const bx = ball.x;
    const by = ball.y + ball.radius;

    if (
      by >= paddle.y &&
      by <= paddle.y + paddle.height &&
      bx >= paddle.x &&
      bx <= paddle.x + paddle.width
    ) {
      // Calculate hit position relative to paddle center (-1 to 1)
      const hitPos = (bx - paddle.centerX) / (paddle.width / 2);

      // Angle: ranges from -60° to -120° based on hit position
      const angle = -Math.PI / 2 + hitPos * (Math.PI / 3);

      ball.vx = Math.cos(angle) * ball.speed;
      ball.vy = Math.sin(angle) * ball.speed;

      // Ensure ball is above paddle
      ball.y = paddle.y - ball.radius - 1;

      return true;
    }
    return false;
  }

  /**
   * Check and resolve ball-brick collisions.
   * Returns array of destroyed bricks for scoring/particles.
   * @param {import('./entities/Ball.js').Ball} ball
   * @param {import('./entities/Brick.js').BrickGrid} brickGrid
   * @returns {import('./entities/Brick.js').Brick[]}
   */
  static ballBrickCollisions(ball, brickGrid) {
    const destroyed = [];

    for (const brick of brickGrid.bricks) {
      if (!brick.alive) continue;

      // AABB vs circle collision
      const closestX = Math.max(brick.x, Math.min(ball.x, brick.x + brick.width));
      const closestY = Math.max(brick.y, Math.min(ball.y, brick.y + brick.height));

      const dx = ball.x - closestX;
      const dy = ball.y - closestY;
      const distSq = dx * dx + dy * dy;

      if (distSq < ball.radius * ball.radius) {
        // Determine collision side
        const overlapLeft = ball.x + ball.radius - brick.x;
        const overlapRight = brick.x + brick.width - (ball.x - ball.radius);
        const overlapTop = ball.y + ball.radius - brick.y;
        const overlapBottom = brick.y + brick.height - (ball.y - ball.radius);

        const minOverlapX = Math.min(overlapLeft, overlapRight);
        const minOverlapY = Math.min(overlapTop, overlapBottom);

        if (minOverlapX < minOverlapY) {
          ball.vx = -ball.vx;
          ball.x += ball.vx > 0 ? minOverlapX : -minOverlapX;
        } else {
          ball.vy = -ball.vy;
          ball.y += ball.vy > 0 ? minOverlapY : -minOverlapY;
        }

        const wasDestroyed = brick.hit();
        if (wasDestroyed) {
          destroyed.push(brick);
        }

        // Speed up slightly on each hit
        ball.increaseSpeed(1.005);

        break; // Only one brick collision per frame
      }
    }

    return destroyed;
  }
}
