/**
 * Brick entity for the Breakout game.
 */

/** Color palette for brick rows */
const BRICK_COLORS = [
  { fill: '#ff2d95', glow: 'rgba(255, 45, 149, 0.6)' },   // Pink
  { fill: '#ff8c00', glow: 'rgba(255, 140, 0, 0.6)' },     // Orange
  { fill: '#ffe048', glow: 'rgba(255, 224, 72, 0.6)' },     // Yellow
  { fill: '#00ff88', glow: 'rgba(0, 255, 136, 0.6)' },      // Green
  { fill: '#00f0ff', glow: 'rgba(0, 240, 255, 0.6)' },      // Cyan
  { fill: '#a855f7', glow: 'rgba(168, 85, 247, 0.6)' },     // Purple
];

export class Brick {
  /**
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {number} hitPoints
   * @param {number} colorIndex
   * @param {boolean} isIndestructible
   */
  constructor(x, y, width, height, hitPoints = 1, colorIndex = 0, isIndestructible = false) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.hitPoints = hitPoints;
    this.maxHitPoints = hitPoints;
    this.alive = true;
    this.isIndestructible = isIndestructible;
    this.colorIndex = colorIndex % BRICK_COLORS.length;
    this.color = BRICK_COLORS[this.colorIndex];

    // Animation
    this.hitFlash = 0;
    this.scale = 1;
  }

  /** Hit the brick, returns true if destroyed */
  hit() {
    if (!this.alive) return false;
    this.hitFlash = 1;

    if (this.isIndestructible) return false;

    this.hitPoints--;
    if (this.hitPoints <= 0) {
      this.alive = false;
      return true;
    }
    return false;
  }

  /** Update animation state */
  update(dt) {
    if (this.hitFlash > 0) {
      this.hitFlash = Math.max(0, this.hitFlash - dt * 5);
    }
    if (!this.alive && this.scale > 0) {
      this.scale = Math.max(0, this.scale - dt * 8);
    }
  }
}

export class BrickGrid {
  constructor() {
    this.bricks = [];
    this.rows = 0;
    this.cols = 0;
  }

  /**
   * Generate brick layout for a level.
   * @param {number} level - current level (1-based)
   * @param {number} canvasWidth
   * @param {number} canvasHeight
   */
  generate(level, canvasWidth, canvasHeight) {
    this.bricks = [];

    const padding = 6;
    const topOffset = canvasHeight * 0.12;
    const sideMargin = 12;
    const availWidth = canvasWidth - sideMargin * 2;

    // Scale up the max visible rows and cols gradually over many levels
    this.cols = Math.min(6 + Math.floor(level / 3), 14);
    this.rows = Math.min(4 + Math.floor(level / 2), 16);

    const brickWidth = (availWidth - padding * (this.cols - 1)) / this.cols;
    const brickHeight = Math.min(22, (canvasHeight * 0.45 - topOffset) / this.rows);

    // HP multiplier scaling with level tiers
    const baseHp = 1 + Math.floor((level - 1) / 5);
    const patternType = level % 10;

    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        const x = sideMargin + col * (brickWidth + padding);
        const y = topOffset + row * (brickHeight + padding);

        // Higher rows naturally have a bit more HP
        let hp = baseHp;
        if (row < 3) hp += 1;

        let shouldPlace = true;
        let isIndestructible = false;

        switch (patternType) {
          case 1: // Full block
            if (row === 0 && col === Math.floor(this.cols / 2) && level > 5) isIndestructible = true;
            break;
          case 2: // Checkerboard
            shouldPlace = (row + col) % 2 === 0;
            break;
          case 3: // Pyramid
            shouldPlace = row <= col && row < this.cols - col;
            if (row === this.rows - 1 && col === Math.floor(this.cols / 2)) isIndestructible = true; // Tip is solid
            break;
          case 4: // Diamond
            const centerCol = Math.floor(this.cols / 2);
            const centerRow = Math.floor(this.rows / 2);
            const dist = Math.abs(col - centerCol) + Math.abs(row - centerRow);
            shouldPlace = dist <= Math.max(centerCol, centerRow);
            if (dist === 0) hp += 3; // Center is tough
            if (dist === 1 && level > 5) isIndestructible = true; // Shielding the center
            break;
          case 5: // X shape
            shouldPlace = col === row || col === (this.cols - 1 - row);
            if (row === Math.floor(this.rows / 2) && level > 5) isIndestructible = true;
            break;
          case 6: // Striped columns
            shouldPlace = col % 2 === 0;
            if (row === 0 && level > 5) isIndestructible = true;
            break;
          case 7: // Hollow box
            shouldPlace = row === 0 || row === this.rows - 1 || col === 0 || col === this.cols - 1;
            hp += 2; // Outer shell is tough
            break;
          case 8: // V shape
            const distToCenter = Math.abs(col - Math.floor(this.cols / 2));
            shouldPlace = row >= distToCenter && row - distToCenter < 3;
            if (row === this.rows - 1 && level > 5) isIndestructible = true;
            break;
          case 9: // U shape
            shouldPlace = col === 0 || col === this.cols - 1 || row === this.rows - 1;
            break;
          case 0: // Boss block (multiples of 10)
            hp += 3;
            if (level > 10 && (row === 0 || row === this.rows - 1) && col % 3 === 0) {
              isIndestructible = true;
            }
            break;
        }

        if (shouldPlace) {
          this.bricks.push(new Brick(x, y, brickWidth, brickHeight, hp, row + col, isIndestructible));
        }
      }
    }
  }

  /** Check if all bricks are destroyed */
  allDestroyed() {
    return this.bricks.filter(b => !b.isIndestructible).every(b => !b.alive);
  }

  /** Get alive bricks count */
  aliveCount() {
    return this.bricks.filter(b => !b.isIndestructible && b.alive).length;
  }

  /** Update all brick animations */
  update(dt) {
    this.bricks.forEach(b => b.update(dt));
  }

  /** Get normalized state for AI (2D grid: 0 = destroyed, >0 = hit points) */
  getState(canvasWidth, canvasHeight) {
    return this.bricks.map(b => ({
      x: b.x / canvasWidth,
      y: b.y / canvasHeight,
      w: b.width / canvasWidth,
      h: b.height / canvasHeight,
      hp: b.alive ? b.hitPoints : 0,
    }));
  }
}

export { BRICK_COLORS };
