/**
 * Level manager - defines brick layouts and difficulty progression.
 */
export class LevelManager {
  constructor() {
    this.currentLevel = 1;
    this.maxLevel = 50;
  }

  /** Get current level number */
  getLevel() {
    return this.currentLevel;
  }

  /** Advance to next level. Returns false if already at max. */
  nextLevel() {
    if (this.currentLevel >= this.maxLevel) {
      // Loop back with increased difficulty modifier
      this.currentLevel = 1;
      return true;
    }
    this.currentLevel++;
    return true;
  }

  /** Reset to level 1 */
  reset() {
    this.currentLevel = 1;
  }

  /** Get ball speed multiplier for current level */
  getSpeedMultiplier() {
    return 1 + (this.currentLevel - 1) * 0.08;
  }

  /** Get paddle width multiplier (shrinks with level) */
  getPaddleWidthMultiplier() {
    return Math.max(0.4, 1 - (this.currentLevel - 1) * 0.05);
  }
}

