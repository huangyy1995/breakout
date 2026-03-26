/**
 * Menu - manages UI screen transitions and HUD updates.
 */
export class Menu {
  constructor() {
    // Screens
    this.startScreen = document.getElementById('start-screen');
    this.pauseScreen = document.getElementById('pause-screen');
    this.gameoverScreen = document.getElementById('gameover-screen');
    this.levelCompleteScreen = document.getElementById('level-complete-screen');
    this.hud = document.getElementById('hud');

    // HUD values
    this.scoreEl = document.getElementById('score-value');
    this.levelEl = document.getElementById('level-value');
    this.livesEl = document.getElementById('lives-value');

    // Score displays
    this.finalScoreEl = document.getElementById('final-score');
    this.levelScoreEl = document.getElementById('level-score');

    this._allScreens = [
      this.startScreen,
      this.pauseScreen,
      this.gameoverScreen,
      this.levelCompleteScreen,
    ];
  }

  /** Hide all screens */
  hideAll() {
    this._allScreens.forEach(s => s.classList.remove('active'));
  }

  /** Show a specific screen */
  show(screen) {
    this.hideAll();
    screen.classList.add('active');
  }

  /** Show start screen */
  showStart() {
    this.show(this.startScreen);
    this.hud.classList.remove('active');
  }

  /** Show pause screen */
  showPause() {
    this.show(this.pauseScreen);
  }

  /** Show game over screen */
  showGameOver(score) {
    this.finalScoreEl.textContent = score;
    this.show(this.gameoverScreen);
  }

  /** Show level complete screen */
  showLevelComplete(score) {
    this.levelScoreEl.textContent = score;
    this.show(this.levelCompleteScreen);
  }

  /** Start playing - hide screens, show HUD */
  showPlaying() {
    this.hideAll();
    this.hud.classList.add('active');
  }

  /** Update HUD values */
  updateHUD(score, level, lives) {
    this.scoreEl.textContent = score;
    this.levelEl.textContent = level;
    this.livesEl.textContent = '●'.repeat(Math.max(0, lives));
  }

  /** Shake the container (on life loss) */
  shakeScreen() {
    const container = document.getElementById('game-container');
    container.classList.add('shake');
    setTimeout(() => container.classList.remove('shake'), 400);
  }
}
