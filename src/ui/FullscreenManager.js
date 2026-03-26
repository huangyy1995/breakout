/**
 * FullscreenManager - Fullscreen API wrapper with orientation lock.
 */
export class FullscreenManager {
  constructor() {
    this.isFullscreen = false;

    document.addEventListener('fullscreenchange', () => {
      this.isFullscreen = !!document.fullscreenElement;
    });
    document.addEventListener('webkitfullscreenchange', () => {
      this.isFullscreen = !!document.webkitFullscreenElement;
    });
  }

  /** Toggle fullscreen on the game container */
  async toggle() {
    const el = document.getElementById('game-container');

    if (!this.isFullscreen) {
      try {
        if (el.requestFullscreen) {
          await el.requestFullscreen();
        } else if (el.webkitRequestFullscreen) {
          await el.webkitRequestFullscreen();
        }
        // Try to lock orientation to portrait on mobile
        this._lockOrientation();
      } catch (err) {
        console.warn('Fullscreen not supported:', err);
      }
    } else {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        await document.webkitExitFullscreen();
      }
    }
  }

  /** Request fullscreen (for mobile auto-fullscreen on game start) */
  async request() {
    if (this.isFullscreen) return;
    await this.toggle();
  }

  _lockOrientation() {
    try {
      if (screen.orientation && screen.orientation.lock) {
        screen.orientation.lock('portrait').catch(() => {});
      }
    } catch (e) {
      // Not supported
    }
  }
}
