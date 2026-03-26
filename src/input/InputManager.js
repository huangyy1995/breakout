/**
 * InputManager - unified input handling for keyboard, mouse, and touch.
 */
export class InputManager {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.inputMode = 'keyboard'; // 'keyboard' | 'touch' | 'mouse'

    // State
    this.direction = 0; // -1 to 1
    this.targetX = null; // absolute canvas X for touch/mouse
    this.launchRequested = false;
    this.pauseRequested = false;

    // Corner-tap cheat tracking (mobile)
    this._cornerTapCount = 0;
    this._lastCornerTapTime = 0;
    /** How many taps are needed to trigger the cheat. */
    this.cornerTapRequired = 5;
    /** Current corner tap progress (0..cornerTapRequired), read by renderer for UI feedback. */
    this.cornerTapCount = 0;
    /** Called when the cheat sequence completes. Set this from outside. */
    this.onCheatActivated = null;

    // Track keys
    this._keys = new Set();

    this._bindEvents();
  }

  _bindEvents() {
    // Keyboard
    window.addEventListener('keydown', (e) => {
      this._keys.add(e.key);
      this.inputMode = 'keyboard';

      if (e.key === 'ArrowLeft' || e.key === 'a') {
        this.direction = -1;
      } else if (e.key === 'ArrowRight' || e.key === 'd') {
        this.direction = 1;
      } else if (e.key === ' ' || e.key === 'ArrowUp') {
        this.launchRequested = true;
        e.preventDefault();
      } else if (e.key === 'Escape' || e.key === 'p') {
        this.pauseRequested = true;
      }
    });

    window.addEventListener('keyup', (e) => {
      this._keys.delete(e.key);

      if (
        (e.key === 'ArrowLeft' || e.key === 'a') && this.direction === -1
      ) {
        this.direction = this._keys.has('ArrowRight') || this._keys.has('d') ? 1 : 0;
      } else if (
        (e.key === 'ArrowRight' || e.key === 'd') && this.direction === 1
      ) {
        this.direction = this._keys.has('ArrowLeft') || this._keys.has('a') ? -1 : 0;
      }
    });

    // Touch
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.inputMode = 'touch';
      this._handleTouch(e.touches[0]);
      this.launchRequested = true;
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      this._handleTouch(e.touches[0]);
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.targetX = null;
    }, { passive: false });

    // Mouse
    this.canvas.addEventListener('mousedown', (e) => {
      this.inputMode = 'mouse';
      this._handleMouse(e);
      this.launchRequested = true;
    });

    this.canvas.addEventListener('mousemove', (e) => {
      if (this.inputMode === 'mouse') {
        this._handleMouse(e);
      }
    });

    // Window-level touch listener for corner-tap cheat.
    // Must be on window (not canvas) so it fires even when the HUD overlay
    // intercepts touches in the upper-right corner.
    window.addEventListener('touchstart', (e) => {
      if (e.touches.length > 0) {
        this._checkCornerTap(e.touches[0]);
      }
    }, { passive: true });
  }

  /**
   * Convert touch to canvas coordinates.
   * @param {Touch} touch
   */
  _handleTouch(touch) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const logicalWidth = this.canvas.width / dpr;
    const scaleX = logicalWidth / rect.width;
    this.targetX = (touch.clientX - rect.left) * scaleX;
  }

  /**
   * Detect taps in the upper-right corner for the skip-level cheat.
   * Uses a window-level listener so it fires even when the HUD overlay is on top.
   * 5 taps within 1.5 s of each other triggers the cheat.
   * @param {Touch} touch
   */
  _checkCornerTap(touch) {
    const rect = document.getElementById('game-canvas').getBoundingClientRect();
    const relX = (touch.clientX - rect.left) / rect.width;
    const relY = (touch.clientY - rect.top) / rect.height;

    // Upper-left corner zone: top 20 %, left 20 %
    if (relX <= 0.20 && relY <= 0.20) {
      const now = Date.now();
      if (now - this._lastCornerTapTime > 1500) {
        this._cornerTapCount = 0; // streak expired — reset
      }
      this._cornerTapCount++;
      this._lastCornerTapTime = now;
      this.cornerTapCount = this._cornerTapCount;

      if (this._cornerTapCount >= this.cornerTapRequired) {
        this._cornerTapCount = 0;
        this.cornerTapCount = 0;
        if (this.onCheatActivated) this.onCheatActivated();
      }
    }
  }

  /**
   * Convert mouse to canvas coordinates.
   * @param {MouseEvent} e
   */
  _handleMouse(e) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const logicalWidth = this.canvas.width / dpr;
    const scaleX = logicalWidth / rect.width;
    this.targetX = (e.clientX - rect.left) * scaleX;
  }

  /** Whether input is position-based (touch/mouse) vs direction-based (keyboard) */
  get isPositionBased() {
    return (this.inputMode === 'touch' || this.inputMode === 'mouse') && this.targetX !== null;
  }

  /** Consume launch request */
  consumeLaunch() {
    const val = this.launchRequested;
    this.launchRequested = false;
    return val;
  }

  /** Consume pause request */
  consumePause() {
    const val = this.pauseRequested;
    this.pauseRequested = false;
    return val;
  }

  /** Reset state */
  reset() {
    this.direction = 0;
    this.targetX = null;
    this.launchRequested = false;
    this.pauseRequested = false;
    this._cornerTapCount = 0;
    this.cornerTapCount = 0;
    this._keys.clear();
  }
}
