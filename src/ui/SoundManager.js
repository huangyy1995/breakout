/**
 * SoundManager — procedurally synthesised sound effects via Web Audio API.
 * No audio files required.
 */
export class SoundManager {
  constructor() {
    this._ctx = null;
    this.enabled = true;
    // Master volume (0–1)
    this._masterGain = null;
    this._init();
  }

  /** Lazily create AudioContext on first user gesture if needed. */
  _init() {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      this._ctx = new Ctx();

      this._masterGain = this._ctx.createGain();
      this._masterGain.gain.value = 0.4;
      this._masterGain.connect(this._ctx.destination);
    } catch (_) {
      // Audio not available — degrade silently
    }
  }

  /** Resume context after a user gesture (required by browsers). */
  _resume() {
    if (this._ctx && this._ctx.state === 'suspended') {
      this._ctx.resume();
    }
  }

  /**
   * Low-level helper: play a shaped oscillator burst.
   * @param {object} opts
   * @param {'sine'|'square'|'sawtooth'|'triangle'} [opts.type='sine']
   * @param {number} [opts.freq=440]      - start frequency (Hz)
   * @param {number} [opts.freqEnd]       - end frequency (slide), defaults to freq
   * @param {number} [opts.duration=0.08] - seconds
   * @param {number} [opts.gain=0.6]      - peak gain (before master)
   * @param {number} [opts.attack=0.005]  - attack time
   * @param {number} [opts.decay=0.05]    - decay time (to zero)
   * @param {number} [opts.startTime=0]   - offset from now
   */
  _tone({ type = 'sine', freq = 440, freqEnd, duration = 0.08,
          gain = 0.6, attack = 0.005, decay, startTime = 0 }) {
    if (!this._ctx || !this.enabled) return;
    this._resume();

    const ctx = this._ctx;
    const now = ctx.currentTime + startTime;
    const dec = decay ?? duration * 0.8;

    const osc = ctx.createOscillator();
    const env = ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, now);
    if (freqEnd !== undefined) {
      osc.frequency.exponentialRampToValueAtTime(freqEnd, now + duration);
    }

    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(gain, now + attack);
    env.gain.exponentialRampToValueAtTime(0.001, now + attack + dec);

    osc.connect(env);
    env.connect(this._masterGain);

    osc.start(now);
    osc.stop(now + attack + dec + 0.01);
  }

  // ─── Public sound methods ─────────────────────────────────────────────────

  /** Ball bounces off a wall. */
  playWallHit() {
    this._tone({ type: 'square', freq: 220, freqEnd: 180, duration: 0.06,
                 gain: 0.18, attack: 0.002, decay: 0.05 });
  }

  /** Ball bounces off the paddle. */
  playPaddleHit() {
    this._tone({ type: 'sine', freq: 300, freqEnd: 380, duration: 0.1,
                 gain: 0.5, attack: 0.003, decay: 0.08 });
  }

  /** Ball hits a brick but doesn't destroy it (HP still remaining). */
  playBrickHit() {
    this._tone({ type: 'square', freq: 480, freqEnd: 420, duration: 0.07,
                 gain: 0.3, attack: 0.002, decay: 0.055 });
  }

  /** Brick is fully destroyed. */
  playBrickDestroy() {
    // A crisp crack with a short noise-like sweep
    this._tone({ type: 'sawtooth', freq: 600, freqEnd: 200, duration: 0.12,
                 gain: 0.45, attack: 0.001, decay: 0.1 });
    this._tone({ type: 'sine',     freq: 900, freqEnd: 400, duration: 0.08,
                 gain: 0.25, attack: 0.001, decay: 0.07, startTime: 0.01 });
  }

  /** Power-up collected. */
  playPowerUp() {
    const notes = [523, 659, 784, 1047]; // C5 E5 G5 C6
    notes.forEach((f, i) => {
      this._tone({ type: 'sine', freq: f, freqEnd: f * 1.04,
                   gain: 0.4, attack: 0.005, decay: 0.08, startTime: i * 0.07 });
    });
  }

  /** Life lost — ball fell below screen. */
  playLifeLost() {
    this._tone({ type: 'sawtooth', freq: 330, freqEnd: 110, duration: 0.35,
                 gain: 0.5, attack: 0.01, decay: 0.32 });
    this._tone({ type: 'sine',     freq: 220, freqEnd: 80,  duration: 0.4,
                 gain: 0.3, attack: 0.02, decay: 0.36, startTime: 0.05 });
  }

  /** Level complete fanfare. */
  playLevelComplete() {
    const melody = [523, 659, 784, 659, 1047];
    melody.forEach((f, i) => {
      this._tone({ type: 'sine', freq: f, freqEnd: f,
                   gain: 0.45, attack: 0.01, decay: 0.12, startTime: i * 0.1 });
    });
    // Harmony
    const harmony = [330, 415, 523, 415, 659];
    harmony.forEach((f, i) => {
      this._tone({ type: 'triangle', freq: f,
                   gain: 0.2, attack: 0.01, decay: 0.12, startTime: i * 0.1 });
    });
  }

  /** Game over. */
  playGameOver() {
    const notes = [330, 277, 220, 165];
    notes.forEach((f, i) => {
      this._tone({ type: 'sawtooth', freq: f, freqEnd: f * 0.92,
                   gain: 0.4, attack: 0.01, decay: 0.18, startTime: i * 0.15 });
    });
  }
}
