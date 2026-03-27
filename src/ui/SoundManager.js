/**
 * SoundManager - Web Audio API, no files needed.
 *
 * Music: pentatonic ambient, 72 BPM, pure sine waves, no drums.
 * C pentatonic major: C D E G A  (no semitones = always consonant)
 * Layers: slow pad chord + gentle arpeggio melody only.
 */

// ---- music constants ----
const BPM   = 72;
const BEAT  = 60 / BPM;         // ~0.833 s per quarter note
const HALF  = BEAT * 2;         // half note
const STEPS = 16;               // 4 bars x 4 beats, loops forever

const _n = (m) => 440 * Math.pow(2, (m - 69) / 12);

// C pentatonic major: C D E G A
const C4=_n(60), D4=_n(62), E4=_n(64), G4=_n(67), A4=_n(69);
const C5=_n(72), D5=_n(74), E5=_n(76), G5=_n(79), A5=_n(81);
const G3=_n(55), A3=_n(57), C3=_n(48), E3=_n(52);

// Pad chords (3 sine voices held for one bar = 4 beats)
// Progression: C - Am - F(sub G3) - G(sub G3) -- pentatonic approximation
const PAD = [
  [C4, E4, G4],   // bar 1: C major
  [A3, C4, E4],   // bar 2: A minor
  [G3, C4, E4],   // bar 3: G sus
  [G3, D4, G4],   // bar 4: G major
];

// Arpeggio melody notes per beat (0 = rest)
// Gentle ascending/descending pentatonic figures
const ARP = [
  C4,  E4,  G4,  C5,   // bar 1 ascend
  A4,  G4,  E4,  D4,   // bar 2 descend
  C4,  E4,  G4,  A4,   // bar 3 mid
  G4,  E4,  D4,  C4,   // bar 4 resolve
];

export class SoundManager {
  constructor() {
    this._ctx         = null;
    this._sfxGain     = null;
    this._musicGain   = null;
    this._musicNodes  = [];
    this._musicPlaying = false;
    this._step        = 0;
    this._nextTime    = 0;
    this._timer       = null;

    // Public toggles (can be set externally)
    this.musicEnabled = true;
    this.sfxEnabled   = true;

    this._init();
  }

  _init() {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      this._ctx = new Ctx();

      this._sfxGain = this._ctx.createGain();
      this._sfxGain.gain.value = 0.45;
      this._sfxGain.connect(this._ctx.destination);

      this._musicGain = this._ctx.createGain();
      this._musicGain.gain.value = 0.5;
      this._musicGain.connect(this._ctx.destination);
    } catch (_) {}
  }

  _resume() {
    if (this._ctx && this._ctx.state === 'suspended') this._ctx.resume();
  }

  // ---- SFX helper ---------------------------------------------------------

  _tone({ type='sine', freq=440, freqEnd, duration=0.08,
          gain=0.5, attack=0.005, decay, startTime=0 }) {
    if (!this._ctx || !this.sfxEnabled) return;
    this._resume();
    const ctx = this._ctx;
    const t   = ctx.currentTime + startTime;
    const dec = decay != null ? decay : duration * 0.8;
    const osc = ctx.createOscillator();
    const env = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t);
    if (freqEnd !== undefined)
      osc.frequency.exponentialRampToValueAtTime(freqEnd, t + duration);
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(gain, t + attack);
    env.gain.exponentialRampToValueAtTime(0.001, t + attack + dec);
    osc.connect(env);
    env.connect(this._sfxGain);
    osc.start(t);
    osc.stop(t + attack + dec + 0.02);
  }

  // ---- Music note helper --------------------------------------------------

  _mNote({ freq, duration, gain=0.12, attack=0.08, release=0.2, time }) {
    if (!this._ctx) return;
    const ctx = this._ctx;
    const osc = ctx.createOscillator();
    const env = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, time);
    env.gain.setValueAtTime(0, time);
    env.gain.linearRampToValueAtTime(gain, time + attack);
    const hold = Math.max(0, duration - release);
    env.gain.setValueAtTime(gain, time + hold);
    env.gain.linearRampToValueAtTime(0, time + duration);
    osc.connect(env);
    env.connect(this._musicGain);
    osc.start(time);
    osc.stop(time + duration + 0.02);
    this._musicNodes.push(osc);
  }

  // ---- Music layers -------------------------------------------------------

  /** Soft pad: 3 sine voices, one full bar */
  _pad(notes, time) {
    const dur = BEAT * 4 * 0.95;
    notes.forEach((f, i) => {
      // tiny detuning on inner voice for warmth
      const f2 = i === 1 ? f * 1.0015 : f;
      this._mNote({ freq: f2, duration: dur, gain: 0.045, attack: 0.25, release: 0.4, time });
    });
  }

  /** Arpeggio: single sine note, short */
  _arp(freq, time) {
    this._mNote({ freq, duration: BEAT * 0.6, gain: 0.09, attack: 0.03, release: 0.15, time });
  }

  // ---- Scheduler ----------------------------------------------------------

  startMusic() {
    if (!this._ctx || this._musicPlaying) return;
    this._resume();
    this._musicPlaying = true;
    this._step         = 0;
    this._nextTime     = this._ctx.currentTime + 0.05;
    this._tick();
    this._timer = setInterval(() => this._tick(), 150);
  }

  stopMusic() {
    if (!this._musicPlaying) return;
    this._musicPlaying = false;
    clearInterval(this._timer);
    this._timer = null;
    if (this._musicGain && this._ctx) {
      const now = this._ctx.currentTime;
      this._musicGain.gain.cancelScheduledValues(now);
      this._musicGain.gain.setValueAtTime(this._musicGain.gain.value, now);
      this._musicGain.gain.linearRampToValueAtTime(0, now + 1.0);
      setTimeout(() => { if (this._musicGain) this._musicGain.gain.value = 0.5; }, 1100);
    }
  }

  pauseMusic() {
    if (this._ctx && this._ctx.state === 'running') this._ctx.suspend();
  }

  resumeMusic() {
    if (this._ctx && this._ctx.state === 'suspended') this._ctx.resume();
  }

  setMusicEnabled(on) {
    this.musicEnabled = on;
    if (!on) {
      this.stopMusic();
    }
  }

  setSfxEnabled(on) {
    this.sfxEnabled = on;
  }

  _tick() {
    if (!this._ctx || !this._musicPlaying) return;
    const ahead = 0.35;
    while (this._nextTime < this._ctx.currentTime + ahead) {
      this._scheduleStep(this._step, this._nextTime);
      this._step     = (this._step + 1) % STEPS;
      this._nextTime += BEAT;
    }
    // prune dead nodes
    this._musicNodes = this._musicNodes.filter(n => {
      try { return n.context && n.context.state !== 'closed'; } catch(_){ return false; }
    });
  }

  _scheduleStep(step, time) {
    const bar  = Math.floor(step / 4);
    const beat = step % 4;

    // Pad fires once per bar
    if (beat === 0) this._pad(PAD[bar], time);

    // Arpeggio on every beat
    const f = ARP[step];
    if (f) this._arp(f, time);
  }

  // ---- SFX ----------------------------------------------------------------

  playWallHit() {
    this._tone({ type:'square', freq:200, freqEnd:170, gain:0.15, attack:0.002, decay:0.04 });
  }
  playPaddleHit() {
    this._tone({ type:'sine', freq:320, freqEnd:400, gain:0.4, attack:0.003, decay:0.07 });
  }
  playBrickHit() {
    this._tone({ type:'square', freq:460, freqEnd:400, gain:0.25, attack:0.002, decay:0.05 });
  }
  playBrickDestroy() {
    this._tone({ type:'sawtooth', freq:600, freqEnd:200, gain:0.4, attack:0.001, decay:0.09 });
    this._tone({ type:'sine', freq:880, freqEnd:380, gain:0.2, attack:0.001, decay:0.07, startTime:0.01 });
  }
  playPowerUp() {
    [523, 659, 784, 1047].forEach((f, i) => {
      this._tone({ type:'sine', freq:f, freqEnd:f*1.04, gain:0.38, attack:0.005, decay:0.08, startTime:i*0.07 });
    });
  }
  playExtraLife() {
    [440, 554, 659, 880].forEach((f, i) => {
      this._tone({ type:'sine', freq:f, gain:0.4, attack:0.01, decay:0.1, startTime:i*0.08 });
    });
  }
  playLifeLost() {
    this._tone({ type:'sawtooth', freq:330, freqEnd:110, gain:0.45, attack:0.01, decay:0.3 });
    this._tone({ type:'sine', freq:220, freqEnd:80, gain:0.25, attack:0.02, decay:0.34, startTime:0.05 });
  }
  playLevelComplete() {
    [523,659,784,659,1047].forEach((f,i)=>{
      this._tone({ type:'sine', freq:f, gain:0.42, attack:0.01, decay:0.12, startTime:i*0.1 });
    });
    [330,415,523,415,659].forEach((f,i)=>{
      this._tone({ type:'triangle', freq:f, gain:0.18, attack:0.01, decay:0.12, startTime:i*0.1 });
    });
  }
  playGameOver() {
    [330,277,220,165].forEach((f,i)=>{
      this._tone({ type:'sawtooth', freq:f, freqEnd:f*0.92, gain:0.38, attack:0.01, decay:0.18, startTime:i*0.15 });
    });
  }
}
