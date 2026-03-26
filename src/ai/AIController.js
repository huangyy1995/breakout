/**
 * AIController - abstraction layer for AI agents to control the game.
 *
 * Usage (from browser console):
 *   const ai = window.__BREAKOUT_AI;
 *   const state = ai.reset();
 *   const { state: nextState, reward, done } = ai.step({ type: 'move', direction: 0.5 });
 *
 * Usage (from Python via WebSocket):
 *   See WebSocketBridge.js
 */
export class AIController {
  /**
   * @param {import('../engine/Game.js').Game} game
   */
  constructor(game) {
    this.game = game;
    this.enabled = false;
    this.renderEnabled = true; // Can disable rendering for faster training
    this.gameSpeed = 1; // Speed multiplier for training
  }

  /** Enable AI control, disable human input */
  enable() {
    this.enabled = true;
    this.game.aiControlled = true;
  }

  /** Disable AI control, re-enable human input */
  disable() {
    this.enabled = false;
    this.game.aiControlled = false;
  }

  /**
   * Get current game state (observation).
   * @returns {object} normalized game state
   */
  getState() {
    return this.game.getState();
  }

  /**
   * Apply an action to the game.
   * @param {object} action - { type: 'move'|'position'|'launch', direction?: number, position?: number }
   */
  applyAction(action) {
    this.game.applyAction(action);
  }

  /**
   * Gym-like reset. Restarts the game and returns initial state.
   * @returns {object} initial observation
   */
  reset() {
    return this.game.resetForAI();
  }

  /**
   * Gym-like step. Apply action, advance one tick, return results.
   * @param {object} action
   * @param {number} [dt] - optional time step
   * @returns {{ state: object, reward: number, done: boolean, info: object }}
   */
  step(action, dt) {
    return this.game.stepForAI(action, dt);
  }

  /**
   * Configure AI training parameters.
   * @param {object} config
   * @param {number} [config.gameSpeed] - game speed multiplier
   * @param {boolean} [config.renderEnabled] - whether to render visuals
   */
  configure(config) {
    if (config.gameSpeed !== undefined) {
      this.gameSpeed = config.gameSpeed;
    }
    if (config.renderEnabled !== undefined) {
      this.renderEnabled = config.renderEnabled;
    }
  }

  /** Get configuration */
  getConfig() {
    return {
      enabled: this.enabled,
      renderEnabled: this.renderEnabled,
      gameSpeed: this.gameSpeed,
    };
  }
}
