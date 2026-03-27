# CLAUDE.md — AI Agent Guide for Breakout Neon Bricks

This file gives AI coding agents (Claude, Copilot, Cursor, etc.) the context needed to work on this codebase effectively.

---

## Project Overview

**What:** A single-page HTML5 Canvas breakout (brick-breaker) game.
**Stack:** Vanilla ES6 JavaScript modules, Vite bundler, no frameworks.
**Target:** Mobile (touch) + Desktop (keyboard/mouse). Deployed via Docker/Nginx to Google Cloud Run.

---

## Commands

```bash
npm install          # Install dependencies (only vite)
npm run dev          # Dev server with HMR → http://localhost:3000
npm run build        # Production build → dist/
npm run preview      # Preview production build
```

There are **no tests, no linter, no TypeScript**. Validate changes by running `npm run build` — Vite will report any import/syntax errors.

---

## Architecture

### Entry Point
- `index.html` — single HTML page, loads `src/main.js` as ES module
- `src/main.js` — bootstraps all systems, wires callbacks, runs the `requestAnimationFrame` game loop

### Core Engine (`src/engine/`)
| File | Responsibility |
|------|---------------|
| `Game.js` | Central state machine (`MENU → PLAYING → PAUSED / GAME_OVER / LEVEL_COMPLETE`). Owns all entities, runs update logic, fires callbacks. Contains multi-ball management, power-up collection, combo scoring. Also has Gym-like AI interface (`getState`, `stepForAI`, `resetForAI`). |
| `Physics.js` | Static collision detection: `ballPaddleCollision()` (reflection angle based on hit position) and `ballBrickCollisions()` (AABB vs circle, returns `{ destroyed[], hit }`) |
| `entities/Ball.js` | Ball entity — position, velocity, wall bounces, trail history, speed scaling. `update()` returns `true` if wall was hit. Has `glowColor`/`midColor` for visual tinting of extra balls. |
| `entities/Brick.js` | `Brick` class (HP, hit flash, death animation) + `BrickGrid` class (generates 10 pattern types based on level number, tracks alive count). |
| `entities/Paddle.js` | Paddle entity — keyboard direction, touch delta, AI position modes. Paddle Y is at `canvasHeight - 110` (raised for mobile). |
| `entities/PowerUp.js` | Falling power-up pill. Three types: `MULTI_BALL`, `SPLIT_BALL`, `EXTRA_LIFE`. Each has unique color/label defined in `POWERUP_STYLES`. |
| `levels/LevelManager.js` | Tracks current level (1-50), provides `getSpeedMultiplier()` and `getPaddleWidthMultiplier()` for difficulty scaling. |

### Input (`src/input/`)
| File | Responsibility |
|------|---------------|
| `InputManager.js` | Unified input: keyboard (`direction`), mouse (`targetX` absolute), touch (`deltaX` relative/incremental). Touch uses `window`-level listener for corner-tap cheat (upper-left 20% zone, 5 taps in 1.5s). Exposes `isPositionBased` (mouse), `isTouchDelta` (touch), `consumeDelta()`, `consumeLaunch()`, `consumePause()`. |

### UI / Rendering (`src/ui/`)
| File | Responsibility |
|------|---------------|
| `Renderer.js` | Canvas 2D rendering — background grid, bricks (with glow/flash/HP/indestructible styles), paddle, ball (with trail + tint colors), power-ups (pulsing pill with label), cheat indicator. Handles DPI scaling. |
| `Menu.js` | DOM-based screen management (start/pause/game-over/level-complete/HUD). `showPlaying()` activates HUD. `updateHUD()` sets score/level/lives. |
| `ParticleSystem.js` | Brick break particles (colored squares) and paddle hit particles (cyan circles with gravity). |
| `SoundManager.js` | Web Audio API — zero audio files. SFX via shaped oscillator bursts (`_tone()`). BGM via pentatonic ambient scheduler (C major, 72 BPM, sine/triangle only, 4-bar loop). Has `musicEnabled`/`sfxEnabled` toggles and `setMusicEnabled()`/`setSfxEnabled()` methods. |
| `FullscreenManager.js` | Fullscreen API wrapper with webkit fallbacks, orientation lock. |

### AI (`src/ai/`)
| File | Responsibility |
|------|---------------|
| `AIController.js` | Wraps `Game` for AI usage. Exposed on `window.__BREAKOUT_AI`. |
| `WebSocketBridge.js` | Connects to a Python WebSocket server for remote AI control. |

### Styling
- `src/styles/index.css` — all CSS. Uses CSS custom properties (`:root` vars). Notable: `#ui-overlay` has `pointer-events: none`, individual screens/HUD re-enable it. Touch bar (`#touch-bar`) appears on first touch during play.

### HTML Structure (`index.html`)
```
#game-container
  #game-canvas
  #ui-overlay
    #start-screen (.screen)
    #pause-screen (.screen)
    #gameover-screen (.screen)
    #level-complete-screen (.screen)
    #hud — score / level / lives / 🎵 / 🔊 / ❚❚
  #touch-bar — "← SLIDE TO MOVE →"
```

---

## Key Design Patterns

### Callback-based events
`Game.js` exposes callbacks that `main.js` wires up:
- `onBrickDestroyed(brick)` → particles + sound
- `onBrickHit(brick)` → sound (hit but not destroyed)
- `onPaddleHit(x, y)` → particles + sound
- `onWallHit()` → sound
- `onLifeLost()` → screen shake + sound
- `onScoreChange(score, level, lives)` → HUD update
- `onStateChange(newState)` → screen transitions + music control
- `onPowerUpCollected(powerUp)` → particles + sound

### Multi-ball management
- `game.ball` = primary ball (always exists)
- `game.extraBalls[]` = spawned by power-ups (max 7)
- When primary ball falls: if extras exist, promote one; otherwise lose a life
- All extra balls rendered with tinted colors (cyan for +BALL, green for SPLIT)

### Touch input model
Touch uses **relative/delta** control (not absolute position), so the paddle doesn't jump to where the finger touches. `InputManager` accumulates `deltaX` on `touchmove`, `Game.update()` consumes it via `input.consumeDelta()` and applies to `paddle.x` directly.

### Physics returns
`Physics.ballBrickCollisions()` returns `{ destroyed: Brick[], hit: Brick|null }`:
- `destroyed` = bricks whose HP reached 0 (for scoring/particles)
- `hit` = brick that was hit but survived (for hit sound)

### Coordinate system
- Logical canvas: width=420, height scales to maintain container aspect ratio
- All AI state normalized to [0, 1]
- DPR scaling handled by `Renderer.resize()`

---

## Common Modification Scenarios

### Adding a new power-up type
1. Add type to `PowerUpType` enum and `POWERUP_STYLES` in `src/engine/entities/PowerUp.js`
2. Add handling in `Game._collectPowerUp()` in `src/engine/Game.js`
3. Optionally add a new SFX method in `SoundManager.js` and wire it in `main.js` `onPowerUpCollected`

### Adding a new brick pattern
1. Add a new `case` in the `switch(patternType)` block inside `BrickGrid.generate()` in `src/engine/entities/Brick.js`
2. Increase `maxLevel` in `LevelManager.js` if needed

### Adding a new sound effect
1. Add a method to `SoundManager.js` using `_tone({ type, freq, ... })`
2. Call it from the appropriate callback in `main.js`

### Changing paddle/ball position
- Paddle Y: `Paddle.js` constructor (`canvasHeight - 110`) + `Game.js` `resize()` method
- Ball attachment: `Ball.attachToPaddle()` positions ball 2px above paddle

### Adding new UI screens
1. Add HTML in `index.html` inside `#ui-overlay` with class `screen`
2. Add show/hide logic in `Menu.js`
3. Wire button handlers in `main.js`

### Modifying background music
- All music is in `SoundManager.js` top-level constants (`PAD`, `ARP`, `BPM`)
- Change `PAD` chords and `ARP` melody arrays to alter the tune
- `_scheduleStep()` controls which layers play on which beats

---

## Important Caveats

- **No test suite.** Use `npm run build` to check for errors.
- **Unicode in source files.** `SoundManager.js` was rewritten via shell heredoc to avoid Unicode dash issues. If editing, avoid using `—` (em-dash) in string replacements.
- **Mobile touch events.** The HUD overlay has `pointer-events: auto` and sits on top of the canvas. Any touch-based feature that needs to work in the top area (like cheat codes) must use `window`-level listeners, not canvas-level.
- **AudioContext autoplay policy.** Browsers require a user gesture before audio plays. `SoundManager._resume()` is called before every sound to handle this. `startMusic()` should only be called after user interaction (it is — via game state change triggered by button click).
- **Ball.update() return value.** Returns `true` if a wall bounce occurred (used for wall-hit sound). Don't change the return to void.
- **Physics.ballBrickCollisions() return shape.** Returns `{ destroyed, hit }` object, not a plain array. Both `Game.update()` and `Game.stepForAI()` destructure this.
