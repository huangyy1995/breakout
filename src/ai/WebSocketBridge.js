/**
 * WebSocketBridge - enables external AI agents (e.g., Python RL) to control the game.
 *
 * Protocol:
 *   Client → Server:
 *     { type: 'reset' }                          → reset game, returns initial state
 *     { type: 'step', action: { ... } }           → take one step
 *     { type: 'state' }                           → get current state
 *     { type: 'config', data: { ... } }           → configure AI controller
 *
 *   Server → Client:
 *     { type: 'state', data: { ... } }            → game state
 *     { type: 'step_result', data: { state, reward, done, info } }
 *     { type: 'config', data: { ... } }           → current config
 *     { type: 'error', message: '...' }
 */
export class WebSocketBridge {
  /**
   * @param {import('./AIController.js').AIController} aiController
   * @param {string} [url] - WebSocket URL (null = use page origin)
   */
  constructor(aiController, url = null) {
    this.aiController = aiController;
    this.url = url;
    this.ws = null;
    this.connected = false;
    this.autoReconnect = true;
    this._reconnectTimer = null;
  }

  /**
   * Start listening for WebSocket connections.
   * In browser context, this creates a WebSocket client that connects
   * to a WebSocket server (which should be set up via nginx proxy or standalone).
   */
  connect(url) {
    const wsUrl = url || this.url || this._getDefaultUrl();
    this.url = wsUrl;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[AI Bridge] Connected to', wsUrl);
        this.connected = true;
        this.aiController.enable();
        this._send({ type: 'connected', data: this.aiController.getConfig() });
      };

      this.ws.onmessage = (event) => {
        this._handleMessage(event.data);
      };

      this.ws.onclose = () => {
        console.log('[AI Bridge] Disconnected');
        this.connected = false;
        this.aiController.disable();

        if (this.autoReconnect) {
          this._reconnectTimer = setTimeout(() => this.connect(), 3000);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('[AI Bridge] WebSocket error:', err);
      };
    } catch (err) {
      console.warn('[AI Bridge] Failed to connect:', err);
    }
  }

  /** Disconnect */
  disconnect() {
    this.autoReconnect = false;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
    this.aiController.disable();
  }

  /** Handle incoming message */
  _handleMessage(raw) {
    try {
      const msg = JSON.parse(raw);

      switch (msg.type) {
        case 'reset': {
          const state = this.aiController.reset();
          this._send({ type: 'state', data: state });
          break;
        }
        case 'step': {
          const result = this.aiController.step(msg.action, msg.dt);
          this._send({ type: 'step_result', data: result });
          break;
        }
        case 'state': {
          const state = this.aiController.getState();
          this._send({ type: 'state', data: state });
          break;
        }
        case 'config': {
          if (msg.data) {
            this.aiController.configure(msg.data);
          }
          this._send({ type: 'config', data: this.aiController.getConfig() });
          break;
        }
        case 'action': {
          // Direct action without stepping (for real-time control)
          this.aiController.applyAction(msg.action);
          break;
        }
        default:
          this._send({ type: 'error', message: `Unknown message type: ${msg.type}` });
      }
    } catch (err) {
      this._send({ type: 'error', message: err.message });
    }
  }

  /** Send message to connected client */
  _send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  /** Get default WebSocket URL based on page origin */
  _getDefaultUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
  }
}
