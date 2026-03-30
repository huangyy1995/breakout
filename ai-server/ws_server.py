"""
ws_server.py - WebSocket relay server between Python trainer and browser game.

Architecture:
  Python (ws client) --> ws_server.py (port 8765) <-- Browser (ws client)

The relay accepts exactly two clients:
  1. Browser game (identified by first message containing type:"connected")
  2. Python trainer (everything else)

Messages from Python are forwarded to Browser and vice versa.
"""

import asyncio
import argparse
import logging
import signal
import sys

import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WS Relay] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class RelayServer:
    def __init__(self):
        self.browser = None
        self.trainer = None
        self._lock = asyncio.Lock()

    async def handler(self, websocket):
        """Handle a new WebSocket connection."""
        client_type = None
        try:
            # Wait for first message to identify client type
            first_msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)

            async with self._lock:
                if '"connected"' in first_msg or '"type":"connected"' in first_msg.replace(" ", ""):
                    if self.browser is not None:
                        log.warning("Browser already connected, replacing")
                        try:
                            await self.browser.close()
                        except Exception:
                            pass
                    self.browser = websocket
                    client_type = "browser"
                    log.info("Browser connected")

                    # Forward the connection message to trainer if present
                    if self.trainer is not None:
                        try:
                            await self.trainer.send(first_msg)
                        except Exception:
                            pass
                else:
                    if self.trainer is not None:
                        log.warning("Trainer already connected, replacing")
                        try:
                            await self.trainer.close()
                        except Exception:
                            pass
                    self.trainer = websocket
                    client_type = "trainer"
                    log.info("Trainer connected")

                    # Forward the first message to browser if present
                    if self.browser is not None:
                        try:
                            await self.browser.send(first_msg)
                        except Exception:
                            pass

            # Relay messages
            async for message in websocket:
                target = None
                async with self._lock:
                    if client_type == "browser":
                        target = self.trainer
                    elif client_type == "trainer":
                        target = self.browser

                if target is not None:
                    try:
                        await target.send(message)
                    except websockets.ConnectionClosed:
                        log.warning(f"Target ({('trainer' if client_type == 'browser' else 'browser')}) disconnected")
                        break
                else:
                    peer = "trainer" if client_type == "browser" else "browser"
                    log.debug(f"No {peer} connected, dropping message")

        except asyncio.TimeoutError:
            log.warning("Client did not send initial message within 30s")
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            log.error(f"Handler error: {e}")
        finally:
            async with self._lock:
                if client_type == "browser" and self.browser == websocket:
                    self.browser = None
                    log.info("Browser disconnected")
                elif client_type == "trainer" and self.trainer == websocket:
                    self.trainer = None
                    log.info("Trainer disconnected")

    @property
    def status(self):
        return {
            "browser": self.browser is not None,
            "trainer": self.trainer is not None,
        }


async def main(host: str, port: int):
    relay = RelayServer()

    # Graceful shutdown
    stop = asyncio.Event()

    def _signal_handler():
        log.info("Shutting down...")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    async with websockets.serve(relay.handler, host, port):
        log.info(f"Relay server listening on ws://{host}:{port}")
        log.info("Waiting for browser and trainer connections...")
        await stop.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breakout RL WebSocket Relay Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        pass
