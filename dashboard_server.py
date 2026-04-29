"""dashboard_server.py — WebSocket server pushing live latency data to the browser dashboard."""

import asyncio
import json
import logging
from collections import deque

import websockets
from websockets.server import WebSocketServerProtocol

log = logging.getLogger("dashboard")

_MAX_HISTORY = 100  # prevent unbounded growth on long sessions
_clients: set[WebSocketServerProtocol] = set()
_history: deque = deque(maxlen=_MAX_HISTORY)


async def push_update(record: dict):
    """Call after each utterance to broadcast to all connected clients."""
    _history.append(record)
    if not _clients:
        return
    msg = json.dumps({"type": "utterance", "data": record, "history": list(_history)})
    await asyncio.gather(*[c.send(msg) for c in _clients], return_exceptions=True)


async def _handler(ws: WebSocketServerProtocol):
    _clients.add(ws)
    log.info(f"Dashboard client connected: {ws.remote_address}")
    await ws.send(json.dumps({"type": "history", "history": list(_history)}))
    try:
        async for _ in ws:
            pass
    finally:
        _clients.discard(ws)
        log.info("Dashboard client disconnected")


async def serve(port: int = 8765):
    async with websockets.serve(_handler, "localhost", port):
        log.info(f"Dashboard WebSocket server on ws://localhost:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
