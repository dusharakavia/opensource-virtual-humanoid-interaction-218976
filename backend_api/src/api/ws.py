from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

from src.api.models import RealtimeEvent, RealtimeEventType


def _now_ms() -> int:
    return int(time.time() * 1000)


def _event_id() -> str:
    return f"evt_{secrets.token_urlsafe(10)}"


def make_event(
    event_type: RealtimeEventType, session_id: Optional[str], data: Dict[str, Any]
) -> RealtimeEvent:
    """Create a realtime event envelope."""
    return RealtimeEvent(
        type=event_type,
        session_id=session_id,
        event_id=_event_id(),
        created_at_ms=_now_ms(),
        data=data or {},
    )


@dataclass
class SessionSockets:
    """Tracks active websockets for a session."""

    sockets: Set[WebSocket] = field(default_factory=set)


class WebSocketHub:
    """Manages WebSocket clients and supports per-session broadcasting."""

    def __init__(self) -> None:
        self._session_sockets: Dict[str, SessionSockets] = {}
        self._lock = asyncio.Lock()

    # PUBLIC_INTERFACE
    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept and register a websocket connection for a session."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self._session_sockets:
                self._session_sockets[session_id] = SessionSockets()
            self._session_sockets[session_id].sockets.add(websocket)

    # PUBLIC_INTERFACE
    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Unregister a websocket connection for a session."""
        async with self._lock:
            group = self._session_sockets.get(session_id)
            if not group:
                return
            group.sockets.discard(websocket)
            if not group.sockets:
                self._session_sockets.pop(session_id, None)

    # PUBLIC_INTERFACE
    async def broadcast(self, session_id: str, event: RealtimeEvent) -> None:
        """Broadcast an event to all connected websockets for a session."""
        async with self._lock:
            group = self._session_sockets.get(session_id)
            sockets = list(group.sockets) if group else []

        if not sockets:
            return

        payload = event.model_dump()
        # Send concurrently; drop failed sockets silently (they'll be cleaned on disconnect).
        await asyncio.gather(
            *[self._safe_send_json(ws, payload) for ws in sockets],
            return_exceptions=True,
        )

    async def _safe_send_json(self, ws: WebSocket, payload: Dict[str, Any]) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            # best-effort: ignore send failures
            return
