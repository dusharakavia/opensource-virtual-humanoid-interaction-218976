from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.api.models import ConversationMessage, ConversationState, Role, SessionInfo


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(12)}"


@dataclass
class SessionRecord:
    """Internal session record stored in memory."""

    info: SessionInfo
    messages: List[ConversationMessage] = field(default_factory=list)


class InMemorySessionStore:
    """In-memory session store with conversation history.

    This is suitable for a single-process prototype. For production, swap with a DB/Redis-backed store.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}

    # PUBLIC_INTERFACE
    def create_session(self, user_id: Optional[str], metadata: Dict[str, Any]) -> SessionInfo:
        """Create a new session and return its info."""
        sid = _new_id("sess")
        info = SessionInfo(
            session_id=sid,
            created_at_ms=_now_ms(),
            user_id=user_id,
            metadata=metadata or {},
        )
        self._sessions[sid] = SessionRecord(info=info)
        return info

    # PUBLIC_INTERFACE
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get a session by id."""
        rec = self._sessions.get(session_id)
        return rec.info if rec else None

    # PUBLIC_INTERFACE
    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions (in-memory)."""
        return [r.info for r in self._sessions.values()]

    # PUBLIC_INTERFACE
    def append_message(self, session_id: str, role: Role, content: str) -> ConversationMessage:
        """Append a message to a session's conversation history."""
        rec = self._sessions.get(session_id)
        if rec is None:
            raise KeyError("session_not_found")

        msg = ConversationMessage(
            id=_new_id("msg"),
            role=role,
            content=content,
            created_at_ms=_now_ms(),
        )
        rec.messages.append(msg)
        return msg

    # PUBLIC_INTERFACE
    def get_conversation_state(self, session_id: str) -> ConversationState:
        """Return the conversation state for a session."""
        rec = self._sessions.get(session_id)
        if rec is None:
            raise KeyError("session_not_found")
        return ConversationState(session_id=session_id, messages=list(rec.messages))
