from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    """Request model to create a new interaction session."""

    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier supplied by the client.",
        examples=["user_123"],
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the session.",
    )


class SessionCreateResponse(BaseModel):
    """Response model returned after creating a new session."""

    session_id: str = Field(
        ...,
        description="Server-generated session identifier.",
        examples=["sess_01HTY0..."],
    )
    created_at_ms: int = Field(
        ...,
        description="Unix epoch time in milliseconds when the session was created.",
        examples=[1710000000000],
    )


class SessionInfo(BaseModel):
    """Session information model."""

    session_id: str = Field(..., description="Session identifier.")
    created_at_ms: int = Field(..., description="Creation time in epoch milliseconds.")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata.")


class Role(str, Enum):
    """Conversation role."""

    user = "user"
    assistant = "assistant"
    system = "system"


class ConversationMessage(BaseModel):
    """A single conversation message."""

    id: str = Field(..., description="Message identifier.")
    role: Role = Field(..., description="Message role.")
    content: str = Field(..., description="Message text content.")
    created_at_ms: int = Field(..., description="Creation time in epoch milliseconds.")


class ConversationState(BaseModel):
    """Conversation history/state for a session."""

    session_id: str = Field(..., description="Session identifier.")
    messages: List[ConversationMessage] = Field(
        default_factory=list,
        description="Ordered list of conversation messages for the session.",
    )


class ChatRequest(BaseModel):
    """REST chat request for a given session."""

    session_id: str = Field(..., description="Session identifier.")
    text: str = Field(..., description="User text input.")
    stream: bool = Field(
        default=False,
        description="If true, use WebSocket streaming for events; REST returns only final response.",
    )


class ChatResponse(BaseModel):
    """REST chat response containing the assistant's final text output."""

    session_id: str = Field(..., description="Session identifier.")
    response_text: str = Field(..., description="Assistant response.")
    message_id: str = Field(..., description="Assistant message id.")
    created_at_ms: int = Field(..., description="Creation time in epoch milliseconds.")


class STTRequest(BaseModel):
    """Speech-to-text request.

    Audio should be provided as base64-encoded bytes for a lightweight demo interface.
    """

    session_id: str = Field(..., description="Session identifier.")
    audio_base64: str = Field(..., description="Base64-encoded audio bytes.")
    mime_type: str = Field(
        default="audio/wav",
        description="MIME type of the provided audio (e.g., audio/wav, audio/webm).",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language hint (e.g., 'en').",
        examples=["en"],
    )


class STTResponse(BaseModel):
    """Speech-to-text response."""

    session_id: str = Field(..., description="Session identifier.")
    text: str = Field(..., description="Transcribed text.")
    provider: str = Field(..., description="STT provider used.")
    confidence: Optional[float] = Field(
        default=None, description="Optional confidence score if available."
    )


class TTSRequest(BaseModel):
    """Text-to-speech request.

    For safe fallback, server may return synthesized audio or a 'not_available' status.
    """

    session_id: str = Field(..., description="Session identifier.")
    text: str = Field(..., description="Text to synthesize.")
    voice: Optional[str] = Field(
        default=None,
        description="Optional voice id/name depending on provider.",
    )
    mime_type: str = Field(
        default="audio/wav",
        description="Desired output MIME type (e.g., audio/wav).",
    )


class TTSResponse(BaseModel):
    """Text-to-speech response."""

    session_id: str = Field(..., description="Session identifier.")
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded synthesized audio bytes if available.",
    )
    provider: str = Field(..., description="TTS provider used.")
    status: Literal["ok", "not_available", "error"] = Field(
        ...,
        description="TTS synthesis status.",
    )
    detail: Optional[str] = Field(default=None, description="Optional status details.")


class RealtimeEventType(str, Enum):
    """Realtime event type for WebSocket streaming."""

    hello = "hello"
    error = "error"

    # Session / connection lifecycle
    session_created = "session.created"
    session_joined = "session.joined"
    session_left = "session.left"

    # Input/output events
    user_text = "input.text"
    stt_partial = "stt.partial"
    stt_final = "stt.final"
    assistant_token = "assistant.token"
    assistant_message = "assistant.message"
    tts_audio = "tts.audio"

    # State snapshot
    conversation_state = "conversation.state"


class RealtimeEvent(BaseModel):
    """Generic realtime event envelope sent over WebSocket."""

    type: RealtimeEventType = Field(..., description="Event type.")
    session_id: Optional[str] = Field(
        default=None, description="Associated session id if applicable."
    )
    event_id: str = Field(..., description="Server-generated event id.")
    created_at_ms: int = Field(..., description="Event time in epoch milliseconds.")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific payload."
    )
