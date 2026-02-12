from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    ChatRequest,
    ChatResponse,
    ConversationState,
    RealtimeEventType,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionInfo,
    STTRequest,
    STTResponse,
    TTSRequest,
    TTSResponse,
)
from src.api.providers import resolve_providers
from src.api.store import InMemorySessionStore
from src.api.ws import WebSocketHub, make_event

openapi_tags = [
    {
        "name": "health",
        "description": "Basic health and capability endpoints.",
    },
    {
        "name": "sessions",
        "description": "Create and inspect interaction sessions.",
    },
    {
        "name": "conversation",
        "description": "Conversation endpoints (REST) for session-scoped interaction.",
    },
    {
        "name": "speech",
        "description": "Speech endpoints (STT/TTS). Default providers are safe fallbacks.",
    },
    {
        "name": "realtime",
        "description": "WebSocket streaming interface for realtime events.",
    },
    {
        "name": "docs",
        "description": "Developer documentation helpers.",
    },
]

app = FastAPI(
    title="Virtual Humanoid Interaction Orchestrator API",
    description=(
        "FastAPI backend orchestrating realtime interaction for a web-based virtual humanoid.\n\n"
        "This service provides:\n"
        "- Session management via REST\n"
        "- Conversation via REST\n"
        "- Realtime streaming events via WebSocket\n"
        "- Pluggable STT/TTS + conversational provider interfaces with safe fallbacks\n\n"
        "WebSocket usage:\n"
        "- Connect to `GET /v1/ws?session_id=<id>`\n"
        "- Server emits events like `assistant.token`, `assistant.message`, `stt.final`, etc.\n"
        "- Client can send JSON messages (see `/v1/docs/realtime` for examples)\n"
    ),
    version="0.2.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_store = InMemorySessionStore()
_hub = WebSocketHub()
_providers = resolve_providers()


@app.get("/", tags=["health"], summary="Health check")
# PUBLIC_INTERFACE
def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        JSON object with a human-friendly status message.
    """
    return {"message": "Healthy"}


@app.get("/v1/capabilities", tags=["health"], summary="Get backend capabilities")
# PUBLIC_INTERFACE
def get_capabilities() -> Dict[str, Any]:
    """Return currently active providers and supported realtime event types."""
    return {
        "providers": {
            "stt": _providers.stt.name,
            "tts": _providers.tts.name,
            "conversation": _providers.convo.name,
        },
        "realtime_event_types": [e.value for e in RealtimeEventType],
    }


@app.post(
    "/v1/sessions",
    tags=["sessions"],
    summary="Create a new session",
    response_model=SessionCreateResponse,
)
# PUBLIC_INTERFACE
def create_session(req: SessionCreateRequest) -> SessionCreateResponse:
    """Create a new interaction session.

    Parameters:
        req: SessionCreateRequest containing optional user_id and metadata.

    Returns:
        SessionCreateResponse with a new session_id and creation timestamp.
    """
    info = _store.create_session(user_id=req.user_id, metadata=req.metadata)
    return SessionCreateResponse(session_id=info.session_id, created_at_ms=info.created_at_ms)


@app.get(
    "/v1/sessions/{session_id}",
    tags=["sessions"],
    summary="Get session info",
    response_model=SessionInfo,
)
# PUBLIC_INTERFACE
def get_session(session_id: str) -> SessionInfo:
    """Get session info by id.

    Raises:
        HTTPException(404) if session does not exist.
    """
    info = _store.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return info


@app.get(
    "/v1/sessions/{session_id}/conversation",
    tags=["sessions"],
    summary="Get conversation state for a session",
    response_model=ConversationState,
)
# PUBLIC_INTERFACE
def get_conversation(session_id: str) -> ConversationState:
    """Return conversation history for a session."""
    try:
        return _store.get_conversation_state(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post(
    "/v1/chat",
    tags=["conversation"],
    summary="Send user text (REST) and receive assistant response",
    response_model=ChatResponse,
)
# PUBLIC_INTERFACE
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a user message and get an assistant response.

    Notes:
        - If `req.stream` is true, the recommended path is to use WebSocket `/v1/ws`
          to receive streaming tokens and audio events. This REST endpoint still returns
          the final response for compatibility.
    """
    session = _store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    _store.append_message(req.session_id, role="user", content=req.text)

    response_text = await _providers.convo.generate(session_id=req.session_id, prompt=req.text)
    msg = _store.append_message(req.session_id, role="assistant", content=response_text)

    return ChatResponse(
        session_id=req.session_id,
        response_text=response_text,
        message_id=msg.id,
        created_at_ms=msg.created_at_ms,
    )


@app.post(
    "/v1/stt",
    tags=["speech"],
    summary="Speech to text (STT)",
    response_model=STTResponse,
)
# PUBLIC_INTERFACE
async def stt(req: STTRequest) -> STTResponse:
    """Transcribe audio (base64) to text using the configured STT provider."""
    session = _store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return await _providers.stt.transcribe(req)


@app.post(
    "/v1/tts",
    tags=["speech"],
    summary="Text to speech (TTS)",
    response_model=TTSResponse,
)
# PUBLIC_INTERFACE
async def tts(req: TTSRequest) -> TTSResponse:
    """Synthesize speech audio (base64) using the configured TTS provider."""
    session = _store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return await _providers.tts.synthesize(req)


@app.get(
    "/v1/docs/realtime",
    tags=["docs"],
    summary="WebSocket realtime usage guide",
)
# PUBLIC_INTERFACE
def realtime_docs() -> Dict[str, Any]:
    """Provide a concise JSON guide for WebSocket usage and message shapes."""
    return {
        "ws_url": "/v1/ws?session_id=<session_id>",
        "client_to_server_messages": [
            {
                "type": "input.text",
                "data": {"text": "hello there"},
                "notes": "Server will stream assistant.token events then assistant.message.",
            },
            {
                "type": "input.stt",
                "data": {"audio_base64": "<base64>", "mime_type": "audio/wav"},
                "notes": "Server will emit stt.final and then assistant responses.",
            },
            {
                "type": "request.state",
                "data": {},
                "notes": "Server emits conversation.state snapshot.",
            },
        ],
        "server_events": [e.value for e in RealtimeEventType],
    }


@app.websocket("/v1/ws")
async def websocket_realtime(
    websocket: WebSocket,
    session_id: str = Query(..., description="Session id to bind this WebSocket to."),
):
    """Realtime WebSocket endpoint for streaming session events.

    Parameters:
        websocket: The WebSocket connection.
        session_id: Query parameter selecting the session this connection belongs to.

    Behavior:
        - On connect: server sends `hello`, `session.joined`, and `conversation.state`.
        - Client messages:
            - { "type": "input.text", "data": { "text": "..." } }
            - { "type": "input.stt", "data": { "audio_base64": "...", "mime_type": "audio/wav" } }
            - { "type": "request.state", "data": {} }
        - Server streams:
            - `assistant.token` events as the assistant responds
            - `assistant.message` when final assistant message is appended
            - `stt.final` after STT requests
            - `tts.audio` if TTS is available (noop by default)
    """
    session = _store.get_session(session_id)
    if session is None:
        # WS can't send normal HTTP errors after handshake unless we accept.
        await websocket.accept()
        await websocket.send_json(
            make_event(
                RealtimeEventType.error,
                session_id=None,
                data={"detail": "Session not found", "session_id": session_id},
            ).model_dump()
        )
        await websocket.close(code=1008)
        return

    await _hub.connect(websocket, session_id=session_id)

    await websocket.send_json(
        make_event(RealtimeEventType.hello, session_id=session_id, data={"message": "connected"}).model_dump()
    )
    await _hub.broadcast(
        session_id,
        make_event(RealtimeEventType.session_joined, session_id=session_id, data={}),
    )

    # Send initial state snapshot to this client
    try:
        state = _store.get_conversation_state(session_id)
        await websocket.send_json(
            make_event(
                RealtimeEventType.conversation_state,
                session_id=session_id,
                data=state.model_dump(),
            ).model_dump()
        )
    except Exception:
        # ignore
        pass

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = (msg.get("type") or "").strip()
            data = msg.get("data") or {}

            if mtype == RealtimeEventType.user_text.value or mtype == "input.text":
                text = (data.get("text") or "").strip()
                if not text:
                    await websocket.send_json(
                        make_event(
                            RealtimeEventType.error,
                            session_id=session_id,
                            data={"detail": "text is required"},
                        ).model_dump()
                    )
                    continue

                _store.append_message(session_id, role="user", content=text)
                await _hub.broadcast(
                    session_id,
                    make_event(RealtimeEventType.user_text, session_id=session_id, data={"text": text}),
                )

                # Stream assistant tokens
                assistant_chunks = []
                async for chunk in _providers.convo.stream(session_id=session_id, prompt=text):
                    assistant_chunks.append(chunk)
                    await _hub.broadcast(
                        session_id,
                        make_event(
                            RealtimeEventType.assistant_token,
                            session_id=session_id,
                            data={"token": chunk},
                        ),
                    )

                assistant_text = "".join(assistant_chunks)
                assistant_msg = _store.append_message(
                    session_id, role="assistant", content=assistant_text
                )
                await _hub.broadcast(
                    session_id,
                    make_event(
                        RealtimeEventType.assistant_message,
                        session_id=session_id,
                        data={
                            "message_id": assistant_msg.id,
                            "text": assistant_text,
                        },
                    ),
                )

                # Optionally synthesize audio and broadcast
                tts_resp = await _providers.tts.synthesize(
                    TTSRequest(session_id=session_id, text=assistant_text)
                )
                if tts_resp.status == "ok" and tts_resp.audio_base64:
                    await _hub.broadcast(
                        session_id,
                        make_event(
                            RealtimeEventType.tts_audio,
                            session_id=session_id,
                            data={
                                "audio_base64": tts_resp.audio_base64,
                                "mime_type": "audio/wav",
                                "provider": tts_resp.provider,
                            },
                        ),
                    )

            elif mtype == "input.stt":
                audio_b64 = (data.get("audio_base64") or "").strip()
                mime_type = (data.get("mime_type") or "audio/wav").strip()
                if not audio_b64:
                    await websocket.send_json(
                        make_event(
                            RealtimeEventType.error,
                            session_id=session_id,
                            data={"detail": "audio_base64 is required"},
                        ).model_dump()
                    )
                    continue

                stt_resp = await _providers.stt.transcribe(
                    STTRequest(session_id=session_id, audio_base64=audio_b64, mime_type=mime_type)
                )
                await _hub.broadcast(
                    session_id,
                    make_event(
                        RealtimeEventType.stt_final,
                        session_id=session_id,
                        data=stt_resp.model_dump(),
                    ),
                )

                # If we got some text, auto-route to conversation
                if stt_resp.text.strip():
                    _store.append_message(session_id, role="user", content=stt_resp.text.strip())

                    assistant_chunks = []
                    async for chunk in _providers.convo.stream(session_id=session_id, prompt=stt_resp.text):
                        assistant_chunks.append(chunk)
                        await _hub.broadcast(
                            session_id,
                            make_event(
                                RealtimeEventType.assistant_token,
                                session_id=session_id,
                                data={"token": chunk},
                            ),
                        )
                    assistant_text = "".join(assistant_chunks)
                    assistant_msg = _store.append_message(
                        session_id, role="assistant", content=assistant_text
                    )
                    await _hub.broadcast(
                        session_id,
                        make_event(
                            RealtimeEventType.assistant_message,
                            session_id=session_id,
                            data={"message_id": assistant_msg.id, "text": assistant_text},
                        ),
                    )

            elif mtype == "request.state":
                try:
                    state = _store.get_conversation_state(session_id)
                    await websocket.send_json(
                        make_event(
                            RealtimeEventType.conversation_state,
                            session_id=session_id,
                            data=state.model_dump(),
                        ).model_dump()
                    )
                except Exception:
                    await websocket.send_json(
                        make_event(
                            RealtimeEventType.error,
                            session_id=session_id,
                            data={"detail": "Unable to get state"},
                        ).model_dump()
                    )

            else:
                await websocket.send_json(
                    make_event(
                        RealtimeEventType.error,
                        session_id=session_id,
                        data={"detail": f"Unknown message type '{mtype}'"},
                    ).model_dump()
                )

    except WebSocketDisconnect:
        # normal
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                make_event(
                    RealtimeEventType.error,
                    session_id=session_id,
                    data={"detail": "Internal error", "error": str(e)},
                ).model_dump()
            )
        except Exception:
            pass
    finally:
        await _hub.disconnect(websocket, session_id=session_id)
        await _hub.broadcast(
            session_id,
            make_event(RealtimeEventType.session_left, session_id=session_id, data={}),
        )
