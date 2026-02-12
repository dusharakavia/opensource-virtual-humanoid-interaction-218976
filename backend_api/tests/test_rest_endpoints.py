import base64

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, dict)
    # Spec says {"message":"Healthy"}; keep tolerant but still meaningful.
    assert data.get("message") == "Healthy"


def test_capabilities(client: TestClient):
    res = client.get("/v1/capabilities")
    assert res.status_code == 200
    data = res.json()

    assert "providers" in data
    assert set(data["providers"].keys()) == {"stt", "tts", "conversation"}

    assert "realtime_event_types" in data
    assert isinstance(data["realtime_event_types"], list)
    # Spot-check a few stable event types from models.RealtimeEventType
    assert "assistant.token" in data["realtime_event_types"]
    assert "conversation.state" in data["realtime_event_types"]


def test_create_and_get_session(client: TestClient):
    create = client.post("/v1/sessions", json={"user_id": "u1", "metadata": {"a": 1}})
    assert create.status_code == 200
    created = create.json()
    assert created["session_id"].startswith("sess_")
    assert isinstance(created["created_at_ms"], int)

    got = client.get(f"/v1/sessions/{created['session_id']}")
    assert got.status_code == 200
    info = got.json()
    assert info["session_id"] == created["session_id"]
    assert info["user_id"] == "u1"
    assert info["metadata"] == {"a": 1}


def test_get_session_404(client: TestClient):
    res = client.get("/v1/sessions/does_not_exist")
    assert res.status_code == 404
    assert res.json()["detail"] == "Session not found"


def test_get_conversation_initial_empty(session_id: str, client: TestClient):
    res = client.get(f"/v1/sessions/{session_id}/conversation")
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert data["messages"] == []


def test_chat_appends_messages_and_returns_response(session_id: str, client: TestClient):
    res = client.post("/v1/chat", json={"session_id": session_id, "text": "hello", "stream": False})
    assert res.status_code == 200
    body = res.json()

    assert body["session_id"] == session_id
    assert isinstance(body["message_id"], str) and body["message_id"].startswith("msg_")
    assert isinstance(body["created_at_ms"], int)
    assert isinstance(body["response_text"], str)
    assert len(body["response_text"]) > 0

    convo = client.get(f"/v1/sessions/{session_id}/conversation")
    assert convo.status_code == 200
    messages = convo.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hello"
    assert messages[1]["role"] == "assistant"
    assert body["response_text"] in messages[1]["content"]


def test_chat_404_for_unknown_session(client: TestClient):
    res = client.post("/v1/chat", json={"session_id": "missing", "text": "hi"})
    assert res.status_code == 404
    assert res.json()["detail"] == "Session not found"


@pytest.mark.parametrize("path", ["/v1/stt", "/v1/tts"])
def test_speech_endpoints_404_for_unknown_session(client: TestClient, path: str):
    if path.endswith("/stt"):
        payload = {"session_id": "missing", "audio_base64": base64.b64encode(b"hi").decode("utf-8")}
    else:
        payload = {"session_id": "missing", "text": "hello"}

    res = client.post(path, json=payload)
    assert res.status_code == 404
    assert res.json()["detail"] == "Session not found"


def test_stt_noop_provider_decodes_base64_text_guess(session_id: str, client: TestClient):
    # Noop STT tries to decode base64 bytes as utf-8 and return it as a "guess".
    audio_base64 = base64.b64encode(b"test transcript").decode("utf-8")
    res = client.post(
        "/v1/stt",
        json={"session_id": session_id, "audio_base64": audio_base64, "mime_type": "audio/wav"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert data["provider"] == "noop-stt"
    assert data["text"] == "test transcript"


def test_tts_noop_provider_returns_not_available_by_default(session_id: str, client: TestClient):
    res = client.post("/v1/tts", json={"session_id": session_id, "text": "hello"})
    assert res.status_code == 200
    data = res.json()

    assert data["session_id"] == session_id
    assert data["provider"] == "noop-tts"
    # Default is safe fallback with not_available
    assert data["status"] in {"not_available", "ok", "error"}
    if data["status"] == "not_available":
        assert data["audio_base64"] is None
