from fastapi.testclient import TestClient


def _assert_event_envelope(ev: dict):
    assert isinstance(ev, dict)
    assert "type" in ev
    assert "event_id" in ev
    assert "created_at_ms" in ev
    assert "data" in ev


def test_ws_rejects_unknown_session_with_error_then_close(client: TestClient):
    # For missing session, backend accepts, sends error event, then closes with code 1008.
    with client.websocket_connect("/v1/ws?session_id=missing_session") as ws:
        ev = ws.receive_json()
        _assert_event_envelope(ev)
        assert ev["type"] == "error"
        assert ev["data"]["detail"] == "Session not found"
        assert ev["data"]["session_id"] == "missing_session"

        # Subsequent receive should fail due to close
        try:
            ws.receive_json()
            assert False, "Expected websocket to close after error"
        except Exception:
            pass


def test_ws_sends_hello_and_conversation_state_on_connect(client: TestClient, session_id: str):
    with client.websocket_connect(f"/v1/ws?session_id={session_id}") as ws:
        ev1 = ws.receive_json()
        _assert_event_envelope(ev1)
        assert ev1["type"] == "hello"
        assert ev1["session_id"] == session_id
        assert ev1["data"]["message"] == "connected"

        # Next, server broadcasts session.joined and sends conversation.state snapshot.
        # Depending on internal scheduling, order can vary, so receive two and assert set.
        ev2 = ws.receive_json()
        ev3 = ws.receive_json()
        for ev in (ev2, ev3):
            _assert_event_envelope(ev)

        types = {ev2["type"], ev3["type"]}
        assert "session.joined" in types
        assert "conversation.state" in types

        state_ev = ev2 if ev2["type"] == "conversation.state" else ev3
        assert state_ev["session_id"] == session_id
        assert state_ev["data"]["session_id"] == session_id
        assert state_ev["data"]["messages"] == []


def test_ws_input_text_requires_text_field(client: TestClient, session_id: str):
    with client.websocket_connect(f"/v1/ws?session_id={session_id}") as ws:
        # Drain initial hello/joined/state
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()

        # Send empty payload; server should respond with error
        ws.send_json({"type": "input.text", "data": {"text": "   "}})
        ev = ws.receive_json()
        _assert_event_envelope(ev)
        assert ev["type"] == "error"
        assert ev["data"]["detail"] == "text is required"
