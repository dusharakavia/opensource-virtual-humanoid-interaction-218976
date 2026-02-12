import os

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="function")
def client():
    """Create a FastAPI TestClient for each test (fresh client; app globals persist)."""
    return TestClient(app)


@pytest.fixture(scope="function")
def session_id(client: TestClient) -> str:
    """Create a session and return its session_id."""
    res = client.post("/v1/sessions", json={"user_id": "test_user", "metadata": {"test": True}})
    assert res.status_code == 200, res.text
    data = res.json()
    assert "session_id" in data
    return data["session_id"]


@pytest.fixture(scope="function", autouse=True)
def _default_env(monkeypatch: pytest.MonkeyPatch):
    """
    Ensure tests run with safe default providers.
    (Providers are resolved at import time in src.api.main; these envs primarily help
    any code paths that read env vars at runtime, like BACKEND_TTS_DUMMY_BEEP.)
    """
    monkeypatch.setenv("BACKEND_CONVO_PROVIDER", os.getenv("BACKEND_CONVO_PROVIDER", "rule-based"))
    monkeypatch.setenv("BACKEND_STT_PROVIDER", os.getenv("BACKEND_STT_PROVIDER", "noop"))
    monkeypatch.setenv("BACKEND_TTS_PROVIDER", os.getenv("BACKEND_TTS_PROVIDER", "noop"))
    monkeypatch.delenv("BACKEND_TTS_DUMMY_BEEP", raising=False)
