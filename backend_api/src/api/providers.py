from __future__ import annotations

import asyncio
import base64
import os
import re
from dataclasses import dataclass
from typing import AsyncGenerator, Protocol

from src.api.models import STTRequest, STTResponse, TTSRequest, TTSResponse


class STTProvider(Protocol):
    """Protocol for speech-to-text providers."""

    name: str

    async def transcribe(self, req: STTRequest) -> STTResponse:
        """Transcribe an audio payload into text."""


class TTSProvider(Protocol):
    """Protocol for text-to-speech providers."""

    name: str

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        """Synthesize audio for the provided text."""


class ConversationalProvider(Protocol):
    """Protocol for conversational AI providers.

    Providers may either:
    - return a final response string, or
    - stream response tokens asynchronously.
    """

    name: str

    async def generate(self, session_id: str, prompt: str) -> str:
        """Generate a final response for the given prompt."""

    async def stream(self, session_id: str, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response tokens/chunks for the given prompt."""


@dataclass(frozen=True)
class ProviderBundle:
    """Resolved providers used by the orchestrator."""

    stt: STTProvider
    tts: TTSProvider
    convo: ConversationalProvider


class SafeNoopSTTProvider:
    """Safe fallback STT provider.

    This provider never fails hard. It returns an empty transcription if audio is not usable.
    """

    name = "noop-stt"

    async def transcribe(self, req: STTRequest) -> STTResponse:
        # Heuristic: sometimes clients send base64-encoded UTF-8 text for demo purposes.
        text_guess = ""
        try:
            raw = base64.b64decode(req.audio_base64.encode("utf-8"), validate=False)
            try:
                decoded = raw.decode("utf-8", errors="ignore").strip()
                if decoded:
                    # Keep it reasonably bounded
                    text_guess = decoded[:2000]
            except Exception:
                text_guess = ""
        except Exception:
            text_guess = ""

        return STTResponse(
            session_id=req.session_id,
            text=text_guess,
            provider=self.name,
            confidence=None,
        )


class SafeNoopTTSProvider:
    """Safe fallback TTS provider.

    By default, it does not synthesize audio (avoids heavy dependencies).
    Optionally, if environment variable BACKEND_TTS_DUMMY_BEEP is set to '1',
    it returns a tiny placeholder WAV-like payload (not a real audio synthesis).
    """

    name = "noop-tts"

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        if os.getenv("BACKEND_TTS_DUMMY_BEEP", "0") == "1":
            # Minimal placeholder bytes; not a valid WAV but good enough for "something returned".
            payload = base64.b64encode(b"NOOP_TTS").decode("utf-8")
            return TTSResponse(
                session_id=req.session_id,
                audio_base64=payload,
                provider=self.name,
                status="ok",
                detail="Dummy audio payload returned by noop TTS provider.",
            )
        return TTSResponse(
            session_id=req.session_id,
            audio_base64=None,
            provider=self.name,
            status="not_available",
            detail="TTS not configured; noop provider active.",
        )


class RuleBasedConversationalProvider:
    """Safe fallback conversational provider with simple, deterministic behavior.

    Useful as a baseline and for environments where an LLM is not configured.
    """

    name = "rule-based"

    async def generate(self, session_id: str, prompt: str) -> str:
        # Keep deterministic and safe.
        cleaned = prompt.strip()
        if not cleaned:
            return "I didn't catch that. Could you say it again?"

        lower = cleaned.lower()

        if any(g in lower for g in ["hello", "hi", "hey"]):
            return "Hello! I'm ready. You can type or speak to me."
        if "your name" in lower:
            return "I'm your virtual humanoid assistant (open-source mode)."
        if "help" in lower:
            return (
                "You can:\n"
                "- Send text via POST /v1/chat\n"
                "- Use WebSocket /v1/ws to stream events\n"
                "- (Optional) Send audio base64 to POST /v1/stt\n"
                "- (Optional) Request speech via POST /v1/tts"
            )

        # A tiny 'echo-with-style' response.
        return f"You said: {cleaned}\n\n(I'm running with a safe fallback conversation provider.)"

    async def stream(self, session_id: str, prompt: str) -> AsyncGenerator[str, None]:
        text = await self.generate(session_id=session_id, prompt=prompt)

        # Stream in small word chunks to simulate token streaming.
        words = re.split(r"(\s+)", text)
        for w in words:
            if not w:
                continue
            yield w
            await asyncio.sleep(0.02)


# PUBLIC_INTERFACE
def resolve_providers() -> ProviderBundle:
    """Resolve STT/TTS/conversation providers based on environment.

    Environment variables (optional):
    - BACKEND_CONVO_PROVIDER: currently supports 'rule-based' (default)
    - BACKEND_STT_PROVIDER: currently supports 'noop' (default)
    - BACKEND_TTS_PROVIDER: currently supports 'noop' (default)

    Returns:
        ProviderBundle: providers to be used by the orchestrator.

    Notes:
        This is intentionally safe-by-default and avoids heavyweight ML model downloads.
        A future step can add real open-source providers (e.g., whisper.cpp, piper) behind
        these interfaces without changing the API surface.
    """
    convo_choice = (os.getenv("BACKEND_CONVO_PROVIDER") or "rule-based").strip().lower()
    stt_choice = (os.getenv("BACKEND_STT_PROVIDER") or "noop").strip().lower()
    tts_choice = (os.getenv("BACKEND_TTS_PROVIDER") or "noop").strip().lower()

    # Conversation provider
    if convo_choice in {"rule-based", "fallback"}:
        convo = RuleBasedConversationalProvider()
    else:
        # Safe fallback
        convo = RuleBasedConversationalProvider()

    # STT provider
    if stt_choice in {"noop", "fallback"}:
        stt = SafeNoopSTTProvider()
    else:
        stt = SafeNoopSTTProvider()

    # TTS provider
    if tts_choice in {"noop", "fallback"}:
        tts = SafeNoopTTSProvider()
    else:
        tts = SafeNoopTTSProvider()

    return ProviderBundle(stt=stt, tts=tts, convo=convo)
