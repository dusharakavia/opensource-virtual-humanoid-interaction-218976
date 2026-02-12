from __future__ import annotations

import asyncio
import base64
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional, Protocol

import httpx

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


def _coerce_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _b64_to_bytes(audio_base64: str) -> bytes:
    return base64.b64decode(audio_base64.encode("utf-8"), validate=False)


class SafeNoopSTTProvider:
    """Safe fallback STT provider.

    This provider never fails hard. It returns an empty transcription if audio is not usable.
    """

    name = "noop-stt"

    async def transcribe(self, req: STTRequest) -> STTResponse:
        # Heuristic: sometimes clients send base64-encoded UTF-8 text for demo purposes.
        text_guess = ""
        try:
            raw = _b64_to_bytes(req.audio_base64)
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


class FasterWhisperSTTProvider:
    """Real open-source STT provider using faster-whisper.

    Requires dependency: faster-whisper

    Environment variables:
    - BACKEND_STT_WHISPER_MODEL: model id/path (default: "base")
      Examples: "base", "small", "medium", "large-v3", or local directory.
    - BACKEND_STT_WHISPER_DEVICE: "cpu" or "cuda" (default: "cpu")
    - BACKEND_STT_WHISPER_COMPUTE_TYPE: e.g. "int8", "float16" (default: "int8")
    """

    name = "faster-whisper"

    def __init__(self) -> None:
        self._model_id = (os.getenv("BACKEND_STT_WHISPER_MODEL") or "base").strip()
        self._device = (os.getenv("BACKEND_STT_WHISPER_DEVICE") or "cpu").strip()
        self._compute_type = (os.getenv("BACKEND_STT_WHISPER_COMPUTE_TYPE") or "int8").strip()
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        # Lazy import so environments without deps still run with fallbacks
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(self._model_id, device=self._device, compute_type=self._compute_type)
        return self._model

    async def transcribe(self, req: STTRequest) -> STTResponse:
        try:
            audio_bytes = _b64_to_bytes(req.audio_base64)
        except Exception:
            return STTResponse(
                session_id=req.session_id,
                text="",
                provider=self.name,
                confidence=None,
            )

        model = self._get_model()

        # faster-whisper can accept a file path; avoid decoding/format dependencies by
        # writing bytes to a temp file and letting ffmpeg/audioread handle it.
        def _run() -> tuple[str, Optional[float]]:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".audio") as f:
                f.write(audio_bytes)
                f.flush()
                segments, info = model.transcribe(
                    f.name,
                    language=req.language,
                    vad_filter=True,
                )
                text_out = "".join(seg.text for seg in segments).strip()
                # faster-whisper info provides language probability; confidence isn't standardized
                conf = getattr(info, "language_probability", None)
                try:
                    conf_f = float(conf) if conf is not None else None
                except Exception:
                    conf_f = None
                return text_out, conf_f

        try:
            text_out, conf = await asyncio.to_thread(_run)
            return STTResponse(
                session_id=req.session_id,
                text=text_out,
                provider=self.name,
                confidence=conf,
            )
        except Exception:
            # Never hard fail the API contract; degrade gracefully
            return STTResponse(
                session_id=req.session_id,
                text="",
                provider=self.name,
                confidence=None,
            )


class SafeNoopTTSProvider:
    """Safe fallback TTS provider.

    By default, it does not synthesize audio (avoids heavy dependencies).
    Optionally, if environment variable BACKEND_TTS_DUMMY_BEEP is set to '1',
    it returns a tiny placeholder payload (not a real audio synthesis).
    """

    name = "noop-tts"

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        if os.getenv("BACKEND_TTS_DUMMY_BEEP", "0") == "1":
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


class PiperTTSProvider:
    """Real open-source TTS provider using Piper.

    This provider shells out to a `piper` binary (recommended for production) or
    a container-provided binary, to avoid heavyweight Python dependencies and
    keep the process simple.

    Requirements:
    - `piper` binary available in PATH, or provide BACKEND_TTS_PIPER_BIN
    - A Piper voice model .onnx file path configured via BACKEND_TTS_PIPER_MODEL

    Environment variables:
    - BACKEND_TTS_PIPER_BIN: path to piper executable (default: "piper")
    - BACKEND_TTS_PIPER_MODEL: path to .onnx model (required)
    - BACKEND_TTS_PIPER_CONFIG: optional path to .onnx.json config
    - BACKEND_TTS_PIPER_SPEAKER: optional speaker id (integer) for multi-speaker voices
    - BACKEND_TTS_PIPER_LENGTH_SCALE: optional float (e.g. 1.0)
    - BACKEND_TTS_PIPER_NOISE_SCALE: optional float
    - BACKEND_TTS_PIPER_NOISE_W: optional float
    """

    name = "piper"

    def __init__(self) -> None:
        self._bin = (os.getenv("BACKEND_TTS_PIPER_BIN") or "piper").strip()
        self._model = (os.getenv("BACKEND_TTS_PIPER_MODEL") or "").strip()
        self._config = (os.getenv("BACKEND_TTS_PIPER_CONFIG") or "").strip()

        self._speaker = (os.getenv("BACKEND_TTS_PIPER_SPEAKER") or "").strip()
        self._length_scale = (os.getenv("BACKEND_TTS_PIPER_LENGTH_SCALE") or "").strip()
        self._noise_scale = (os.getenv("BACKEND_TTS_PIPER_NOISE_SCALE") or "").strip()
        self._noise_w = (os.getenv("BACKEND_TTS_PIPER_NOISE_W") or "").strip()

    async def synthesize(self, req: TTSRequest) -> TTSResponse:
        if not self._model:
            return TTSResponse(
                session_id=req.session_id,
                audio_base64=None,
                provider=self.name,
                status="not_available",
                detail="BACKEND_TTS_PIPER_MODEL not set.",
            )

        if req.mime_type and req.mime_type != "audio/wav":
            # We keep contract: the API returns audio_base64 and frontend assumes wav;
            # Piper produces WAV; if caller requests something else, still return WAV.
            pass

        def _run() -> bytes:
            import subprocess

            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as out_f:
                cmd = [self._bin, "--model", self._model, "--output_file", out_f.name]
                if self._config:
                    cmd += ["--config", self._config]
                if self._speaker:
                    cmd += ["--speaker", self._speaker]
                if self._length_scale:
                    cmd += ["--length_scale", self._length_scale]
                if self._noise_scale:
                    cmd += ["--noise_scale", self._noise_scale]
                if self._noise_w:
                    cmd += ["--noise_w", self._noise_w]

                # Piper reads text from stdin
                subprocess.run(
                    cmd,
                    input=req.text.encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                out_f.seek(0)
                return out_f.read()

        try:
            wav_bytes = await asyncio.to_thread(_run)
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            return TTSResponse(
                session_id=req.session_id,
                audio_base64=audio_b64,
                provider=self.name,
                status="ok",
                detail=None,
            )
        except Exception as e:
            return TTSResponse(
                session_id=req.session_id,
                audio_base64=None,
                provider=self.name,
                status="error",
                detail=f"Piper synthesis failed: {e}",
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


class LlamaCppConversationalProvider:
    """Local open-source LLM provider using llama-cpp-python.

    This runs fully locally (given a model file) and supports token streaming.

    Environment variables:
    - BACKEND_LLM_LLAMA_MODEL_PATH: required, path to .gguf model
    - BACKEND_LLM_LLAMA_N_CTX: context length (default 2048)
    - BACKEND_LLM_LLAMA_N_THREADS: threads (optional)
    - BACKEND_LLM_LLAMA_N_GPU_LAYERS: gpu layers (optional; requires proper build)
    - BACKEND_LLM_SYSTEM_PROMPT: optional system prompt
    """

    name = "llama-cpp"

    def __init__(self) -> None:
        self._model_path = (os.getenv("BACKEND_LLM_LLAMA_MODEL_PATH") or "").strip()
        self._n_ctx = int((os.getenv("BACKEND_LLM_LLAMA_N_CTX") or "2048").strip())
        self._n_threads = os.getenv("BACKEND_LLM_LLAMA_N_THREADS")
        self._n_gpu_layers = os.getenv("BACKEND_LLM_LLAMA_N_GPU_LAYERS")
        self._system = (os.getenv("BACKEND_LLM_SYSTEM_PROMPT") or "").strip()
        self._llm: Any = None

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        if not self._model_path:
            raise RuntimeError("BACKEND_LLM_LLAMA_MODEL_PATH is required for llama-cpp provider")

        from llama_cpp import Llama  # type: ignore

        kwargs: dict[str, Any] = {"model_path": self._model_path, "n_ctx": self._n_ctx}
        if self._n_threads:
            kwargs["n_threads"] = int(self._n_threads)
        if self._n_gpu_layers:
            kwargs["n_gpu_layers"] = int(self._n_gpu_layers)
        self._llm = Llama(**kwargs)
        return self._llm

    def _make_prompt(self, user_prompt: str) -> str:
        if self._system:
            # Keep simple "system + user" format
            return f"System: {self._system}\nUser: {user_prompt}\nAssistant:"
        return f"User: {user_prompt}\nAssistant:"

    async def generate(self, session_id: str, prompt: str) -> str:
        chunks: list[str] = []
        async for c in self.stream(session_id=session_id, prompt=prompt):
            chunks.append(c)
        return "".join(chunks)

    async def stream(self, session_id: str, prompt: str) -> AsyncGenerator[str, None]:
        llm = self._get_llm()
        full_prompt = self._make_prompt(prompt)

        def _iter_tokens() -> list[str]:
            # llama-cpp-python streaming yields dict events with "choices"
            out: list[str] = []
            for ev in llm.create_completion(
                prompt=full_prompt,
                max_tokens=512,
                temperature=0.7,
                stream=True,
            ):
                try:
                    token = ev["choices"][0].get("text") or ""
                except Exception:
                    token = ""
                if token:
                    out.append(token)
            return out

        # Run token generation off the event loop; yield progressively.
        tokens = await asyncio.to_thread(_iter_tokens)
        for t in tokens:
            yield t
            await asyncio.sleep(0)


class OpenAICompatibleConversationalProvider:
    """Conversational provider using an OpenAI-compatible local server (open-source).

    This supports servers like:
    - vLLM (OpenAI compatible)
    - llama.cpp server (--api)
    - LocalAI
    - Ollama (via its OpenAI-compat layer, if enabled)

    Environment variables:
    - BACKEND_LLM_OPENAI_BASE_URL: required (e.g. "http://localhost:8000/v1")
    - BACKEND_LLM_OPENAI_API_KEY: optional (many local servers ignore it)
    - BACKEND_LLM_OPENAI_MODEL: required model id on that server
    - BACKEND_LLM_SYSTEM_PROMPT: optional system prompt
    - BACKEND_LLM_STREAM: "1" to stream (default "1")
    """

    name = "openai-compatible"

    def __init__(self) -> None:
        self._base_url = (os.getenv("BACKEND_LLM_OPENAI_BASE_URL") or "").strip().rstrip("/")
        self._api_key = (os.getenv("BACKEND_LLM_OPENAI_API_KEY") or "").strip()
        self._model = (os.getenv("BACKEND_LLM_OPENAI_MODEL") or "").strip()
        self._system = (os.getenv("BACKEND_LLM_SYSTEM_PROMPT") or "").strip()
        self._stream_default = _coerce_bool(os.getenv("BACKEND_LLM_STREAM"), default=True)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def generate(self, session_id: str, prompt: str) -> str:
        chunks: list[str] = []
        async for c in self.stream(session_id=session_id, prompt=prompt):
            chunks.append(c)
        return "".join(chunks)

    async def stream(self, session_id: str, prompt: str) -> AsyncGenerator[str, None]:
        if not self._base_url or not self._model:
            # Misconfigured: degrade gracefully
            text = await RuleBasedConversationalProvider().generate(session_id=session_id, prompt=prompt)
            for w in re.split(r"(\s+)", text):
                if w:
                    yield w
            return

        messages: list[dict[str, str]] = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            try:
                async with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
                    resp.raise_for_status()
                    # Streaming is SSE-like: lines that begin with "data: ..."
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data = line[len("data:") :].strip()
                        else:
                            data = line.strip()

                        if data == "[DONE]":
                            break

                        # Parse minimal OpenAI stream chunk
                        try:
                            import json

                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta") or {}
                            content = delta.get("content") or ""
                            if content:
                                yield content
                        except Exception:
                            # ignore malformed lines
                            continue
            except Exception:
                text = await RuleBasedConversationalProvider().generate(session_id=session_id, prompt=prompt)
                for w in re.split(r"(\s+)", text):
                    if w:
                        yield w


# PUBLIC_INTERFACE
def resolve_providers() -> ProviderBundle:
    """Resolve STT/TTS/conversation providers based on environment.

    Environment variables:
    - BACKEND_CONVO_PROVIDER:
        - "rule-based" (default, safe fallback)
        - "llama-cpp" (local llama-cpp-python; requires BACKEND_LLM_LLAMA_MODEL_PATH)
        - "openai-compatible" (for vLLM/LocalAI/llama.cpp server; requires BACKEND_LLM_OPENAI_BASE_URL + BACKEND_LLM_OPENAI_MODEL)
    - BACKEND_STT_PROVIDER:
        - "noop" (default, safe fallback)
        - "faster-whisper" (requires faster-whisper)
    - BACKEND_TTS_PROVIDER:
        - "noop" (default, safe fallback)
        - "piper" (requires piper binary + BACKEND_TTS_PIPER_MODEL)

    Returns:
        ProviderBundle: providers to be used by the orchestrator.

    Notes:
        - Providers are lazy-loaded where possible to preserve safe operation in minimal envs.
        - API/WS contracts are preserved (same request/response shapes).
    """
    convo_choice = (os.getenv("BACKEND_CONVO_PROVIDER") or "rule-based").strip().lower()
    stt_choice = (os.getenv("BACKEND_STT_PROVIDER") or "noop").strip().lower()
    tts_choice = (os.getenv("BACKEND_TTS_PROVIDER") or "noop").strip().lower()

    # Conversation provider
    if convo_choice in {"rule-based", "fallback"}:
        convo: ConversationalProvider = RuleBasedConversationalProvider()
    elif convo_choice in {"llama-cpp", "llamacpp"}:
        try:
            convo = LlamaCppConversationalProvider()
            # Validate minimal config early; if misconfigured fall back
            if not (os.getenv("BACKEND_LLM_LLAMA_MODEL_PATH") or "").strip():
                raise RuntimeError("Missing BACKEND_LLM_LLAMA_MODEL_PATH")
        except Exception:
            convo = RuleBasedConversationalProvider()
    elif convo_choice in {"openai-compatible", "openai", "vllm", "localai"}:
        convo = OpenAICompatibleConversationalProvider()
    else:
        convo = RuleBasedConversationalProvider()

    # STT provider
    if stt_choice in {"noop", "fallback"}:
        stt: STTProvider = SafeNoopSTTProvider()
    elif stt_choice in {"faster-whisper", "whisper"}:
        try:
            stt = FasterWhisperSTTProvider()
        except Exception:
            stt = SafeNoopSTTProvider()
    else:
        stt = SafeNoopSTTProvider()

    # TTS provider
    if tts_choice in {"noop", "fallback"}:
        tts: TTSProvider = SafeNoopTTSProvider()
    elif tts_choice in {"piper"}:
        tts = PiperTTSProvider()
    else:
        tts = SafeNoopTTSProvider()

    return ProviderBundle(stt=stt, tts=tts, convo=convo)
