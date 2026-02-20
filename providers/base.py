"""
MRAgent — Base Provider Interface
Abstract base class for all API providers with retry logic and rate limiting.

Created: 2026-02-15
"""

import time
from abc import ABC, abstractmethod
from typing import Generator

from utils.logger import get_logger, log_api_call

logger = get_logger("providers.base")


class RateLimiter:
    """Simple rate limiter that tracks requests per minute."""

    def __init__(self, max_rpm: int = 35):
        self.max_rpm = max_rpm
        self._timestamps: list[float] = []

    def wait_if_needed(self):
        """Block if we've hit the rate limit. Clears old timestamps."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._timestamps = [t for t in self._timestamps if now - t < 60]

        if len(self._timestamps) >= self.max_rpm:
            sleep_time = 60 - (now - self._timestamps[0]) + 0.1
            if sleep_time > 0:
                logger.warning(f"Rate limit reached ({self.max_rpm} RPM). Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self._timestamps.append(time.time())

    @property
    def requests_remaining(self) -> int:
        now = time.time()
        recent = [t for t in self._timestamps if now - t < 60]
        return max(0, self.max_rpm - len(recent))


class BaseProvider(ABC):
    """
    Abstract base class for all API providers.

    Subclasses should implement the specific API methods they support.
    Not all providers support all methods — use `supports()` to check.
    """

    def __init__(self, name: str, rate_limit_rpm: int = 35):
        self.name = name
        self.rate_limiter = RateLimiter(max_rpm=rate_limit_rpm)
        self.logger = get_logger(f"providers.{name}")
        self._call_count = 0
        self._error_count = 0

    def supports(self, capability: str) -> bool:
        """Check if this provider supports a capability: 'chat', 'image', 'tts', 'stt', 'search'."""
        method_map = {
            "chat": "chat",
            "image": "generate_image",
            "tts": "text_to_speech",
            "stt": "speech_to_text",
            "search": "search",
        }
        method_name = method_map.get(capability)
        if not method_name:
            return False
        method = getattr(self, method_name, None)
        # Check if the method is actually implemented (not just the abstract stub)
        return method is not None and not getattr(method, "__isabstractmethod__", False)

    def _track_call(self, endpoint: str, model: str = "",
                    duration_ms: float = 0, status: str = "ok",
                    tokens_used: int = 0):
        """Log and track an API call."""
        self._call_count += 1
        if status != "ok":
            self._error_count += 1
        log_api_call(self.logger, self.name, endpoint, model, duration_ms, status, tokens_used)

    def _retry_call(self, func, max_retries: int = 3, base_delay: float = 1.0):
        """Execute a function with exponential backoff retry."""
        last_exception = None
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                return func()
            except Exception as e:
                last_exception = e
                # Fail fast on unrecoverable authentication or routing errors
                if hasattr(e, "status_code") and getattr(e, "status_code") in (401, 403, 404):
                    self.logger.error(f"Unrecoverable API error {getattr(e, 'status_code')}: {e}. Skipping retries.")
                    raise e
                    
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), 30.0)
                    time.sleep(delay)
        raise last_exception

    @property
    def stats(self) -> dict:
        """Return provider usage statistics."""
        return {
            "provider": self.name,
            "total_calls": self._call_count,
            "errors": self._error_count,
            "rate_limit_remaining": self.rate_limiter.requests_remaining,
        }


class LLMProvider(BaseProvider):
    """Base class for LLM (chat) providers."""

    @abstractmethod
    def chat(self, messages: list[dict], model: str = "",
             stream: bool = True, tools: list[dict] = None,
             temperature: float = 0.7, max_tokens: int = 4096) -> dict | Generator:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            model: Model ID to use
            stream: Whether to stream the response
            tools: List of tool/function definitions for function calling
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response

        Returns:
            If stream=False: dict with {"content": str, "tool_calls": list, "usage": dict}
            If stream=True: Generator yielding {"delta": str} chunks
        """
        pass


class ImageProvider(BaseProvider):
    """Base class for image generation providers."""

    @abstractmethod
    def generate_image(self, prompt: str, model: str = "",
                       width: int = 1024, height: int = 1024,
                       steps: int = 50, cfg_scale: float = 5.0,
                       seed: int = 0) -> dict:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            model: Model ID to use
            width/height: Image dimensions
            steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (0 for random)

        Returns:
            dict with {"base64": str, "seed": int, "filepath": Path}
        """
        pass


class TTSProvider(BaseProvider):
    """Base class for text-to-speech providers."""

    @abstractmethod
    def text_to_speech(self, text: str, voice: str = "",
                       language: str = "en-US",
                       sample_rate: int = 44100) -> bytes:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize
            voice: Voice name/ID
            language: Language code
            sample_rate: Audio sample rate in Hz

        Returns:
            Raw audio bytes (PCM/WAV)
        """
        pass


class STTProvider(BaseProvider):
    """Base class for speech-to-text providers."""

    @abstractmethod
    def speech_to_text(self, audio_bytes: bytes,
                       language: str = "en-US",
                       sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: Raw audio data
            language: Language code
            sample_rate: Audio sample rate in Hz

        Returns:
            Transcribed text string
        """
        pass


class SearchProvider(BaseProvider):
    """Base class for web search providers."""

    @abstractmethod
    def search(self, query: str, count: int = 5) -> list[dict]:
        """
        Search the web.

        Args:
            query: Search query
            count: Number of results

        Returns:
            List of {"title": str, "url": str, "description": str} dicts
        """
        pass
