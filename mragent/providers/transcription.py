"""Voice transcription provider using Groq Whisper — fast & free tier.

Groq offers extremely fast transcription (whisper-large-v3-turbo is recommended:
faster, cheaper, multilingual). Free tier: ~2000 audio requests/day.
Sign up at https://console.groq.com — no credit card required.
"""

import os
from pathlib import Path

import httpx
from loguru import logger

_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq free tier limit: 25 MB


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    Default model: whisper-large-v3-turbo (faster + cheaper than v3).
    """

    def __init__(self, api_key: str | None = None, model: str = "whisper-large-v3-turbo"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path, language: str | None = None) -> str:
        """
        Transcribe an audio file using Groq Whisper.

        Args:
            file_path: Path to the audio file (WAV, MP3, FLAC, M4A, OGG, WebM).
                       Max size: 25 MB.
            language: Optional ISO-639-1 language code (e.g. "en", "fr"). Auto-detected if None.

        Returns:
            Transcribed text string, or "" on error.
        """
        if not self.api_key:
            logger.warning(
                "Groq API key not configured for transcription. "
                "Get a free key at https://console.groq.com and add it to "
                "~/.mragent/config.json under providers.groq.api_key"
            )
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        file_size = path.stat().st_size
        if file_size > _MAX_AUDIO_BYTES:
            logger.error(
                "Audio file too large ({:.1f} MB). Groq free tier supports up to 25 MB.",
                file_size / 1024 / 1024,
            )
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files: dict = {
                        "file": (path.name, f),
                        "model": (None, self.model),
                        "response_format": (None, "json"),
                    }
                    if language:
                        files["language"] = (None, language)

                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0,
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""
