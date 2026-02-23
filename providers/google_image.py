"""
MRAgent — Google AI Studio Image Generation Provider
Uses Gemini's native image generation (gemini-2.5-flash-image) via google-genai SDK.
Free tier with quota — falls back to NVIDIA FLUX when exhausted.

Created: 2026-02-23
"""

import time
import base64
from pathlib import Path

from providers.base import ImageProvider
from config.settings import IMAGES_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp_short

logger = get_logger("providers.google_image")

# Default model for image generation
GOOGLE_IMAGE_MODEL = "gemini-2.5-flash-preview-image-generation"


class GoogleImageProvider(ImageProvider):
    """
    Google AI Studio image generation via Gemini native image generation.
    Uses the free-tier google-genai SDK.
    """

    def __init__(self):
        super().__init__(name="google_image", rate_limit_rpm=10)
        import os
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_KEY", "")
        self._available = bool(self.api_key)
        if self._available:
            self.logger.info("Google Image provider initialized (Gemini native image gen)")
        else:
            self.logger.debug("Google Image provider not available (GOOGLE_AI_STUDIO_KEY not set)")

    @property
    def available(self) -> bool:
        return self._available

    def generate_image(self, prompt: str, model: str = None, **kwargs) -> dict:
        """
        Generate an image using Gemini's native image generation.

        Args:
            prompt: Text description of the image to generate.
            model: Model override (default: gemini-2.5-flash-preview-image-generation)

        Returns:
            {"base64": str, "filepath": Path, "model": str, "prompt": str}

        Raises:
            Exception on quota exhaustion or API errors.
        """
        if not self._available:
            raise ValueError("GOOGLE_AI_STUDIO_KEY not set")

        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed. Install with: pip install google-genai")

        target_model = model or GOOGLE_IMAGE_MODEL
        self.logger.info(f"Generating image via Google: model={target_model}, prompt='{prompt[:60]}...'")
        start_time = time.time()

        try:
            client = genai.Client(api_key=self.api_key)

            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
            )

            # Extract image from response parts
            b64_image = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    # inline_data.data is raw bytes
                    image_bytes = part.inline_data.data
                    b64_image = base64.b64encode(image_bytes).decode("utf-8")
                    break

            if not b64_image:
                raise ValueError("No image returned from Google API (may be quota or safety filter)")

            # Save to disk
            timestamp = get_timestamp_short()
            filename = f"img_{timestamp}_google_gemini.png"
            filepath = IMAGES_DIR / filename
            filepath.write_bytes(base64.b64decode(b64_image))

            duration_ms = (time.time() - start_time) * 1000
            self._track_call("image/generate", target_model, duration_ms, status="ok")
            self.logger.info(f"Image saved: {filepath} ({duration_ms:.0f}ms)")

            return {
                "base64": b64_image,
                "filepath": filepath,
                "model": target_model,
                "prompt": prompt,
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._track_call("image/generate", target_model, duration_ms, status=f"error: {e}")
            error_str = str(e).lower()
            # Detect quota exhaustion
            if any(kw in error_str for kw in ["quota", "429", "rate limit", "resource exhausted"]):
                self.logger.warning(f"Google image quota exhausted: {e}")
                raise QuotaExhaustedError(f"Google image API quota exhausted: {e}") from e
            raise


class QuotaExhaustedError(Exception):
    """Raised when the Google API free tier quota is exhausted."""
    pass
