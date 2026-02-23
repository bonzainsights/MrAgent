"""
MRAgent — Google AI Studio Image Generation Provider
Uses Gemini's generate_content API with gemini-2.5-flash-image model.
Free tier: ~100 images/day — falls back to NVIDIA FLUX when exhausted.

Created: 2026-02-23
Updated: 2026-02-23 — Use generate_content() with gemini-2.5-flash-image
"""

import time
import base64
from pathlib import Path
from io import BytesIO

from providers.base import ImageProvider
from config.settings import IMAGES_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp_short

logger = get_logger("providers.google_image")

# Correct model for free-tier image generation via generate_content
GOOGLE_IMAGE_MODEL = "gemini-2.5-flash-image"


class GoogleImageProvider(ImageProvider):
    """
    Google AI Studio image generation via Gemini generate_content API.
    Uses the free-tier google-genai SDK.
    """

    def __init__(self):
        super().__init__(name="google_image", rate_limit_rpm=10)
        import os
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_KEY", "")
        self._available = bool(self.api_key)
        if self._available:
            self.logger.info("Google Image provider initialized (gemini-2.5-flash-image)")
        else:
            self.logger.debug("Google Image provider not available (GOOGLE_AI_STUDIO_KEY not set)")

    @property
    def available(self) -> bool:
        return self._available

    def generate_image(self, prompt: str, model: str = None,
                       aspect_ratio: str = "1:1", **kwargs) -> dict:
        """
        Generate an image using Gemini's generate_content API.

        Args:
            prompt: Text description of the image to generate.
            model: Ignored — always uses gemini-2.5-flash-image.
            aspect_ratio: Hint added to prompt (1:1, 4:3, 3:4, 16:9, 9:16).

        Returns:
            {"base64": str, "filepath": Path, "model": str, "prompt": str}
        """
        if not self._available:
            raise ValueError("GOOGLE_AI_STUDIO_KEY not set")

        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed. Install with: pip install google-genai")

        target_model = GOOGLE_IMAGE_MODEL

        # Add aspect ratio hint to prompt if not square
        enhanced_prompt = prompt
        if aspect_ratio and aspect_ratio != "1:1":
            ratio_hints = {
                "16:9": "widescreen landscape format",
                "9:16": "tall portrait format",
                "4:3": "standard landscape format",
                "3:4": "standard portrait format",
            }
            hint = ratio_hints.get(aspect_ratio, "")
            if hint:
                enhanced_prompt = f"{prompt}, {hint}"

        self.logger.info(f"Generating image via Google: model={target_model}, prompt='{prompt[:60]}...'")
        start_time = time.time()

        try:
            client = genai.Client(api_key=self.api_key)

            # Use generate_content — this is the correct API for gemini-2.5-flash-image
            response = client.models.generate_content(
                model=target_model,
                contents=[enhanced_prompt],
            )

            # Extract image from response parts
            image_bytes = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image_bytes = part.inline_data.data
                    break

            if not image_bytes:
                raise ValueError("No image returned from Google API (may be safety filter or quota)")

            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Save to disk
            timestamp = get_timestamp_short()
            filename = f"img_{timestamp}_google_gemini.png"
            filepath = IMAGES_DIR / filename
            filepath.write_bytes(image_bytes)

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
