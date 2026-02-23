"""
MRAgent — Google AI Studio Image Generation Provider
Uses Gemini's generate_images API (gemini-2.5-flash-image) via google-genai SDK.
Free tier: 100 images/day — falls back to NVIDIA FLUX when exhausted.

Created: 2026-02-23
Updated: 2026-02-23 — Use generate_images() API per Google docs
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

# Default model for image generation
GOOGLE_IMAGE_MODEL = "gemini-2.5-flash-image"

# Valid aspect ratios
VALID_ASPECT_RATIOS = {"1:1", "4:3", "3:4", "16:9", "9:16"}


class GoogleImageProvider(ImageProvider):
    """
    Google AI Studio image generation via Gemini generate_images API.
    Uses the free-tier google-genai SDK (100 images/day).
    """

    def __init__(self):
        super().__init__(name="google_image", rate_limit_rpm=10)
        import os
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_KEY", "")
        self._available = bool(self.api_key)
        if self._available:
            self.logger.info("Google Image provider initialized (Gemini generate_images)")
        else:
            self.logger.debug("Google Image provider not available (GOOGLE_AI_STUDIO_KEY not set)")

    @property
    def available(self) -> bool:
        return self._available

    def generate_image(self, prompt: str, model: str = None,
                       aspect_ratio: str = "1:1", **kwargs) -> dict:
        """
        Generate an image using Gemini's generate_images API.

        Args:
            prompt: Text description of the image to generate.
            model: Ignored for Google — always uses gemini-2.5-flash-image.
            aspect_ratio: Image aspect ratio (1:1, 4:3, 3:4, 16:9, 9:16).

        Returns:
            {"base64": str, "filepath": Path, "model": str, "prompt": str}
        """
        if not self._available:
            raise ValueError("GOOGLE_AI_STUDIO_KEY not set")

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai not installed. Install with: pip install google-genai")

        # Always use Google's own model
        target_model = GOOGLE_IMAGE_MODEL

        # Validate aspect ratio
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            aspect_ratio = "1:1"

        self.logger.info(f"Generating image via Google: model={target_model}, "
                         f"aspect={aspect_ratio}, prompt='{prompt[:60]}...'")
        start_time = time.time()

        try:
            client = genai.Client(api_key=self.api_key)

            response = client.models.generate_images(
                model=target_model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    safety_filter_level="block_some",
                ),
            )

            # Extract image from response
            if not response.generated_images:
                raise ValueError("No image returned from Google API (may be quota or safety filter)")

            image_bytes = response.generated_images[0].image.image_bytes
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
