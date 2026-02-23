"""
MRAgent — Image Generation Tool
Allows the LLM to generate images during conversation.
Supports aspect ratio selection and Google→FLUX automatic fallback.

Created: 2026-02-16
Updated: 2026-02-23 — Google AI Studio primary, aspect ratio support
"""

from tools.base import Tool
from utils.logger import get_logger

logger = get_logger("tools.image_gen")


class GenerateImageTool(Tool):
    """Generate an image from a text prompt using Google AI Studio or NVIDIA."""

    name = "generate_image"
    description = (
        "Generate an image from a text description. "
        "Returns the file path of the saved image. "
        "Use this when the user asks you to create, draw, or generate an image. "
        "You can specify aspect_ratio based on user preferences."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the image to generate",
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Image aspect ratio. Use '16:9' for landscape/widescreen, '9:16' for portrait/phone, '4:3' for standard landscape, '3:4' for standard portrait, '1:1' for square (default)",
                "enum": ["1:1", "4:3", "3:4", "16:9", "9:16"],
            },
        },
        "required": ["prompt"],
    }

    def execute(self, prompt: str, aspect_ratio: str = "1:1") -> str:
        try:
            from providers import get_image
            from agents.prompt_enhancer import PromptEnhancer
            from pathlib import Path

            # Enhance the prompt for better results
            enhanced = PromptEnhancer().build_image_prompt(prompt)
            self.logger.info(f"Generating image: {enhanced[:80]}...")

            provider = get_image()
            provider_name = getattr(provider, 'name', 'unknown')
            used_fallback = False

            try:
                result = provider.generate_image(
                    enhanced,
                    aspect_ratio=aspect_ratio,
                )
            except Exception as primary_err:
                # If Google provider fails (quota, error), fallback to NVIDIA
                if provider_name == "google_image":
                    self.logger.warning(f"Google image failed ({primary_err}), falling back to NVIDIA FLUX")
                    from providers.nvidia_image import NvidiaImageProvider
                    fallback = NvidiaImageProvider()
                    result = fallback.generate_image(enhanced, model="flux-dev")
                    used_fallback = True
                else:
                    raise

            filepath = result["filepath"]
            filename = Path(filepath).name
            used_model = result.get("model", "unknown")
            self.logger.info(f"Image saved: {filepath}")

            fallback_note = " *(fell back to NVIDIA FLUX)*" if used_fallback else ""

            # Return with image tag on its own line for reliable markdown parsing
            return (
                f"Image generated successfully{fallback_note}. "
                f"Model: {used_model}. Aspect: {aspect_ratio}.\n\n"
                f"![{prompt}](/api/images/{filename})\n\n"
                f"Saved to: {filepath}"
            )

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return f"Image generation failed: {e}"
