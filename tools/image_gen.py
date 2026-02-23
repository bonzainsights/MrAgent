"""
MRAgent — Image Generation Tool
Allows the LLM to generate images during conversation.

Created: 2026-02-16
"""

from tools.base import Tool
from utils.logger import get_logger

logger = get_logger("tools.image_gen")


class GenerateImageTool(Tool):
    """Generate an image from a text prompt using NVIDIA NIM."""

    name = "generate_image"
    description = (
        "Generate an image from a text description. "
        "Returns the file path of the saved image. "
        "Use this when the user asks you to create, draw, or generate an image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the image to generate",
            },
            "model": {
                "type": "string",
                "description": "Image model: 'flux-dev' (default, best quality) or 'sd-3-medium'",
                "enum": ["flux-dev", "sd-3-medium"],
            },
        },
        "required": ["prompt"],
    }

    def execute(self, prompt: str, model: str = "flux-dev") -> str:
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
                result = provider.generate_image(enhanced, model=model)
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
            used_model = result.get("model", model)
            self.logger.info(f"Image saved: {filepath}")

            fallback_note = " *(fell back to NVIDIA FLUX)*" if used_fallback else ""

            # Return markdown image so web UI renders it inline
            return (
                f"✅ Image generated!{fallback_note}\n\n"
                f"![{prompt}](/api/images/{filename})\n\n"
                f"**Prompt:** {prompt}\n"
                f"**Model:** {used_model}\n"
                f"**Saved to:** `{filepath}`"
            )

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return f"❌ Image generation failed: {e}"
