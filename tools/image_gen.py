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

            result = get_image().generate_image(enhanced, model=model)
            filepath = result["filepath"]
            filename = Path(filepath).name
            self.logger.info(f"Image saved: {filepath}")

            # Return markdown image so web UI renders it inline
            return (
                f"✅ Image generated!\n\n"
                f"![{prompt}](/api/images/{filename})\n\n"
                f"**Prompt:** {prompt}\n"
                f"**Model:** {model}\n"
                f"**Saved to:** `{filepath}`"
            )

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return f"❌ Image generation failed: {e}"
