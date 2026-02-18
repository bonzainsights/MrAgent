"""
MRAgent — Text-to-Speech Provider
Uses edge-tts (Microsoft Edge's free online TTS) for high-quality neural speech.
"""

import os
import asyncio
import edge_tts
from utils.logger import get_logger

logger = get_logger("providers.tts")

# Friendly Neural Voices
# en-US-EmmaNeural: Friendly, warm
# en-US-AnaNeural: Child/Teen, very natural
# en-US-AriaNeural: Professional, clear
# en-GB-SoniaNeural: British, professional
VOICES = [
    "en-US-EmmaNeural",
    "en-US-AnaNeural",
    "en-US-AriaNeural", 
    "en-GB-SoniaNeural"
]

# Set a warmer default
DEFAULT_VOICE = "en-US-EmmaNeural"

async def text_to_speech(text: str, output_file: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Convert text to speech and save to output_file.
    
    Args:
        text: The text to speak.
        output_file: Path to save the audio file.
        voice: The voice to use.
        
    Returns:
        The path to the output file if successful, else None.
    """
    try:
        if not text:
            return None
            
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        
        logger.info(f"TTS generated: {output_file} ({len(text)} chars)")
        return output_file
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None
