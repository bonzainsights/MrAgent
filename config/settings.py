"""
MRAgent — Configuration System
Loads API keys from .env, manages model registry, and handles config snapshots.

Created: 2026-02-15
"""

import os
import json
import platform
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# Load .env
# ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = _PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
IMAGES_DIR = DATA_DIR / "images"
CONFIG_BACKUP_DIR = DATA_DIR / "config_backups"
CHAT_DB_PATH = DATA_DIR / "chats.db"

# Create directories on import
for d in [DATA_DIR, LOGS_DIR, IMAGES_DIR, CONFIG_BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# NVIDIA API Keys & URLs
# ──────────────────────────────────────────────
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

NVIDIA_KEYS = {
    "kimi_k2_5":     os.getenv("NVIDIA_KIMI_K2_5", ""),
    "glm5":          os.getenv("NVIDIA_GLM5", ""),
    "sd_35_large":   os.getenv("NVIDIA_SD_35_LARGE", ""),
    "whisper_lv3":   os.getenv("NVIDIA_WHISPER_LV3", ""),
    "flux_dev":      os.getenv("NVIDIA_FLUX_DEV", ""),
    "magpie_tts":    os.getenv("NVIDIA_MAGPIE_TTS", ""),
    "gemma_3n":      os.getenv("NVIDIA_GEMMA_3N", ""),
    "qwen3_coder":   os.getenv("NVIDIA_QWEN3_CODER", ""),
    "llama_33_70b":  os.getenv("NVIDIA_LLAMA3_3_70B_INSTRUCT", ""),
    "gpt_oss_120b":  os.getenv("NVIDIA_GPT_OSS_120B", ""),
}

# Other provider keys
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────
# Each model has:
#   id          — NVIDIA NIM model ID (verified from build.nvidia.com)
#   key         — maps to NVIDIA_KEYS dict above
#   type        — "llm" or "image"
#   categories  — list of modes this model can serve: thinking, fast, code
#   context_window   — max tokens
#   supports_tools   — whether it accepts function calling
#   description      — shown in /model list
#
# MODES map to categories:
#   mode thinking → picks default from models tagged "thinking"
#   mode fast     → picks default from models tagged "fast"
#   mode code     → picks default from models tagged "code"
#   mode auto     → classifies message and picks the best category

MODEL_REGISTRY = {
    # ── Reasoning + General ──
    "kimi-k2.5": {
        "id": "moonshotai/kimi-k2.5",
        "key": "kimi_k2_5",
        "type": "llm",
        "categories": ["thinking", "fast", "code"],
        "context_window": 131_072,
        "supports_tools": True,
        "description": "All-rounder: reasoning, coding & fast replies",
    },
    "glm5": {
        "id": "z-ai/glm5",
        "key": "glm5",
        "type": "llm",
        "categories": ["thinking"],
        "context_window": 128_000,
        "supports_tools": True,
        "description": "Strong reasoning & tool use by Z.ai",
    },

    # ── Fast ──
    "gemma-3n": {
        "id": "google/gemma-3n-e4b-it",
        "key": "gemma_3n",
        "type": "llm",
        "categories": ["fast"],
        "context_window": 32_000,
        "supports_tools": False,
        "description": "Fastest responses, lightweight (default fast)",
    },

    # ── Code ──
    "qwen3-coder": {
        "id": "qwen/qwen3-coder-480b-a35b-instruct",
        "key": "qwen3_coder",
        "type": "llm",
        "categories": ["code"],
        "context_window": 262_144,
        "supports_tools": True,
        "description": "480B MoE agentic coder, 1M context (default code)",
    },

    # ── Fallback ──
    "llama-3.3-70b": {
        "id": "meta/llama-3.3-70b-instruct",
        "key": "llama_33_70b",
        "type": "llm",
        "categories": ["thinking", "fast"],
        "context_window": 128_000,
        "supports_tools": True,
        "description": "Reliable fallback, strong general-purpose LLM",
    },

    "gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "key": "gpt_oss_120b",
        "type": "llm",
        "categories": ["thinking", "fast", "code"],
        "context_window": 128_000,
        "supports_tools": True,
        "description": "Open reasoning & coding model (Default)",
    },

    # Image Models — verified from build.nvidia.com (2026-02-16)
    # Note: SD 3.5 Large has NO hosted API — download only. Using SD 3 Medium instead.
    "sd-3-medium": {
        "id": "stabilityai/stable-diffusion-3-medium",
        "key": "sd_35_large",
        "type": "image",
        "description": "High-quality image generation (Stable Diffusion 3 Medium)",
    },
    "flux-dev": {
        "id": "black-forest-labs/flux.1-dev",
        "key": "flux_dev",
        "type": "image",
        "description": "Fast creative image generation",
    },

    # Voice Models
    "magpie-tts": {
        "id": "nvidia/magpie-tts-multilingual",
        "key": "magpie_tts",
        "type": "tts",
        "description": "Natural multilingual text-to-speech",
    },
    "whisper-lv3": {
        "id": "nvidia/whisper-large-v3",
        "key": "whisper_lv3",
        "type": "stt",
        "description": "Accurate speech-to-text transcription",
    },
}


# ──────────────────────────────────────────────
# Default Settings
# ──────────────────────────────────────────────
DEFAULTS = {
    "default_llm": "gpt-oss-120b",
    "default_image_model": "sd-3.5-large",
    "model_selection_mode": "auto",    # auto | thinking | fast | code
    "voice_enabled": False,
    "web_port": 7860,
    "max_tool_retries": 3,
    "rate_limit_rpm": 35,              # Stay under NVIDIA's 40 RPM
    "context_threshold": 0.8,          # Summarize at 80% context usage
    "max_config_backups": 3,
    "log_level": "INFO",
}


# ──────────────────────────────────────────────
# System Info (for context injection)
# ──────────────────────────────────────────────
SYSTEM_INFO = {
    "os": platform.system(),
    "os_version": platform.version(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "machine": platform.machine(),
    "hostname": platform.node(),
}


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def get_api_key(model_name: str) -> str:
    """Get the API key for a given model name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    key_name = MODEL_REGISTRY[model_name]["key"]
    key = NVIDIA_KEYS.get(key_name, "")
    if not key:
        raise ValueError(f"API key not set for {model_name} (env var: NVIDIA_{key_name.upper()})")
    return key


def get_model_id(model_name: str) -> str:
    """Get the NVIDIA NIM model ID for a given friendly name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]["id"]


def get_llm_models() -> dict:
    """Return all registered LLM models."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v["type"] == "llm"}


def get_available_models() -> dict:
    """Return all models that have valid API keys configured."""
    available = {}
    for name, info in MODEL_REGISTRY.items():
        key_name = info["key"]
        if NVIDIA_KEYS.get(key_name):
            available[name] = info
    return available


def validate_config() -> dict:
    """Check which API keys are configured and return a status report."""
    report = {"valid": [], "missing": [], "warnings": []}
    for name, info in MODEL_REGISTRY.items():
        key_name = info["key"]
        if NVIDIA_KEYS.get(key_name):
            report["valid"].append(name)
        else:
            report["missing"].append(name)

    if not BRAVE_SEARCH_API_KEY:
        report["warnings"].append("BRAVE_SEARCH_API_KEY not set — web search disabled")
    if not TELEGRAM_BOT_TOKEN:
        report["warnings"].append("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled")

    return report


def export_config() -> dict:
    """Export current config as a dict (for backup/restore). Keys are NOT included."""
    return {
        "defaults": DEFAULTS.copy(),
        "timestamp": datetime.now().isoformat(),
        "system_info": SYSTEM_INFO.copy(),
        "available_models": list(get_available_models().keys()),
    }


def save_config_backup():
    """Save current config snapshot for rollback."""
    config = export_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CONFIG_BACKUP_DIR / f"config_{timestamp}.json"
    backup_path.write_text(json.dumps(config, indent=2))

    # Keep only last N backups
    backups = sorted(CONFIG_BACKUP_DIR.glob("config_*.json"))
    while len(backups) > DEFAULTS["max_config_backups"]:
        backups[0].unlink()
        backups.pop(0)

    return backup_path


def load_config_backup(steps_back: int = 1) -> dict | None:
    """Load a previous config backup. steps_back=1 means the most recent backup."""
    backups = sorted(CONFIG_BACKUP_DIR.glob("config_*.json"))
    if not backups or steps_back > len(backups):
        return None
    target = backups[-steps_back]
    return json.loads(target.read_text())
