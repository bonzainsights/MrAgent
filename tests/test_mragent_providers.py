"""Tests for MRAgent-specific provider additions."""

from mragent.config.schema import Config
from mragent.providers.registry import PROVIDERS, find_by_name


# ── Provider Registry ──────────────────────────────────────────────

def test_nvidia_nim_registry_entry():
    """NVIDIA NIM should be in the provider registry."""
    spec = find_by_name("nvidia_nim")
    assert spec is not None, "nvidia_nim must be in PROVIDERS"
    assert spec.display_name == "NVIDIA NIM"
    assert spec.is_gateway is True
    assert spec.default_api_base == "https://integrate.api.nvidia.com/v1"
    assert spec.detect_by_key_prefix == "nvapi-"


def test_nvidia_nim_config_field():
    """Config should have an nvidia_nim field."""
    cfg = Config()
    assert hasattr(cfg.providers, "nvidia_nim"), "ProvidersConfig must have nvidia_nim field"
    assert cfg.providers.nvidia_nim.api_key == ""


def test_qwen_oauth_registry_entry():
    """Qwen OAuth should be in the registry with OAuth + direct flags."""
    spec = find_by_name("qwen_oauth")
    assert spec is not None, "qwen_oauth must be in PROVIDERS"
    assert spec.is_oauth is True
    assert spec.is_direct is True
    assert spec.env_key == ""
    assert "chat.qwen.ai" in spec.default_api_base


def test_qwen_oauth_config_field():
    """Config should have a qwen_oauth field."""
    cfg = Config()
    assert hasattr(cfg.providers, "qwen_oauth"), "ProvidersConfig must have qwen_oauth field"


def test_mragent_env_prefix():
    """Config env prefix must be MRAGENT_."""
    cfg = Config()
    assert cfg.model_config["env_prefix"] == "MRAGENT_"


def test_mragent_default_workspace():
    """Default workspace should be under ~/.mragent."""
    cfg = Config()
    assert ".mragent" in cfg.agents.defaults.workspace


def test_mragent_web_ui_config():
    """GatewayConfig should include WebUIConfig with default port 6326."""
    cfg = Config()
    assert hasattr(cfg.gateway, "web"), "GatewayConfig must have web (WebUIConfig)"
    assert cfg.gateway.web.port == 6326


# ── Qwen OAuth Provider ────────────────────────────────────────────

def test_qwen_oauth_provider_init(tmp_path, monkeypatch):
    """QwenOAuthProvider should initialise without error."""
    from mragent.providers.qwen_oauth_provider import QwenOAuthProvider, _TOKEN_FILE

    # Point token file to tmp location so test is isolated
    monkeypatch.setattr(
        "mragent.providers.qwen_oauth_provider._TOKEN_FILE",
        tmp_path / "qwen_token.json",
    )
    provider = QwenOAuthProvider(default_model="qwen-plus")
    assert provider.default_model == "qwen-plus"
    assert provider._token is None  # no cached token in tmp


def test_qwen_oauth_provider_get_default_model():
    """get_default_model should return the configured model."""
    from mragent.providers.qwen_oauth_provider import QwenOAuthProvider
    p = QwenOAuthProvider(default_model="qwen-turbo")
    assert p.get_default_model() == "qwen-turbo"


# ── Groq Transcription ─────────────────────────────────────────────

def test_groq_transcription_default_model():
    """Transcription provider should default to turbo model."""
    from mragent.providers.transcription import GroqTranscriptionProvider
    t = GroqTranscriptionProvider()
    assert "turbo" in t.model.lower(), "Should default to whisper-large-v3-turbo"
