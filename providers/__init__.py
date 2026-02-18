# MRAgent Providers Package
"""
Provider registry — singleton access to all API providers.

Usage:
    from providers import get_llm, get_image, get_tts, get_stt, get_search
    response = get_llm().chat(messages)
"""

from utils.logger import get_logger

logger = get_logger("providers")

# Lazy-loaded singletons
_llm_provider = None
_image_provider = None
_tts_provider = None
_stt_provider = None
_search_provider = None


def get_llm():
    """Get the NVIDIA LLM provider (singleton)."""
    global _llm_provider
    if _llm_provider is None:
        from providers.nvidia_llm import NvidiaLLMProvider
        _llm_provider = NvidiaLLMProvider()
        logger.info("LLM provider ready")
    return _llm_provider


def get_image():
    """Get the NVIDIA Image provider (singleton)."""
    global _image_provider
    if _image_provider is None:
        from providers.nvidia_image import NvidiaImageProvider
        _image_provider = NvidiaImageProvider()
        logger.info("Image provider ready")
    return _image_provider


def get_tts():
    """Get the NVIDIA TTS provider (singleton)."""
    global _tts_provider
    if _tts_provider is None:
        from providers.nvidia_tts import NvidiaTTSProvider
        _tts_provider = NvidiaTTSProvider()
        logger.info("TTS provider ready")
    return _tts_provider


def get_stt():
    """Get the NVIDIA STT provider (singleton)."""
    global _stt_provider
    if _stt_provider is None:
        from providers.nvidia_stt import NvidiaSTTProvider
        _stt_provider = NvidiaSTTProvider()
        logger.info("STT provider ready")
    return _stt_provider


def get_search(provider_name: str = None):
    """
    Get the search provider.
    args:
        provider_name: 'brave' or 'google'. If None, uses SEARCH_PROVIDER env var (default: brave).
    """
    import os
    target = provider_name or os.getenv("SEARCH_PROVIDER", "brave")

    if target == "google":
        return _get_google_search()
    elif target == "langsearch":
        return _get_langsearch()
    else:
        return _get_brave_search()


def _get_brave_search():
    global _search_provider
    if _search_provider is None or _search_provider.name != "brave_search":
        from providers.brave_search import BraveSearchProvider
        _search_provider = BraveSearchProvider()
        logger.info("Brave Search provider ready")
    return _search_provider


def _get_google_search():
    global _search_provider
    if _search_provider is None or _search_provider.name != "google_search":
        from providers.google_search import GoogleSearchProvider
        _search_provider = GoogleSearchProvider()
        logger.info("Google Search provider ready")
    return _search_provider


def _get_langsearch():
    global _search_provider
    if _search_provider is None or _search_provider.name != "langsearch":
        from providers.langsearch import LangSearchProvider
        _search_provider = LangSearchProvider()
        logger.info("LangSearch provider ready")
    return _search_provider


def get_all_status() -> dict:
    """Return status of all providers."""
    status = {}
    try:
        status["llm"] = get_llm().stats
    except Exception as e:
        status["llm"] = {"error": str(e)}
    try:
        status["image"] = get_image().stats
    except Exception as e:
        status["image"] = {"error": str(e)}
    try:
        status["tts"] = {"available": get_tts().available}
    except Exception as e:
        status["tts"] = {"error": str(e)}
    try:
        status["stt"] = {"available": get_stt().available}
    except Exception as e:
        status["stt"] = {"error": str(e)}
    try:
        status["search"] = {"available": get_search().available}
    except Exception as e:
        status["search"] = {"error": str(e)}
    return status
