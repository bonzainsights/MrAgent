"""
MRAgent — Model Selector
Auto-selects the best model based on task type, with multiple models per category.

Users can:
  - Switch modes: mode thinking / fast / code / auto
  - Pick a specific model: model glm5 / model deepseek-r1
  - The prompt shows [current-model · mode] at all times

Created: 2026-02-15
Updated: 2026-02-16 — Multi-model categories with defaults
"""

import re

from config.settings import MODEL_REGISTRY
from utils.logger import get_logger

logger = get_logger("agents.model_selector")

# Default model for each mode (first pick when switching mode)
MODE_DEFAULTS = {
    "thinking": "gpt-oss-120b",
    "fast": "gemma-3n",
    "code": "qwen3-coder",
    "browsing": "llama-3.3-70b",
    "general": "gpt-oss-120b",
}

# Keyword patterns for auto-classification
PATTERNS = {
    "code": [
        r"\b(code|function|class|bug|debug|implement|refactor|script|program)\b",
        r"\b(python|javascript|html|css|sql|bash|api|json|xml)\b",
        r"\b(error|traceback|exception|syntax|compile|runtime)\b",
        r"\b(git|commit|push|pull|merge|branch|deploy)\b",
        r"\b(pip|npm|install|package|dependency|import)\b",
    ],
    "browsing": [
        r"\b(search|find|look up|fetch|download|browse|web|news|headline|topic|trend)\b",
        r"\b(url|link|site|page|website)\b",
    ],
    "thinking": [
        r"\b(analyze|explain|compare|evaluate|reason|think|plan)\b",
        r"\b(design|architect|strategy|approach|tradeoff|pros and cons)\b",
        r"\b(why|how does|what if|should i|which is better)\b",
        r"\b(complex|detailed|thorough|comprehensive|in-depth)\b",
        r"\b(step[- ]?by[- ]?step|break down|decompose)\b",
        # Creative / tool-requiring tasks → need tool-capable model
        r"\b(image|picture|photo|draw|paint|illustrat|generat|creat)\b",
        r"\b(file|folder|directory|read|write|save|delete|move|rename)\b",
        r"\b(run|execute|terminal|command|shell|screenshot)\b",
    ],
    "fast": [
        r"\b(hi|hello|hey|thanks|ok|yes|no|sure)\b",
        r"\b(what is|who is|when|where|define|meaning)\b",
        r"\b(quick|simple|short|brief|tldr|summary)\b",
        r"\b(translate|convert|format|list|count)\b",
    ],
}


class ModelSelector:
    """
    Selects the optimal model based on task classification.

    Modes:
    - auto: classify the message and pick the best model
    - thinking: use reasoning model (default: kimi-k2.5, options: glm5, deepseek-r1)
    - fast: use fast model (default: gemma-3n, option: kimi-k2.5, llama-3.3-70b)
    - code: use code model (default: qwen3-coder, options: deepseek-r1, kimi-k2.5)
    """

    def __init__(self, mode: str = "auto"):
        """
        Args:
            mode: Selection mode — "auto", "thinking", "fast", "code"
        """
        self.mode = mode
        self.logger = get_logger("agents.model_selector")

    def select(self, message: str, override: str = None) -> str:
        """
        Select the best model for the given message.

        Args:
            message: The user's message text
            override: Force a specific model name

        Returns:
            Model friendly name (e.g. "kimi-k2.5", "gemma-3n")
        """
        if override:
            self.logger.info(f"Model override: {override}")
            return override

        if self.mode in MODE_DEFAULTS:
            model = MODE_DEFAULTS[self.mode]
            self.logger.debug(f"Mode {self.mode} → {model}")
            return model

        # Auto mode: classify the task
        return self._classify(message)

    def _classify(self, message: str) -> str:
        """Classify a message and return the best model name."""
        msg_lower = message.lower()
        scores = {"code": 0, "thinking": 0, "fast": 0}

        for category, patterns in PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, msg_lower)
                scores[category] += len(matches)

        # Get the category with highest score
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # If no clear signal, default to general model
        if best_score == 0:
            model = MODE_DEFAULTS["general"]
            self.logger.debug(f"No pattern match, defaulting to: {model}")
        else:
            model = MODE_DEFAULTS[best_category]
            self.logger.debug(
                f"Classified as '{best_category}' (scores: {scores}), model: {model}"
            )

        # Safety: if model doesn't support tools but message likely needs tools,
        # upgrade to a tool-capable model
        model_info = MODEL_REGISTRY.get(model, {})
        if not model_info.get("supports_tools", False):
            needs_tools = bool(re.search(
                r"\b(image|generat|draw|search|find|file|run|execute|screenshot|fetch|web)\b",
                msg_lower
            ))
            if needs_tools:
                model = MODE_DEFAULTS["thinking"]  # gpt-oss-120b supports tools
                self.logger.info(f"Upgraded to tool-capable model: {model}")

        return model

    def set_mode(self, mode: str):
        """Change the selection mode."""
        if mode not in ("auto", "thinking", "fast", "code"):
            raise ValueError(f"Invalid mode: {mode}. Use: auto, thinking, fast, code")
        self.mode = mode
        self.logger.info(f"Model selection mode changed to: {mode}")

    @staticmethod
    def get_models_for_mode(mode: str) -> list[str]:
        """Return all model names that support a given mode/category."""
        return [
            name for name, info in MODEL_REGISTRY.items()
            if info.get("type") == "llm" and mode in info.get("categories", [])
        ]

    @staticmethod
    def get_default_for_mode(mode: str) -> str:
        """Return the default model for a mode."""
        return MODE_DEFAULTS.get(mode, MODE_DEFAULTS["general"])
