"""Qwen Portal provider — secure Device Authorization Flow, no API key required.

Qwen Portal (portal.qwen.ai) offers free requests via OAuth. This provider:
1. Uses `oauth-cli-kit` to handle the Device Authorization Flow.
2. Caches the token locally at ~/.mragent/qwen_portal_token.json.
3. Automatically opens a browser for one-click authorization.
4. Calls the Qwen Portal OpenAI-compatible API.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# Try to import oauth-cli-kit; it's a dependency in pyproject.toml
try:
    from oauth_cli_kit import get_token as get_oauth_token
except ImportError:
    get_oauth_token = None

from mragent.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_TOKEN_FILE = Path.home() / ".mragent" / "qwen_portal_token.json"
_QWEN_PORTAL_API = "https://portal.qwen.ai/v1/chat/completions"
_DEFAULT_MODEL = "qwen-plus"


class QwenPortalProvider(LLMProvider):
    """
    Qwen Portal provider using Device Authorization Flow.

    No manual cookie copying — authenticates via portal.qwen.ai.
    Uses oauth-cli-kit to manage the token and browser interaction.
    """

    def __init__(self, default_model: str = _DEFAULT_MODEL):
        super().__init__(api_key=None, api_base=_QWEN_PORTAL_API)
        self.default_model = default_model
        self._token_data: dict[str, Any] | None = None
        self._load_cached_token()

    def _load_cached_token(self) -> None:
        """Load cached OAuth token from disk if valid."""
        if not _TOKEN_FILE.exists():
            return
        try:
            data = json.loads(_TOKEN_FILE.read_text(encoding="utf-8"))
            # oauth-cli-kit tokens usually have an 'access' and 'expiry'
            if data.get("access") and data.get("expiry", 0) > time.time() + 60:
                self._token_data = data
                logger.info("Qwen Portal: loaded cached token")
        except Exception as e:
            logger.warning("Qwen Portal: failed to load cached token: {}", e)

    async def _ensure_token(self) -> str:
        """Return a valid access token, initiating device flow if needed."""
        # Use env var for headless/CI override
        env_token = os.environ.get("QWEN_PORTAL_TOKEN")
        if env_token:
            return env_token

        if self._token_data and self._token_data.get("expiry", 0) > time.time() + 60:
            return self._token_data["access"]

        if not get_oauth_token:
            raise RuntimeError("oauth-cli-kit is not installed. Please run `pip install oauth-cli-kit`.")

        # Initiating Device Flow
        logger.info("Qwen Portal: initiating device authorization flow...")
        
        # get_token is typically synchronous and handles browser/polling.
        # We run it in a thread to avoid blocking the async loop.
        try:
            # We pass a client_id if known, otherwise it defaults to qwen-portal patterns
            token = await asyncio.to_thread(get_oauth_token, client_id="qwen-portal")
            
            # Extract and cache
            self._token_data = {
                "access": token.access,
                "expiry": token.expiry if hasattr(token, "expiry") else (time.time() + 3600),
                "account_id": getattr(token, "account_id", None)
            }
            
            _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            _TOKEN_FILE.write_text(json.dumps(self._token_data), encoding="utf-8")
            
            return self._token_data["access"]
        except Exception as e:
            logger.error("Qwen Portal auth failed: {}", e)
            raise RuntimeError(f"Qwen Portal authentication failed: {e}")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to Qwen Portal."""
        resolved_model = model or self.default_model
        if resolved_model.startswith("qwen-portal/"):
             resolved_model = resolved_model.replace("qwen-portal/", "", 1)
        
        messages = self._sanitize_empty_content(messages)

        try:
            token = await self._ensure_token()
        except Exception as e:
            return LLMResponse(content=str(e), finish_reason="error")

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "MRAgent/1.0",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(_QWEN_PORTAL_API, json=payload, headers=headers)

                if response.status_code == 401:
                    # Token might have expired early
                    self._token_data = None
                    _TOKEN_FILE.unlink(missing_ok=True)
                    return LLMResponse(
                        content="Qwen Portal session expired. Please run again to re-authorize.",
                        finish_reason="error",
                    )

                response.raise_for_status()
                data = response.json()
                return self._parse_openai_response(data)

        except httpx.HTTPStatusError as e:
            logger.error("Qwen Portal HTTP error: {} — {}", e.response.status_code, e.response.text[:300])
            return LLMResponse(content=f"Qwen Portal error {e.response.status_code}: {e.response.text[:200]}", finish_reason="error")
        except Exception as e:
            logger.error("Qwen Portal error: {}", e)
            return LLMResponse(content=f"Qwen Portal error: {e}", finish_reason="error")

    def _parse_openai_response(self, data: dict[str, Any]) -> LLMResponse:
        """Standard OpenAI response parser."""
        import json_repair
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="Qwen Portal returned no choices", finish_reason="error")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or None
        finish_reason = choice.get("finish_reason", "stop")

        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json_repair.loads(args)
                except Exception:
                    args = {"raw": args}
            tool_calls.append(ToolCallRequest(
                id=tc.get("id", "tc_0"),
                name=fn.get("name", ""),
                arguments=args,
            ))

        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens": usage_raw.get("total_tokens", 0),
        }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def get_default_model(self) -> str:
        return self.default_model
