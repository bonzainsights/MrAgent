"""Qwen OAuth provider — free browser-based auth, no API key required.

Qwen offers ~2000 free requests/day via browser OAuth. This provider:
1. Checks for a cached token at ~/.mragent/qwen_token.json
2. If missing/expired, opens a browser to qwen.ai for login
3. Calls the Qwen OpenAI-compatible API endpoint

Compatible models (free tier):
  - qwen-plus     (recommended default)
  - qwen-turbo
  - qwen-max      (more capable, lower rate limit)
"""

from __future__ import annotations

import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from mragent.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_TOKEN_FILE = Path.home() / ".mragent" / "qwen_token.json"
_QWEN_CHAT_API = "https://chat.qwen.ai/api/chat/completions"
_QWEN_AUTH_URL = "https://chat.qwen.ai"
_DEFAULT_MODEL = "qwen-plus"


class QwenOAuthProvider(LLMProvider):
    """
    Free Qwen provider via browser-based OAuth.

    No API key needed — authenticates via qwen.ai account.
    Free tier: ~2000 requests/day, 60 req/min.

    Usage:
        In ~/.mragent/config.json set:
          "agents": {"defaults": {"provider": "qwen_oauth",
                                   "model": "qwen-plus"}}
    """

    def __init__(self, default_model: str = _DEFAULT_MODEL):
        super().__init__(api_key=None, api_base=_QWEN_CHAT_API)
        self.default_model = default_model
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._load_cached_token()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _load_cached_token(self) -> None:
        """Load cached OAuth token from disk if not expired."""
        if not _TOKEN_FILE.exists():
            return
        try:
            data = json.loads(_TOKEN_FILE.read_text(encoding="utf-8"))
            token = data.get("token", "")
            expiry = float(data.get("expiry", 0))
            if token and expiry > time.time() + 60:
                self._token = token
                self._token_expiry = expiry
                logger.info("Qwen OAuth: loaded cached token (expires in {:.0f}s)", expiry - time.time())
        except Exception as e:
            logger.warning("Qwen OAuth: failed to load cached token: {}", e)

    def _save_token(self, token: str, expiry: float) -> None:
        """Persist OAuth token to disk."""
        _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_FILE.write_text(
            json.dumps({"token": token, "expiry": expiry}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_token(self) -> str:
        """Return a valid token, prompting browser auth if needed."""
        # Check env var first (useful in CI or scripts)
        env_token = os.environ.get("QWEN_OAUTH_TOKEN", "")
        if env_token:
            return env_token

        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        # Need to authenticate
        self._browser_auth_flow()

        if not self._token:
            raise RuntimeError(
                "Qwen OAuth: authentication failed. "
                "Set QWEN_OAUTH_TOKEN env var or log in at https://chat.qwen.ai"
            )
        return self._token

    def _browser_auth_flow(self) -> None:
        """Prompt the user to log in via browser and paste their token."""
        from rich.console import Console
        console = Console()

        console.print("\n[bold cyan]🤖 MRAgent — Qwen OAuth Login[/bold cyan]")
        console.print("Qwen offers ~2000 free requests/day. No credit card needed.\n")
        console.print("Steps:")
        console.print("  1. Opening [link=https://chat.qwen.ai]https://chat.qwen.ai[/link] in your browser")
        console.print("  2. Log in with your Alibaba / Qwen account")
        console.print("  3. Open DevTools → Application → Cookies → chat.qwen.ai")
        console.print("  4. Copy the value of the [bold]token[/bold] cookie")
        console.print("  5. Paste it below\n")

        try:
            webbrowser.open(_QWEN_AUTH_URL)
        except Exception:
            pass

        try:
            token = input("Paste your Qwen token here: ").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if not token:
            console.print("[red]No token provided — skipping Qwen auth.[/red]")
            return

        # Token is good for ~7 days; store with 6-day expiry to be safe
        expiry = time.time() + 6 * 86400
        self._token = token
        self._token_expiry = expiry
        self._save_token(token, expiry)
        console.print("[green]✓[/green] Qwen token saved to ~/.mragent/qwen_token.json (6d expiry)\n")

    # ------------------------------------------------------------------
    # LLM chat interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to Qwen's free API."""
        resolved_model = model or self.default_model
        messages = self._sanitize_empty_content(messages)

        try:
            token = self._ensure_token()
        except RuntimeError as e:
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
                response = await client.post(_QWEN_CHAT_API, json=payload, headers=headers)

                if response.status_code == 401:
                    # Token expired — clear cache and report
                    self._token = None
                    _TOKEN_FILE.unlink(missing_ok=True)
                    return LLMResponse(
                        content=(
                            "Qwen OAuth token expired. Run `mragent agent` again to re-authenticate, "
                            "or delete ~/.mragent/qwen_token.json."
                        ),
                        finish_reason="error",
                    )

                response.raise_for_status()
                data = response.json()
                return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            logger.error("Qwen API HTTP error: {} — {}", e.response.status_code, e.response.text[:300])
            return LLMResponse(content=f"Qwen API error {e.response.status_code}: {e.response.text[:200]}", finish_reason="error")
        except Exception as e:
            logger.error("Qwen API error: {}", e)
            return LLMResponse(content=f"Qwen API error: {e}", finish_reason="error")

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """Parse Qwen OpenAI-compatible response."""
        try:
            import json_repair
            choices = data.get("choices", [])
            if not choices:
                return LLMResponse(content="Qwen returned no choices", finish_reason="error")

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
                    args = json_repair.loads(args)
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
        except Exception as e:
            return LLMResponse(content=f"Error parsing Qwen response: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        """Get the default Qwen model."""
        return self.default_model
