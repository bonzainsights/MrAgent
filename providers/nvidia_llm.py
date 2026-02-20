"""
MRAgent — NVIDIA LLM Provider
Chat completions via NVIDIA NIM using the OpenAI-compatible SDK.
Supports multi-model, streaming, and function calling (tool use).

Created: 2026-02-15
"""

import time
from typing import Generator

from openai import OpenAI

from providers.base import LLMProvider
from config.settings import NVIDIA_BASE_URL, NVIDIA_KEYS, MODEL_REGISTRY, get_api_key
from utils.logger import get_logger

logger = get_logger("providers.nvidia_llm")


class NvidiaLLMProvider(LLMProvider):
    """
    NVIDIA NIM LLM provider using OpenAI-compatible API.

    Supports models: Kimi K2.5, GLM-5, Gemma 3N, Qwen3 Coder.
    All accessed via the same OpenAI SDK with NVIDIA's base URL.
    """

    def __init__(self, rate_limit_rpm: int = 35):
        super().__init__(name="nvidia_llm", rate_limit_rpm=rate_limit_rpm)
        # Create one client per API key (models may use different keys)
        self._clients: dict[str, OpenAI] = {}
        self.logger.info("NVIDIA LLM provider initialized")

    def _get_client(self, model_name: str) -> OpenAI:
        """Get or create an OpenAI client for the given model."""
        api_key = get_api_key(model_name)

        if api_key not in self._clients:
            self._clients[api_key] = OpenAI(
                base_url=NVIDIA_BASE_URL,
                api_key=api_key,
            )
            self.logger.debug(f"Created OpenAI client for model: {model_name}")

        return self._clients[api_key]

    def _resolve_model(self, model: str) -> tuple[str, str]:
        """
        Resolve a model name to (friendly_name, nvidia_model_id).

        Accepts both friendly names ("kimi-k2.5") and full IDs ("moonshotai/kimi-k2.5").
        """
        # Check if it's already a friendly name
        if model in MODEL_REGISTRY:
            return model, MODEL_REGISTRY[model]["id"]

        # Check if it's a full model ID
        for name, info in MODEL_REGISTRY.items():
            if info.get("id") == model and info.get("type") in ("llm", "vlm"):
                return name, model

        # Fallback: assume it's a valid NIM model ID
        self.logger.warning(f"Unknown model '{model}', passing through as-is")
        return model, model

    def chat(self, messages: list[dict], model: str = "kimi-k2.5",
             stream: bool = True, tools: list[dict] = None,
             temperature: float = 0.7, max_tokens: int = 4096) -> dict | Generator:
        """
        Send a chat completion to NVIDIA NIM.

        Args:
            messages: Chat history [{"role": "user", "content": "Hello"}]
            model: Model name (friendly or full ID)
            stream: Stream response chunks
            tools: Tool definitions for function calling
            temperature: Creativity (0.0-2.0)
            max_tokens: Max response length

        Returns:
            If stream=False: {"content": str, "tool_calls": list, "usage": dict}
            If stream=True: Generator yielding {"delta": str} or {"tool_calls": list}
        """
        # Define the parameter step-down chain
        fallback_chain = [
            "gpt-oss-120b",
            "llama-3.3-70b",
            "glm5",
            "kimi-k2.5",
            "qwen3-coder",
            "gemma-3n"
        ]

        models_to_try = []
        if model in fallback_chain:
            start_idx = fallback_chain.index(model)
            models_to_try = fallback_chain[start_idx:]
        else:
            models_to_try = [model] + fallback_chain

        last_exception = None

        for current_model in models_to_try:
            try:
                friendly_name, model_id = self._resolve_model(current_model)
                client = self._get_client(friendly_name)

                self.logger.debug(f"Chat request trying: model={model_id} (param size step-down)")

                start_time = time.time()

                def _make_request():
                    kwargs = {
                        "model": model_id,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": stream,
                    }

                    model_info = MODEL_REGISTRY.get(friendly_name, {})
                    if tools and model_info.get("supports_tools", True):
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"
                    elif tools and not model_info.get("supports_tools", True):
                        self.logger.debug(f"Skipping tools for {friendly_name} — unsupported")

                    return client.chat.completions.create(**kwargs)

                response = self._retry_call(_make_request)
                duration_ms = (time.time() - start_time) * 1000

                if stream:
                    return self._handle_stream(response, friendly_name, model_id, duration_ms)
                else:
                    return self._handle_response(response, friendly_name, model_id, duration_ms)

            except Exception as e:
                self.logger.warning(f"Model {current_model} failed ({e}). Stepping down to next fallback model...")
                last_exception = e
                # Fall through and let loop try the next model in the chain
                continue

        # If exhausted all
        duration_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
        self._track_call("chat/completions", model, duration_ms, status=f"error: exhausted fallbacks")
        raise last_exception

    def _handle_response(self, response, friendly_name: str,
                         model_id: str, duration_ms: float) -> dict:
        """Process a non-streaming response."""
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content or "",
            "tool_calls": [],
            "usage": {},
            "model": model_id,
            "finish_reason": choice.finish_reason,
        }

        # Extract tool calls if present
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # Extract usage stats
        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        tokens = result["usage"].get("total_tokens", 0)
        self._track_call("chat/completions", model_id, duration_ms,
                         status="ok", tokens_used=tokens)

        return result

    def _handle_stream(self, response, friendly_name: str,
                       model_id: str, start_duration_ms: float) -> Generator:
        """Process a streaming response, yielding chunks."""
        full_content = ""
        tool_calls_buffer = {}

        try:
            for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Text content
                if delta.content:
                    full_content += delta.content
                    yield {"delta": delta.content, "type": "content"}

                # Tool calls (streamed incrementally)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or "",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            tool_calls_buffer[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_buffer[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_buffer[idx]["function"]["arguments"] += tc.function.arguments

                # Check for finish
                if chunk.choices[0].finish_reason:
                    if tool_calls_buffer:
                        yield {
                            "tool_calls": list(tool_calls_buffer.values()),
                            "type": "tool_calls",
                        }
                    yield {
                        "finish_reason": chunk.choices[0].finish_reason,
                        "type": "finish",
                        "full_content": full_content,
                    }

        finally:
            duration_ms = (time.time() * 1000) - (start_duration_ms - (start_duration_ms % 1))
            self._track_call("chat/completions", model_id, duration_ms,
                             status="ok", tokens_used=len(full_content) // 4)

    def list_models(self) -> list[dict]:
        """Return available LLM models with their info."""
        available = []
        for name, info in MODEL_REGISTRY.items():
            if info["type"] not in ("llm", "vlm"):
                continue
            key_name = info["key"]
            has_key = bool(NVIDIA_KEYS.get(key_name))
            available.append({
                "name": name,
                "id": info["id"],
                "category": info.get("category", ""),
                "context_window": info.get("context_window", 0),
                "description": info.get("description", ""),
                "available": has_key,
            })
        return available
