"""
MRAgent — Core Agent Loop
The main brain: Receive → Enhance → Plan → Execute → Observe → Respond.

Created: 2026-02-15
"""

import threading
import json
import time
from typing import Callable, Generator

from agents.context_manager import ContextManager
from agents.model_selector import ModelSelector
from agents.prompt_enhancer import PromptEnhancer
from providers import get_llm
from tools import create_tool_registry
from tools.base import ToolRegistry
from utils.logger import get_logger
from utils.helpers import generate_id

logger = get_logger("agents.core")

MAX_TOOL_ITERATIONS = 10  # Safety: max tool call rounds per turn


class AgentCore:
    """
    The main MRAgent agent.

    Implements a ReAct-style loop:
    1. Receive user input (text or transcribed voice)
    2. Enhance the prompt with context
    3. Send to LLM with tool definitions
    4. If LLM returns tool calls → execute tools → feed results back → repeat
    5. When LLM returns final text → output to user
    6. Store everything in conversation history
    """

    def __init__(self, model_mode: str = "auto", model_override: str = None):
        """
        Args:
            model_mode: Model selection mode (auto/thinking/fast/code)
            model_override: Force a specific model name
        """
        self.model_selector = ModelSelector(mode=model_mode)
        self.prompt_enhancer = PromptEnhancer()
        self.context_manager = ContextManager()
        self.tool_registry: ToolRegistry = create_tool_registry()
        self.model_override = model_override
        self.chat_id = generate_id("chat_")
        self._response_callbacks: list[Callable] = []
        self._lock = threading.Lock()

        # Initialize with system prompt
        system_msg = self.prompt_enhancer.get_system_prompt()
        self.context_manager.add_message(system_msg)

        logger.info(
            f"Agent initialized: chat_id={self.chat_id}, "
            f"mode={model_mode}, tools={self.tool_registry.count}"
        )

    def on_response(self, callback: Callable):
        """Register a callback for streaming response chunks."""
        self._response_callbacks.append(callback)

    def _emit(self, event_type: str, data: str):
        """Emit a response event to all registered callbacks."""
        for cb in self._response_callbacks:
            try:
                cb(event_type, data)
            except Exception:
                pass

    def chat(self, user_message: str, stream: bool = True) -> str:
        """
        Process a user message through the agent loop.
        
        Thread-safe: serializes access to prevent concurrent history modification.
        """
        with self._lock:
            return self._chat_unsafe(user_message, stream)

    def _chat_unsafe(self, user_message: str, stream: bool = True) -> str:
        """Internal chat logic (not thread-safe)."""
        turn_start = time.time()

        # 1. Enhance & add user message
        enhanced = self.prompt_enhancer.enhance_user_message(user_message)
        self.context_manager.add_message({"role": "user", "content": enhanced})

        # 2. Select model
        model = self.model_selector.select(user_message, override=self.model_override)
        self.context_manager.set_model(model)
        self._emit("model", model)

        logger.info(f"User: '{user_message[:80]}...' → model: {model}")

        # 3. Agent loop (may iterate for tool calls)
        final_response = self._agent_loop(model, stream)

        # 4. Store assistant response
        self.context_manager.add_message({"role": "assistant", "content": final_response})

        turn_duration = time.time() - turn_start
        logger.info(f"Turn complete: {turn_duration:.1f}s, response: {len(final_response)} chars")

        # Check if we should suggest a new chat
        if self.context_manager.needs_new_chat():
            self._emit("suggestion", "💡 This conversation is getting long. Consider starting a new chat with /newchat")

        return final_response

    def _agent_loop(self, model: str, stream: bool) -> str:
        """
        Core ReAct loop: LLM call → tool execution → repeat.

        Returns the final text response after all tool calls are resolved.
        """
        llm = get_llm()
        tools_schema = self.tool_registry.get_openai_tools()

        for iteration in range(MAX_TOOL_ITERATIONS):
            messages = self.context_manager.messages

            logger.debug(f"Agent loop iteration {iteration + 1}, messages: {len(messages)}")

            if stream:
                response = self._handle_streaming(llm, messages, model, tools_schema)
            else:
                response = llm.chat(
                    messages=messages,
                    model=model,
                    stream=False,
                    tools=tools_schema,
                    temperature=0.7,
                )

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if tool_calls:
                # Execute tools and add results to context
                self._execute_tool_calls(tool_calls, response)
                continue  # Loop back for another LLM call
            else:
                # Final text response
                return response.get("content", response.get("full_content", ""))

        # Safety: too many iterations
        logger.warning(f"Agent loop hit max iterations ({MAX_TOOL_ITERATIONS})")
        return "I've made too many tool calls in this turn. Let me give you what I have so far."

    def _handle_streaming(self, llm, messages: list, model: str,
                          tools_schema: list) -> dict:
        """Handle a streaming response, emitting chunks and collecting the full result."""
        full_content = ""
        tool_calls = []

        stream = llm.chat(
            messages=messages,
            model=model,
            stream=True,
            tools=tools_schema,
            temperature=0.7,
        )

        for chunk in stream:
            chunk_type = chunk.get("type", "")

            if chunk_type == "content":
                delta = chunk.get("delta", "")
                full_content += delta
                self._emit("delta", delta)

            elif chunk_type == "tool_calls":
                tool_calls = chunk.get("tool_calls", [])
                self._emit("tool_calls", json.dumps(tool_calls))

            elif chunk_type == "finish":
                full_content = chunk.get("full_content", full_content)
                if not tool_calls:
                    tool_calls = chunk.get("tool_calls", [])

        return {
            "content": full_content,
            "full_content": full_content,
            "tool_calls": tool_calls,
        }

    def _execute_tool_calls(self, tool_calls: list, assistant_response: dict):
        """Execute tool calls and add results to conversation context."""
        # Normalize tool_calls: ensure each has "type": "function" (required by NVIDIA API)
        normalized_tool_calls = []
        for tc in tool_calls:
            normalized = {
                "id": tc.get("id", generate_id("tc_")),
                "type": "function",
                "function": tc.get("function", {}),
            }
            normalized_tool_calls.append(normalized)

        # Add assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response.get("content", "") or None,
            "tool_calls": normalized_tool_calls,
        }
        self.context_manager.add_message(assistant_msg)

        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name", "")
            func_args_str = tc.get("function", {}).get("arguments", "{}")
            tc_id = tc.get("id", generate_id("tc_"))

            # Parse arguments
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                func_args = {}
                logger.warning(f"Failed to parse tool args: {func_args_str[:100]}")

            self._emit("tool_start", f"🔧 Running: {func_name}({json.dumps(func_args)[:100]})")

            # Execute the tool
            result = self.tool_registry.execute(func_name, **func_args)

            self._emit("tool_result", f"✅ {func_name}: {result[:200]}")

            # Add tool result to context
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            }
            self.context_manager.add_message(tool_msg)

        logger.info(f"Executed {len(tool_calls)} tool call(s)")

    # ──────────────────────────────────────────────
    # User-facing commands
    # ──────────────────────────────────────────────

    def set_model_mode(self, mode: str):
        """Change model selection mode: auto/thinking/fast/code."""
        self.model_selector.set_mode(mode)
        self._emit("info", f"Model mode set to: {mode}")

    def set_model(self, model_name: str):
        """Override the model to use."""
        self.model_override = model_name
        self._emit("info", f"Model set to: {model_name}")

    def new_chat(self):
        """Start a new conversation."""
        old_id = self.chat_id
        self.chat_id = generate_id("chat_")
        self.context_manager.clear()

        # Re-add system prompt
        system_msg = self.prompt_enhancer.get_system_prompt()
        self.context_manager.add_message(system_msg)

        logger.info(f"New chat started: {old_id} → {self.chat_id}")
        self._emit("info", "🔄 New chat started")

    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            "chat_id": self.chat_id,
            "model_mode": self.model_selector.mode,
            "model_override": self.model_override,
            "tools": self.tool_registry.count,
            "context": self.context_manager.get_stats(),
        }
