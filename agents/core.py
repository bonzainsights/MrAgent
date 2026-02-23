"""
MRAgent — Core Agent Loop
The main brain: Receive → Enhance → Plan → Execute → Observe → Respond.

Created: 2026-02-15
"""

import threading
import json
import time
import re
import base64
from pathlib import Path
from typing import Callable, Generator

from agents.context_manager import ContextManager
from agents.model_selector import ModelSelector
from agents.prompt_enhancer import PromptEnhancer
from config.settings import MODEL_REGISTRY
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
        self.approval_callback: Callable[[str], bool] = None
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

        # 1. Enhance & format user message (parse images if present)
        enhanced = self.prompt_enhancer.enhance_user_message(user_message)
        
        # Parse for [Attached Image: /path/to/image]
        image_matches = re.findall(r'\[Attached Image: (.*?)\]', enhanced)
        if image_matches:
            # Multi-modal array
            content_array = []
            
            # Remove the tags from the text part to avoid confusing the LLM with duplicate info
            text_only = re.sub(r'\[Attached Image: .*?\]', '', enhanced).strip()
            if text_only:
                content_array.append({"type": "text", "text": text_only})
                
            for img_path_str in image_matches:
                img_path = Path(img_path_str.strip())
                if img_path.exists():
                    try:
                        with open(img_path, "rb") as bf:
                            encoded_img = base64.b64encode(bf.read()).decode('utf-8')
                            
                        # Determine MIME type roughly
                        ext = img_path.suffix.lower()
                        mime_type = "image/jpeg"
                        if ext == ".png": mime_type = "image/png"
                        elif ext == ".webp": mime_type = "image/webp"
                        
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_img}"
                            }
                        })
                    except Exception as e:
                        logger.error(f"Failed to read attached image {img_path}: {e}")
            
            if not content_array:
                # Fallback to string if parsing failed
                content_array = enhanced
            
            self.context_manager.add_message({"role": "user", "content": content_array})
        else:
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
        
        # Check if model supports tools
        model_info = MODEL_REGISTRY.get(model, {})
        supports_tools = model_info.get("supports_tools", False)
        
        tools_schema = self.tool_registry.get_openai_tools() if supports_tools else None
        
        # Max tools limit
        for iteration in range(MAX_TOOL_ITERATIONS):
            # Get messages, filtering out tools if model doesn't support them
            messages = self.context_manager.get_messages(include_tools=supports_tools)

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
        from config.settings import AUTONOMY_SETTINGS
        import fnmatch

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

        trust_level = AUTONOMY_SETTINGS.get("trust_level", "balanced")

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

            # Execute the tool with tiered approval logic
            result = None

            # Helper: check if a command matches auto-approve patterns
            def matches_auto_approve(cmd: str) -> bool:
                """Check if command matches any auto-approve pattern from settings."""
                if not cmd:
                    return False
                patterns = AUTONOMY_SETTINGS.get("auto_approve_patterns", [])
                for pattern in patterns:
                    if fnmatch.fnmatch(cmd.strip(), pattern):
                        return True
                return False

            # Helper: check if a terminal command is safe (read-only)
            def is_safe_command(cmd: str) -> bool:
                if not cmd:
                    return False
                # Unsafe operators: chaining or redirecting
                unsafe_patterns = ['&&', '||', ';', '|', '>', '<', '`', '$(']
                for p in unsafe_patterns:
                    if p in cmd:
                        return False

                # Extract base command
                base_cmd = cmd.strip().split()[0] if cmd.strip() else ""

                # Whitelisted safe read-only commands
                safe_cmds = {
                    'ls', 'pwd', 'echo', 'cat', 'git status', 'git log', 'git diff',
                    'grep', 'find', 'which', 'whoami', 'date', 'tree', 'head', 'tail', 'less'
                }

                # Check for two-word safe commands like git status
                if len(cmd.split()) >= 2:
                    two_word_cmd = " ".join(cmd.split()[:2])
                    if two_word_cmd in safe_cmds:
                        return True

                return base_cmd in safe_cmds

            # ── Tiered Approval Logic ──
            if func_name == "execute_terminal":
                cmd_to_run = func_args.get("command", "")

                if trust_level == "autonomous":
                    # Autonomous: run everything, just log it
                    logger.info(f"[AUTONOMOUS] Auto-running: {cmd_to_run}")
                    self._emit("info", f"⚡ [autonomous] Running: {cmd_to_run[:80]}")

                elif trust_level == "balanced":
                    # Balanced: auto-run if safe OR matches patterns; otherwise ask
                    if not is_safe_command(cmd_to_run) and not matches_auto_approve(cmd_to_run):
                        if self.approval_callback:
                            tool_desc = f"⚠️ Agent wants to run a command:\n```bash\n{cmd_to_run}\n```"
                            self._emit("approval_required", tool_desc)
                            # Send async Telegram notification
                            self._notify_pending_approval(func_name, cmd_to_run)
                            approved = self.approval_callback(tool_desc)
                            if not approved:
                                result = "❌ Execution rejected by user."
                        else:
                            logger.info(f"[BALANCED] Auto-approved (no callback): {cmd_to_run}")
                    else:
                        logger.info(f"[BALANCED] Pattern-approved: {cmd_to_run}")

                else:  # cautious
                    if not is_safe_command(cmd_to_run):
                        if self.approval_callback:
                            tool_desc = f"⚠️ Agent wants to run an unsafe command:\n```bash\n{cmd_to_run}\n```"
                            self._emit("approval_required", tool_desc)
                            approved = self.approval_callback(tool_desc)
                            if not approved:
                                result = "❌ Execution rejected by user."

            elif func_name == "run_code":
                if trust_level == "autonomous":
                    logger.info(f"[AUTONOMOUS] Auto-running code snippet")
                    self._emit("info", "⚡ [autonomous] Running code...")
                elif trust_level == "cautious":
                    if self.approval_callback:
                        tool_desc = f"⚠️ Agent wants to run code:\n```python\n{func_args.get('code', '')[:200]}...\n```"
                        self._emit("approval_required", tool_desc)
                        approved = self.approval_callback(tool_desc)
                        if not approved:
                            result = "❌ Execution rejected by user."
                # balanced: auto-approve code execution (it's sandboxed)

            if result is None:
                result = self.tool_registry.execute(func_name, **func_args)

            self._emit("tool_result", f"✅ {func_name}: {result[:200]}")

            # Add tool result to context
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            }
            self.context_manager.add_message(tool_msg)

        logger.info(f"Executed {len(tool_calls)} tool call(s) [trust={trust_level}]")

    def _notify_pending_approval(self, tool_name: str, command: str):
        """Send a Telegram notification when an approval is pending (balanced mode)."""
        from config.settings import AUTONOMY_SETTINGS, TELEGRAM_BOT_TOKEN
        if not AUTONOMY_SETTINGS.get("notify_on_pending", False) or not TELEGRAM_BOT_TOKEN:
            return

        try:
            import threading
            def _send():
                try:
                    from skills.telegram import TelegramSkill
                    skill = TelegramSkill()
                    allowed_chats = skill.allowed_chats
                    if not allowed_chats:
                        return
                    msg = (
                        f"⚠️ *MRAgent Approval Pending*\n\n"
                        f"Tool: `{tool_name}`\n"
                        f"Command: `{command[:200]}`\n\n"
                        f"Waiting for your approval in the terminal..."
                    )
                    for chat_id in allowed_chats:
                        skill._send_message(chat_id, msg)
                except Exception as e:
                    logger.debug(f"Telegram notification failed: {e}")

            # Fire-and-forget in a thread to not block the approval prompt
            threading.Thread(target=_send, daemon=True).start()
        except Exception:
            pass

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
