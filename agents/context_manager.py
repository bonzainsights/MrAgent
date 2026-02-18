"""
MRAgent — Context Manager
Token counting, sliding window, auto-summarization, and context optimization.

Created: 2026-02-15
"""

from utils.logger import get_logger
from utils.helpers import estimate_tokens

logger = get_logger("agents.context_manager")

# Default context windows per model
CONTEXT_WINDOWS = {
    "kimi-k2.5": 32_000,      # Capped for speed (native is 131k)
    "glm5": 128_000,
    "gemma-3n": 32_000,
    "qwen3-coder": 262_144,
    "llama-3.3-70b": 128_000,
    "gpt-oss-120b": 128_000,  # Explicitly defined
}

DEFAULT_CONTEXT_WINDOW = 32_000
RESPONSE_RESERVE = 8_000  # Reserve tokens for response
THRESHOLD = 0.8  # Summarize at 80% usage


class ContextManager:
    """
    Manages conversation context to stay within model token limits.

    Key features:
    - Track token usage per message
    - Auto-summarize when approaching limit
    - Keep recent messages in full detail
    - Store full history in memory for retrieval
    """

    def __init__(self, model_name: str = "kimi-k2.5"):
        self.model_name = model_name
        self.max_tokens = CONTEXT_WINDOWS.get(model_name, DEFAULT_CONTEXT_WINDOW)
        self.available_tokens = self.max_tokens - RESPONSE_RESERVE
        self._messages: list[dict] = []
        self._token_counts: list[int] = []  # Token count per message
        self._total_tokens = 0
        self._summary: str = ""
        self._full_history: list[dict] = []  # All messages ever, for retrieval
        self.logger = get_logger("agents.context_manager")

    @property
    def usage_ratio(self) -> float:
        """Current token usage as a fraction (0.0 - 1.0)."""
        return self._total_tokens / self.available_tokens if self.available_tokens > 0 else 1.0

    @property
    def tokens_used(self) -> int:
        return self._total_tokens

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.available_tokens - self._total_tokens)

    @property
    def messages(self) -> list[dict]:
        """Get current active messages (default: include tools)."""
        return self.get_messages(include_tools=True)

    def get_messages(self, include_tools: bool = True) -> list[dict]:
        """
        Get active messages, optionally filtering out tool interactions.
        
        Args:
            include_tools: If False, strips 'tool' messages and 'tool_calls' 
                          from assistant messages.
        """
        result = []
        if self._summary:
            result.append({
                "role": "system",
                "content": f"[Previous conversation summary]\n{self._summary}",
            })
            
        raw_messages = self._messages
        
        if include_tools:
            result.extend(raw_messages)
            return result
            
        # Filter tools if not supported
        for msg in raw_messages:
            role = msg.get("role")
            
            # Skip tool outputs
            if role == "tool":
                continue
                
            # Copy message to avoid mutating original
            new_msg = msg.copy()
            
            # Strip tool calls from assistant messages
            if role == "assistant" and "tool_calls" in new_msg:
                del new_msg["tool_calls"]
                # If message becomes empty (content None/empty), skip it entirely
                # to avoid "Assistant -> Assistant" patterns or empty messages.
                if not new_msg.get("content"):
                    continue
            
            result.append(new_msg)
            
        return result

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def add_message(self, message: dict):
        """
        Add a message and track its token count.

        Args:
            message: {"role": "user/assistant/system/tool", "content": "..."}
        """
        tokens = self._count_message_tokens(message)
        self._messages.append(message)
        self._token_counts.append(tokens)
        self._total_tokens += tokens
        self._full_history.append(message)

        self.logger.debug(
            f"Added message: role={message['role']}, "
            f"tokens={tokens}, total={self._total_tokens}/{self.available_tokens} "
            f"({self.usage_ratio:.1%})"
        )

        # Check if we need to summarize
        if self.usage_ratio >= THRESHOLD:
            self._auto_summarize()

    def add_messages(self, messages: list[dict]):
        """Add multiple messages at once."""
        for msg in messages:
            self.add_message(msg)

    def set_model(self, model_name: str):
        """Update the model and its context window."""
        self.model_name = model_name
        self.max_tokens = CONTEXT_WINDOWS.get(model_name, DEFAULT_CONTEXT_WINDOW)
        self.available_tokens = self.max_tokens - RESPONSE_RESERVE
        self.logger.info(f"Context window set to {self.max_tokens} for {model_name}")

        # May need to summarize if new model has smaller window
        if self.usage_ratio >= THRESHOLD:
            self._auto_summarize()

    def _count_message_tokens(self, message: dict) -> int:
        """Estimate tokens for a message."""
        tokens = 4  # Message overhead
        content = message.get("content", "")
        if isinstance(content, str):
            tokens += estimate_tokens(content)
        elif isinstance(content, list):
            # Multi-modal messages (text + images)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    tokens += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    tokens += 85  # ~85 tokens for image reference

        # Tool calls
        if "tool_calls" in message:
            for tc in message.get("tool_calls", []):
                if isinstance(tc, dict):
                    tokens += estimate_tokens(str(tc))

        return tokens

    def _auto_summarize(self):
        """
        Summarize older messages to free up context.
        Keeps the system prompt and last N messages in full.
        """
        if len(self._messages) <= 4:
            return  # Not enough to summarize

        keep_recent = 6  # Keep last 6 messages in full
        to_summarize = self._messages[:-keep_recent]
        to_keep = self._messages[-keep_recent:]

        if not to_summarize:
            return

        # Build a simple summary of the older messages
        summary_parts = []
        if self._summary:
            summary_parts.append(self._summary)

        for msg in to_summarize:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                # Truncate individual messages in summary
                preview = content[:200] + "..." if len(content) > 200 else content
                summary_parts.append(f"[{role}]: {preview}")

        self._summary = "\n".join(summary_parts)

        # Replace messages with just the recent ones
        old_total = self._total_tokens
        self._messages = to_keep
        self._token_counts = [self._count_message_tokens(m) for m in to_keep]
        self._total_tokens = sum(self._token_counts) + estimate_tokens(self._summary)

        saved = old_total - self._total_tokens
        self.logger.info(
            f"Auto-summarized: freed {saved} tokens "
            f"({old_total} → {self._total_tokens}), "
            f"summarized {len(to_summarize)} messages, "
            f"kept {len(to_keep)} recent"
        )

    def get_summary(self) -> str:
        """Get the current conversation summary."""
        return self._summary

    def get_full_history(self) -> list[dict]:
        """Get the complete message history (for storage/search)."""
        return self._full_history.copy()

    def clear(self):
        """Clear all messages and start fresh."""
        self._messages.clear()
        self._token_counts.clear()
        self._total_tokens = 0
        self._summary = ""
        self.logger.info("Context cleared")

    def needs_new_chat(self) -> bool:
        """
        Heuristic: suggest starting a new chat when:
        - Summary is getting very long (topic drift)
        - History is very deep (>50 messages summarized)
        """
        if len(self._full_history) > 50 and len(self._summary) > 2000:
            return True
        return False

    def get_stats(self) -> dict:
        """Return context manager statistics."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "used_tokens": self._total_tokens,
            "remaining_tokens": self.tokens_remaining,
            "usage_ratio": f"{self.usage_ratio:.1%}",
            "active_messages": len(self._messages),
            "full_history_messages": len(self._full_history),
            "has_summary": bool(self._summary),
        }
