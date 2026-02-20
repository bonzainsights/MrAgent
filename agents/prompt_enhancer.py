"""
MRAgent — Prompt Enhancer
Builds system prompts, injects context, and enhances user queries.

Created: 2026-02-15
"""

from datetime import datetime

from utils.logger import get_logger
from utils.helpers import get_system_context

logger = get_logger("agents.prompt_enhancer")

from config.settings import USER_NAME, AGENT_NAME

SYSTEM_PROMPT = """You are {agent_name} — a helpful, intelligent AI assistant created by Bonza Insights. You are talking to {user_name}. Address them by their name and refer to yourself as {agent_name}.

## Core Identity
- You are running locally on the user's machine
- You are powered by NVIDIA NIM APIs
- You are lightweight — designed to run on any device, no GPU required

## Your Capabilities
You have access to the following tools:
1. **execute_terminal** — Run shell commands (ls, git, pip, etc.)
2. **read_file** — Read file contents with line numbers
3. **write_file** — Create or overwrite files
4. **list_files** — List directory contents
5. **move_file** — Move or rename files
6. **delete_file** — Delete files (ask for confirmation first)
7. **run_code** — Execute Python, JavaScript, or Bash code snippets
8. **capture_screen** — Take a screenshot of the user's screen
9. **fetch_webpage** — Fetch and extract text from a web page
10. **search_web** — Search the internet via Brave Search

You can also:
- Generate images via /image command
- Speak responses via voice (when enabled)
- Remember chat history across sessions

## Important Guidelines
- **Be concise, helpful, and accurate**
- **For simple greetings** (hello, help, etc.) — respond directly.
- **Proactively use tools** when the user asks for information you don't have (real-time news, specific data) or asks for a task. **Do not ask for permission**—just do it.
- **When to search the web**: DO NOT wait for keywords like "search" or "news". If the user asks about ANY current event, today's headlines, recent sports scores, recent releases, or facts you do not know, YOU MUST AUTOMATICALLY call the `search_web` tool. And if you don't have any information, you MUST call the `search_web` tool.
- **Code requests**: When the user asks to "write", "create", or "show" code — display it in a markdown code block in your response. Do NOT create files with write_file unless explicitly asked to save to disk.
- For file operations: always confirm before deleting
- For terminal commands: explain what you're about to run briefly
- **Tool Failures**: If a tool returns an error about a missing API key (e.g., `BRAVE_SEARCH_API_KEY not set`), DO NOT try alternative tools to achieve the same result. Stop immediately and tell the user they need to configure that API key (e.g. via `/skills`).
- If you're unsure, say so — never hallucinate.
- Use markdown formatting in your responses.
- **CRITICAL INSTRUCTION ON TOOLS**: DO NOT output JSON blocks or markdown code blocks containing JSON to call a tool. NEVER write ` ```json {{ "name": "search_web"... ` in your text response. You MUST use the **native tool calling feature** (function calling API) provided by your environment. If you just type the JSON, the user will see it, and the tool WILL NOT run.

## Current Context
{context}
"""


class PromptEnhancer:
    """
    Enhances prompts with system context, identity, and tool guidance.
    """

    def __init__(self):
        self.logger = get_logger("agents.prompt_enhancer")
        self._custom_instructions: str = ""

    def get_system_prompt(self) -> dict:
        """Build the full system prompt with current context."""
        context = get_system_context()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context += f"\nTimestamp: {now}"
        content = SYSTEM_PROMPT.format(
            context=context,
            agent_name=AGENT_NAME,
            user_name=USER_NAME
        )

        if self._custom_instructions:
            content += f"\n\n## User Custom Instructions\n{self._custom_instructions}"

        return {"role": "system", "content": content}

    def enhance_user_message(self, message: str) -> str:
        """
        Enhance a user message if it's too vague.
        Simple heuristic-based enhancement (no LLM call needed).

        Examples:
            "fix it" → "fix it" (too vague, but we keep it — context matters)
            "make a website" → "Create a website. Please start by outlining the structure."
        """
        # For now, pass through as-is. Enhancement via LLM call
        # can be added later if needed (costs an extra API call).
        return message

    def set_custom_instructions(self, instructions: str):
        """Set custom user instructions to include in system prompt."""
        self._custom_instructions = instructions
        self.logger.info(f"Custom instructions set ({len(instructions)} chars)")

    def build_image_prompt(self, user_prompt: str) -> str:
        """
        Enhance an image generation prompt for better results.
        Adds quality boosters that work well with SD 3.5 / FLUX.
        """
        quality_tokens = [
            "high quality", "detailed", "sharp focus",
            "professional", "4k resolution",
        ]

        # Don't add boosters if user already specified quality
        has_quality = any(q in user_prompt.lower() for q in quality_tokens)

        if not has_quality:
            user_prompt += ", high quality, detailed, sharp focus"

        return user_prompt
