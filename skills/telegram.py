"""
MRAgent — Telegram Skill
Provides Telegram messaging capabilities.
"""

import os
import requests
from typing import List

from skills.base import Skill
from tools.base import Tool


class TelegramSkill(Skill):
    name = "telegram"
    description = "Chat capabilities via Telegram Bot API"

    def get_tools(self) -> List[Tool]:
        return [
            TelegramSendTool(),
        ]


class TelegramTool(Tool):
    """Base tool for Telegram operations."""

    def _get_api_key(self) -> str:
        key = os.getenv("TELEGRAM_BOT_TOKEN")
        if not key:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN in .env")
        return key

    def _get_chat_id(self) -> str:
        chat_id = os.getenv("ALLOWED_TELEGRAM_CHATS")
        # Optional: tool args can override this, but env var is default target
        return chat_id

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        api_key = self._get_api_key()
        url = f"https://api.telegram.org/bot{api_key}{endpoint}"
        
        try:
            if method == "GET":
                resp = requests.get(url, params=data, timeout=10)
            else:
                resp = requests.post(url, json=data, timeout=10)
            
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            return {"error": f"HTTP Error: {e.response.text if e.response else str(e)}"}
        except Exception as e:
            return {"error": str(e)}


class TelegramSendTool(TelegramTool):
    name = "send_telegram"
    description = "Send a message to a Telegram chat."
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Text message to send",
            },
            "chat_id": {
                "type": "string",
                "description": "Target chat ID (optional, defaults to env var)",
            },
        },
        "required": ["message"],
    }

    def execute(self, message: str, chat_id: str = None) -> str:
        target_chat_id = chat_id or self._get_chat_id()
        if not target_chat_id:
            return "❌ Error: No chat_id provided and ALLOWED_TELEGRAM_CHATS not set in .env"

        payload = {
            "chat_id": target_chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        
        result = self._request("POST", "/sendMessage", payload)
        
        if "error" in result:
            return f"❌ Failed to send Telegram message: {result['error']}"
            
        return f"✅ Message sent to Telegram chat {target_chat_id}!"
