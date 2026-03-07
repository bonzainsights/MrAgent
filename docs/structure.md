# 📂 MRAgent File Structure

Welcome to the inner workings of MRAgent! We keep things lean, modular, and easy to navigate. Here's the layout of our digital headquarters.

## 🏗 The `mragent/` Core

- **`agent/`** 🧠
  The brain! Contains the core logic, decision-making loops, and the reasoning engine.
- **`skills/`** 🛠
  The toolbox. This is where the agent's actual capabilities (searching, tmux control, etc.) live.
- **`providers/`** 🔌
  External connections. Logic for interacting with LLM providers (NVIDIA NIM, Groq) and transcription services.
- **`channels/`** 📡
  Communication portals. How the agent talks to the world (Telegram, Dingtalk, Slack, Matrix).
- **`web/`** 🌐
  The beautiful UI you see in your browser. Contains the `server.py` and the glassmorphism `index.html`.
- **`session/`** 📖
  Memory lane. Manages chat history, session state, and persistence.
- **`config/`** ⚙️
  The command center. Pydantic-powered settings and configuration loading.
- **`cli/`** ⌨️
  Command-line power. The code behind the `mragent` and `nanobot` commands.
- **`cron/`** ⏰
  The agent's calendar. Logic for scheduled tasks and autonomous reminders.
- **`bus/`** 🚌
  The internal message highway. Connects different components through an event-driven architecture.
- **`templates/`** 📝
  The prompt bank. Markdown templates that guide the LLM's behavior.
- **`utils/`** 🧰
  Common helpers and shortcut functions used across the codebase.

## 🗺 Root Directory Highlights

- **`tests/`** 🧪
  Where we make sure nothing breaks.
- **`docs/`** 📚
  (You are here!) The friendly guidebooks for developers.
- **`pyproject.toml`** 📦
  Modern project manifest defining dependencies and entry points.

---

_Keep it simple. Keep it fast. Keep it MRAgent._
