<div align="center">

# 🤖 MRAgent

**Small · Fast · Secure · Completely Free · Open Source**

_Your personal AI agent that runs anywhere — terminal or browser._

[![License: MIT](https://img.shields.io/badge/License-MIT-violet.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Free](https://img.shields.io/badge/Cost-Free-brightgreen.svg)](#-free-providers)
[![Built on nanobot](https://img.shields.io/badge/Built%20on-nanobot-orange.svg)](https://github.com/HKUDS/nanobot)

</div>

---

MRAgent is a lightweight, privacy-first personal AI agent you can run from your terminal or browser in seconds. It uses free API tiers from NVIDIA, Qwen, and Groq — **no credit card required**.

> Built on and inspired by **[nanobot](https://github.com/HKUDS/nanobot)** — full credit to the HKUDS team for the excellent foundation. MRAgent adds free provider integrations, a web UI, and a tighter focus on being completely cost-free.

---

## ✨ Features

- 🆓 **Completely free** — NVIDIA NIM free credits, Qwen OAuth (no key!), Groq free tier
- 🧙 **Interactive setup** — `mragent setup` fetches live models from NVIDIA, pick with arrow keys
- 🌐 **Web UI** on port 6326 — dark glassmorphism chat, markdown, streaming
- ⚙️ **In-browser model picker** — click ⚙️, paste key, select model, save — no config editing
- 🎙️ **Voice input** — mic → Groq Whisper transcription → agent
- 🔧 **Tools** — web search, file system, shell, MCP servers, cron scheduler
- 📡 **Channels** — Telegram, Discord, WhatsApp, Slack, Email, and more
- 🔒 **Secure** — runs 100% locally, no data sent anywhere except the LLM
- ⚡ **Fast** — lightweight Python, no heavy frameworks
- 🐍 **Easy install** — `pip install bonza-mragent`

---

## 🚀 Quick Start

```bash
# Install
pip install bonza-mragent

# Initialize config
mragent onboard

# Interactive setup — pick your NVIDIA key + model with arrow keys
mragent setup

# Terminal chat
mragent agent -m "Hello!"

# Web UI (opens browser at http://localhost:6326)
mragent web
```

---

## 💚 Free Providers

MRAgent works out of the box with **zero cost** using these providers:

### 🟢 NVIDIA NIM — Free GPU-Accelerated Models

Access Llama, Qwen, Nemotron, Mistral, and more — live from NVIDIA's model catalog.

**Option A — Interactive wizard (recommended):**

```bash
mragent setup
```

This fetches all available models from NVIDIA, lets you pick a **family** (Meta, Qwen, NVIDIA, Mistral…) and then a **specific model** using ↑↓ arrow keys, and saves everything to `~/.mragent/config.json` automatically.

**Option B — Web UI settings drawer:**

1. Run `mragent web`
2. Click the ⚙️ button in the top-right header
3. Paste your `nvapi-...` key → click **Fetch Available Models**
4. Filter by family, pick a model, click **Save & Apply**

**Option C — Manual config edit:**

```json
{
  "providers": {
    "nvidia_nim": { "apiKey": "nvapi-YOUR_KEY_HERE" }
  },
  "agents": {
    "defaults": { "model": "meta/llama-3.1-8b-instruct" }
  }
}
```

**Option D — Environment variable:**

```bash
export MRAGENT_PROVIDERS__NVIDIA_NIM__API_KEY=nvapi-...
export MRAGENT_AGENTS__DEFAULTS__MODEL=meta/llama-3.1-8b-instruct
```

Get a free key at **[build.nvidia.com](https://build.nvidia.com)**.

Popular free NIM models:
| Model | Context | Notes |
|---|---|---|
| `meta/llama-3.1-8b-instruct` | 128K | Fast, great for chat |
| `meta/llama-3.3-70b-instruct` | 128K | High quality |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | 128K | Most capable |
| `qwen/qwen2.5-72b-instruct` | 128K | Strong reasoning |
| `mistralai/mistral-7b-instruct-v0.3` | 32K | Very fast |

---

### 🟣 Qwen OAuth — No API Key, ~2000 Free Req/Day

Authenticate with your Alibaba/Qwen account — no API key management.

```bash
mragent provider login qwen-oauth
# Opens browser → log in → paste token → done!
```

Then set in config:

```json
{
  "agents": {
    "defaults": {
      "provider": "qwen_oauth",
      "model": "qwen-plus"
    }
  }
}
```

---

### 🔵 Groq — Fast LLM + Voice Transcription

Groq's free tier provides ultra-fast inference and Whisper voice transcription.

1. Sign up at **[console.groq.com](https://console.groq.com)** (no credit card)
2. Create an API key
3. Add to config:

```json
{
  "providers": {
    "groq": {
      "apiKey": "gsk_YOUR_KEY_HERE"
    }
  }
}
```

**Voice**: Set `providers.groq.api_key` and the 🎙️ mic button in the web UI will transcribe your speech using `whisper-large-v3-turbo`.

---

## 🌐 Web UI

```bash
mragent web                    # port 6326 (default)
mragent web --port 8080        # custom port
mragent web --no-open          # don't auto-open browser
```

Features:

- 💬 Real-time streaming chat with markdown rendering
- ⚙️ **Settings drawer** — paste NVIDIA key, fetch live model list, pick with arrow keys, save
- 🎙️ Voice input via microphone (requires Groq key)
- 🌙 Dark glassmorphism design
- 📱 Mobile responsive
- ⌨️ Keyboard shortcuts (Enter to send, Shift+Enter for newline)

---

## 💻 Terminal

```bash
# Single message
mragent agent -m "What's the weather like in Paris?"

# Interactive session
mragent agent

# Interactive setup (NVIDIA key + model picker)
mragent setup

# Gateway mode (runs channels: Telegram, Discord, etc.)
mragent gateway

# Check status
mragent status
```

---

## ⚙️ Configuration

Config file: `~/.mragent/config.json`

```json
{
  "agents": {
    "defaults": {
      "model": "meta/llama-3.1-8b-instruct",
      "provider": "auto",
      "max_tokens": 8192,
      "temperature": 0.1
    }
  },
  "providers": {
    "nvidia_nim": { "apiKey": "nvapi-..." },
    "groq": { "apiKey": "gsk_..." }
  },
  "tools": {
    "web": {
      "search": { "apiKey": "" }
    }
  }
}
```

---

## 📡 Channels (Gateway Mode)

Connect to messaging platforms by enabling them in config:

```json
{
  "channels": {
    "telegram": { "enabled": true, "token": "YOUR_BOT_TOKEN" },
    "discord": { "enabled": true, "token": "YOUR_BOT_TOKEN" }
  }
}
```

Run: `mragent gateway`

---

## 🛠️ Development

```bash
git clone https://github.com/your-fork/Bonza
cd Bonza
pip install -e ".[dev]"
pytest tests/ -v

# Run web UI in dev mode
mragent web --no-open
```

---

## 📚 Documentation

Detailed guides and developer references are available in the **[docs/](docs/README.md)** folder:

- **[Project Structure](docs/structure.md)** — Explains how the files and directories are organized.
- **[Developer Guide](docs/contributing.md)** — Guidelines on coding traditions and adding new features.

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Credits

MRAgent is built on **[nanobot](https://github.com/HKUDS/nanobot)** by the HKUDS team. The core agent loop, channel integrations, cron scheduler, MCP support, and provider architecture are all from that excellent project.

MRAgent adds:

- NVIDIA NIM free provider integration
- `mragent setup` — interactive wizard with live model fetching + arrow-key picker
- Web UI ⚙️ settings drawer — fetch + select NVIDIA models in-browser
- Qwen OAuth (free, no API key) provider
- Groq voice upgrade (whisper-large-v3-turbo)
- Web UI on port 6326
- MRAgent branding and config paths (`~/.mragent`)

---

<div align="center">
<sub>🤖 MRAgent — Free AI for everyone</sub>
</div>
