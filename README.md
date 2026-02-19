<p align="center">
  <h1 align="center">🤖 MRAgent</h1>
  <p align="center">
    <a href="https://pypi.org/project/mragent/">
      <img src="https://img.shields.io/pypi/v/mragent.svg?style=flat-square" alt="PyPI version" />
    </a>
    <a href="https://github.com/bonzainsights/MrAgent/releases">
      <img src="https://img.shields.io/github/v/release/bonzainsights/MrAgent?style=flat-square" alt="GitHub release" />
    </a>
    <br/>
    <strong>A lightweight, open-source AI Agent powered by free APIs</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#api-providers">API Providers</a> •
    <a href="#roadmap">Roadmap</a>
  </p>
</p>

---

## ✨ Overview

**MRAgent** is a lightweight AI agent that connects to **free-tier LLM and multimodal APIs** to deliver a powerful, personal assistant experience — without expensive subscriptions. It combines text generation, image generation, text-to-speech, speech-to-text, screen monitoring, web browsing, code execution, terminal access, and file management into a single, extensible agent.

> **Philosophy:** Leverage the best free APIs available (primarily from NVIDIA and other open-source providers) to build an agent that rivals commercial solutions.

---

## 🚀 Features

| Capability               | Description                                                        | Status         |
| ------------------------ | ------------------------------------------------------------------ | -------------- |
| 💬 **LLM Chat**          | Multi-model text generation (GPT-OSS, Kimi, GLM-5, Llama 3.3)      | ✅ Implemented |
| 🎨 **Image Generation**  | Text-to-image via Stable Diffusion 3.5 Large & FLUX Dev            | ✅ Implemented |
| 🗣️ **Text-to-Speech**    | Natural voice synthesis via **Edge TTS** (Free, Neutral)           | ✅ Implemented |
| 👂 **Speech-to-Text**    | Audio transcription via **Groq Whisper v3** (Ultra-fast)           | ✅ Implemented |
| 📧 **Email Skill**       | Send & receive emails via AgentMail (Interactive `/email` command) | ✅ Implemented |
| 📱 **Telegram Bot**      | Chat, Voice, & Image interaction                                   | ✅ Implemented |
| 💓 **VivreCard**         | Background Scheduler & Heartbeat System                            | ✅ Implemented |
| 🛡️ **Poneglyph**         | System Guardian & Doctor (Auto-diagnostics)                        | ✅ Implemented |
| 🌐 **Web Browsing**      | Autonomous internet surfing and information gathering              | ✅ Implemented |
| 🖥️ **Screen Monitoring** | Capture and analyze screen content in real-time                    | ✅ Implemented |
| 💻 **Code Execution**    | Write, run, and debug code in multiple languages                   | ✅ Implemented |
| 🔧 **Terminal Access**   | Execute shell commands and system operations                       | ✅ Implemented |
| 📁 **File Management**   | Navigate, create, move, and organize files                         | ✅ Implemented |
| 🔍 **Web Search**        | Search the internet via Brave Search API                           | ✅ Implemented |

---

## 🏗️ Architecture

```
MRAgent/
├── README.md
├── .env.example          # Template for API keys
├── .gitignore
├── requirements.txt      # Python dependencies
├── main.py               # Entry point
├── config/
│   └── settings.py       # Configuration & API key management
├── data/
│   ├── mragent.json      # 🆕 Poneglyph Configuration
│   └── vivrecard_jobs.json # Scheduled jobs
├── core/
│   └── poneglyph.py      # 🆕 System Guardian & Doctor
├── agents/
│   ├── core.py           # Core agent orchestration loop
│   ├── vivrecard.py      # 🆕 Scheduler system
│   ├── planner.py        # Task planning & decomposition
│   └── executor.py       # Action execution engine
├── skills/               # Modular Skills System
│   ├── base.py           # Base skill interface
│   ├── agentmail.py      # Email skill
│   └── telegram.py       # Telegram skill
├── providers/
│   ├── base.py           # Base API provider interface
│   ├── nvidia_llm.py     # NVIDIA LLM provider (GPT-OSS, Kimi, GLM)
│   ├── nvidia_image.py   # NVIDIA image generation (SD 3.5, FLUX)
│   ├── tts.py            # Edge TTS provider
│   ├── nvidia_stt.py     # Groq STT provider
│   └── brave_search.py   # Brave Search API
├── tools/
│   ├── browser.py        # Web browsing automation
│   ├── terminal.py       # Shell command execution
│   └── ...
├── ui/
│   ├── cli.py            # Command-line interface
│   ├── telegram_bot.py   # Telegram bot interface
│   └── web.py            # Flask Web Interface
└── utils/
    ├── logger.py         # Logging utilities
    └── helpers.py        # Shared helper functions
```

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.10+**
- Free API keys (see [API Providers](#api-providers))

### Installation

### Installation

#### **One-Line Installer (Recommended)**

**Mac/Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/bonzainsights/MrAgent/main/install.sh | bash
```

**Windows (PowerShell):**

```powershell
iwr https://raw.githubusercontent.com/bonzainsights/MrAgent/main/install.bat | iex
```

#### **Pip (Python Package)**

If you have Python 3.10+ installed:

```bash
pip install git+https://github.com/bonzainsights/MRAgent.git
mragent
```

#### **Homebrew (macOS)**

```bash
brew install --HEAD https://raw.githubusercontent.com/bonzainsights/MRAgent/main/Formula/mragent.rb
```

#### **Manual Setup (Advanced)**

```bash
# Clone the repository
git clone https://github.com/bonzainsights/MrAgent.git
cd MRAgent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Quick Start

```bash
# Run the agent (CLI mode + Web UI)
python main.py

# Run as Telegram bot
python main.py --mode telegram

# Run System Diagnostic
python main.py doctor
```

---

## 🔑 API Providers

MRAgent is built around **free-tier APIs** to keep costs at zero. Here are the current providers:

### NVIDIA NIM (Primary)

| Model                      | Purpose             | API        |
| -------------------------- | ------------------- | ---------- |
| GPT-OSS-120B               | Reasoning (Primary) | NVIDIA NIM |
| Kimi K2.5                  | General-purpose LLM | NVIDIA NIM |
| GLM-5                      | Reasoning & code    | NVIDIA NIM |
| Llama 3.3 70B              | Reliable fallback   | NVIDIA NIM |
| Qwen2.5 Coder              | Code generation     | NVIDIA NIM |
| Stable Diffusion 3.5 Large | Image generation    | NVIDIA NIM |
| FLUX.1 Dev                 | Image generation    | NVIDIA NIM |

### Other Free Providers

| Provider         | Purpose             | Service                      |
| ---------------- | ------------------- | ---------------------------- |
| **Groq**         | Speech-to-Text      | Whisper Large v3 (Free)      |
| **Edge TTS**     | Text-to-Speech      | Microsoft Edge Neural (Free) |
| **AgentMail**    | Email               | AgentMail.to (Free)          |
| **Brave Search** | Web search          | Brave Search API (Free)      |
| **Telegram**     | Messaging Interface | Telegram Bot API (Free)      |

> 💡 **Adding new providers?** Implement the base interface in `providers/base.py` and register your provider in the config.

---

## 🗺️ Roadmap

- [x] Project setup & repository initialization
- [x] Core agent loop with task planning
- [x] NVIDIA LLM integration (multi-model)
- [x] Image generation pipeline
- [x] Text-to-speech (Edge TTS)
- [x] Speech-to-text (Groq Whisper)
- [x] Telegram bot interface (Voice & Image support)
- [x] Web Interface (Chat & Voice)
- [x] Email Integration (AgentMail)
- [x] VivreCard Scheduler
- [x] Poneglyph System (Guardian & Doctor)
- [x] Brave Search integration
- [x] Terminal & code execution tools
- [x] File management system
- [x] Screen monitoring & analysis
- [x] Web browsing automation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

MRAgent uses free-tier API keys which may have rate limits and usage quotas. The agent is designed to work within these constraints. Never commit your `.env` file or expose API keys publicly.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/bonzainsights">Bonza Insights</a> & <a href="https://github.com/achbj">achbj</a>
</p>
