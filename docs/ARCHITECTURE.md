# 🏗️ MRAgent — Detailed Architecture Plan

> **Version:** 0.1.0 | **Created:** 2026-02-15 | **Last Updated:** 2026-02-23 | **Status:** Active  
> **Goal:** Build a lightweight Jarvis-like AI agent that runs on low-end hardware using free NVIDIA NIM APIs.

---

## 1. Design Principles

| Principle              | How We Achieve It                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| **Lightweight**        | No ML models loaded locally. All inference via API. SQLite for storage. Minimal deps (~15 packages). |
| **Cross-platform**     | Pure Python, no OS-specific deps. Works on Windows/Mac/Linux/old PCs.                                |
| **API-first**          | NVIDIA NIM primary. OpenAI-compatible SDK means easy provider swapping.                              |
| **Privacy-respecting** | All data stored locally. No telemetry. User controls what gets sent to APIs.                         |
| **Modular**            | Each capability (LLM, image, voice, tools) is a swappable plugin.                                    |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   CLI    │  │  Web (Flask) │  │  Telegram Bot             │  │
│  │  (rich)  │  │  :7860       │  │  (python-telegram-bot)   │  │
│  └────┬─────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│       └───────────────┬┴─────────────────────┘                  │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    AGENT CORE                             │   │
│  │  ┌─────────────┐ ┌────────────┐ ┌───────────────────┐   │   │
│  │  │  Prompt      │ │  Context   │ │  Model Selector   │   │   │
│  │  │  Enhancer    │ │  Manager   │ │  (auto/think/fast)│   │   │
│  │  └─────────────┘ └────────────┘ └───────────────────┘   │   │
│  │                                                          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  Agent Loop: Receive → Plan → Execute → Respond  │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────┬────────────────────────┬──────────────────┘   │
│                 ▼                        ▼                       │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │    PROVIDER LAYER     │  │        TOOL SYSTEM             │   │
│  │                       │  │                                │   │
│  │  ┌─────────────────┐ │  │  ┌──────────┐ ┌────────────┐  │   │
│  │  │  NVIDIA LLM     │ │  │  │ Terminal │ │ File Mgr   │  │   │
│  │  │  (OpenAI SDK)   │ │  │  └──────────┘ └────────────┘  │   │
│  │  ├─────────────────┤ │  │  ┌──────────┐ ┌────────────┐  │   │
│  │  │  NVIDIA ImageGen│ │  │  │ Code Run │ │ Screen Cap │  │   │
│  │  │  (REST API)     │ │  │  └──────────┘ └────────────┘  │   │
│  │  ├─────────────────┤ │  │  ┌──────────┐ ┌────────────┐  │   │
│  │  │  NVIDIA TTS     │ │  │  │ Browser  │ │ Web Search │  │   │
│  │  │  (Riva gRPC)    │ │  │  └──────────┘ └────────────┘  │   │
│  │  ├─────────────────┤ │  │  ┌──────────┐ ┌────────────┐  │   │
│  │  │  NVIDIA STT     │ │  │  │ PDF Read │ │ Image Gen  │  │   │
│  │  │  (Riva gRPC)    │ │  │  └──────────┘ └────────────┘  │   │
│  │  └─────────────────┘ │                                       │
│  └──────────────────────┘                                       │
│                 ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  MEMORY & STORAGE                         │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │   │
│  │  │ Chat Store   │ │ Config       │ │ Chat Summaries   │ │   │
│  │  │ (SQLite)     │ │ Backup/Roll  │ │ & Cross-Ref      │ │   │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                 ▼                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  VOICE PIPELINE                           │   │
│  │  Mic → VAD → STT → Agent → TTS → Speaker                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. NVIDIA NIM API Integration Map

### 3.1 LLM Chat (OpenAI-Compatible REST)

| Model       | NVIDIA NIM ID                         | Use Case                       | Context Window |
| ----------- | ------------------------------------- | ------------------------------ | -------------- |
| Kimi K2.5   | `moonshotai/kimi-k2.5`                | Complex reasoning, agent swarm | 131K           |
| GLM5        | `z-ai/glm5`                           | Reasoning & code, tool calling | 128K           |
| Gemma 3N    | `google/gemma-3n-e4b-it`              | Fast lightweight responses     | 32K            |
| Qwen3 Coder | `qwen/qwen3-coder-480b-a35b-instruct` | Agentic coding, 480B MoE       | 262K           |
| Llama 3.3   | `meta/llama-3.3-70b-instruct`         | General fallback               | 128K           |

**Endpoint:** `https://integrate.api.nvidia.com/v1/chat/completions`  
**Auth:** Bearer token from env  
**SDK:** `openai` Python package (set `base_url` to NVIDIA)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_KIMI_K2_5"]
)

# Standard chat completion
response = client.chat.completions.create(
    model="moonshotai/kimi-k2.5",
    messages=[{"role": "user", "content": "Hello MRAgent!"}],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)

# Function calling (tool use)
response = client.chat.completions.create(
    model="moonshotai/kimi-k2.5",
    messages=messages,
    tools=[{
        "type": "function",
        "function": {
            "name": "execute_terminal",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"]
            }
        }
    }],
    tool_choice="auto"
)
```

### 3.2 Image Generation (REST API)

**Stable Diffusion 3.5 Large:**

```
POST https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-5-large
Authorization: Bearer {NVIDIA_SD_35_LARGE}
Content-Type: application/json

{
    "text_prompts": [{"text": "A cyberpunk city at sunset", "weight": 1}],
    "cfg_scale": 5,
    "height": 1024,
    "width": 1024,
    "steps": 50,
    "seed": 0
}
→ Response: {"artifacts": [{"base64": "...", "seed": 12345}]}
```

**FLUX Dev:**

```
POST https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev
Same auth/format pattern
```

### 3.3 Text-to-Speech (Riva gRPC)

```python
import riva.client

auth = riva.client.Auth(
    uri="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    metadata_args=[
        ["function-id", "0149dedb-2be8-4195-b9a0-e57e0e14f972"],
        ["authorization", f"Bearer {os.environ['NVIDIA_MAGPIE_TTS']}"]
    ]
)

tts = riva.client.SpeechSynthesisService(auth)
resp = tts.synthesize(
    text="Hello, I am MRAgent!",
    voice_name="English-US.Female-1",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=44100
)
# resp.audio → raw PCM bytes, play via sounddevice
```

### 3.4 Speech-to-Text (Riva gRPC)

```python
asr = riva.client.ASRService(auth_whisper)
config = riva.client.StreamingRecognitionConfig(
    config=riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
        sample_rate_hertz=16000,
    ),
    interim_results=True,
)
# Stream microphone audio → get real-time transcriptions
```

### 3.5 Brave Search (REST)

```python
response = requests.get(
    "https://api.search.brave.com/res/v1/web/search",
    headers={"X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]},
    params={"q": query, "count": 5, "safesearch": "moderate"}
)
results = response.json()["web"]["results"]
# Extract: title, url, description for each result
```

---

## 4. Agent Core — The Brain

### 4.1 Main Loop (ReAct Pattern)

```
User Input
    │
    ▼
┌─ ENHANCE PROMPT ──────────────────────┐
│  - Add system prompt (identity, time)  │
│  - Add context (OS, cwd, open files)   │
│  - Rewrite vague queries               │
└──────────┬────────────────────────────┘
           ▼
┌─ SELECT MODEL ────────────────────────┐
│  auto: classify → fast/think/code     │
│  manual: user-selected model          │
└──────────┬────────────────────────────┘
           ▼
┌─ LLM CALL (with tools) ──────────────┐
│  Send messages + tool definitions      │
│  Get response (text or tool_calls)     │
└──────────┬────────────────────────────┘
           ▼
     ┌─ Has tool_calls? ─┐
     │                    │
    YES                  NO
     │                    │
     ▼                    ▼
┌─ EXECUTE TOOLS ──┐  ┌─ OUTPUT ─────────┐
│  Run each tool    │  │  Display text     │
│  Format results   │  │  or TTS playback  │
│  Append to msgs   │  │  Save to history  │
└──────┬───────────┘  └─────────────────┘
       │
       └──► Loop back to LLM CALL
```

### 4.2 Context Management Strategy

```
Max Context: ~128K tokens (Kimi K2.5)

Active Window:
┌─────────────────────────────────┐
│ System Prompt           (~500)  │
│ Context Injection       (~200)  │
│ Chat Summary (if any)   (~500)  │
│ Recent Messages     (~100,000)  │
│ Tool Results            (~var)  │
│ ─────────── Buffer ──────────── │
│ Reserved for Response  (~8,000) │
└─────────────────────────────────┘

When context reaches 80%:
1. Summarize oldest messages → store summary
2. Replace old messages with summary
3. Keep last 10 messages in full
4. Full history always in SQLite

Auto New Chat:
- When summary quality degrades (topic drift detected)
- User can force: /newchat
- Previous chat accessible via /history
```

### 4.3 Model Auto-Selection Logic

```python
def select_model(user_message: str, mode: str = "auto") -> str:
    if mode == "thinking":
        return "moonshotai/kimi-k2.5"  # Best reasoning
    elif mode == "fast":
        return "google/gemma-3n-e4b-it"  # Fastest

    # Auto mode: classify the task
    keywords_code = ["code", "function", "bug", "implement", "debug", "script"]
    keywords_complex = ["analyze", "plan", "design", "compare", "explain why"]

    msg_lower = user_message.lower()
    if any(k in msg_lower for k in keywords_code):
        return "qwen/qwen3-coder-480b-a35b-instruct"
    elif any(k in msg_lower for k in keywords_complex):
        return "moonshotai/kimi-k2.5"
    else:
        return "meta/llama-3.3-70b-instruct"
```

---

## 5. Memory Architecture

### 5.1 SQLite Schema

```sql
-- Lightweight: one file, no server, <1MB for 10K messages
CREATE TABLE chats (
    id TEXT PRIMARY KEY,     -- UUID
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary TEXT,            -- Auto-generated summary
    token_count INTEGER DEFAULT 0
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT REFERENCES chats(id),
    role TEXT,               -- system/user/assistant/tool
    content TEXT,
    tool_calls TEXT,         -- JSON if assistant made tool calls
    tool_call_id TEXT,       -- if this is a tool result
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_estimate INTEGER
);

CREATE TABLE config_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot BLOB,           -- JSON of full config
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Keep only last 3 rows via trigger
```

### 5.2 Config Backup/Rollback

```
data/
├── chats.db              # SQLite database
├── config_backups/
│   ├── config_001.json   # 3 steps ago
│   ├── config_002.json   # 2 steps ago
│   └── config_003.json   # most recent
├── images/               # Generated images
└── logs/
    └── mragent.log       # Rotating log file
```

---

## 6. Kimi K2.5 Agent Swarm

Kimi K2.5's unique capability: coordinate up to 100 sub-agents with 1,500 tool calls per session.

### How We Use It:

```
User: "Create me a portfolio website"
          │
          ▼
┌─ ORCHESTRATOR (Kimi K2.5) ─────────┐
│  Decompose task into sub-tasks:      │
│  1. Plan site structure              │
│  2. Generate content                 │
│  3. Write HTML/CSS                   │
│  4. Create images                    │
│  5. Review & fix                     │
│                                      │
│  For each sub-task → tool calls:     │
│  - file_write("index.html", ...)     │
│  - file_write("styles.css", ...)     │
│  - generate_image("hero banner")     │
│  - terminal("python -m http.server") │
└──────────────────────────────────────┘
```

The swarm works through Kimi's **native multi-step tool calling** — no separate swarm framework needed. We just provide all tools and let Kimi orchestrate.

---

## 7. Directory Structure (Final)

```
MRAgent/
├── main.py                    # Entry point & startup
├── requirements.txt           # Dependencies
├── .env                       # API keys (gitignored)
├── .env.example               # Template
├── README.md
│
├── config/
│   └── settings.py            # Config, model registry, autonomy settings
│
├── providers/
│   ├── base.py                # Abstract provider interface
│   ├── nvidia_llm.py          # NVIDIA LLM (OpenAI SDK)
│   ├── nvidia_image.py        # NVIDIA Image Gen (REST)
│   ├── tts.py                 # Edge TTS
│   ├── nvidia_stt.py          # Groq STT
│   ├── brave_search.py        # Brave Search API
│   ├── google_search.py       # Google Custom Search API
│   └── langsearch.py          # LangSearch API
│
├── agents/
│   ├── core.py                # Main agent loop + tiered approval
│   ├── prompt_enhancer.py     # Prompt rewriting & context injection
│   ├── context_manager.py     # Token counting & sliding window
│   ├── model_selector.py      # Auto model selection
│   └── watcher.py             # Eagle Eye screen monitor
│
├── tools/
│   ├── base.py                # Base tool interface + OpenAI schema
│   ├── terminal.py            # Shell command execution
│   ├── file_manager.py        # File operations
│   ├── code_runner.py         # Code execution sandbox
│   ├── screen.py              # Screen capture & diff
│   ├── browser.py             # Web fetch & search (with sanitizer)
│   ├── pdf_reader.py          # PDF text extraction
│   └── image_gen.py           # Image generation tool
│
├── skills/
│   ├── agentmail.py           # Email skill
│   └── telegram.py            # Telegram skill
│
├── memory/
│   └── chat_store.py          # SQLite chat storage
│
├── ui/
│   ├── cli.py                 # Rich CLI (commands, menus, autonomy)
│   ├── web.py                 # Flask browser interface
│   └── telegram_bot.py        # Telegram bot
│
├── utils/
│   ├── sanitizer.py           # Prompt injection defense
│   ├── logger.py              # Logging system
│   └── helpers.py             # Shared utilities
│
├── core/
│   └── poneglyph.py           # System Guardian & Doctor
│
├── data/                      # Runtime data (gitignored)
│   ├── chats.db
│   ├── config_backups/
│   ├── images/
│   └── logs/
│
└── docs/
    └── ARCHITECTURE.md        # This file
```

---

## 8. Dependency Budget

Target: **< 50MB installed** (excluding pip cache)

| Package               | Size   | Purpose                            |
| --------------------- | ------ | ---------------------------------- |
| `openai`              | ~5MB   | LLM API client (OpenAI-compatible) |
| `nvidia-riva-client`  | ~30MB  | gRPC client for TTS/STT            |
| `requests`            | ~0.5MB | HTTP client for REST APIs          |
| `python-dotenv`       | ~0.1MB | .env file loading                  |
| `rich`                | ~3MB   | Terminal UI                        |
| `prompt-toolkit`      | ~2MB   | Interactive CLI input              |
| `Pillow`              | ~5MB   | Image handling                     |
| `sounddevice`         | ~0.5MB | Audio capture/playback             |
| `numpy`               | ~15MB  | Audio array handling               |
| `flask`               | ~1MB   | Web server (optional)              |
| `pyautogui`           | ~1MB   | Screen capture                     |
| `beautifulsoup4`      | ~0.5MB | HTML parsing                       |
| `python-telegram-bot` | ~1MB   | Telegram interface (optional)      |

**Total: ~66MB** (acceptable for low-end devices, no GPU needed)

---

## 9. Rate Limit Strategy

NVIDIA NIM free tier: ~40 requests/min

```python
class RateLimiter:
    def __init__(self, max_rpm=35):  # Leave 5 RPM headroom
        self.max_rpm = max_rpm
        self.requests = []  # Timestamps of recent requests

    def wait_if_needed(self):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < 60]
        if len(self.requests) >= self.max_rpm:
            sleep_time = 60 - (now - self.requests[0])
            time.sleep(sleep_time)
        self.requests.append(time.time())
```

---

## 10. Implementation Phases & Timeline

| Phase        | What                                                 | Estimated Effort |
| ------------ | ---------------------------------------------------- | ---------------- |
| **Phase 2**  | Core foundation (config, logging, main.py)           | ~2 hours         |
| **Phase 3**  | Provider layer (NVIDIA LLM, image, TTS, STT, search) | ~4 hours         |
| **Phase 4**  | Tool system (terminal, files, code, screen, browser) | ~3 hours         |
| **Phase 5**  | Agent core (loop, prompt enhancer, context manager)  | ~4 hours         |
| **Phase 6**  | Memory & history (SQLite, config backup)             | ~2 hours         |
| **Phase 7**  | Voice pipeline (mic → STT → agent → TTS)             | ~3 hours         |
| **Phase 8**  | User interfaces (CLI, web, Telegram)                 | ~4 hours         |
| **Phase 9**  | Advanced features (swarm, screen monitor, planning)  | ~4 hours         |
| **Phase 10** | Testing & polish                                     | ~3 hours         |

**Total: ~29 hours of focused implementation**

---

## 11. Security & Autonomy Architecture

### 11.1 Prompt Injection Defense (2-Layer)

```
External Data (web pages, search results, PDFs)
    │
    ▼
┌─ LAYER 1: Sanitizer (utils/sanitizer.py) ────┐
│  1. strip_dangerous_patterns()                 │
│     - Regex detection of injection patterns    │
│     - Removes: "ignore instructions",          │
│       "share API keys", embedded bash/python   │
│  2. sanitize_external_data()                   │
│     - Wraps in structural markers:             │
│     ═══ [UNTRUSTED EXTERNAL DATA] ═══          │
│     ... content ...                            │
│     ═══ [END UNTRUSTED EXTERNAL DATA] ═══      │
└──────────────────────────────────────────────┘
    │
    ▼
┌─ LAYER 2: System Prompt (prompt_enhancer.py) ─┐
│  - LLM instructed to NEVER follow             │
│    instructions inside UNTRUSTED markers       │
│  - Data treated as DISPLAY-ONLY               │
│  - Report injection attempts to user           │
└──────────────────────────────────────────────┘
```

### 11.2 Tiered Approval System

```
Tool Call (execute_terminal / run_code)
    │
    ▼
┌─ Check AUTONOMY_SETTINGS.trust_level ─┐
│                                        │
├── "autonomous" ──► Run immediately     │
│                    Log action           │
│                                        │
├── "balanced" ────► Check patterns:     │
│   │ is_safe_command() OR               │
│   │ fnmatch(cmd, auto_approve_patterns)│
│   ├── Match ──► Auto-run               │
│   └── No match ──► Ask user            │
│                    + Telegram notify    │
│                                        │
├── "cautious" ────► Always ask user     │
│                                        │
└────────────────────────────────────────┘
```

---

_This document will be updated as implementation progresses. Each phase will be committed separately with detailed logging._
