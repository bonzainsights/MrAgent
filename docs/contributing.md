# 🛠 Contributing to MRAgent

Want to add a new feature or fix a bug? Awesome! We follow a simple, catchy "tradition" here to keep the code clean, fast, and secure.

## 📝 The Coding Tradition

1. **Concise First Lines** ✍️
   Always start your Python files with a docstring that summarizes exactly what it does in one line.
2. **Async by Default** ⚡️
   MRAgent lives in the future. We use `async`/`await` for almost everything to stay responsive.
3. **Type Hints Everywhere** 🏷
   We use Python 3.11+ syntax. Use `str | None` instead of `Optional[str]`. This keeps the code readable and easy to debug.
4. **Loguru is Your Friend** 📝
   Use `from loguru import logger` for all debugging and info messages. No naked `print()` statements allowed!
5. **Clean & Flat** 📐
   We prefer flat hierarchies over deep nesting. If a function is too long, break it out—but keep it logical.

## 📁 Adding New Files

Following the tradition is easy:

- **Adding a Skill?** 🛠
  Drop it into `mragent/skills/`. Each skill should be its own directory with a `SKILL.md` for the agent to read.
- **Adding a Channel?** 📡
  Create a new file in `mragent/channels/`. Follow the structure of `telegram.py` or `dingtalk.py`.
- **Adding a Provider?** 🔌
  New LLM or API? Put it in `mragent/providers/` and inherit from the base provider class.

## 🛠 Pro-Tips for Clean Code

- **Imports**: Group your imports (Standard Lib -> Third Party -> MRAgent).
- **Style**: We follow `ruff` formatting. Keep it tidy!
- **UI**: If you're touching the web UI, keep the "Glassmorphism" aesthetic alive. Use CSS variables for everything.

---

_Build it fast. Build it clean. Build it for everyone._
