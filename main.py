#!/usr/bin/env python3
"""
MRAgent — Lightweight AI Agent
A Jarvis-like AI assistant powered by free NVIDIA NIM APIs.

Usage:
    python main.py                     # CLI + Web on port 16226 (default)
    python main.py --mode cli          # CLI only
    python main.py --mode web          # Browser UI only
    python main.py --mode telegram     # Telegram bot
    python main.py --voice             # CLI with voice input/output
    python main.py --debug             # Verbose logging to terminal

Created: 2026-02-15
Repository: https://github.com/bonzainsights/MRAgent
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import get_logger
from config.settings import (
    DEFAULTS, validate_config, save_config_backup, SYSTEM_INFO
)
from agents.vivrecard import VivreCard
from core.poneglyph import Poneglyph

logger = get_logger("main")


# ──────────────────────────────────────────────
# Version
# ──────────────────────────────────────────────
__version__ = "0.2.0"



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MRAgent — Lightweight AI Agent powered by free NVIDIA NIM APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                   Start in CLI mode\n"
            "  python main.py --mode web        Start browser UI\n"
            "  python main.py --mode telegram   Start Telegram bot\n"
            "  python main.py --voice           Enable voice I/O\n"
            "  python main.py --model kimi-k2.5 Use specific model\n"
        ),
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "web", "both", "telegram", "watch"],
        default="both",
        help="Interface mode: both (cli+web, default), cli, web, telegram, watch",
    )
    parser.add_argument(
        "--voice", "-v",
        action="store_true",
        default=False,
        help="Enable voice input/output",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override default LLM model (e.g. kimi-k2.5, llama-3.3-70b, qwen3-coder)",
    )
    parser.add_argument(
        "--model-mode",
        choices=["auto", "thinking", "fast", "code"],
        default="auto",
        help="Model selection mode (default: auto)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("PORT", 16226)),
        help="Port for web UI (default: from .env or 16226)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"MRAgent v{__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────
BANNER = r"""
 ╔══════════════════════════════════════════════╗
 ║   ███╗   ███╗██████╗  █████╗  ██████╗       ║
 ║   ████╗ ████║██╔══██╗██╔══██╗██╔════╝       ║
 ║   ██╔████╔██║██████╔╝███████║██║  ███╗      ║
 ║   ██║╚██╔╝██║██╔══██╗██╔══██║██║   ██║      ║
 ║   ██║ ╚═╝ ██║██║  ██║██║  ██║╚██████╔╝      ║
 ║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝       ║
 ╚══════════════════════════════════════════════╝
"""


def print_startup_info(args: argparse.Namespace):
    """Print startup banner and configuration summary."""
    from config.settings import AGENT_NAME, USER_NAME, AUTONOMY_SETTINGS

    print(BANNER)

    # Validate API keys
    report = validate_config()

    # Compact status line
    trust = AUTONOMY_SETTINGS.get("trust_level", "balanced")
    trust_icons = {"cautious": "🔒", "balanced": "⚖️", "autonomous": "⚡"}
    model_count = len(report["valid"])
    mode_str = f"{args.mode}" + (" +voice" if args.voice else "")

    print(f"  {AGENT_NAME} v{__version__} | {SYSTEM_INFO['os']} {SYSTEM_INFO['platform']}")
    print(f"  Mode: {mode_str} | Models: {model_count} | Trust: {trust_icons.get(trust, '❓')} {trust}")

    if report["missing"]:
        print(f"  ⚠ Missing keys: {', '.join(report['missing'][:3])}{'...' if len(report['missing']) > 3 else ''}")
    for warning in report["warnings"][:2]:
        print(f"  ⚠ {warning}")
    print()

    # Log (not printed to console)
    logger.info(f"MRAgent v{__version__} starting — mode={args.mode}")
    logger.info(f"System: {SYSTEM_INFO['os']} {SYSTEM_INFO['platform']}")
    logger.info(f"Models: {', '.join(report['valid'])}")

    # Save config snapshot at startup
    backup_path = save_config_backup()
    logger.debug(f"Config backup saved: {backup_path}")

    # Auto-clean old data (images >7d, uploads >3d, logs >30d)
    from utils.cleanup import run_startup_cleanup
    run_startup_cleanup()

    # Start VivreCard Scheduler
    vivrecard = VivreCard()
    vivrecard.start()
    logger.info("VivreCard Scheduler started in background")


def run_cli(args: argparse.Namespace):
    """Launch the CLI interface."""
    logger.info("Starting CLI interface...")
    try:
        from ui.cli import CLIInterface
        cli = CLIInterface(
            voice_enabled=args.voice,
            model_override=args.model,
            model_mode=args.model_mode,
        )
        cli.run()
    except ImportError as e:
        if "rich" in str(e) or "prompt" in str(e):
            logger.error(f"CLI dependencies missing: {e}")
            logger.info("Install with: pip install rich prompt-toolkit")
        else:
            logger.error(f"Failed to start CLI due to missing module: {e}")
        sys.exit(1)


def run_web(args: argparse.Namespace):
    """Launch the web UI."""
    logger.info(f"Starting web UI on port {args.port}...")
    try:
        from ui.web import create_app
        app = create_app()
        app.run(host="0.0.0.0", port=args.port, debug=args.debug)
    except ImportError as e:
        logger.error(f"Web dependencies missing: {e}")
        logger.info("Install with: pip install flask")
        sys.exit(1)


def run_telegram(args: argparse.Namespace):
    """Launch the Telegram bot."""
    logger.info("Starting Telegram bot...")
    try:
        from ui.telegram_bot import TelegramBot
        bot = TelegramBot()
        bot.run()
    except ImportError as e:
        logger.error(f"Telegram dependencies missing: {e}")
        logger.info("Install with: pip install python-telegram-bot")
        sys.exit(1)


def run_both(args: argparse.Namespace):
    """Launch Web UI in background + CLI in foreground simultaneously."""
    import threading

    print("  🌐 Starting Web UI in background on port %d..." % args.port)
    logger.info(f"Dual mode: starting web UI on port {args.port} + CLI")

    def _web_thread():
        try:
            from ui.web import create_app
            app = create_app()
            # Suppress Flask's default request logging in dual mode
            import logging as _logging
            _logging.getLogger("werkzeug").setLevel(_logging.WARNING)
            app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Web UI failed: {e}")

    web_thread = threading.Thread(target=_web_thread, daemon=True)
    web_thread.start()

    print(f"  ✅ Web UI running at http://localhost:{args.port}")
    print()

    # Launch Telegram Bot if token is present
    import os
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.info("Checks found TELEGRAM_BOT_TOKEN, starting Telegram bot in background...")
        print("  🤖 Starting Telegram Bot in background...")
        
        def _telegram_thread():
            try:
                # Suppress telegram logs to avoid cluttering CLI
                import logging as _logging
                _logging.getLogger("httpx").setLevel(_logging.WARNING)
                _logging.getLogger("telegram").setLevel(_logging.WARNING)
                
                from ui.telegram_bot import TelegramBot
                bot = TelegramBot()
                # Run async loop for telegram
                import asyncio
                asyncio.run(bot.run_async())
            except Exception as e:
                logger.error(f"Telegram Bot failed: {e}")

        telegram_thread = threading.Thread(target=_telegram_thread, daemon=True)
        telegram_thread.start()
        print("  ✅ Telegram Bot running")

    # Run CLI in foreground
    run_cli(args)


def run_watch(args: argparse.Namespace):
    """Launch Eagle Eye Watcher mode."""
    logger.info("Starting Eagle Eye Watcher...")
    try:
        from agents.watcher import EagleEyeWatcher
        # Default to 2s interval, 5% threshold
        watcher = EagleEyeWatcher(interval=2.0, diff_threshold=5.0)
        watcher.start()
    except ImportError as e:
        logger.error(f"Watcher dependencies missing: {e}")
        logger.info("Ensure pillow, edge-tts, and pygame/afplay are available.")
        sys.exit(1)


def run_install_wizard():
    """Interactive wizard to guide new users to set up API keys."""
    from config.settings import validate_config, _APP_DATA_DIR
    from config import settings
    
    report = validate_config()
    if report["valid"]:
        return  # Keys exist, skip wizard

    print("\n" + "="*60)
    print(" 🚀 Welcome to MRAgent Setup Wizard")
    print("="*60)
    print("It looks like you don't have any API keys configured yet.")
    print("To get started for FREE, you need an NVIDIA NIM API Key.")
    print("\n1. Go to: https://build.nvidia.com")
    print("2. Sign in or create an account.")
    print("3. Click 'Get API Key' and copy the generated key.")
    print("="*60 + "\n")
    
    while True:
        key = input("Paste your NVIDIA api key here (or press Enter to skip): ").strip()
        if not key:
            print("Skipping wizard. You may have limited functionality.")
            break
            
        if len(key) > 20: 
            env_path = _APP_DATA_DIR / ".env"
            
            updates = [
                f"\n# Auto-configured by Startup Wizard",
                f"NVIDIA_API_KEY={key}",
            ]
            
            with open(env_path, "a", encoding="utf-8") as f:
                f.write("\n".join(updates) + "\n")
                
            print("\n✅ API Key saved to .env!")
            print("Reloading environment...\n")
            
            os.environ["NVIDIA_API_KEY"] = key
            
            settings.NVIDIA_KEYS["kimi_k2_5"] = key
            settings.NVIDIA_KEYS["glm5"] = key
            settings.NVIDIA_KEYS["sd_35_large"] = key
            settings.NVIDIA_KEYS["whisper_lv3"] = key
            settings.NVIDIA_KEYS["flux_dev"] = key
            settings.NVIDIA_KEYS["magpie_tts"] = key
            settings.NVIDIA_KEYS["gemma_3n"] = key
            settings.NVIDIA_KEYS["qwen3_coder"] = key
            settings.NVIDIA_KEYS["llama_33_70b"] = key
            settings.NVIDIA_KEYS["gpt_oss_120b"] = key
            settings.NVIDIA_KEYS["llama_32_11b_vision"] = key
            break
        else:
            print("❌ That doesn't look like a valid API key. Please try again.")

def run_identity_wizard():
    """Interactive wizard to configure User and Agent names on first boot."""
    from config.settings import _APP_DATA_DIR
    from config import settings
    import os
    
    # Check if USER_NAME is already in .env
    env_path = _APP_DATA_DIR / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            if "USER_NAME=" in f.read():
                return

    print("\n" + "="*60)
    print(" 📛 Persona & Identity Setup")
    print("="*60)
    print("Let's personalize your experience!")
    
    user_name = input("What is your name? [default: User]: ").strip()
    if not user_name:
        user_name = "User"
        
    agent_name = input("What would you like to call me? [default: MRAgent]: ").strip()
    if not agent_name:
        agent_name = "MRAgent"

    updates = [
        f"\n# Identity Configuration",
        f"USER_NAME={user_name}",
        f"AGENT_NAME={agent_name}",
    ]
    
    with open(env_path, "a", encoding="utf-8") as f:
        f.write("\n".join(updates) + "\n")
        
    print(f"\n✅ Nice to meet you, {user_name}! I will answer to {agent_name}.")
    print("="*60 + "\n")
    
    # Dynamically update the running environment for this session
    os.environ["USER_NAME"] = user_name
    os.environ["AGENT_NAME"] = agent_name
    
    # Update settings globals
    settings.USER_NAME = user_name
    settings.AGENT_NAME = agent_name

def main():
    """Main entry point."""
    
    # Initialize Poneglyph (The Guardian)
    poneglyph = Poneglyph()

    # Handle 'doctor' command early
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        if "--fix" in sys.argv:
            poneglyph.run_fixer()
        else:
            poneglyph.check_health()
            poneglyph.report()
        return

    # Poneglyph Guardian Check before startup
    if not poneglyph.check_health():
         logger.warning("System health check reported issues. Run 'python main.py doctor' for details.")

    # Run startup wizard if missing keys
    run_install_wizard()
    run_identity_wizard()

    args = parse_args()

    # Set debug logging if requested — shows all logs in terminal too
    if args.debug:
        DEFAULTS["log_level"] = "DEBUG"
        import logging
        # Set root mragent logger to DEBUG
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
        # Also configure any future loggers
        logging.getLogger("mragent").setLevel(logging.DEBUG)

    # Override settings from args
    if args.model:
        DEFAULTS["default_llm"] = args.model
    if args.model_mode:
        DEFAULTS["model_selection_mode"] = args.model_mode
    DEFAULTS["voice_enabled"] = args.voice
    DEFAULTS["web_port"] = args.port

    print_startup_info(args)
    
    # Test Poneglyph LLM Diagnostics
    # Launch selected interface
    try:
        if args.mode == "both":
            run_both(args)
        else:
            mode_runners = {
                "cli": run_cli,
                "web": run_web,
                "telegram": run_telegram,
                "watch": run_watch,
            }
            runner = mode_runners[args.mode]
            runner(args)
    except KeyboardInterrupt:
        print("\nMRAgent shutting down. Goodbye! 👋")
        logger.info("MRAgent shutting down via Ctrl+C")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        # Use Poneglyph to analyze and suggest fixes
        poneglyph.analyze_error(e, context="main execution loop")
        sys.exit(1)


if __name__ == "__main__":
    main()
