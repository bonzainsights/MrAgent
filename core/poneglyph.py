"""
Poneglyph System — The Guardian and Doctor of MrAgent.

"The Poneglyphs are mysterious stone blocks that hold history... 
and the key to finding the One Piece."

This module provides:
1. The Guardian: Runtime error handling and resilience.
2. The Doctor: System diagnostics and health checks.
3. The Fixer: Auto-remediation for common issues.
"""

import os
import sys
import json
import logging
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

from config.settings import DATA_DIR, _APP_DATA_DIR

logger = logging.getLogger("core.poneglyph")

CONFIG_FILE = DATA_DIR / "mragent.json"

class Poneglyph:
    def __init__(self):
        self.config = self.load_config()
        self.system_health = "UNKNOWN"
        self.issues: List[Dict[str, Any]] = []

    def load_config(self) -> Dict[str, Any]:
        """Load the Poneglyph configuration (mragent.json)."""
        if not CONFIG_FILE.exists():
            logger.warning("Poneglyph config missing. Creating default.")
            # Create default config if missing (basic recovery)
            default_config = {
                "system": {"version": "1.0.0", "safe_mode": True, "auto_heal": True, "llm_diagnostics": True},
                "modules": {"core": {"critical": True}, "vivrecard": {"enabled": True}},
                "recovery": {"max_retries": 3}
            }
            try:
                CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
                CONFIG_FILE.write_text(json.dumps(default_config, indent=2))
                return default_config
            except Exception as e:
                logger.error(f"Failed to create default config: {e}")
                return {}
        
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception as e:
            logger.error(f"Failed to read Poneglyph config: {e}")
            return {}

    def check_health(self) -> bool:
        """Run all diagnostic checks and return overall health status."""
        self.issues = []
        logger.info("Reading the Poneglyph... (Running Diagnostics)")

        checks = [
            self._check_environment,
            self._check_dependencies,
            self._check_connectivity
        ]

        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                logger.error(f"Diagnostic check failed: {e}")
                self.issues.append({"type": "internal_error", "details": str(e), "severity": "high"})
                all_passed = False

        self.system_health = "HEALTHY" if all_passed else "UNHEALTHY"
        return all_passed

    def _check_environment(self) -> bool:
        """Verify environment variables."""
        # Check for .env file
        env_path = _APP_DATA_DIR / ".env"
        if not env_path.exists():
            self.issues.append({
                "type": "config",
                "component": ".env",
                "details": "Missing .env file",
                "severity": "high",
                "fix_action": "create_env_template"
            })
            return False
        
        # Check for critical keys (simplified checks for now)
        # In a real scenario, we'd check loaded os.environ values
        return True

    def _check_dependencies(self) -> bool:
        """Verify installed packages match requirements.txt."""
        req_file = Path("requirements.txt")
        if not req_file.exists():
            self.issues.append({"type": "config", "details": "Missing requirements.txt", "severity": "medium"})
            return False

        # Basic check: try to import critical modules
        missing = []
        critical_modules = ["requests", "schedule", "croniter"] # Add more as needed
        for mod in critical_modules:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)

        if missing:
            self.issues.append({
                "type": "dependency",
                "details": f"Missing critical Python packages: {', '.join(missing)}",
                "severity": "high",
                "fix_action": "install_dependencies"
            })
            return False
        return True
    
    def _check_connectivity(self) -> bool:
        """Check connection to external services (LLM, Internet)."""
        import socket
        try:
            # Simple internet check
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            self.issues.append({
                "type": "connectivity",
                "details": "No internet connection available",
                "severity": "medium"
            })
            return False

    def report(self):
        """Print the diagnostic report to the console."""
        print("\n📜 --- Poneglyph Diagnostic Report --- 📜\n")
        print(f"System Status: {self.system_health}")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print("-" * 40)
        
        if not self.issues:
             print("✅ All systems functioning within normal parameters.")
        else:
            for issue in self.issues:
                severity = issue.get("severity", "medium").upper()
                icon = "🛑" if severity == "HIGH" else "⚠️"
                print(f"{icon} [{severity}] {issue.get('type')}: {issue.get('details')}")
                if "fix_action" in issue:
                    print(f"   ↳ Fix available: Run 'python main.py doctor --fix'")
        print("-" * 40 + "\n")

    def analyze_error(self, e: Exception, context: str = ""):
        """
        The Guardian: Analyze a runtime exception and decide on action.
        """
        logger.error(f"🛑 [Guardian] Exception detected in {context}: {e}")
        
        # 1. Capture Traceback
        tb = traceback.format_exc()
        
        # 2. Check config for LLM analysis
        if self.config.get("system", {}).get("llm_diagnostics", False):
            self._consult_llm(e, tb, context)
        else:
            print(f"\n⚠️  [Guardian] Exception: {e}")
            print(f"Run 'python main.py doctor' to check for system issues.\n")

    def _consult_llm(self, e: Exception, traceback_str: str, context: str):
        """
        Send error details to LLM for analysis.
        """
        print(f"\n🧠 [Poneglyph] Consulting the Ancient Texts (LLM)...")
        
        try:
            from providers.nvidia_llm import NvidiaLLMProvider
            
            # Use a lightweight reasoning model if available, or default
            llm = NvidiaLLMProvider()
            
            previous_context = "" 
            # TODO: Add recent log context if available
            
            messages = [
                {"role": "system", "content": (
                    "You are the Poneglyph Guardian, a system doctor for an AI Agent. "
                    "Analyze the following Python exception and traceback. "
                    "Provide a concise diagnosis and concrete fix instructions. "
                    "Be Technical but clear. Focus on 'Why it broke' and 'How to fix it'."
                )},
                {"role": "user", "content": f"Context: {context}\n\nError: {str(e)}\n\nTraceback:\n{traceback_str}"}
            ]
            
            response = llm.chat(messages, model="kimi-k2.5", stream=False)
            analysis = response["content"]
            
            print(f"\n📜 [Poneglyph Analysis]")
            print(f"{analysis}\n")
            
        except Exception as llm_error:
             logger.error(f"Failed to consult LLM: {llm_error}")
             print(f"⚠️  [Poneglyph] LLM consultation failed. Fallback error info:\n{e}")

    def run_fixer(self):
        """Attempt to auto-fix identified issues."""
        print("\n🔧 --- Poneglyph Fixer --- 🔧\n")
        
        # Re-run diagnostics to get current issues
        self.check_health()
        
        if not self.issues:
            print("✅ No issues found to fix.")
            return

        for issue in self.issues:
            action = issue.get("fix_action")
            if action == "install_dependencies":
                print(f"🛠️  Fixing dependencies...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                    print("✅ Dependencies installed.")
                except subprocess.CalledProcessError:
                    print("❌ Failed to install dependencies.")
            
            elif action == "create_env_template":
                 print("🛠️  Creating .env from example...")
                 try:
                     env_path = _APP_DATA_DIR / ".env"
                     if Path(".env.example").exists():
                         env_path.write_text(Path(".env.example").read_text())
                         print("✅ .env created. Please edit it with your keys.")
                     else:
                         print("❌ .env.example missing.")
                 except Exception as e:
                     print(f"❌ Failed to create .env: {e}")
            
            else:
                print(f"⚠️  No auto-fix available for: {issue.get('details')}")

        print("\n✨ Fix process complete. Run 'python main.py doctor' to verify.")

