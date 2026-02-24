"""
MRAgent — Terminal Tool
Executes shell commands via subprocess with timeout and safety checks.

Created: 2026-02-15
"""

import os
import subprocess

from tools.base import Tool


# Commands that are never allowed
BLACKLISTED_COMMANDS = [
    "rm -rf /", "rm -rf /*",
    "mkfs", "dd if=/dev/zero",
    ":(){:|:&};:",  # fork bomb
    "shutdown", "reboot", "halt",
]

# Maximum output length to avoid huge outputs clogging context
MAX_OUTPUT_LENGTH = 8000


class TerminalTool(Tool):
    """Execute shell commands and return output."""

    name = "execute_terminal"
    description = (
        "Execute a shell command on the user's system. "
        "Returns stdout and stderr. Use for running scripts, "
        "listing files, installing packages, git operations, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "working_directory": {
                "type": "string",
                "description": "Optional working directory for the command",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
            },
        },
        "required": ["command"],
    }

    def execute(self, command: str, working_directory: str = None,
                timeout: int = 30) -> str:
        """Execute a shell command."""
        # Safety check
        cmd_lower = command.lower().strip()
        for blocked in BLACKLISTED_COMMANDS:
            if blocked in cmd_lower:
                return f"⚠️ Blocked: command matches safety blacklist ({blocked})"

        cwd = working_directory or os.getcwd()

        # Validate working directory exists
        if not os.path.isdir(cwd):
            return f"❌ Working directory does not exist: {cwd}. Use list_files to verify paths before operating."

        self.logger.info(f"Executing: {command} (cwd={cwd}, timeout={timeout}s)")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ},
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n--- stderr ---\n"
                output += result.stderr

            # Truncate if too long
            if len(output) > MAX_OUTPUT_LENGTH:
                output = output[:MAX_OUTPUT_LENGTH] + f"\n... (truncated, {len(output)} total chars)"

            if result.returncode != 0:
                output = f"Exit code: {result.returncode}\n{output}"

            return output.strip() or "(no output)"

        except subprocess.TimeoutExpired:
            return f"⏰ Command timed out after {timeout}s: {command}"
        except FileNotFoundError:
            return f"❌ Working directory not found: {cwd}"
        except Exception as e:
            return f"❌ Error executing command: {e}"
