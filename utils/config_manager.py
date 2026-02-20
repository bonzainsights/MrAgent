"""
MRAgent — Config Manager
Utilities for safely updating configuration files (.env).
"""

import os
from pathlib import Path

from utils.logger import get_logger
from config.settings import _APP_DATA_DIR

logger = get_logger("utils.config")

def update_env_key(key: str, value: str, env_path: str = None) -> bool:
    """
    Update or add a key-value pair in the .env file.
    Preserves existing comments and structure.
    """
    if env_path is None:
        path = _APP_DATA_DIR / ".env"
    else:
        path = Path(env_path)
    
    if not path.exists():
        # Create new if doesn't exist
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{key}={value}\n")
            logger.info(f"Created {env_path} with {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to create {env_path}: {e}")
            return False

    # Read existing lines
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read {env_path}: {e}")
        return False

    key_found = False
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Check if line starts with key= (ignoring comments)
        if stripped.startswith(f"{key}=") and not stripped.startswith("#"):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)
    
    if not key_found:
        # Append to end if not found
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"{key}={value}\n")

    # Write back
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info(f"Updated {key} in {env_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write {env_path}: {e}")
        return False
