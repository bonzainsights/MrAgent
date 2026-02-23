"""
MRAgent — Data Cleanup Utility
Auto-cleans old generated images, uploads, and temporary files
to prevent disk bloat. Users should save important files elsewhere.

Created: 2026-02-23
"""

import time
from pathlib import Path

from config.settings import DATA_DIR, IMAGES_DIR
from utils.logger import get_logger

logger = get_logger("utils.cleanup")

# Default retention: 7 days for images/uploads, 30 days for logs
DEFAULT_RETENTION = {
    "images": 7,      # days
    "uploads": 3,     # days — short, users should save elsewhere
    "logs": 30,       # days
}


def cleanup_old_files(
    retention_days: dict = None,
    dry_run: bool = False,
) -> dict:
    """
    Remove files older than retention_days from data directories.

    Args:
        retention_days: Dict of {directory_name: days}. Uses DEFAULT_RETENTION if None.
        dry_run: If True, only report what would be deleted.

    Returns:
        Dict with summary: {"deleted": int, "freed_bytes": int, "errors": int}
    """
    retention = retention_days or DEFAULT_RETENTION
    now = time.time()

    stats = {"deleted": 0, "freed_bytes": 0, "errors": 0, "details": []}

    dirs_to_clean = {
        "images": IMAGES_DIR,
        "uploads": DATA_DIR / "uploads",
        "logs": DATA_DIR / "logs",
    }

    for dir_name, dir_path in dirs_to_clean.items():
        if not dir_path.exists():
            continue

        max_age_days = retention.get(dir_name, 7)
        max_age_seconds = max_age_days * 86400

        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue

            # Skip important files
            if file_path.name.startswith(".") or file_path.name == "mragent.log":
                continue

            file_age = now - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_size = file_path.stat().st_size
                if dry_run:
                    stats["details"].append(
                        f"[DRY RUN] Would delete: {file_path.name} "
                        f"({file_size / 1024:.1f}KB, {file_age / 86400:.0f}d old)"
                    )
                else:
                    try:
                        file_path.unlink()
                        stats["deleted"] += 1
                        stats["freed_bytes"] += file_size
                        logger.info(f"Cleaned up: {file_path.name} ({file_size / 1024:.1f}KB)")
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Failed to delete {file_path}: {e}")

    if stats["deleted"] > 0:
        freed_mb = stats["freed_bytes"] / (1024 * 1024)
        logger.info(f"Cleanup complete: {stats['deleted']} files removed, {freed_mb:.1f}MB freed")

    return stats


def run_startup_cleanup():
    """Run a quick cleanup on startup (non-blocking)."""
    import threading

    def _cleanup():
        try:
            stats = cleanup_old_files()
            if stats["deleted"] > 0:
                freed_mb = stats["freed_bytes"] / (1024 * 1024)
                logger.info(f"Startup cleanup: removed {stats['deleted']} old files ({freed_mb:.1f}MB)")
        except Exception as e:
            logger.debug(f"Startup cleanup error: {e}")

    threading.Thread(target=_cleanup, daemon=True).start()
