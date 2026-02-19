"""
VivreCard — Proactive Scheduler for MrAgent
Named after the life paper from One Piece, this scheduler keeps the agent 'alive'
and performing periodic tasks.

Created: 2026-02-19
"""

import time
import json
import threading
import schedule
from pathlib import Path
from datetime import datetime, timezone
from croniter import croniter

from utils.logger import get_logger
from config.settings import DATA_DIR

logger = get_logger("agents.vivrecard")

JOBS_FILE = DATA_DIR / "vivrecard_jobs.json"


class VivreCard(threading.Thread):
    """
    Background scheduler thread that executes tasks based on cron expressions.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.jobs = []
        self._lock = threading.Lock()
        
        # Ensure jobs file exists
        if not JOBS_FILE.exists():
            self._create_default_jobs()

    def _create_default_jobs(self):
        """Create a default jobs file if none exists."""
        defaults = [
            {
                "id": "heartbeat",
                "schedule": "* * * * *",  # Every minute
                "type": "log",
                "payload": "💓 VivreCard heartbeat: I am alive and checking in!",
                "enabled": True
            }
        ]
        try:
            JOBS_FILE.write_text(json.dumps(defaults, indent=2))
        except Exception as e:
            logger.error(f"Failed to create default jobs file: {e}")

    def load_jobs(self):
        """Load jobs from the JSON configuration."""
        with self._lock:
            try:
                if JOBS_FILE.exists():
                    data = json.loads(JOBS_FILE.read_text())
                    self.jobs = [j for j in data if j.get("enabled", True)]
                    logger.info(f"Loaded {len(self.jobs)} VivreCard jobs")
                else:
                    self.jobs = []
            except Exception as e:
                logger.error(f"Failed to load VivreCard jobs: {e}")
                self.jobs = []

    def get_next_run(self, cron_expression: str) -> float:
        """Calculate the next run time timestamp for a cron expression."""
        try:
            # Use UTC to align with time.time()
            iter = croniter(cron_expression, datetime.now(timezone.utc))
            return iter.get_next(float)
        except Exception as e:
            logger.error(f"Invalid cron expression '{cron_expression}': {e}")
            return 0

    def execute_job(self, job: dict):
        """Execute a single job based on its type."""
        job_id = job.get("id", "unknown")
        job_type = job.get("type", "log")
        payload = job.get("payload", "")

        logger.debug(f"Executing job {job_id} ({job_type})")

        try:
            if job_type == "log":
                logger.info(f"[VivreCard] {payload}")
                # Also print to stdout for visibility in CLI
                print(f"\n[VivreCard] {payload}\n")

            elif job_type == "shell":
                import subprocess
                result = subprocess.run(payload, shell=True, capture_output=True, text=True)
                logger.info(f"[VivreCard] Shell output: {result.stdout.strip()}")

            # TODO: Add 'agent' type to trigger AgentCore tasks
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

    def run(self):
        """Main scheduler loop."""
        self.running = True
        logger.info("VivreCard Scheduler started")

        # Initial load and schedule calculation
        self.load_jobs()
        
        # Track next run times: {job_id: timestamp}
        next_runs = {}
        for job in self.jobs:
            next_runs[job["id"]] = self.get_next_run(job["schedule"])

        while self.running:
            try:
                now = time.time()
                
                # check for due jobs
                for job in self.jobs:
                    job_id = job["id"]
                    run_time = next_runs.get(job_id, 0)
                    
                    if run_time > 0 and now >= run_time:
                        # Execute in a separate thread to not block the scheduler
                        threading.Thread(target=self.execute_job, args=(job,)).start()
                        
                        # Schedule next run
                        next_runs[job_id] = self.get_next_run(job["schedule"])

                # Reload jobs periodically (naive implementation: check file mod time?)
                # For now, just sleep
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)  # Prevent tight loop on error

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("VivreCard Scheduler stopping...")
