from __future__ import annotations

import logging
import subprocess
import threading
import time
from orchestra.engine.cue import CueEngine

logger = logging.getLogger(__name__)

class CueGitHubPoller:
    """Simple poller using git log to detect remote changes as a proxy for webhooks (since gh CLI missing)."""
    def __init__(self, engine: CueEngine, interval: int = 300):
        self.engine = engine
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        rules = [r for r in self.engine.rules if r.event_type.startswith("github.")]
        if not rules:
            logger.info("No github rules in cue.yaml. GitHub poller not started.")
            return

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Started GitHub poller (interval: {self.interval}s)")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self):
        last_commit = self._get_last_commit()
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            current_commit = self._get_last_commit()
            if current_commit and current_commit != last_commit:
                logger.info(f"Detected new GitHub commit: {current_commit}")
                self.engine.trigger("github.push", {"CUE_GH_COMMIT": current_commit})
                last_commit = current_commit

    def _get_last_commit(self) -> str | None:
        try:
            # Try to fetch and get last origin/main commit
            subprocess.run(["git", "fetch"], capture_output=True, timeout=10)
            res = subprocess.run(
                ["git", "rev-parse", "origin/main"],
                capture_output=True, text=True, timeout=5
            )
            return res.stdout.strip() if res.returncode == 0 else None
        except Exception:
            return None
