from __future__ import annotations

import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from orchestra.engine.cue import CueEngine
from orchestra.engine.cue_scheduler import CueScheduler
from orchestra.engine.cue_github import CueGitHubPoller

logger = logging.getLogger(__name__)

class OrchestraFileHandler(FileSystemEventHandler):
    def __init__(self, engine: CueEngine, watch_dir: str):
        self.engine = engine
        self.watch_dir = Path(watch_dir).resolve()

    def on_modified(self, event):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        try:
            rel_path = str(path.relative_to(self.watch_dir))
        except ValueError:
            rel_path = str(path)
            
        logger.info(f"File modified: {rel_path}")
        self.engine.trigger("file.changed", {"CUE_FILE_PATH": rel_path, "CUE_EVENT": "modified"})

    def on_created(self, event):
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        try:
            rel_path = str(path.relative_to(self.watch_dir))
        except ValueError:
            rel_path = str(path)
            
        logger.info(f"File created: {rel_path}")
        self.engine.trigger("file.changed", {"CUE_FILE_PATH": rel_path, "CUE_EVENT": "created"})

class CueWatcher:
    def __init__(self, cue_file: str = ".orchestra/cue.yaml", watch_dir: str = "."):
        self.engine = CueEngine(cue_file)
        self.watch_dir = watch_dir
        self.observer = Observer()
        self.handler = OrchestraFileHandler(self.engine, self.watch_dir)
        self.scheduler = CueScheduler(self.engine)
        self.github = CueGitHubPoller(self.engine)

    def start(self):
        # File watcher
        has_file_rules = any(r.event_type == "file.changed" for r in self.engine.rules)
        if has_file_rules:
            self.observer.schedule(self.handler, self.watch_dir, recursive=True)
            self.observer.start()
            logger.info(f"Started Cue file watcher on {self.watch_dir}")
        else:
            logger.info("No file.changed rules in cue.yaml. Watcher not started.")

        # Scheduler
        self.scheduler.start()
        
        # GitHub
        self.github.start()

    def stop(self):
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped Cue file watcher")
        self.scheduler.stop()
        self.github.stop()
