from __future__ import annotations

import queue
import threading
import logging
from typing import Callable, Any, Dict

logger = logging.getLogger(__name__)

class CommandQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def enqueue(self, command_name: str, payload: Dict[str, Any], callback: Callable = None):
        """Enqueue a command for the agent."""
        self._queue.put({"command": command_name, "payload": payload, "callback": callback})
        logger.info(f"Enqueued command: {command_name}")
        self._ensure_worker()

    def _ensure_worker(self):
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._stop_event.clear()
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
                cmd = item["command"]
                payload = item["payload"]
                cb = item["callback"]
                
                logger.info(f"Processing queued command: {cmd}")
                
                # Mock processing - this would route to actual handlers
                result = {"status": "success", "command": cmd, "processed": True}
                
                if cb:
                    try:
                        cb(result)
                    except Exception as e:
                        logger.error(f"Callback failed: {e}")
                
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing command queue: {e}")

    def stop(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
