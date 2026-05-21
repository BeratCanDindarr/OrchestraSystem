from __future__ import annotations

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from orchestra.engine.cue import CueEngine

logger = logging.getLogger(__name__)

class CueScheduler:
    def __init__(self, engine: CueEngine):
        self.engine = engine
        self.scheduler = BackgroundScheduler()

    def start(self):
        rules = [r for r in self.engine.rules if r.event_type == "time.scheduled"]
        if not rules:
            logger.info("No time.scheduled rules in cue.yaml. Scheduler not started.")
            return

        for rule in rules:
            cron = rule.params.get("cron")
            interval = rule.params.get("interval_minutes")
            
            if cron:
                try:
                    # cron format: "minute hour day month day_of_week"
                    self.scheduler.add_job(
                        self._trigger_rule,
                        'cron',
                        args=[rule],
                        **self._parse_cron(cron)
                    )
                    logger.info(f"Scheduled cron job for rule '{rule.name}': {cron}")
                except Exception as e:
                    logger.error(f"Failed to schedule cron job for rule '{rule.name}': {e}")
            elif interval:
                self.scheduler.add_job(
                    self._trigger_rule,
                    'interval',
                    minutes=int(interval),
                    args=[rule]
                )
                logger.info(f"Scheduled interval job for rule '{rule.name}': every {interval} mins")

        if self.scheduler.get_jobs():
            self.scheduler.start()
            logger.info("Started Cue background scheduler")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Stopped Cue background scheduler")

    def _trigger_rule(self, rule):
        logger.info(f"Triggering scheduled rule: {rule.name}")
        self.engine.trigger("time.scheduled", {"CUE_RULE_NAME": rule.name})

    def _parse_cron(self, cron_str: str) -> dict:
        parts = cron_str.split()
        mapping = ['minute', 'hour', 'day', 'month', 'day_of_week']
        return {mapping[i]: p for i, p in enumerate(parts) if i < len(mapping)}
