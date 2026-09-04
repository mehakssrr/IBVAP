import logging
from datetime import datetime

class EventLogger:
    def __init__(self, log_file="events.log"):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger("IBVAP")

    def log_event(self, event_type: str, details: dict):
        msg = f"{event_type} | " + " | ".join(f"{k}={v}" for k, v in details.items())
        self.logger.info(msg)
