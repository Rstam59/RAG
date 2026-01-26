import json
import logging
import os
import time
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: Optional[str] = None) -> None:
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, lvl, logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    # avoid duplicate handlers in reloads/notebooks
    root.handlers.clear()
    root.addHandler(handler)


def log(logger: logging.Logger, msg: str, **fields: Any) -> None:
    logger.info(msg, extra={"extra": fields})
