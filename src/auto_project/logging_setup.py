from __future__ import annotations
import logging
import os

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger(name: str = "auto_project") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(_LEVEL)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
