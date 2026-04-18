"""Logging configuration for the LLM pipeline."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Returns a logger that writes to both the console (INFO+) and a log file (DEBUG+).
    All pipeline modules should call this with their __name__.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — DEBUG and above (full detail)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
