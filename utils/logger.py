"""Logging utilities.

Usage:
    from utils.logger import setup_logger
    logger = setup_logger("train", log_file=run_dir / "train.log")
    logger.info("Starting epoch 1")
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "fish", log_file: Path = None) -> logging.Logger:
    """
    Create (or retrieve) a named logger writing to stdout and, optionally, to a file.

    Calling this function multiple times with the same name is safe — handlers are
    not duplicated if the logger was already configured.

    Args:
        name:     Logger name shown in each log line.
        log_file: If provided, a FileHandler is added at this path.
                  Parent directories are created automatically.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured; avoid duplicate handlers

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
