"""
utils/logger.py — Setup logging chuẩn cho toàn framework.
"""

import logging
import sys
import io
from pathlib import Path


def setup_logger(
    name: str = "backtest",
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not logger.handlers:
        # Create UTF-8 wrapper for stdout
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        ch = logging.StreamHandler(utf8_stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
