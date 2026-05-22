"""
utils/logger.py — Setup logging chuẩn cho toàn framework.
"""

import logging
import sys
import io
import pandas as pd
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

def log_dataframe_info(
    logger: logging.Logger,
    df: pd.DataFrame,
    name: str = "Data",
) -> None:
    """
    Log basic dataframe quality statistics.
    """

    if df.empty:
        logger.warning(f"{name}: DataFrame is empty")
        return

    logger.info(
        f"{name}: shape={df.shape} "
        f"range=[{df.index.min()} -> {df.index.max()}]"
    )

    missing = int(df.isna().sum().sum())
    duplicates = int(df.index.duplicated().sum())

    logger.info(
        f"{name}: missing={missing:,} "
        f"duplicate_index={duplicates:,}"
    )

    if {"open", "high", "low", "close"}.issubset(df.columns):
        bad_hl = int((df["high"] < df["low"]).sum())

        logger.info(
            f"{name}: "
            f"close=[{df['close'].min():.4f}, {df['close'].max():.4f}] "
            f"bad_high_low={bad_hl}"
        )

    if "volume" in df.columns:
        zero_vol = int((df["volume"] == 0).sum())

        logger.info(
            f"{name}: "
            f"volume_mean={df['volume'].mean():,.2f} "
            f"zero_volume={zero_vol:,}"
        )