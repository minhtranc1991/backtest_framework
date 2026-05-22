"""
data_loader.py — Tải, làm sạch và cache dữ liệu OHLCV từ Binance hoặc CSV.
"""

from __future__ import annotations
import os
import time
import logging
import hashlib
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def str_to_millis(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%d/%m/%Y")
    return int(dt.timestamp() * 1000)


def _cache_key(symbol: str, interval: str, start: str, end: str) -> str:
    raw = f"{symbol}_{interval}_{start}_{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────
# BINANCE FETCHER
# ─────────────────────────────────────────────
def fetch_binance_kline_df(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str,
    retry: int = 3,
) -> pd.DataFrame:
    """Tải dữ liệu OHLCV từ Binance REST API, hỗ trợ phân trang tự động."""
    BASE_URL = "https://api.binance.com/api/v3/klines"

    start_time = str_to_millis(start_str)
    end_dt = datetime.strptime(end_str, "%d/%m/%Y").replace(
        hour=23, minute=59, second=59
    )
    end_time = int(end_dt.timestamp() * 1000)

    all_klines: list = []
    logger.info(f"⏳ Fetching {symbol} [{interval}] {start_str} → {end_str}")

    while start_time < end_time:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000,
        }
        success = False
        for attempt in range(retry):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                if resp.status_code != 200:
                    logger.warning(f"HTTP {resp.status_code}: {resp.text}")
                    break
                data = resp.json()
                if not data:
                    return _build_df(all_klines)
                all_klines.extend(data)
                start_time = data[-1][0] + 1
                time.sleep(0.05)
                success = True
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(1)
        if not success:
            break

    return _build_df(all_klines)


def _build_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        "timestamp", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume",
             "taker_buy_base_volume", "taker_buy_quote_volume"]].astype(float)
    return df


# ─────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV từ file CSV.
    Cần cột: timestamp/date, Open, High, Low, Close, Volume.
    """
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().title() for c in df.columns]
    ts_col = next((c for c in df.columns if "Time" in c or "Date" in c), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df.set_index(ts_col, inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────
# DATA CLEANER
# ─────────────────────────────────────────────
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch DataFrame: xử lý NaN, duplicate, OHLC validity."""
    if df.empty:
        return df

    # 1. Loại bỏ duplicate index
    df = df[~df.index.duplicated(keep="first")]

    # 2. Sort theo thời gian
    df.sort_index(inplace=True)

    # 3. Fill forward NaN (tối đa 5 candle liên tiếp)
    df.ffill(limit=5, inplace=True)

    # 4. Drop nếu vẫn còn NaN ở OHLCV cốt lõi
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    # 5. Sanity check OHLC
    invalid = (
        (df["High"] < df["Low"]) |
        (df["Close"] < 0) |
        (df["Volume"] < 0)
    )
    n_invalid = invalid.sum()
    if n_invalid > 0:
        logger.warning(f"Dropping {n_invalid} invalid OHLCV rows.")
        df = df[~invalid]

    return df


# ─────────────────────────────────────────────
# CACHE MANAGER
# ─────────────────────────────────────────────
class DataCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> pd.DataFrame:
        logger.info(f"📦 Loading from cache: {key}")
        return pd.read_parquet(self._path(key))

    def save(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(key))
        logger.info(f"💾 Saved to cache: {key}")


# ─────────────────────────────────────────────
# MAIN DATA LOADER
# ─────────────────────────────────────────────
class DataLoader:
    """Facade duy nhất để lấy dữ liệu OHLCV đã làm sạch."""

    def __init__(
        self,
        source: str = "binance",
        cache_dir: str = "cache",
        use_cache: bool = True,
    ):
        self.source = source
        self.use_cache = use_cache
        self.cache = DataCache(cache_dir)

    def load(
        self,
        symbol: str,
        interval: str,
        start_str: str,
        end_str: str,
        csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        key = _cache_key(symbol, interval, start_str, end_str)

        if self.use_cache and self.cache.exists(key):
            return self.cache.load(key)

        if self.source == "binance":
            df = fetch_binance_kline_df(symbol, interval, start_str, end_str)
        elif self.source == "csv":
            if not csv_path:
                raise ValueError("csv_path required for source='csv'")
            df = load_csv(csv_path)
        else:
            raise ValueError(f"Unknown source: {self.source}")

        df = clean_ohlcv(df)

        if self.use_cache and not df.empty:
            self.cache.save(key, df)

        logger.info(f"✅ Loaded {len(df)} candles for {symbol} [{interval}]")
        return df
