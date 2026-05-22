"""
config.py — Toàn bộ tham số inject vào framework, không hard-code trong engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, List


# ─────────────────────────────────────────────
# INDICATOR REGISTRY — bật / tắt indicator
# ─────────────────────────────────────────────
INDICATOR_REGISTRY = {
    # Price
    "high": True, "low": True, "open": True, "close": True,
    "volume": True, "delta_price": True, "percent_price_tl": True,

    # Volume-based
    "vma": True, "vwma": True, "vo": True, "vwap": True,
    "upper_vwap": True, "lower_vwap": True, "cvd": True, "delta_cvd": True,

    # On-chart
    "ma": True, "ema": True,
    "ichimoku": True,          # tenkan, kijun, chikou, senkou_a, senkou_b
    "keltner": True,           # ema_kc, upAtr, downAtr
    "bollinger": True,         # upper, lower, basis
    "williams_fractal": True,
    "pivot_points": True,      # PP, R1-R5, S1-S5
    "aroon": True,
    "sar": True,
    "supertrend": True,

    # Oscillators
    "macd": True, "stoch": True, "rsi": True,
    "atr": True, "mfi": True, "stoch_fast": True,
    "rsi_sma": True, "atr_sma": True, "mfi_sma": True,
    "adx": True, "cci": True,
    "rsi_divergence": True, "atr_divergence": True, "mfi_divergence": True,
}

# Tham số mặc định cho từng indicator
INDICATOR_PARAMS = {
    "ma_period": 20,
    "ema_period": 20,
    "ema_fast": 12,
    "ema_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "atr_period": 14,
    "mfi_period": 14,
    "adx_period": 14,
    "cci_period": 20,
    "bb_period": 20,
    "bb_std": 2.0,
    "kc_period": 20,
    "kc_atr_mult": 1.5,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "aroon_period": 25,
    "sar_acceleration": 0.02,
    "sar_max_acceleration": 0.2,
    "supertrend_period": 10,
    "supertrend_mult": 3.0,
    "ichimoku_tenkan": 9,
    "ichimoku_kijun": 26,
    "ichimoku_senkou_b": 52,
    "vwap_band_mult": 1.0,
    "vma_period": 20,
    "vo_fast": 5,
    "vo_slow": 14,
}


# ─────────────────────────────────────────────
# SIGNAL GENERATOR CONFIG
# ─────────────────────────────────────────────
@dataclass
class SignalConfig:
    max_conditions: int = 4          # tối đa số điều kiện trong một signal
    max_indicators: int = 3          # tối đa số indicator khác nhau
    composition: Literal["AND", "OR", "MIXED"] = "AND"
    search_mode: Literal["random", "evolutionary"] = "random"
    population_size: int = 50        # cho evolutionary search
    generations: int = 20
    mutation_rate: float = 0.2
    crossover_rate: float = 0.5
    random_seed: int = 42


# ─────────────────────────────────────────────
# BACKTEST CONFIG
# ─────────────────────────────────────────────
@dataclass
class BacktestConfig:
    # Data
    warmup_candles: int = 50

    # Costs
    maker_fee: float = 0.0002        # 0.02%
    taker_fee: float = 0.0004        # 0.04%
    slippage_model: Literal["fixed", "pct", "atr"] = "pct"
    slippage_value: float = 0.0001   # 0.01%
    spread: float = 0.0

    # Capital
    initial_balance: float = 10_000.0
    leverage: float = 1.0
    margin_mode: Literal["spot", "futures", "margin"] = "futures"

    # Risk — plugins được truyền vào lúc runtime
    sizing_plugin: object = None     # SizingPlugin instance
    stop_plugin: object = None       # StopPlugin instance
    max_concurrent_trades: int = 3
    max_portfolio_exposure: float = 0.9   # 90% balance
    max_daily_loss: float = 0.05          # 5%
    max_drawdown: float = 0.20            # 20%

    # Entry
    entry_timing: Literal["next_open", "limit"] = "next_open"
    cooldown_candles: int = 0
    allow_pyramiding: bool = False
    position_mode: Literal["long_only", "short_only", "long_short"] = "long_short"


# ─────────────────────────────────────────────
# OPTIMIZER CONFIG
# ─────────────────────────────────────────────
@dataclass
class OptimizerConfig:
    n_strategies: int = 100
    top_k: int = 10
    fitness_weights: dict = field(default_factory=lambda: {
        "sharpe":         0.35,
        "total_return":   0.25,
        "max_drawdown":  -0.20,
        "profit_factor":  0.20,
    })
    overfitting_penalty: float = 0.1   # penalty nếu n_trades < min_trades
    min_trades: int = 20
    multi_objective: bool = False
    random_seed: int = 42


# ─────────────────────────────────────────────
# DATA CONFIG
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    start_str: str = "01/01/2023"
    end_str: str = "01/01/2024"
    source: Literal["binance", "csv"] = "binance"
    csv_path: Optional[str] = None
    cache_dir: str = "cache"
    use_cache: bool = True
