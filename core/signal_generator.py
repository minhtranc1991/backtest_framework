"""
signal_generator.py — Tự động sinh và đánh giá signal trading.
Hỗ trợ random search và evolutionary search.
"""

from __future__ import annotations
import random
import logging
import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any, Tuple
from config.config import SignalConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# OPERATOR DEFINITIONS
# ─────────────────────────────────────────────
OPERATORS = ["gt", "lt", "gte", "lte", "crossover_up", "crossover_dn",
             "above_threshold", "below_threshold", "slope_positive", "slope_negative"]

# Indicators grouped by scale for valid comparisons
SCALE_GROUPS = {
    "price_scale": [
        "Close", "Open", "High", "Low", "ma", "ema", "ema_fast", "ema_slow",
        "tenkan", "kijun", "senkou_a", "senkou_b", "ema_kc", "upAtr", "downAtr",
        "upper", "lower", "bb_basis", "upper_band", "lower_band",
        "up_trend", "down_trend", "super_trend",
        "pp", "r1", "s1", "r2", "s2", "r3", "s3",
        "vwap", "upper_vwap", "lower_vwap", "sar",
    ],
    "oscillator_0_100": ["rsi", "mfi", "slowk", "slowd", "fastk", "fastd", "aroon_up", "aroon_down"],
    "macd_scale":       ["macd", "macd_signal", "macd_hist"],
    "adx_scale":        ["adx", "plus_di", "minus_di"],
    "cci_scale":        ["cci"],
    "volume_scale":     ["Volume", "vma", "vwma"],
    "pct_scale":        ["percent_price_tl", "vo"],
    "cvd_scale":        ["cvd", "delta_cvd"],
    "delta_scale":      ["delta_price"],
    "bool_scale":       ["wf_bull", "wf_bear"],
    "divergence":       ["rsi_divergence", "atr_divergence", "mfi_divergence"],
}

# Map: indicator name → scale group
_IND_TO_SCALE: Dict[str, str] = {}
for group, inds in SCALE_GROUPS.items():
    for ind in inds:
        _IND_TO_SCALE[ind] = group


# ─────────────────────────────────────────────
# CONDITION DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class Condition:
    """Một điều kiện đơn: left_ind OP right_ind_or_threshold."""
    left: str
    operator: str
    right: Any       # str (indicator name) hoặc float (threshold)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.left not in df.columns:
            return pd.Series(False, index=df.index)

        left_s = df[self.left]

        # right là indicator
        if isinstance(self.right, str):
            if self.right not in df.columns:
                return pd.Series(False, index=df.index)
            right_s = df[self.right]
        else:
            right_s = float(self.right)

        if self.operator == "gt":
            return left_s > right_s
        elif self.operator == "lt":
            return left_s < right_s
        elif self.operator == "gte":
            return left_s >= right_s
        elif self.operator == "lte":
            return left_s <= right_s
        elif self.operator == "crossover_up":
            if isinstance(right_s, float):
                return (left_s > right_s) & (left_s.shift(1) <= right_s)
            return (left_s > right_s) & (left_s.shift(1) <= right_s.shift(1))
        elif self.operator == "crossover_dn":
            if isinstance(right_s, float):
                return (left_s < right_s) & (left_s.shift(1) >= right_s)
            return (left_s < right_s) & (left_s.shift(1) >= right_s.shift(1))
        elif self.operator == "above_threshold":
            return left_s > right_s
        elif self.operator == "below_threshold":
            return left_s < right_s
        elif self.operator == "slope_positive":
            return left_s.diff() > 0
        elif self.operator == "slope_negative":
            return left_s.diff() < 0
        return pd.Series(False, index=df.index)

    def __repr__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class Strategy:
    """Một strategy hoàn chỉnh: danh sách điều kiện long/short."""
    long_conditions:  List[Condition] = field(default_factory=list)
    short_conditions: List[Condition] = field(default_factory=list)
    composition: Literal["AND", "OR"] = "AND"
    strategy_id: str = ""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về DataFrame với cột signal: 1=long, -1=short, 0=flat."""
        signals = pd.Series(0, index=df.index)

        if self.long_conditions:
            long_mask = self._combine([c.evaluate(df) for c in self.long_conditions])
            signals[long_mask] = 1

        if self.short_conditions:
            short_mask = self._combine([c.evaluate(df) for c in self.short_conditions])
            signals[short_mask] = -1

        # Không cho phép long và short cùng lúc → giữ long
        return signals

    def _combine(self, masks: List[pd.Series]) -> pd.Series:
        if not masks:
            return pd.Series(False)
        result = masks[0]
        for m in masks[1:]:
            if self.composition == "AND":
                result = result & m
            else:
                result = result | m
        return result.fillna(False)

    def __repr__(self) -> str:
        long_str  = f" {self.composition} ".join(str(c) for c in self.long_conditions)
        short_str = f" {self.composition} ".join(str(c) for c in self.short_conditions)
        return (f"Strategy[{self.strategy_id}]\n"
                f"  LONG:  {long_str or '—'}\n"
                f"  SHORT: {short_str or '—'}")


# ─────────────────────────────────────────────
# STRATEGY GENERATOR
# ─────────────────────────────────────────────
class StrategyGenerator:
    """Sinh strategy ngẫu nhiên hoặc evolutionary."""

    def __init__(self, config: SignalConfig, available_indicators: List[str]):
        self.cfg = config
        self.available = [i for i in available_indicators if i in _IND_TO_SCALE]
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        self._counter = 0

    # ── Tạo một điều kiện đơn ──────────────────
    def _random_condition(self) -> Optional[Condition]:
        if not self.available:
            return None
        left = random.choice(self.available)
        scale = _IND_TO_SCALE.get(left)

        # Chọn operator hợp lý theo loại indicator
        if scale in ("bool_scale", "divergence"):
            op = random.choice(["gt", "crossover_up"])
            return Condition(left, op, 0.5)

        if scale == "oscillator_0_100":
            op = random.choice(["gt", "lt", "crossover_up", "crossover_dn",
                                 "above_threshold", "below_threshold"])
            threshold = random.choice([20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80])
            return Condition(left, op, float(threshold))

        if scale == "macd_scale":
            # So sánh macd vs signal hoặc vs 0
            peers = [i for i in self.available if _IND_TO_SCALE.get(i) == scale and i != left]
            if peers and random.random() < 0.6:
                right = random.choice(peers)
            else:
                right = 0.0
            op = random.choice(["gt", "lt", "crossover_up", "crossover_dn"])
            return Condition(left, op, right)

        if scale == "price_scale":
            peers = [i for i in self.available if _IND_TO_SCALE.get(i) == scale and i != left]
            if peers:
                right = random.choice(peers)
                op = random.choice(["gt", "lt", "crossover_up", "crossover_dn", "gte", "lte"])
                return Condition(left, op, right)
            return None

        # Các trường hợp còn lại: slope
        op = random.choice(["slope_positive", "slope_negative", "gt", "lt"])
        return Condition(left, op, 0.0)

    def _random_conditions(self, n: int) -> List[Condition]:
        conds = []
        used_inds: set = set()
        attempts = 0
        while len(conds) < n and attempts < n * 5:
            c = self._random_condition()
            attempts += 1
            if c is None:
                continue
            # Không dùng lại quá nhiều indicator
            if c.left in used_inds and len(used_inds) < self.cfg.max_indicators:
                continue
            conds.append(c)
            used_inds.add(c.left)
        return conds

    def _new_id(self) -> str:
        self._counter += 1
        return f"S{self._counter:04d}"

    def random_strategy(self) -> Strategy:
        n_long  = random.randint(1, self.cfg.max_conditions)
        n_short = random.randint(1, self.cfg.max_conditions)
        comp = self.cfg.composition if self.cfg.composition != "MIXED" \
               else random.choice(["AND", "OR"])
        return Strategy(
            long_conditions=self._random_conditions(n_long),
            short_conditions=self._random_conditions(n_short),
            composition=comp,
            strategy_id=self._new_id(),
        )

    def generate_population(self, n: int) -> List[Strategy]:
        return [self.random_strategy() for _ in range(n)]

    # ── Evolutionary operators ─────────────────
    def mutate(self, strategy: Strategy) -> Strategy:
        s = copy.deepcopy(strategy)
        s.strategy_id = self._new_id()
        target = random.choice(["long", "short"])
        pool = s.long_conditions if target == "long" else s.short_conditions
        if not pool:
            return s

        action = random.choice(["replace", "add", "remove"])
        if action == "replace" and pool:
            idx = random.randint(0, len(pool) - 1)
            new_c = self._random_condition()
            if new_c:
                pool[idx] = new_c
        elif action == "add" and len(pool) < self.cfg.max_conditions:
            new_c = self._random_condition()
            if new_c:
                pool.append(new_c)
        elif action == "remove" and len(pool) > 1:
            pool.pop(random.randint(0, len(pool) - 1))

        return s

    def crossover(self, s1: Strategy, s2: Strategy) -> Tuple[Strategy, Strategy]:
        c1, c2 = copy.deepcopy(s1), copy.deepcopy(s2)
        c1.strategy_id = self._new_id()
        c2.strategy_id = self._new_id()
        if len(s1.long_conditions) > 1 and len(s2.long_conditions) > 1:
            point = random.randint(1, min(len(s1.long_conditions), len(s2.long_conditions)) - 1)
            c1.long_conditions = s1.long_conditions[:point] + s2.long_conditions[point:]
            c2.long_conditions = s2.long_conditions[:point] + s1.long_conditions[point:]
        return c1, c2


# ─────────────────────────────────────────────
# SIGNAL GENERATOR FACADE
# ─────────────────────────────────────────────
class SignalGenerator:
    """
    Nhận DataFrame chứa indicator, sinh tập strategy, trả về signals.
    """

    def __init__(self, config: SignalConfig):
        self.cfg = config

    def generate_signals_for(
        self, strategy: Strategy, df: pd.DataFrame
    ) -> pd.Series:
        return strategy.generate_signals(df)

    def build_generator(self, df: pd.DataFrame) -> StrategyGenerator:
        available = [c for c in df.columns if c in _IND_TO_SCALE]
        return StrategyGenerator(self.cfg, available)
