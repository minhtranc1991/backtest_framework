"""
evaluation.py — Tính toán metrics hiệu suất và kiểm tra độ ổn định chiến lược.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from core.backtester import BacktestResult, TradeRecord

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════
class PerformanceEvaluator:
    """Tính đầy đủ các metrics từ BacktestResult."""

    @staticmethod
    def evaluate(result: BacktestResult,
                 candles_per_year: int = 8760) -> Dict:
        """
        candles_per_year: 8760 cho 1h, 365 cho 1d, 52560 cho 1m, ...
        """
        equity  = result.equity_curve
        trades  = result.trade_history
        dd      = result.drawdown_series

        if equity.empty or len(equity) < 2:
            return {}

        pnls   = [t.pnl for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Basic
        initial = equity.iloc[0]
        final   = equity.iloc[-1]
        total_return_pct = (final / initial - 1) * 100

        # CAGR (prevent overflow by clamping ratio)
        n_years = len(equity) / candles_per_year
        if initial > 0 and np.isfinite(final):
            ratio = final / initial
            # Clamp ratio to prevent overflow in power operation
            ratio = np.clip(ratio, 1e-10, 1e10)
            cagr = (ratio ** (1 / max(n_years, 1e-6)) - 1) * 100
        else:
            cagr = 0

        # Risk metrics
        rets = equity.pct_change(fill_method=None).dropna()
        ann  = np.sqrt(candles_per_year)
        sharpe  = float(rets.mean() / (rets.std() + 1e-10) * ann)
        neg_std = rets[rets < 0].std()
        sortino = float(rets.mean() / (neg_std + 1e-10) * ann)
        max_dd  = float(dd.min() * 100)
        calmar  = total_return_pct / (abs(max_dd) + 1e-8)

        # Trade metrics
        n_trades = len(trades)
        winrate  = len(wins) / (n_trades + 1e-10)
        avg_win  = np.mean(wins)  if wins   else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / (abs(sum(losses)) + 1e-10) if losses else np.inf
        expectancy    = np.mean(pnls) if pnls else 0

        # RR
        avg_rr = abs(avg_win / (avg_loss + 1e-10))

        # Holding
        avg_hold = np.mean([t.holding_period for t in trades]) if trades else 0

        # Long/short ratio
        n_long  = sum(1 for t in trades if t.side == "long")
        n_short = sum(1 for t in trades if t.side == "short")

        # Exit breakdown
        exit_counts = {}
        for t in trades:
            k = t.exit_reason.value
            exit_counts[k] = exit_counts.get(k, 0) + 1

        return {
            "total_return_pct":   round(total_return_pct, 2),
            "cagr_pct":           round(cagr, 2),
            "sharpe":             round(sharpe, 3),
            "sortino":            round(sortino, 3),
            "max_drawdown_pct":   round(max_dd, 2),
            "calmar":             round(calmar, 3),
            "winrate":            round(winrate, 3),
            "profit_factor":      round(profit_factor, 3),
            "expectancy":         round(expectancy, 2),
            "n_trades":           n_trades,
            "avg_holding_period": round(avg_hold, 1),
            "long_trades":        n_long,
            "short_trades":       n_short,
            "avg_rr":             round(avg_rr, 3),
            "avg_win":            round(avg_win, 2),
            "avg_loss":           round(avg_loss, 2),
            "exit_breakdown":     exit_counts,
        }


# ═══════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════
class WalkForwardValidator:
    """
    Chia dữ liệu thành nhiều fold: in-sample và out-of-sample.
    Mỗi fold: train trên IS, test trên OOS, lấy metric OOS.
    """

    def __init__(self, n_splits: int = 5, oos_ratio: float = 0.2):
        self.n_splits  = n_splits
        self.oos_ratio = oos_ratio

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Trả về danh sách (is_df, oos_df)."""
        n = len(df)
        fold_size = n // self.n_splits
        splits = []
        for i in range(self.n_splits):
            end_is  = (i + 1) * fold_size
            end_oos = min(end_is + int(fold_size * self.oos_ratio), n)
            if end_is >= n:
                break
            is_df  = df.iloc[:end_is]
            oos_df = df.iloc[end_is:end_oos]
            if len(oos_df) > 10:
                splits.append((is_df, oos_df))
        return splits

    def run(
        self,
        df: pd.DataFrame,
        strategy,
        engine_factory,
        indicator_builder,
        signal_generator,
    ) -> Dict:
        """
        engine_factory: callable() → ExecutionEngine (fresh instance)
        """
        folds = self.split(df)
        oos_metrics: List[Dict] = []

        for fold_idx, (is_df, oos_df) in enumerate(folds):
            logger.info(f"Walk-forward fold {fold_idx+1}/{len(folds)}")
            try:
                # Build indicators on OOS data
                oos_feat = indicator_builder.build(oos_df)
                # Generate signals
                signals = signal_generator.generate_signals_for(strategy, oos_feat)
                # Run backtest
                engine  = engine_factory()
                result  = engine.run(oos_feat, signals)
                metrics = PerformanceEvaluator.evaluate(result)
                metrics["fold"] = fold_idx + 1
                oos_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Fold {fold_idx+1} failed: {e}")

        if not oos_metrics:
            return {}

        summary = {
            "n_folds":          len(oos_metrics),
            "avg_sharpe":       np.mean([m.get("sharpe", 0) for m in oos_metrics]),
            "avg_return":       np.mean([m.get("total_return_pct", 0) for m in oos_metrics]),
            "avg_drawdown":     np.mean([m.get("max_drawdown_pct", 0) for m in oos_metrics]),
            "avg_winrate":      np.mean([m.get("winrate", 0) for m in oos_metrics]),
            "stability_score":  self._stability(oos_metrics),
            "fold_results":     oos_metrics,
        }
        return summary

    def _stability(self, metrics: List[Dict]) -> float:
        """Tỷ lệ fold có return dương."""
        positives = sum(1 for m in metrics if m.get("total_return_pct", -1) > 0)
        return round(positives / len(metrics), 3)


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS TEST
# ═══════════════════════════════════════════════════════════════════
class RobustnessTest:
    """
    Test độ ổn định bằng cách pertub dữ liệu nhẹ và so sánh kết quả.
    """

    def __init__(self, n_trials: int = 10, noise_pct: float = 0.001,
                 random_seed: int = 42):
        self.n_trials   = n_trials
        self.noise_pct  = noise_pct
        self.rng        = np.random.default_rng(random_seed)

    def run(self, df: pd.DataFrame, strategy, engine_factory,
            indicator_builder, signal_generator) -> Dict:
        results = []
        for trial in range(self.n_trials):
            noisy_df = self._add_noise(df)
            try:
                feat    = indicator_builder.build(noisy_df)
                signals = signal_generator.generate_signals_for(strategy, feat)
                engine  = engine_factory()
                result  = engine.run(feat, signals)
                m = PerformanceEvaluator.evaluate(result)
                results.append(m)
            except Exception as e:
                logger.warning(f"Robustness trial {trial} failed: {e}")

        if not results:
            return {}

        returns = [r.get("total_return_pct", 0) for r in results]
        sharpes = [r.get("sharpe", 0) for r in results]
        return {
            "n_trials":       self.n_trials,
            "return_mean":    round(np.mean(returns), 2),
            "return_std":     round(np.std(returns), 2),
            "sharpe_mean":    round(np.mean(sharpes), 3),
            "sharpe_std":     round(np.std(sharpes), 3),
            "consistency":    round(np.mean(returns) / (np.std(returns) + 1e-8), 3),
        }

    def _add_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        noisy = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col in noisy.columns:
                noise = self.rng.normal(0, self.noise_pct, len(noisy))
                noisy[col] = noisy[col] * (1 + noise)
        return noisy
