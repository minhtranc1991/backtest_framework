"""
utils/plotting.py — Visualize equity curve, drawdown, trade distribution.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional
from core.backtester import BacktestResult, TradeRecord


def plot_equity_curve(result: BacktestResult, title: str = "Equity Curve",
                      save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1]})

    # Equity
    ax1 = axes[0]
    ax1.plot(result.equity_curve.index, result.equity_curve.values,
             color="#2196F3", linewidth=1.5, label="Equity")
    ax1.fill_between(result.equity_curve.index, result.equity_curve.values,
                     result.equity_curve.iloc[0], alpha=0.1, color="#2196F3")
    ax1.set_ylabel("Equity (USD)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    dd = result.drawdown_series * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    # Trade PnL dots
    ax3 = axes[2]
    trades = result.trade_history
    if trades:
        times = [t.exit_time for t in trades]
        pnls  = [t.pnl for t in trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax3.scatter(times, pnls, c=colors, alpha=0.7, s=20)
        ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.set_ylabel("Trade PnL")
    ax3.grid(alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trade_distribution(trades: List[TradeRecord],
                            save_path: Optional[str] = None) -> None:
    pnls = [t.pnl for t in trades]
    if not pnls:
        print("No trades to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(pnls, bins=40, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("PnL Distribution")
    axes[0].set_xlabel("PnL (USD)")

    holds = [t.holding_period for t in trades]
    axes[1].hist(holds, bins=30, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[1].set_title("Holding Period Distribution")
    axes[1].set_xlabel("Candles")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_metrics_table(metrics: dict) -> None:
    print("\n" + "=" * 45)
    print(f"  {'PERFORMANCE METRICS':^41}")
    print("=" * 45)
    rows = [
        ("Total Return",     f"{metrics.get('total_return_pct', 0):.2f}%"),
        ("CAGR",             f"{metrics.get('cagr_pct', 0):.2f}%"),
        ("Sharpe Ratio",     f"{metrics.get('sharpe', 0):.3f}"),
        ("Sortino Ratio",    f"{metrics.get('sortino', 0):.3f}"),
        ("Max Drawdown",     f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("Calmar Ratio",     f"{metrics.get('calmar', 0):.3f}"),
        ("Win Rate",         f"{metrics.get('winrate', 0)*100:.1f}%"),
        ("Profit Factor",    f"{metrics.get('profit_factor', 0):.3f}"),
        ("Expectancy",       f"${metrics.get('expectancy', 0):.2f}"),
        ("# Trades",         str(metrics.get('n_trades', 0))),
        ("Avg Hold (candles)", f"{metrics.get('avg_holding_period', 0):.1f}"),
        ("Long / Short",     f"{metrics.get('long_trades',0)} / {metrics.get('short_trades',0)}"),
    ]
    for label, val in rows:
        print(f"  {label:<28} {val:>12}")
    print("=" * 45 + "\n")
