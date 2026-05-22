"""
backtester.py — Backtesting engine hoàn chỉnh, tách biệt khỏi signal module.
Kiến trúc: ExecutionEngine / OrderManager / PositionManager /
           CapitalManager / RiskManager / StopManager / PortfolioManager / TradeLogger
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, Dict

from config.config import BacktestConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════
class ExitReason(Enum):
    LIQUIDATION    = "liquidation"
    FORCED_EXIT    = "forced_exit"
    STOP_LOSS      = "stop_loss"
    TAKE_PROFIT    = "take_profit"
    TRAILING_STOP  = "trailing_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    TIME_EXIT      = "time_exit"


@dataclass
class EntrySignal:
    timestamp: datetime
    side: Literal["long", "short"]
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = None
    expiry_candles: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class PendingOrder:
    side: str
    limit_price: float
    quantity: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: int
    expiry_candles: Optional[int]


@dataclass
class Position:
    pos_id: int
    side: Literal["long", "short"]
    entry_price: float
    quantity: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_time: datetime
    entry_candle: int
    max_holding_period: Optional[int]
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trailing_stop: Optional[float] = None


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    fee: float
    slippage: float
    pnl: float
    pnl_pct: float
    holding_period: int
    exit_reason: ExitReason


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trade_history: List[TradeRecord]
    performance_metrics: Dict


# ═══════════════════════════════════════════════════════════════════
# PLUGIN INTERFACES
# ═══════════════════════════════════════════════════════════════════
class SizingPlugin(ABC):
    @abstractmethod
    def compute_size(self, balance: float, entry: float,
                     stop: Optional[float], data: dict) -> float: ...


class StopPlugin(ABC):
    @abstractmethod
    def compute_sl(self, entry: float, side: str, data: dict) -> float: ...
    @abstractmethod
    def compute_tp(self, entry: float, sl: float, side: str) -> float: ...
    def update_trailing(self, current_stop: float, close: float, data: dict) -> float:
        return current_stop


# ── Built-in Sizing Plugins ──────────────────
class FixedFractionSizing(SizingPlugin):
    def __init__(self, fraction: float = 0.1):
        self.fraction = fraction

    def compute_size(self, balance, entry, stop, data):
        return (balance * self.fraction) / entry


class RiskBasedSizing(SizingPlugin):
    def __init__(self, risk_pct: float = 0.01):
        self.risk_pct = risk_pct

    def compute_size(self, balance, entry, stop, data):
        if stop is None or abs(entry - stop) < 1e-8:
            return (balance * 0.01) / entry
        return (balance * self.risk_pct) / abs(entry - stop)


class VolatilityAdjustedSizing(SizingPlugin):
    def __init__(self, target_vol: float = 0.01):
        self.target_vol = target_vol

    def compute_size(self, balance, entry, stop, data):
        atr = data.get("atr", entry * 0.01)
        if atr < 1e-8:
            atr = entry * 0.01
        return (balance * self.target_vol) / atr


# ── Built-in Stop Plugins ───────────────────
class FixedPctStop(StopPlugin):
    def __init__(self, sl_pct: float = 0.02, rr: float = 2.0):
        self.sl_pct = sl_pct
        self.rr = rr

    def compute_sl(self, entry, side, data):
        return entry * (1 - self.sl_pct) if side == "long" else entry * (1 + self.sl_pct)

    def compute_tp(self, entry, sl, side):
        risk = abs(entry - sl)
        return entry + self.rr * risk if side == "long" else entry - self.rr * risk


class ATRStop(StopPlugin):
    def __init__(self, atr_mult: float = 2.0, rr: float = 2.0):
        self.atr_mult = atr_mult
        self.rr = rr

    def compute_sl(self, entry, side, data):
        atr = data.get("atr", entry * 0.01)
        return entry - self.atr_mult * atr if side == "long" else entry + self.atr_mult * atr

    def compute_tp(self, entry, sl, side):
        risk = abs(entry - sl)
        return entry + self.rr * risk if side == "long" else entry - self.rr * risk

    def update_trailing(self, current_stop, close, data):
        atr = data.get("atr", close * 0.01)
        new_stop = close - self.atr_mult * atr
        return max(current_stop, new_stop)


# ═══════════════════════════════════════════════════════════════════
# SUB-MANAGERS
# ═══════════════════════════════════════════════════════════════════
class CapitalManager:
    def __init__(self, initial_balance: float, leverage: float, margin_mode: str):
        self.balance        = initial_balance
        self.initial        = initial_balance
        self.leverage       = leverage
        self.margin_mode    = margin_mode
        self.used_margin    = 0.0
        self._daily_start   = initial_balance
        self._last_day: Optional[datetime] = None

    @property
    def available_margin(self):
        return self.balance - self.used_margin

    def check_liquidation(self, pos: Position, low: float, high: float) -> bool:
        if self.margin_mode == "spot":
            return False
        liq_price = (pos.entry_price * pos.quantity * (1 / self.leverage) * 0.9) / pos.quantity
        if pos.side == "long":
            return low <= liq_price
        return high >= liq_price

    def reset_daily(self, timestamp: datetime) -> None:
        day = timestamp.date()
        if self._last_day != day:
            self._daily_start = self.balance
            self._last_day = day

    def daily_loss_exceeded(self, max_daily_loss: float) -> bool:
        return (self._daily_start - self.balance) / (self._daily_start + 1e-8) > max_daily_loss

    def max_drawdown_exceeded(self, max_dd: float) -> bool:
        return (self.initial - self.balance) / (self.initial + 1e-8) > max_dd


class OrderManager:
    def __init__(self, slippage_model: str, slippage_value: float,
                 maker_fee: float, taker_fee: float, spread: float):
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.maker_fee  = maker_fee
        self.taker_fee  = taker_fee
        self.spread     = spread
        self.pending: List[PendingOrder] = []

    def calc_slippage(self, price: float, data: dict) -> float:
        if self.slippage_model == "fixed":
            return self.slippage_value
        elif self.slippage_model == "pct":
            return price * self.slippage_value
        elif self.slippage_model == "atr":
            atr = data.get("atr", price * 0.001)
            return self.slippage_value * atr
        return 0.0

    def calc_fee(self, fill_price: float, qty: float, order_type: str = "market") -> float:
        rate = self.taker_fee if order_type == "market" else self.maker_fee
        return fill_price * qty * rate

    def fill_market(self, side: str, open_price: float, qty: float, data: dict) -> Tuple[float, float, float]:
        slip = self.calc_slippage(open_price, data)
        fill = open_price + slip if side == "long" else open_price - slip
        fee  = self.calc_fee(fill, qty, "market")
        return fill, slip, fee

    def check_pending_fills(self, candle_idx: int, low: float, high: float,
                            data: dict) -> List[Tuple[PendingOrder, float, float, float]]:
        """Kiểm tra và fill pending limit orders."""
        filled, remaining = [], []
        for order in self.pending:
            expired = (order.expiry_candles is not None and
                       candle_idx - order.created_at >= order.expiry_candles)
            if expired:
                continue
            if order.side == "long"  and low  <= order.limit_price:
                slip = self.calc_slippage(order.limit_price, data)
                fill = order.limit_price + slip
                fee  = self.calc_fee(fill, order.quantity, "limit")
                filled.append((order, fill, slip, fee))
            elif order.side == "short" and high >= order.limit_price:
                slip = self.calc_slippage(order.limit_price, data)
                fill = order.limit_price - slip
                fee  = self.calc_fee(fill, order.quantity, "limit")
                filled.append((order, fill, slip, fee))
            else:
                remaining.append(order)
        self.pending = remaining
        return filled


class PositionManager:
    def __init__(self):
        self._positions: List[Position] = []
        self._next_id = 1

    def open(self, pos: Position) -> None:
        pos.pos_id = self._next_id
        self._next_id += 1
        self._positions.append(pos)

    def close(self, pos: Position) -> None:
        self._positions = [p for p in self._positions if p.pos_id != pos.pos_id]

    def all(self) -> List[Position]:
        return list(self._positions)

    def update_unrealized(self, price: float) -> None:
        for p in self._positions:
            if p.side == "long":
                p.unrealized_pnl = (price - p.entry_price) * p.quantity * p.leverage
            else:
                p.unrealized_pnl = (p.entry_price - price) * p.quantity * p.leverage

    @property
    def count(self) -> int:
        return len(self._positions)

    def sides(self) -> List[str]:
        return [p.side for p in self._positions]


class TradeLogger:
    def __init__(self):
        self._records: List[TradeRecord] = []

    def log(self, record: TradeRecord) -> None:
        self._records.append(record)

    def records(self) -> List[TradeRecord]:
        return list(self._records)


# ═══════════════════════════════════════════════════════════════════
# EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════
from typing import Tuple   # moved here to avoid forward ref

class ExecutionEngine:
    """
    Candle loop chính. Hoàn toàn tách biệt khỏi signal module.
    Nhận signal dưới dạng pd.Series (index=timestamp, values: 1=long, -1=short, 0=flat).
    """

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        sizing_plugin = config.sizing_plugin or FixedFractionSizing(0.1)
        stop_plugin   = config.stop_plugin   or FixedPctStop(0.02, 2.0)

        self.capital   = CapitalManager(config.initial_balance, config.leverage, config.margin_mode)
        self.orders    = OrderManager(config.slippage_model, config.slippage_value,
                                      config.maker_fee, config.taker_fee, config.spread)
        self.positions = PositionManager()
        self.logger_t  = TradeLogger()
        self.sizing    = sizing_plugin
        self.stops     = stop_plugin

        self._last_trade_candle = -999
        self._equity_curve: List[float] = []

    # ── Main Run ──────────────────────────────
    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """
        df:      OHLCV + indicator DataFrame
        signals: pd.Series aligned with df.index (1=long, -1=short, 0=flat)
        """
        n = len(df)
        warmup = self.cfg.warmup_candles
        equity_list: List[float] = []

        for i in range(n):
            row       = df.iloc[i]
            ts        = df.index[i]
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            data      = row.to_dict()

            # [0] Skip warmup
            if i < warmup:
                equity_list.append(self.capital.balance)
                continue

            # [1] Update open positions (liquidation → SL/TP → time exit → trailing)
            self._update_positions(i, ts, o, h, l, c, data, signals)

            # [2] Check pending limit orders filled by this candle
            fills = self.orders.check_pending_fills(i, l, h, data)
            for (order, fill_price, slip, fee) in fills:
                self._open_position(order.side, fill_price, order.quantity,
                                    slip, fee, ts, i,
                                    order.stop_loss, order.take_profit, data)

            # [3] Get signal from PREVIOUS candle (no lookahead)
            if i > warmup:
                prev_signal = signals.iloc[i - 1] if i - 1 < len(signals) else 0
            else:
                prev_signal = 0

            # [4] Check entry conditions
            self.capital.reset_daily(ts)
            can_enter = self._can_enter(i)

            # [5] Execute entry
            if can_enter and prev_signal != 0:
                side = "long" if prev_signal == 1 else "short"
                if self.cfg.position_mode == "long_only"  and side == "short": pass
                elif self.cfg.position_mode == "short_only" and side == "long":  pass
                else:
                    self._execute_entry(side, o, i, ts, data, signals)

            # [6] Update equity
            total_unrealized = sum(p.unrealized_pnl for p in self.positions.all())
            self.positions.update_unrealized(c)
            equity_list.append(self.capital.balance + total_unrealized)

        return self._build_result(df.index, equity_list)

    # ── Position lifecycle ─────────────────────
    def _update_positions(self, i, ts, o, h, l, c, data, signals):
        for pos in list(self.positions.all()):
            # Liquidation (highest priority)
            if self.capital.check_liquidation(pos, l, h):
                exit_price = l if pos.side == "long" else h
                self._close_position(pos, exit_price, ts, i, data, ExitReason.LIQUIDATION)
                continue

            # Update trailing stop
            if pos.trailing_stop is not None:
                pos.trailing_stop = self.stops.update_trailing(pos.trailing_stop, c, data)

            # Stop loss
            if pos.stop_loss is not None:
                triggered = (pos.side == "long"  and l <= pos.stop_loss) or \
                            (pos.side == "short" and h >= pos.stop_loss)
                if triggered:
                    exit_price = pos.stop_loss
                    self._close_position(pos, exit_price, ts, i, data, ExitReason.STOP_LOSS)
                    continue

            # Trailing stop
            if pos.trailing_stop is not None:
                triggered = (pos.side == "long"  and l <= pos.trailing_stop) or \
                            (pos.side == "short" and h >= pos.trailing_stop)
                if triggered:
                    exit_price = pos.trailing_stop
                    self._close_position(pos, exit_price, ts, i, data, ExitReason.TRAILING_STOP)
                    continue

            # Take profit
            if pos.take_profit is not None:
                triggered = (pos.side == "long"  and h >= pos.take_profit) or \
                            (pos.side == "short" and l <= pos.take_profit)
                if triggered:
                    exit_price = pos.take_profit
                    self._close_position(pos, exit_price, ts, i, data, ExitReason.TAKE_PROFIT)
                    continue

            # Signal reversal
            if i > 0 and i < len(signals):
                cur_sig = signals.iloc[i - 1]
                if (pos.side == "long" and cur_sig == -1) or \
                   (pos.side == "short" and cur_sig == 1):
                    self._close_position(pos, o, ts, i, data, ExitReason.SIGNAL_REVERSAL)
                    continue

            # Time exit
            if pos.max_holding_period and (i - pos.entry_candle) >= pos.max_holding_period:
                self._close_position(pos, c, ts, i, data, ExitReason.TIME_EXIT)

    def _can_enter(self, i: int) -> bool:
        if self.capital.daily_loss_exceeded(self.cfg.max_daily_loss):
            return False
        if self.capital.max_drawdown_exceeded(self.cfg.max_drawdown):
            return False
        if i - self._last_trade_candle < self.cfg.cooldown_candles:
            return False
        if self.positions.count >= self.cfg.max_concurrent_trades:
            return False
        exposure = self.capital.used_margin / (self.capital.balance + 1e-8)
        if exposure >= self.cfg.max_portfolio_exposure:
            return False
        return True

    def _execute_entry(self, side, open_price, i, ts, data, signals):
        if not self.cfg.allow_pyramiding and side in self.positions.sides():
            return

        sl = self.stops.compute_sl(open_price, side, data)
        tp = self.stops.compute_tp(open_price, sl, side)
        qty = self.sizing.compute_size(self.capital.balance, open_price, sl, data)
        qty = max(qty, 1e-8)

        fill, slip, fee = self.orders.fill_market(side, open_price, qty, data)
        cost = fill * qty / self.cfg.leverage
        if cost + fee > self.capital.available_margin:
            return   # margin không đủ

        self.capital.used_margin += cost
        self.capital.balance     -= fee
        self._last_trade_candle   = i

        pos = Position(
            pos_id=0, side=side,
            entry_price=fill, quantity=qty,
            leverage=self.cfg.leverage,
            stop_loss=sl, take_profit=tp,
            entry_time=ts, entry_candle=i,
            max_holding_period=None,
        )
        self.positions.open(pos)

    def _open_position(self, side, fill_price, qty, slip, fee, ts, i, sl, tp, data):
        if sl is None:
            sl = self.stops.compute_sl(fill_price, side, data)
        if tp is None:
            tp = self.stops.compute_tp(fill_price, sl, side)
        cost = fill_price * qty / self.cfg.leverage
        self.capital.used_margin += cost
        self.capital.balance     -= fee
        self._last_trade_candle   = i
        pos = Position(pos_id=0, side=side, entry_price=fill_price, quantity=qty,
                       leverage=self.cfg.leverage, stop_loss=sl, take_profit=tp,
                       entry_time=ts, entry_candle=i, max_holding_period=None)
        self.positions.open(pos)

    def _close_position(self, pos: Position, exit_price: float,
                        ts: datetime, i: int, data: dict,
                        reason: ExitReason) -> None:
        slip = self.orders.calc_slippage(exit_price, data)
        if pos.side == "long":
            fill = exit_price - slip
            pnl  = (fill - pos.entry_price) * pos.quantity * pos.leverage
        else:
            fill = exit_price + slip
            pnl  = (pos.entry_price - fill) * pos.quantity * pos.leverage

        fee  = self.orders.calc_fee(fill, pos.quantity, "market")
        pnl -= fee

        cost = pos.entry_price * pos.quantity / pos.leverage
        self.capital.used_margin = max(0, self.capital.used_margin - cost)
        self.capital.balance    += pnl + cost

        pnl_pct = pnl / (cost + 1e-8) * 100

        self.logger_t.log(TradeRecord(
            entry_time=pos.entry_time, exit_time=ts,
            side=pos.side, entry_price=pos.entry_price, exit_price=fill,
            quantity=pos.quantity, leverage=pos.leverage,
            fee=fee, slippage=slip, pnl=pnl, pnl_pct=pnl_pct,
            holding_period=i - pos.entry_candle,
            exit_reason=reason,
        ))
        self.positions.close(pos)

    # ── Build result ───────────────────────────
    def _build_result(self, index, equity_list: List[float]) -> BacktestResult:
        equity = pd.Series(equity_list, index=index[:len(equity_list)])
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / (rolling_max + 1e-8)

        trades = self.logger_t.records()
        pnls   = [t.pnl for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return  = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if len(equity) else 0
        n_candles     = len(equity)
        returns_daily = equity.pct_change().dropna()
        sharpe = (returns_daily.mean() / (returns_daily.std() + 1e-8)) * np.sqrt(252 * 24) \
                 if len(returns_daily) > 1 else 0
        sortino_neg = returns_daily[returns_daily < 0].std()
        sortino = (returns_daily.mean() / (sortino_neg + 1e-8)) * np.sqrt(252 * 24) \
                  if sortino_neg > 1e-8 else 0
        max_dd = drawdown.min() * 100

        profit_factor = sum(wins) / (abs(sum(losses)) + 1e-8) if losses else np.inf
        winrate = len(wins) / (len(trades) + 1e-8)
        avg_rr  = np.mean([abs(w) / (abs(l) + 1e-8) for w, l in zip(wins, losses[:len(wins)])]) \
                  if wins and losses else 0

        metrics = {
            "total_return_pct":   round(total_return, 2),
            "sharpe":             round(sharpe, 3),
            "sortino":            round(sortino, 3),
            "max_drawdown_pct":   round(max_dd, 2),
            "calmar":             round(total_return / (abs(max_dd) + 1e-8), 3),
            "winrate":            round(winrate, 3),
            "profit_factor":      round(profit_factor, 3),
            "n_trades":           len(trades),
            "avg_holding_period": round(np.mean([t.holding_period for t in trades]), 1) if trades else 0,
            "long_trades":        sum(1 for t in trades if t.side == "long"),
            "short_trades":       sum(1 for t in trades if t.side == "short"),
            "expectancy":         round(np.mean(pnls), 2) if pnls else 0,
        }

        return BacktestResult(equity_curve=equity, drawdown_series=drawdown,
                              trade_history=trades, performance_metrics=metrics)
