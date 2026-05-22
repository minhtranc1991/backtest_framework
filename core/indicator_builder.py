"""
indicator_builder.py — Tính toán toàn bộ indicator, modular, có registry.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from config.config import INDICATOR_REGISTRY, INDICATOR_PARAMS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# BASE HELPERS
# ─────────────────────────────────────────────
def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _divergence(ind: pd.Series, price: pd.Series, window: int = 14) -> pd.Series:
    """Phát hiện bullish/bearish divergence đơn giản.
    +1 = bullish divergence, -1 = bearish divergence, 0 = none.
    """
    result = pd.Series(0, index=ind.index)
    for i in range(window, len(ind)):
        p_slice = price.iloc[i - window: i + 1]
        i_slice = ind.iloc[i - window: i + 1]
        p_min_idx, p_max_idx = p_slice.idxmin(), p_slice.idxmax()
        i_min_idx, i_max_idx = i_slice.idxmin(), i_slice.idxmax()
        if p_min_idx == p_slice.index[-1] and i_min_idx != i_slice.index[-1]:
            result.iloc[i] = 1   # bullish
        elif p_max_idx == p_slice.index[-1] and i_max_idx != i_slice.index[-1]:
            result.iloc[i] = -1  # bearish
    return result


# ─────────────────────────────────────────────
# INDIVIDUAL INDICATOR FUNCTIONS
# ─────────────────────────────────────────────
def calc_price_features(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    out = {}
    out["delta_price"]     = df["Close"].diff()
    out["percent_price_tl"] = df["Close"].pct_change() * 100
    return out


def calc_vma(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["vma_period"]
    return {"vma": df["Volume"].rolling(period).mean()}


def calc_vwma(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["ma_period"]
    vw = (df["Close"] * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    return {"vwma": vw}


def calc_vo(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    fast = df["Volume"].ewm(span=p["vo_fast"], adjust=False).mean()
    slow = df["Volume"].ewm(span=p["vo_slow"], adjust=False).mean()
    return {"vo": (fast - slow) / slow * 100}


def calc_vwap(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    typ = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tv = (typ * df["Volume"]).cumsum()
    cum_v  = df["Volume"].cumsum()
    vwap   = cum_tv / cum_v
    std    = df["Close"].expanding().std()
    mult   = p["vwap_band_mult"]
    return {
        "vwap":       vwap,
        "upper_vwap": vwap + mult * std,
        "lower_vwap": vwap - mult * std,
    }


def calc_cvd(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    """CVD dùng taker_buy_base_volume nếu có, fallback ước tính."""
    if "taker_buy_base_volume" in df.columns:
        buy_vol  = df["taker_buy_base_volume"]
        sell_vol = df["Volume"] - buy_vol
    else:
        # Ước tính: close>open → mua, ngược lại bán
        buy_mask = df["Close"] >= df["Open"]
        buy_vol  = df["Volume"].where(buy_mask, 0)
        sell_vol = df["Volume"].where(~buy_mask, 0)
    delta = buy_vol - sell_vol
    cvd   = delta.cumsum()
    return {"cvd": cvd, "delta_cvd": delta}


def calc_ma(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    return {"ma": df["Close"].rolling(p["ma_period"]).mean()}


def calc_ema(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    return {
        "ema":      _ema(df["Close"], p["ema_period"]),
        "ema_fast": _ema(df["Close"], p["ema_fast"]),
        "ema_slow": _ema(df["Close"], p["ema_slow"]),
    }


def calc_ichimoku(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    tenkan_p  = p["ichimoku_tenkan"]
    kijun_p   = p["ichimoku_kijun"]
    senkou_p  = p["ichimoku_senkou_b"]

    tenkan  = (df["High"].rolling(tenkan_p).max() + df["Low"].rolling(tenkan_p).min()) / 2
    kijun   = (df["High"].rolling(kijun_p).max()  + df["Low"].rolling(kijun_p).min())  / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
    senkou_b = ((df["High"].rolling(senkou_p).max() + df["Low"].rolling(senkou_p).min()) / 2).shift(kijun_p)
    chikou  = df["Close"].shift(-kijun_p)
    return {
        "tenkan":   tenkan,
        "kijun":    kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou":   chikou,
    }


def calc_keltner(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["kc_period"]
    mult   = p["kc_atr_mult"]
    mid    = _ema(df["Close"], period)
    atr    = _atr(df, period)
    return {
        "ema_kc":    mid,
        "upAtr":     mid + mult * atr,
        "downAtr":   mid - mult * atr,
        "upper_band": mid + mult * atr,
        "lower_band": mid - mult * atr,
        "basis_band": mid,
    }


def calc_bollinger(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["bb_period"]
    std_mult = p["bb_std"]
    basis = df["Close"].rolling(period).mean()
    std   = df["Close"].rolling(period).std()
    return {
        "bb_basis": basis,
        "upper":    basis + std_mult * std,
        "lower":    basis - std_mult * std,
    }


def calc_williams_fractal(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    n = 2
    high, low = df["High"], df["Low"]
    bull = pd.Series(False, index=df.index)
    bear = pd.Series(False, index=df.index)
    for i in range(n, len(df) - n):
        if all(low.iloc[i] < low.iloc[i - j] for j in range(1, n + 1)) and \
           all(low.iloc[i] < low.iloc[i + j] for j in range(1, n + 1)):
            bull.iloc[i] = True
        if all(high.iloc[i] > high.iloc[i - j] for j in range(1, n + 1)) and \
           all(high.iloc[i] > high.iloc[i + j] for j in range(1, n + 1)):
            bear.iloc[i] = True
    return {"wf_bull": bull.astype(float), "wf_bear": bear.astype(float)}


def calc_pivot_points(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    pp = (df["High"].shift(1) + df["Low"].shift(1) + df["Close"].shift(1)) / 3
    r1 = 2 * pp - df["Low"].shift(1)
    s1 = 2 * pp - df["High"].shift(1)
    r2 = pp + (df["High"].shift(1) - df["Low"].shift(1))
    s2 = pp - (df["High"].shift(1) - df["Low"].shift(1))
    r3 = df["High"].shift(1) + 2 * (pp - df["Low"].shift(1))
    s3 = df["Low"].shift(1)  - 2 * (df["High"].shift(1) - pp)
    rng = df["High"].shift(1) - df["Low"].shift(1)
    r4, s4 = r3 + rng, s3 - rng
    r5, s5 = r4 + rng, s4 - rng
    return {"pp": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2,
            "r3": r3, "s3": s3, "r4": r4, "s4": s4, "r5": r5, "s5": s5}


def calc_aroon(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["aroon_period"]
    aroon_up   = df["High"].rolling(period + 1).apply(lambda x: x.argmax()) / period * 100
    aroon_down = df["Low"].rolling(period + 1).apply(lambda x: x.argmin())  / period * 100
    return {"aroon_up": aroon_up, "aroon_down": aroon_down}


def calc_sar(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    try:
        import talib
        sar = talib.SAR(df["High"], df["Low"],
                        acceleration=p["sar_acceleration"],
                        maximum=p["sar_max_acceleration"])
    except ImportError:
        # Fallback: simplified SAR
        sar = _simple_sar(df["High"], df["Low"],
                          p["sar_acceleration"], p["sar_max_acceleration"])
    return {"sar": sar}


def _simple_sar(high: pd.Series, low: pd.Series, acc: float, max_acc: float) -> pd.Series:
    """Simplified Parabolic SAR không dùng TA-Lib."""
    sar = low.copy()
    ep  = high.copy()
    af  = acc
    bull = True
    for i in range(2, len(sar)):
        if bull:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])
            sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2])
            if low.iloc[i] < sar.iloc[i]:
                bull = False
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i]  = low.iloc[i]
                af = acc
            else:
                if high.iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = high.iloc[i]
                    af = min(af + acc, max_acc)
        else:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])
            sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2])
            if high.iloc[i] > sar.iloc[i]:
                bull = True
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i]  = high.iloc[i]
                af = acc
            else:
                if low.iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = low.iloc[i]
                    af = min(af + acc, max_acc)
    return sar


def calc_supertrend(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period, mult = p["supertrend_period"], p["supertrend_mult"]
    atr = _atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper_raw = hl2 + mult * atr
    lower_raw = hl2 - mult * atr

    upper = upper_raw.copy()
    lower = lower_raw.copy()
    trend = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        upper.iloc[i] = upper_raw.iloc[i] if upper_raw.iloc[i] < upper.iloc[i-1] or df["Close"].iloc[i-1] > upper.iloc[i-1] else upper.iloc[i-1]
        lower.iloc[i] = lower_raw.iloc[i] if lower_raw.iloc[i] > lower.iloc[i-1] or df["Close"].iloc[i-1] < lower.iloc[i-1] else lower.iloc[i-1]
        if df["Close"].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    supertrend = pd.Series(np.where(trend == 1, lower, upper), index=df.index)
    return {"up_trend": lower, "down_trend": upper, "super_trend": supertrend}


def calc_macd(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    fast   = _ema(df["Close"], p["ema_fast"])
    slow   = _ema(df["Close"], p["ema_slow"])
    macd   = fast - slow
    signal = _ema(macd, p["macd_signal"])
    return {"macd": macd, "macd_signal": signal, "macd_hist": macd - signal}


def calc_stoch(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    k_period = p["stoch_k"]
    d_period = p["stoch_d"]
    smooth   = p["stoch_smooth"]
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    fastk = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    fastd = fastk.rolling(d_period).mean()
    slowk = fastk.rolling(smooth).mean()
    slowd = slowk.rolling(d_period).mean()
    return {"fastk": fastk, "fastd": fastd, "slowk": slowk, "slowd": slowd}


def calc_rsi(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    rsi = _rsi(df["Close"], p["rsi_period"])
    rsi_sma = rsi.rolling(p["rsi_period"]).mean()
    div = _divergence(rsi, df["Close"])
    return {"rsi": rsi, "rsi_sma_line": rsi_sma, "rsi_divergence": div}


def calc_atr_ind(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["atr_period"]
    atr = _atr(df, period)
    atr_sma = atr.rolling(period).mean()
    div = _divergence(atr, df["Close"])
    return {"atr": atr, "atr_sma_line": atr_sma, "atr_divergence": div}


def calc_mfi(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["mfi_period"]
    typ = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = typ * df["Volume"]
    pos_mf = raw_mf.where(typ > typ.shift(1), 0)
    neg_mf = raw_mf.where(typ < typ.shift(1), 0)
    mfr = pos_mf.rolling(period).sum() / (neg_mf.rolling(period).sum() + 1e-10)
    mfi = 100 - (100 / (1 + mfr))
    mfi_sma = mfi.rolling(period).mean()
    div = _divergence(mfi, df["Close"])
    return {"mfi": mfi, "mfi_sma_line": mfi_sma, "mfi_divergence": div}


def calc_adx(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["adx_period"]
    atr = _atr(df, period)
    up   = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm  = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    plus_di  = 100 * _ema(plus_dm, period)  / (atr + 1e-10)
    minus_di = 100 * _ema(minus_dm, period) / (atr + 1e-10)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = _ema(dx, period)
    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def calc_cci(df: pd.DataFrame, p: dict) -> Dict[str, pd.Series]:
    period = p["cci_period"]
    typ   = (df["High"] + df["Low"] + df["Close"]) / 3
    mean  = typ.rolling(period).mean()
    mad   = typ.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci   = (typ - mean) / (0.015 * mad + 1e-10)
    return {"cci": cci}


# ─────────────────────────────────────────────
# INDICATOR REGISTRY MAP
# ─────────────────────────────────────────────
CALC_MAP = {
    "delta_price":       calc_price_features,
    "vma":               calc_vma,
    "vwma":              calc_vwma,
    "vo":                calc_vo,
    "vwap":              calc_vwap,
    "cvd":               calc_cvd,
    "ma":                calc_ma,
    "ema":               calc_ema,
    "ichimoku":          calc_ichimoku,
    "keltner":           calc_keltner,
    "bollinger":         calc_bollinger,
    "williams_fractal":  calc_williams_fractal,
    "pivot_points":      calc_pivot_points,
    "aroon":             calc_aroon,
    "sar":               calc_sar,
    "supertrend":        calc_supertrend,
    "macd":              calc_macd,
    "stoch":             calc_stoch,
    "rsi":               calc_rsi,
    "atr":               calc_atr_ind,
    "mfi":               calc_mfi,
    "adx":               calc_adx,
    "cci":               calc_cci,
}


# ─────────────────────────────────────────────
# INDICATOR BUILDER (main class)
# ─────────────────────────────────────────────
class IndicatorBuilder:
    """
    Tính toán tất cả indicator được bật trong INDICATOR_REGISTRY.
    Không tính lại indicator đã tồn tại trong DataFrame.
    """

    def __init__(
        self,
        registry: Dict[str, bool] = None,
        params: Dict[str, Any] = None,
    ):
        self.registry = registry or INDICATOR_REGISTRY
        self.params   = params   or INDICATOR_PARAMS

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về DataFrame gốc + toàn bộ indicator đã tính."""
        result = df.copy()
        computed: set = set(result.columns)

        for name, enabled in self.registry.items():
            if not enabled:
                continue
            if name not in CALC_MAP:
                continue
            try:
                new_cols = CALC_MAP[name](result, self.params)
                for col, series in new_cols.items():
                    if col not in computed:
                        result[col] = series
                        computed.add(col)
            except Exception as e:
                logger.warning(f"Indicator [{name}] failed: {e}")

        logger.info(f"✅ Built {len(result.columns) - len(df.columns)} indicator columns.")
        return result
