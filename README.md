# 🚀 Backtest Framework - Khung Kiểm Thử Chiến Lược Trading

Một khung công tác (framework) backtesting hiện đại, mạnh mẽ và toàn diện được thiết kế để **tự động phát sinh, kiểm thử và tối ưu hóa** các chiến lược trading cryptocurrency trên các sàn giao dịch như **Binance**. Framework này hỗ trợ cả **long** và **short**, **leverage**, **quản lý rủi ro** toàn diện, và **walk-forward validation** để ngăn chặn overfitting.

---

## 📋 Mục Lục
- [🎯 Tính Năng Chính](#-tính-năng-chính)
- [🏗️ Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [📦 Yêu Cầu Cài Đặt](#-yêu-cầu-cài-đặt)
- [⚙️ Hướng Dẫn Cài Đặt](#-hướng-dẫn-cài-đặt)
- [🔧 Cấu Hình Chi Tiết](#-cấu-hình-chi-tiết)
- [🎮 Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
- [🧠 Module Chính](#-module-chính)
- [📊 Ví Dụ Chạy Đầy Đủ](#-ví-dụ-chạy-đầy-đủ)
- [📈 Metrics & KPI](#-metrics--kpi)
- [🔒 Quản Lý Rủi Ro](#-quản-lý-rủi-ro)
- [🧬 Tối Ưu Hóa Chiến Lược](#-tối-ưu-hóa-chiến-lược)
- [✅ Walk-Forward Validation](#-walk-forward-validation)
- [📊 Hiển Thị & Báo Cáo](#-hiển-thị--báo-cáo)
- [🐛 Gỡ Lỗi & Troubleshooting](#-gỡ-lỗi--troubleshooting)
- [📝 License & Đóng Góp](#-license--đóng-góp)

---

## 🎯 Tính Năng Chính

### ✨ Core Features
- **📊 Tải Dữ Liệu Linh Hoạt**
  - Hỗ trợ **Binance REST API** (lịch sử đầy đủ)
  - Hỗ trợ **CSV** custom data
  - **Caching** tự động với PyArrow Parquet
  - Hỗ trợ đa timeframe (1m, 5m, 15m, 1h, 4h, 1d, ...)

- **📈 40+ Chỉ Báo Kỹ Thuật**
  - **Giá**: MA, EMA, Ichimoku, Keltner, Bollinger Bands, SAR, Supertrend
  - **Volume**: VMA, VWMA, VWAP, OBV, CVD (Cumulative Volume Delta)
  - **Oscillator**: RSI, MACD, Stochastic, ADX, CCI, MFI, Aroon
  - **Pattern Recognition**: Williams Fractals, Pivot Points
  - **Chỉ báo Mở Rộng**: Divergence Detection (RSI, ATR, MFI), Slope Analysis

- **🤖 Tự Động Phát Sinh Chiến Lược**
  - **Random Search**: Sinh ngẫu nhiên N chiến lược
  - **Evolutionary Search**: Tìm kiếm tối ưu với Genetic Algorithm
  - **Dynamic Signal Generation**: Kết hợp các chỉ báo theo quy tắc AND/OR
  - **Validation**: Kiểm tra tính hợp lệ các phép so sánh (scale matching)

- **⚡ Backtesting Engine Hoàn Chỉnh**
  - **Execution**: Thực hiện lệnh market/limit, phí maker/taker
  - **Position Management**: Long/Short, Pyramiding, Concurrent trades
  - **Stop Management**: Stop-Loss, Take-Profit, Trailing Stop, Liquidation
  - **Order Management**: Pending orders, Order expiry, Cooldown periods
  - **Portfolio Management**: Max portfolio exposure, Daily loss limits

- **💰 Quản Lý Tài Chính & Rủi Ro**
  - **Position Sizing**: Risk-based sizing, ATR-based stops
  - **Leverage Support**: Futures trading với margin
  - **Fee Modeling**: Maker/Taker fees, Slippage simulation
  - **Drawdown Tracking**: Max drawdown, Daily drawdown limits
  - **Multi-Trade Management**: Concurrent positions với exposure limits

- **📊 Metrics & Đánh Giá**
  - **Return Metrics**: Total Return, CAGR, Monthly/Annual Returns
  - **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - **Trade Metrics**: Win Rate, Profit Factor, Average Win/Loss
  - **Distribution**: Tail risk, Consecutive wins/losses
  - **Consistency**: Recovery factor, Expectancy

- **🔍 Walk-Forward Validation**
  - **Out-of-Sample Testing**: Tránh overfitting
  - **Rolling Windows**: In-sample optimization, Out-of-sample test
  - **Anchored vs Rolling**: Hai chế độ cross-validation
  - **Parameter Stability**: Kiểm tra tính ổn định giữa các window

- **📈 Visualization & Reports**
  - **Equity Curve Plots**: Biểu đồ lợi nhuận theo thời gian
  - **Drawdown Charts**: Biểu đồ Drawdown, Max DD
  - **Trade Analysis**: Entry/exit points, Trade PnL distribution
  - **Performance Tables**: Metrics summary, Monthly/Annual returns

---

## 🏗️ Kiến Trúc Hệ Thống

```
backtest_framework/
├── main.py                          # Entry point - ví dụ đầy đủ
├── requirements.txt                 # Dependencies
├── config/
│   ├── __init__.py
│   └── config.py                    # Tất cả cấu hình (DataConfig, BacktestConfig, ...)
├── core/
│   ├── __init__.py
│   ├── data_loader.py              # Tải dữ liệu từ Binance/CSV, caching
│   ├── indicator_builder.py        # 40+ chỉ báo kỹ thuật
│   ├── signal_generator.py         # Phát sinh tín hiệu, random/evolutionary search
│   ├── backtester.py               # Backtesting engine chính (ExecutionEngine)
│   ├── optimizer.py                # Tối ưu hóa chiến lược, fitness functions
│   └── evaluation.py               # Tính metrics, PerformanceEvaluator, WalkForwardValidator
├── strategies/
│   ├── __init__.py
│   └── (chứa các chiến lược tùy chỉnh - optional)
├── utils/
│   ├── __init__.py
│   ├── logger.py                   # Logging configuration
│   └── plotting.py                 # Visualization, plot equity curves
└── results/                         # Output folder (tự tạo)
    ├── run.log                      # Log file
    └── (các biểu đồ và báo cáo)
```

### 🔄 Pipeline Chính

```
1. 📥 Load Data (Binance/CSV)
   ↓
2. 📊 Build Indicators (40+ technical indicators)
   ↓
3. 🤖 Optimize Strategies (Generate N strategies via random/evolutionary search)
   ↓
4. ⚙️ Backtest Each Strategy (Execute trades, track metrics)
   ↓
5. 📈 Evaluate & Rank (Fitness scoring, top K selection)
   ↓
6. 🔍 Walk-Forward Validation (Out-of-sample validation)
   ↓
7. 📊 Report & Visualize (Print metrics, plot equity curves)
```

---

## 📦 Yêu Cầu Cài Đặt

### Python Version
- **Python 3.10+** (khuyến nghị Python 3.11+)

### Dependencies Chính
```
numpy>=1.24          # Xử lý số học
pandas>=2.0          # DataFrames
requests>=2.31       # HTTP requests cho Binance API
pyarrow>=12.0        # Parquet caching
matplotlib>=3.7      # Visualization
scipy>=1.11          # Tính toán khoa học (optim, stats)
```

### Optional Dependencies
- **TA-Lib** (talib-binary): Để sử dụng SAR chính xác hơn
  ```bash
  pip install TA-Lib
  ```
  Nếu không cài, framework sẽ sử dụng SAR built-in

---

## ⚙️ Hướng Dẫn Cài Đặt

### 1️⃣ Clone hoặc Download Project

```bash
cd /path/to/backtest_framework
```

### 2️⃣ Tạo Virtual Environment (Khuyến Nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ (Tùy Chọn) Cài TA-Lib cho SAR

```bash
pip install TA-Lib
# hoặc
pip install talib-binary
```

### 5️⃣ Kiểm Tra Cài Đặt

```bash
python -c "import pandas; import numpy; import requests; print('✓ Installation OK')"
```

---

## 🔧 Cấu Hình Chi Tiết

Tất cả cấu hình nằm trong [config/config.py](config/config.py). Dưới đây là các tham số chính:

### 📊 DataConfig - Cấu Hình Dữ Liệu

```python
from config.config import DataConfig

data_cfg = DataConfig(
    symbol="BTCUSDT",              # Ký hiệu giao dịch (Binance)
    interval="1h",                 # Timeframe: 1m, 5m, 15m, 1h, 4h, 1d, ...
    start_str="01/01/2023",        # Ngày bắt đầu (DD/MM/YYYY)
    end_str="01/12/2023",          # Ngày kết thúc (DD/MM/YYYY)
    source="binance",              # "binance" hoặc "csv"
    cache_dir="cache",             # Thư mục cache Parquet
    use_cache=True,                # Sử dụng cache (tiết kiệm API calls)
)
```

**Symbols hỗ trợ**: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, v.v. (tất cả pairs trên Binance)

**Intervals hỗ trợ**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

### ⚙️ BacktestConfig - Cấu Hình Backtesting

```python
from config.config import BacktestConfig
from core.backtester import RiskBasedSizing, ATRStop

backtest_cfg = BacktestConfig(
    # ─────────── Data ───────────
    warmup_candles=60,             # Số candle để warm-up indicators trước trading
    
    # ─────────── Fees & Slippage ───────────
    maker_fee=0.0002,              # 0.02% (Binance maker fee)
    taker_fee=0.0004,              # 0.04% (Binance taker fee)
    slippage_model="pct",          # "pct" hoặc "fixed"
    slippage_value=0.0001,         # 0.01% slippage per trade
    
    # ─────────── Capital ───────────
    initial_balance=10_000,        # Số tiền ban đầu (USDT)
    leverage=2.0,                  # Leverage (1.0 = spot, >1 = margin/futures)
    margin_mode="futures",         # "spot" hoặc "futures"
    
    # ─────────── Position Sizing & Stops ───────────
    sizing_plugin=RiskBasedSizing(risk_pct=0.01),   # Risk 1% per trade
    stop_plugin=ATRStop(atr_mult=2.0, rr=2.5),      # SL = 2*ATR, TP = 2.5*RR
    
    # ─────────── Position Management ───────────
    max_concurrent_trades=2,       # Tối đa 2 trades mở cùng lúc
    max_portfolio_exposure=0.8,    # Max 80% của balance in trades
    max_daily_loss=0.05,           # Stop trading nếu mất 5% trong ngày
    max_drawdown=0.20,             # Stop trading nếu drawdown > 20%
    
    # ─────────── Order Execution ───────────
    entry_timing="next_open",      # "next_open" hoặc "current_close"
    cooldown_candles=2,            # Chờ 2 candles trước khi mở vị trí mới
    allow_pyramiding=False,        # Không stack thêm vào vị trí hiện tại
    
    # ─────────── Trading Mode ───────────
    position_mode="long_short",    # "long_only", "short_only", hoặc "long_short"
)
```

### 🤖 SignalConfig - Cấu Hình Phát Sinh Signal

```python
from config.config import SignalConfig

signal_cfg = SignalConfig(
    max_conditions=3,              # Tối đa 3 điều kiện per signal
    max_indicators=3,              # Tối đa 3 chỉ báo khác nhau
    composition="AND",             # "AND" hoặc "OR" (kết hợp các điều kiện)
    search_mode="random",          # "random" hoặc "evolutionary"
    population_size=30,            # Kích thước population cho evolutionary
    generations=10,                # Số generation cho evolutionary
    random_seed=42,                # Seed cho reproducibility
)
```

### 🎯 OptimizerConfig - Cấu Hình Tối Ưu Hóa

```python
from config.config import OptimizerConfig

opt_cfg = OptimizerConfig(
    n_strategies=50,               # Sinh 50 chiến lược
    top_k=5,                       # Chọn top 5 tốt nhất
    fitness_weights={              # Hàm fitness custom
        "sharpe":           0.35,   # Trọng số Sharpe Ratio
        "total_return_pct": 0.25,   # Trọng số Return
        "max_drawdown_pct": -0.20,  # Minimize Drawdown (âm)
        "profit_factor":    0.20,   # Trọng số Profit Factor
    },
    overfitting_penalty=0.1,       # Penalty nếu quá ít trades
    min_trades=15,                 # Tối thiểu 15 trades để valid
    multi_objective=False,         # False = weighted sum; True = Pareto front
    random_seed=42,
)
```

### 📋 INDICATOR_REGISTRY - Bật/Tắt Chỉ Báo

Trong [config/config.py](config/config.py), bạn có thể bật/tắt từng chỉ báo:

```python
INDICATOR_REGISTRY = {
    # Price-based
    "ma": True,          # Moving Average
    "ema": True,         # Exponential MA
    "ichimoku": True,    # Ichimoku Cloud
    "bollinger": True,   # Bollinger Bands
    "keltner": True,     # Keltner Channels
    "supertrend": True,  # Supertrend
    
    # Oscillators
    "rsi": True,         # Relative Strength Index
    "macd": True,        # MACD
    "stoch": True,       # Stochastic
    "adx": True,         # ADX
    "cci": True,         # Commodity Channel Index
    "mfi": True,         # Money Flow Index
    
    # Volume
    "vma": True,         # Volume MA
    "vwap": True,        # VWAP
    "cvd": True,         # Cumulative Volume Delta
    "obv": True,         # On-Balance Volume
    
    # Pattern
    "williams_fractal": True,  # Williams Fractal
    "pivot_points": True,      # Pivot Points
    "aroon": True,             # Aroon Indicator
    "sar": True,               # Parabolic SAR
    
    # ... and 20+ more
}
```

Tham số mặc định cho chỉ báo được lưu trong `INDICATOR_PARAMS`:

```python
INDICATOR_PARAMS = {
    "ma_period": 20,
    "ema_period": 20,
    "rsi_period": 14,
    "atr_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    # ... và 40+ tham số khác
}
```

---

## 🎮 Hướng Dẫn Sử Dụng

### Chạy Pipeline Đầy Đủ

```bash
python main.py
```

**main.py** chứa ví dụ end-to-end:

```python
import logging
from config.config import DataConfig, BacktestConfig, SignalConfig, OptimizerConfig
from core.data_loader import DataLoader
from core.indicator_builder import IndicatorBuilder
from core.signal_generator import SignalGenerator
from core.backtester import ExecutionEngine, RiskBasedSizing, ATRStop
from core.optimizer import Optimizer
from core.evaluation import PerformanceEvaluator, WalkForwardValidator
from utils.plotting import plot_equity_curve, print_metrics_table

# 1. Cấu hình
data_cfg = DataConfig(symbol="BTCUSDT", interval="1h", ...)
backtest_cfg = BacktestConfig(initial_balance=10_000, ...)
signal_cfg = SignalConfig(search_mode="random", ...)
opt_cfg = OptimizerConfig(n_strategies=50, ...)

# 2. Tải dữ liệu
loader = DataLoader(data_cfg)
df = loader.load()

# 3. Tính chỉ báo
builder = IndicatorBuilder(df, config.INDICATOR_REGISTRY)
df = builder.build_all_indicators()

# 4. Sinh chiến lược
gen = StrategyGenerator(signal_cfg, config.INDICATOR_REGISTRY)
strategies = gen.generate_n_strategies(opt_cfg.n_strategies)

# 5. Tối ưu hóa
optimizer = Optimizer(df, backtest_cfg, opt_cfg)
ranked = optimizer.optimize(strategies)

# 6. Walk-Forward Validation
validator = WalkForwardValidator(df, backtest_cfg)
wf_results = validator.validate(ranked[:opt_cfg.top_k])

# 7. Báo cáo
print_metrics_table(ranked)
plot_equity_curve(ranked[0].metrics["equity_curve"])
```

### Tùy Chỉnh: Chỉ Backtest Một Chiến Lược

```python
from core.signal_generator import Strategy, Condition
from core.backtester import ExecutionEngine

# Tạo chiến lược tùy chỉnh
my_strategy = Strategy(
    name="Custom RSI Strategy",
    long_entry=[
        Condition(left="rsi", operator="lt", right=30),
        Condition(left="Close", operator="gt", right="ema"),
    ],
    long_exit=[
        Condition(left="rsi", operator="gt", right=70),
    ],
    short_entry=[
        Condition(left="rsi", operator="gt", right=70),
    ],
    short_exit=[
        Condition(left="rsi", operator="lt", right=30),
    ],
    composition="AND",
)

# Backtest
engine = ExecutionEngine(df, backtest_cfg)
result = engine.run(my_strategy)
metrics = PerformanceEvaluator.evaluate(result)
print(metrics)
```

### Tùy Chỉnh: Sử Dụng Dữ Liệu CSV

```python
from config.config import DataConfig
from core.data_loader import DataLoader

data_cfg = DataConfig(
    symbol="CUSTOM",
    source="csv",
    csv_path="my_data.csv",  # Cột: Open, High, Low, Close, Volume
    # ...
)

loader = DataLoader(data_cfg)
df = loader.load()
```

---

## 🧠 Module Chính

### 1. **data_loader.py** - Tải & Xử Lý Dữ Liệu

| Hàm/Class | Mô Tả |
|-----------|-------|
| `DataLoader` | Class chính tải từ Binance/CSV |
| `fetch_binance_kline_df()` | Tải từ Binance REST API với phân trang |
| `load_csv()` | Tải từ file CSV |
| `_cache_key()` | Tạo key cho cache |
| `_load_from_cache()` / `_save_to_cache()` | Quản lý cache Parquet |

**Ví dụ:**
```python
from core.data_loader import DataLoader
from config.config import DataConfig

data_cfg = DataConfig(symbol="BTCUSDT", interval="1h", ...)
loader = DataLoader(data_cfg)
df = loader.load()  # Returns: DataFrame with OHLCV
```

### 2. **indicator_builder.py** - Tính Chỉ Báo

| Chỉ Báo | Loại | Thành Phần |
|---------|------|----------|
| MA, EMA | Trend | Moving Averages |
| Ichimoku | Trend | Tenkan, Kijun, Senkou A/B, Chikou |
| Bollinger | Volatility | Upper/Lower band, Basis |
| Keltner | Volatility | EMA ± ATR |
| Supertrend | Trend | Up/Down trend lines |
| RSI, Stochastic, MACD, ADX, CCI, MFI | Oscillators | Momentum indicators |
| VWAP, OBV, CVD, VMA, VWMA | Volume | Volume-weighted indicators |
| Pivot Points, Williams Fractal, Aroon | Pattern | Pattern recognition |
| SAR, Divergences | Advanced | Parabolic SAR, RSI/ATR/MFI divergences |

**Ví dụ:**
```python
from core.indicator_builder import IndicatorBuilder
from config.config import INDICATOR_REGISTRY

builder = IndicatorBuilder(df, INDICATOR_REGISTRY)
df = builder.build_all_indicators()
# df bây giờ có 60+ cột (indicators)
```

### 3. **signal_generator.py** - Phát Sinh Tín Hiệu

| Class/Hàm | Mô Tả |
|-----------|-------|
| `Condition` | Một điều kiện (e.g., RSI < 30) |
| `Signal` | Entry/exit signals |
| `Strategy` | Toàn bộ entry/exit logic |
| `StrategyGenerator` | Sinh ngẫu nhiên chiến lược |
| `EAValidator` | Kiểm tra điều kiện |

**Ví dụ:**
```python
from core.signal_generator import StrategyGenerator
from config.config import SignalConfig

signal_cfg = SignalConfig(search_mode="random", n_strategies=50)
gen = StrategyGenerator(signal_cfg, INDICATOR_REGISTRY)
strategies = gen.generate_n_strategies(50)
```

### 4. **backtester.py** - Backtesting Engine

**Core Classes:**
- `ExecutionEngine`: Chạy backtest chính
- `OrderManager`: Quản lý lệnh (pending, filled)
- `PositionManager`: Quản lý vị trí (long/short)
- `CapitalManager`: Quản lý tài chính (balance, equity)
- `RiskManager`: Kiểm tra điều kiện rủi ro (max DD, daily loss)
- `StopManager`: Quản lý stop-loss/take-profit/trailing stop
- `PortfolioManager`: Quản lý danh mục (concurrent trades, exposure)

**Data Classes:**
- `Position`, `TradeRecord`, `BacktestResult`

**Ví dụ:**
```python
from core.backtester import ExecutionEngine

engine = ExecutionEngine(df, backtest_cfg)
result = engine.run(strategy)
# result: BacktestResult chứa equity_curve, trade_history, metrics
```

### 5. **optimizer.py** - Tối Ưu Hóa Chiến Lược

| Hàm/Class | Mô Tả |
|-----------|-------|
| `compute_fitness()` | Tính fitness score |
| `Optimizer` | Sinh, backtest, rank chiến lược |
| `RankedStrategy` | Strategy + metrics + fitness |
| `pareto_front()` | Multi-objective optimization |
| `elitism()` | Giữ best strategies |

**Ví dụ:**
```python
from core.optimizer import Optimizer

optimizer = Optimizer(df, backtest_cfg, opt_cfg)
ranked = optimizer.optimize(strategies)  # List[RankedStrategy]
print(ranked[0].fitness)  # Best strategy fitness score
```

### 6. **evaluation.py** - Đánh Giá & Validation

| Class | Mô Tả |
|-------|-------|
| `PerformanceEvaluator` | Tính toàn bộ metrics |
| `WalkForwardValidator` | Out-of-sample validation |

**Metrics Tính Toán:**
- **Return**: Total Return %, CAGR, Monthly/Annual Returns
- **Risk**: Sharpe, Sortino, Calmar, Max Drawdown
- **Trade**: Win Rate, Profit Factor, Avg Win/Loss
- **Distribution**: Consecutive wins/losses, Tail risk

**Ví dụ:**
```python
from core.evaluation import PerformanceEvaluator, WalkForwardValidator

# Metrics
metrics = PerformanceEvaluator.evaluate(result, candles_per_year=8760)
print(f"Sharpe: {metrics['sharpe']}")

# Walk-Forward Validation
validator = WalkForwardValidator(df, backtest_cfg)
wf_summary = validator.validate(strategy)  # Dict with in-sample vs out-of-sample metrics
```

### 7. **plotting.py** - Visualization

| Hàm | Mô Tả |
|-----|-------|
| `plot_equity_curve()` | Vẽ equity curve |
| `plot_drawdown()` | Vẽ drawdown |
| `plot_trades()` | Vẽ entry/exit points |
| `print_metrics_table()` | In bảng metrics |

**Ví dụ:**
```python
from utils.plotting import plot_equity_curve, print_metrics_table

plot_equity_curve(result.equity_curve, title="Strategy Performance")
print_metrics_table([ranked[0], ranked[1], ranked[2]])
```

---

## 📊 Ví Dụ Chạy Đầy Đủ

### Scenario: Optimize RSI + EMA Strategy trên BTC 1h

**main.py:**

```python
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logger
setup_logger("backtest", logging.INFO, "results/run.log")

import pandas as pd
from config.config import (
    DataConfig, BacktestConfig, SignalConfig, OptimizerConfig,
    INDICATOR_REGISTRY, INDICATOR_PARAMS
)
from core.data_loader import DataLoader
from core.indicator_builder import IndicatorBuilder
from core.signal_generator import StrategyGenerator
from core.backtester import ExecutionEngine, RiskBasedSizing, ATRStop
from core.optimizer import Optimizer
from core.evaluation import PerformanceEvaluator, WalkForwardValidator
from utils.plotting import plot_equity_curve, print_metrics_table

logger = logging.getLogger("backtest")
os.makedirs("results", exist_ok=True)


def main():
    # ═══════════════════════════════════════════════════════════════
    # 1. CẤU HÌNH
    # ═══════════════════════════════════════════════════════════════
    data_cfg = DataConfig(
        symbol="BTCUSDT",
        interval="1h",
        start_str="01/01/2023",
        end_str="01/12/2023",
        source="binance",
        cache_dir="cache",
        use_cache=True,
    )

    backtest_cfg = BacktestConfig(
        warmup_candles=60,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_model="pct",
        slippage_value=0.0001,
        initial_balance=10_000,
        leverage=2.0,
        margin_mode="futures",
        sizing_plugin=RiskBasedSizing(risk_pct=0.01),
        stop_plugin=ATRStop(atr_mult=2.0, rr=2.5),
        max_concurrent_trades=2,
        max_portfolio_exposure=0.8,
        max_daily_loss=0.05,
        max_drawdown=0.20,
        entry_timing="next_open",
        cooldown_candles=2,
        allow_pyramiding=False,
        position_mode="long_short",
    )

    signal_cfg = SignalConfig(
        max_conditions=3,
        max_indicators=3,
        composition="AND",
        search_mode="random",
        population_size=30,
        generations=10,
        random_seed=42,
    )

    opt_cfg = OptimizerConfig(
        n_strategies=50,
        top_k=5,
        fitness_weights={
            "sharpe": 0.35,
            "total_return_pct": 0.25,
            "max_drawdown_pct": -0.20,
            "profit_factor": 0.20,
        },
        overfitting_penalty=0.1,
        min_trades=15,
        multi_objective=False,
        random_seed=42,
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. TẢI DỮ LIỆU
    # ═══════════════════════════════════════════════════════════════
    logger.info("📥 Loading data...")
    loader = DataLoader(data_cfg)
    df = loader.load()
    logger.info(f"✓ Loaded {len(df)} candles for {data_cfg.symbol}")

    # ═══════════════════════════════════════════════════════════════
    # 3. TÍNH INDICATORS
    # ═══════════════════════════════════════════════════════════════
    logger.info("📊 Building indicators...")
    builder = IndicatorBuilder(df, INDICATOR_REGISTRY)
    df = builder.build_all_indicators()
    logger.info(f"✓ Built {len(df.columns)} columns (OHLCV + indicators)")

    # ═══════════════════════════════════════════════════════════════
    # 4. SINH CHIẾN LƯỢC
    # ═══════════════════════════════════════════════════════════════
    logger.info("🤖 Generating strategies...")
    gen = StrategyGenerator(signal_cfg, INDICATOR_REGISTRY)
    strategies = gen.generate_n_strategies(opt_cfg.n_strategies)
    logger.info(f"✓ Generated {len(strategies)} strategies")

    # ═══════════════════════════════════════════════════════════════
    # 5. TỐI ƯU HÓA
    # ═══════════════════════════════════════════════════════════════
    logger.info("⚡ Optimizing strategies...")
    optimizer = Optimizer(df, backtest_cfg, opt_cfg)
    ranked = optimizer.optimize(strategies)
    logger.info(f"✓ Top 5 strategies ranked")

    # ═══════════════════════════════════════════════════════════════
    # 6. WALK-FORWARD VALIDATION
    # ═══════════════════════════════════════════════════════════════
    logger.info("🔍 Walk-forward validation...")
    validator = WalkForwardValidator(df, backtest_cfg)
    for i, rs in enumerate(ranked[:opt_cfg.top_k]):
        wf_result = validator.validate(rs.strategy)
        rs.wf_summary = wf_result
        logger.info(f"  Strategy {i+1}: IS sharpe={wf_result['is_sharpe']:.2f}, "
                   f"OOS sharpe={wf_result['oos_sharpe']:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 7. BÁOCÁO & VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    logger.info("📈 Generating reports...")
    print_metrics_table(ranked[:opt_cfg.top_k])

    # Plot top strategy
    top_strategy = ranked[0]
    plot_equity_curve(
        top_strategy.metrics["equity_curve"],
        title=f"Top Strategy: {top_strategy.strategy.name}"
    )

    logger.info("✓ Done! Check results/ folder for plots.")


if __name__ == "__main__":
    main()
```

**Output mong đợi:**

```
2026-05-22 10:30:15 | INFO | 📥 Loading data...
2026-05-22 10:30:25 | INFO | ✓ Loaded 8760 candles for BTCUSDT
2026-05-22 10:30:26 | INFO | 📊 Building indicators...
2026-05-22 10:30:30 | INFO | ✓ Built 65 columns (OHLCV + indicators)
2026-05-22 10:30:30 | INFO | 🤖 Generating strategies...
2026-05-22 10:30:32 | INFO | ✓ Generated 50 strategies
2026-05-22 10:30:32 | INFO | ⚡ Optimizing strategies...
2026-05-22 10:31:15 | INFO | ✓ Top 5 strategies ranked
2026-05-22 10:31:15 | INFO | 🔍 Walk-forward validation...
2026-05-22 10:32:00 | INFO | ✓ Done! Check results/ folder for plots.

═══════════════════════════════════════════════════════════════
                      TOP 5 STRATEGIES
═══════════════════════════════════════════════════════════════
Rank | Fitness | Sharpe | Return % | Max DD % | Win Rate | Trades
─────┼─────────┼────────┼──────────┼──────────┼──────────┼────────
  1  |  1.234  |  1.56  |  45.32   |  -18.5   |  62.3%   |  52
  2  |  1.102  |  1.42  |  38.21   |  -21.3   |  59.1%   |  48
  3  |  0.987  |  1.21  |  32.45   |  -19.8   |  58.5%   |  45
  4  |  0.856  |  1.08  |  28.12   |  -22.1   |  56.7%   |  42
  5  |  0.743  |  0.95  |  24.56   |  -24.3   |  54.2%   |  38
```

---

## 📈 Metrics & KPI

Framework tính toàn bộ metrics hiệu suất từ trade history. Dưới đây là chi tiết:

### 💰 Return Metrics

| Metric | Công Thức | Giải Thích |
|--------|----------|-----------|
| **Total Return %** | (Final / Initial - 1) × 100 | Lợi nhuận tổng cộng (%) |
| **CAGR** | (Final / Initial)^(1/years) - 1 | Lợi nhuận hằng năm |
| **Monthly Return** | Nhóm theo tháng | Lợi nhuận từng tháng |
| **Annual Return** | Nhóm theo năm | Lợi nhuận từng năm |

### 📊 Risk Metrics

| Metric | Công Thức | Giải Thích |
|--------|----------|-----------|
| **Sharpe Ratio** | Mean(Return) / Std(Return) × √252 | Lợi nhuận điều chỉnh theo rủi ro |
| **Sortino Ratio** | Mean(Return) / Std(Negative Returns) × √252 | Chỉ tính downside risk |
| **Calmar Ratio** | CAGR / Max Drawdown | Lợi nhuận trên max loss |
| **Max Drawdown %** | (Trough / Peak - 1) × 100 | Mức sụt giảm lớn nhất |
| **Volatility** | Std(Daily Returns) × √252 | Độ biến động |

### 🎯 Trade Metrics

| Metric | Công Thức | Giải Thích |
|--------|----------|-----------|
| **Win Rate %** | Winning Trades / Total Trades × 100 | % trades lãi |
| **Profit Factor** | Sum(Wins) / Sum(Losses) | Tổng lãi / tổng lỗ |
| **Average Win** | Sum(Wins) / Count(Wins) | Lãi trung bình/trade |
| **Average Loss** | Sum(Losses) / Count(Losses) | Lỗ trung bình/trade |
| **Expectancy** | Avg Win × Win Rate - Avg Loss × (1 - Win Rate) | Kỳ vọng lãi/trade |
| **Recovery Factor** | Total Profit / Max Drawdown | Khả năng phục hồi |

### 📈 Consistency Metrics

| Metric | Giải Thích |
|--------|-----------|
| **Consecutive Wins** | Chuỗi trades thắng dài nhất |
| **Consecutive Losses** | Chuỗi trades thua dài nhất |
| **Tail Ratio** | (99th percentile) / (1st percentile) |
| **Return Distribution** | Skewness, Kurtosis |

---

## 🔒 Quản Lý Rủi Ro

Framework có các plugin quản lý rủi ro toàn diện:

### 1. **Position Sizing**

**RiskBasedSizing** - Risk fixed % của balance:

```python
from core.backtester import RiskBasedSizing

sizing = RiskBasedSizing(risk_pct=0.01)  # Risk 1% per trade
# position_size = balance × risk_pct / (entry_price - stop_loss)
```

**FixedSizing** - Size cố định:

```python
from core.backtester import FixedSizing

sizing = FixedSizing(qty=0.1)  # Luôn 0.1 BTC
```

### 2. **Stop Management**

**ATRStop** - Stop dựa trên ATR:

```python
from core.backtester import ATRStop

stop = ATRStop(atr_mult=2.0, rr=2.5)
# stop_loss = entry_price ± 2.0 × ATR
# take_profit = entry_price ± 2.0 × ATR × 2.5 (RR = 2.5)
```

**PercentStop** - Stop theo %:

```python
from core.backtester import PercentStop

stop = PercentStop(sl_pct=0.02, tp_pct=0.05)
# stop_loss = entry_price × (1 - 0.02)
# take_profit = entry_price × (1 + 0.05)
```

### 3. **Constraints**

```python
BacktestConfig(
    max_concurrent_trades=2,        # Tối đa 2 trades mở
    max_portfolio_exposure=0.8,     # Max 80% balance
    max_daily_loss=0.05,            # Stop nếu mất 5% trong ngày
    max_drawdown=0.20,              # Stop nếu DD > 20%
    cooldown_candles=2,             # Chờ 2 candles trước entry mới
    allow_pyramiding=False,         # Không stack vào vị trí
)
```

### 4. **Trailing Stop & Liquidation**

```python
# Tự động tính trailing stop nếu trade có lãi
# Liquidation kiểm tra margin level (cho futures)
```

---

## 🧬 Tối Ưu Hóa Chiến Lược

### Phương Pháp 1: Random Search

Sinh N chiến lược ngẫu nhiên, backtest tất cả, chọn top K tốt nhất.

```python
signal_cfg = SignalConfig(
    search_mode="random",
    max_conditions=3,
    max_indicators=3,
)

gen = StrategyGenerator(signal_cfg, INDICATOR_REGISTRY)
strategies = gen.generate_n_strategies(50)  # 50 random strategies
```

**Ưu điểm**: Nhanh, không bị local optima
**Nhược điểm**: Có thể miss tối ưu global

### Phương Pháp 2: Evolutionary Search

Sử dụng Genetic Algorithm: crossover, mutation, selection.

```python
signal_cfg = SignalConfig(
    search_mode="evolutionary",
    population_size=30,
    generations=10,
    mutation_rate=0.2,
    crossover_rate=0.5,
)

gen = StrategyGenerator(signal_cfg, INDICATOR_REGISTRY)
strategies = gen.generate_n_strategies(30)  # Initial population
```

**Ưu điểm**: Khám phá tốt hơn, convergence tốt
**Nhược điểm**: Chậm hơn random search

### Hàm Fitness Tùy Chỉnh

```python
OptimizerConfig(
    fitness_weights={
        "sharpe":           0.35,   # Ưu tiên risk-adjusted return
        "total_return_pct": 0.25,   # Ưu tiên lợi nhuận tuyệt đối
        "max_drawdown_pct": -0.20,  # Minimize drawdown
        "profit_factor":    0.20,   # Ưu tiên trades lãi
    },
    overfitting_penalty=0.1,        # Penalty nếu quá ít trades
    min_trades=15,                  # Tối thiểu 15 trades
)
```

Fitness = 0.35×Sharpe + 0.25×Return - 0.20×|MaxDD| + 0.20×PF - 0.1×overfitting_penalty

### Multi-Objective Optimization (Pareto Front)

```python
OptimizerConfig(
    multi_objective=True,  # Pareto front instead of weighted sum
)

# Trả về non-dominated solutions: maximize Sharpe + Return, minimize Drawdown
```

---

## ✅ Walk-Forward Validation

Walk-Forward Validation tránh overfitting bằng cách:
1. **In-Sample Window**: Optimize strategy trên data cũ
2. **Out-of-Sample Window**: Test trên data mới không dùng optimize

### Ví Dụ

```python
from core.evaluation import WalkForwardValidator

validator = WalkForwardValidator(df, backtest_cfg)

# Cấu hình mặc định: 
# - Is window: 60% data
# - OOS window: 20% data
# - Rolling forward: 10% data

wf_result = validator.validate(strategy)
print(f"In-Sample Sharpe:  {wf_result['is_sharpe']}")
print(f"Out-Of-Sample Sharpe: {wf_result['oos_sharpe']}")
print(f"OOS Degradation: {wf_result['degradation_pct']:.1f}%")
```

**Output mong đợi:**

```python
{
    "is_sharpe": 1.45,          # In-sample Sharpe
    "oos_sharpe": 0.92,         # Out-of-sample Sharpe
    "is_return": 45.2,          # In-sample return %
    "oos_return": 28.5,         # Out-of-sample return %
    "degradation_pct": 37.0,    # Overfitting indicator (%)
    "is_windows": 5,            # Number of in-sample windows
    "oos_windows": 5,           # Number of out-of-sample windows
}
```

**Cảnh báo**: Nếu `degradation_pct > 50%`, chiến lược overfit

---

## 📊 Hiển Thị & Báo Cáo

### Vẽ Biểu Đồ

```python
from utils.plotting import (
    plot_equity_curve,
    plot_drawdown,
    plot_trades,
    print_metrics_table,
)

# Equity curve
plot_equity_curve(
    result.equity_curve,
    title="Strategy Performance",
    save_path="results/equity_curve.png"
)

# Drawdown
plot_drawdown(
    result.drawdown_series,
    title="Drawdown Over Time",
    save_path="results/drawdown.png"
)

# Trades visualization
plot_trades(
    df,
    result.trade_history,
    title="Entry/Exit Points",
    save_path="results/trades.png"
)
```

### In Bảng Metrics

```python
from utils.plotting import print_metrics_table

# Print top 5 strategies
print_metrics_table(ranked[:5])

# Output:
# ╔════╦═════════╦════════╦═════════╦═════════╦════════╦═════════╗
# ║ ID ║ Fitness ║ Sharpe ║ Return% ║ MaxDD%  ║ WinRate║ Trades  ║
# ╠════╬═════════╬════════╬═════════╬═════════╬════════╬═════════╣
# ║ 1  ║  1.234  ║  1.56  ║  45.32  ║ -18.50  ║ 62.3%  ║   52    ║
# ...
```

### Custom Reporting

```python
import pandas as pd

# Tạo custom report
report = pd.DataFrame([
    {
        "Strategy": s.strategy.name,
        "Sharpe": s.metrics["sharpe"],
        "Return %": s.metrics["total_return_pct"],
        "Win Rate": s.metrics["win_rate"] * 100,
        "Trades": s.metrics["n_trades"],
    }
    for s in ranked[:10]
])

report.to_csv("results/strategy_summary.csv", index=False)
```

---

## 🐛 Gỡ Lỗi & Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'pandas'"

```bash
pip install -r requirements.txt
```

### ❌ "ConnectionError: Failed to connect to Binance"

- Kiểm tra internet connection
- Binance API có thể bị rate-limit (1200 requests/minute)
- Sử dụng cache: `use_cache=True` trong DataConfig

```python
data_cfg = DataConfig(
    use_cache=True,
    cache_dir="cache",  # Sẽ load từ cache nếu có
)
```

### ❌ "Strategy has no valid conditions"

- Kiểm tra INDICATOR_REGISTRY, bắt buộc phải bật indicators
- Hoặc tùy chỉnh các conditions

```python
from config.config import INDICATOR_REGISTRY
print([k for k, v in INDICATOR_REGISTRY.items() if v])  # Indicators enabled
```

### ❌ "No trades generated"

- Kiểm tra signal conditions
- Tăng `max_conditions`, `max_indicators` trong SignalConfig
- Giảm volatility filters

### ❌ "Memory error với dataset lớn"

- Giảm timeframe (e.g., từ 1m → 5m)
- Giảm số chiến lược (`n_strategies`)
- Sử dụng chunking trong optimizer

### ❌ Overfitting cao (WF degradation > 50%)

- Tăng `min_trades` constraint
- Thêm `overfitting_penalty` trong fitness
- Giảm `max_conditions`, `max_indicators`
- Sử dụng multi-objective optimization

### ✅ Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print strategy details
print(ranked[0].strategy)

# Check metrics
print(ranked[0].metrics)

# Inspect trades
for trade in result.trade_history[:5]:
    print(f"{trade.entry_time} → {trade.exit_time}: {trade.pnl} USDT")
```

---

## 📝 License & Đóng Góp

Dự án này được xây dựng cho **nghiên cứu & giáo dục**. 

**Disclaimer**: 
- ⚠️ Trading là có rủi ro. Backtest result không đảm bảo performance tương lai
- ⚠️ Luôn backtest kỹ trước khi trade thực
- ⚠️ Bắt đầu với small position size

**Contributions**: 
- Issues & pull requests được chào đón
- Fork & customize cho nhu cầu của bạn
- Chia sẻ cải tiến

---

## 🚀 Quick Start Checklist

- [ ] Install Python 3.10+
- [ ] `pip install -r requirements.txt`
- [ ] Cấu hình `config/config.py`
- [ ] Chạy `python main.py`
- [ ] Kiểm tra `results/` folder
- [ ] Tinh chỉnh parameters
- [ ] Walk-forward validation
- [ ] Deploy strategy

---

## 📞 Support & Questions

- Xem [main.py](main.py) để ví dụ đầy đủ
- Đọc docstrings trong từng module
- Kiểm tra [config/config.py](config/config.py) để cấu hình
- Tìm hiểu thêm về technical indicators trong [core/indicator_builder.py](core/indicator_builder.py)

---

## 🎓 Tham Khảo Thêm

- **Backtesting Theory**: https://www.investopedia.com/terms/b/backtesting.asp
- **Technical Analysis**: https://www.investopedia.com/terms/t/technicalanalysis.asp
- **Binance API**: https://binance-docs.github.io/apidocs/
- **Risk Management**: https://en.wikipedia.org/wiki/Risk_management
- **Walk-Forward Analysis**: https://en.wikipedia.org/wiki/Walk_forward_optimization

---

**Framework Version**: 1.0.0  
**Last Updated**: May 22, 2026  
**Author**: AI Backtesting Framework Team  

Happy Trading! 🚀📈
