"""
main.py — Ví dụ chạy end-to-end toàn bộ framework.

Pipeline:
    1. Tải dữ liệu (Binance hoặc CSV)
    2. Tính toán indicators
    3. Optimize strategy (random/evolutionary)
    4. Backtest top strategy đầy đủ
    5. Walk-forward validation
    6. In kết quả + plot
"""

import logging
import sys
import os
import warnings
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

# Thêm root vào sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Suppress non-critical numerical warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
# warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*pct_change.*fill_method.*")

from utils.paths         import create_run_dir
run_dir = create_run_dir()
log_file = run_dir / "run.log"
from utils.logger import setup_logger, log_dataframe_info
logger = setup_logger(log_file=str(log_file))

from config.config          import DataConfig, BacktestConfig, SignalConfig, OptimizerConfig
from core.data_loader       import DataLoader
from core.indicator_builder import IndicatorBuilder
from core.signal_generator  import SignalGenerator
from core.backtester        import ExecutionEngine, ATRStop, RiskBasedSizing
from core.evaluation        import PerformanceEvaluator, WalkForwardValidator
from core.optimizer         import Optimizer
from utils.plotting         import plot_equity_curve, print_metrics_table

logger = logging.getLogger("backtest")
os.makedirs("results", exist_ok=True)


def main():
    # ─────────────────────────────────────────
    # 1. CẤU HÌNH
    # ─────────────────────────────────────────
    
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
        warmup_candles    = 60,
        maker_fee         = 0.0002,
        taker_fee         = 0.0004,
        slippage_model    = "pct",
        slippage_value    = 0.0001,
        initial_balance   = 10_000,
        leverage          = 2.0,
        margin_mode       = "futures",
        sizing_plugin     = RiskBasedSizing(risk_pct=0.01),
        stop_plugin       = ATRStop(atr_mult=2.0, rr=2.5),
        max_concurrent_trades  = 2,
        max_portfolio_exposure = 0.8,
        max_daily_loss         = 0.05,
        max_drawdown           = 0.20,
        entry_timing           = "next_open",
        cooldown_candles       = 2,
        allow_pyramiding       = False,
        position_mode          = "long_short",
    )

    signal_cfg = SignalConfig(
        max_conditions  = 3,
        max_indicators  = 3,
        composition     = "AND",
        search_mode     = "random",    # hoặc "evolutionary"
        population_size = 30,
        generations     = 10,
        random_seed     = 42,
    )

    opt_cfg = OptimizerConfig(
        n_strategies       = 50,
        top_k              = 5,
        fitness_weights    = {
            "sharpe":           0.35,
            "total_return_pct": 0.25,
            "max_drawdown_pct": -0.20,
            "profit_factor":    0.20,
        },
        overfitting_penalty = 0.1,
        min_trades          = 15,
        multi_objective     = False,
        random_seed         = 42,
    )

    # ─────────────────────────────────────────
    # 2. TẢI DỮ LIỆU
    # ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    loader = DataLoader(source=data_cfg.source, cache_dir=data_cfg.cache_dir,
                        use_cache=data_cfg.use_cache)
    df_raw = loader.load(data_cfg.symbol, data_cfg.interval,
                         data_cfg.start_str, data_cfg.end_str)

    if df_raw.empty:
        logger.error("Không tải được dữ liệu. Thoát.")
        return

    logger.info(f"Loaded {len(df_raw)} candles | {df_raw.index[0]} -> {df_raw.index[-1]}")
    log_dataframe_info(logger, df_raw, name=f"Raw Data [{data_cfg.symbol} {data_cfg.interval}]")
    # ─────────────────────────────────────────
    # 3. TÍNH INDICATOR
    # ─────────────────────────────────────────
    logger.info("STEP 2: Building indicators")
    ind_builder = IndicatorBuilder()
    df_feat = ind_builder.build(df_raw)
    logger.info(f"Feature DataFrame: {df_feat.shape[1]} columns")

    # ─────────────────────────────────────────
    # 4. OPTIMIZE STRATEGY
    # ─────────────────────────────────────────
    logger.info("STEP 3: Optimizing strategies")
    optimizer = Optimizer(opt_cfg, signal_cfg, backtest_cfg)
    top_strategies = optimizer.run(df_feat)

    if not top_strategies:
        logger.error("Không tìm được strategy hợp lệ.")
        return

    best = top_strategies[0]
    logger.info(f"\n[BEST STRATEGY]\n{best.strategy}")
    print_metrics_table(best.metrics)

    # ─────────────────────────────────────────
    # 5. FULL BACKTEST trên best strategy
    # ─────────────────────────────────────────
    logger.info("STEP 4: Full backtest on best strategy")
    sig_gen  = SignalGenerator(signal_cfg)
    signals  = sig_gen.generate_signals_for(best.strategy, df_feat)
    engine   = ExecutionEngine(backtest_cfg)
    result   = engine.run(df_feat, signals)
    metrics  = PerformanceEvaluator.evaluate(result)
    print_metrics_table(metrics)

    # ─────────────────────────────────────────
    # 6. WALK-FORWARD VALIDATION
    # ─────────────────────────────────────────
    logger.info("STEP 5: Walk-forward validation")
    wf = WalkForwardValidator(n_splits=4, oos_ratio=0.25)
    wf_result = wf.run(
        df_feat, best.strategy,
        engine_factory=lambda: ExecutionEngine(backtest_cfg),
        indicator_builder=ind_builder,
        signal_generator=sig_gen,
    )
    if wf_result:
        logger.info(
            f"Walk-forward: avg_sharpe={wf_result['avg_sharpe']:.3f} | "
            f"stability={wf_result['stability_score']:.0%}"
        )

    # ─────────────────────────────────────────
    # 7. PLOT
    # ─────────────────────────────────────────
    plot_equity_curve(result, title=f"Best Strategy [{best.strategy.strategy_id}]",
                      save_path=run_dir / "equity_curve.png")

    # ─────────────────────────────────────────
    # 8. IN TOP-K
    # ─────────────────────────────────────────
    logger.info("\n[TOP STRATEGIES RANKING]")
    print(f"\n{'Rank':<6} {'ID':<8} {'Fitness':>9} {'Sharpe':>8} {'Return%':>9} {'DD%':>8} {'WRate':>7} {'N':>5}")
    print("-" * 65)
    for r in top_strategies:
        m = r.metrics
        print(f"{r.rank:<6} {r.strategy.strategy_id:<8} {r.fitness:>9.3f} "
              f"{m.get('sharpe',0):>8.3f} {m.get('total_return_pct',0):>9.2f} "
              f"{m.get('max_drawdown_pct',0):>8.2f} {m.get('winrate',0):>7.2%} "
              f"{m.get('n_trades',0):>5}")


if __name__ == "__main__":
    main()
