"""
optimizer.py — Sinh nhiều strategy, backtest và rank theo fitness function.
Hỗ trợ: random search, evolutionary search, multi-objective (Pareto), overfitting penalty.
"""

from __future__ import annotations
import logging
import random
import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple

from config.config import BacktestConfig, SignalConfig, OptimizerConfig
from core.signal_generator import Strategy, SignalGenerator, StrategyGenerator
from core.indicator_builder import IndicatorBuilder
from core.backtester import ExecutionEngine
from core.evaluation import PerformanceEvaluator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# FITNESS FUNCTION
# ═══════════════════════════════════════════════════════════════════
def compute_fitness(metrics: Dict, weights: Dict[str, float],
                    n_trades: int, min_trades: int,
                    overfitting_penalty: float) -> float:
    """
    Ví dụ mặc định:
        fitness = 0.35*sharpe + 0.25*return - 0.20*|drawdown| + 0.20*profit_factor
    """
    if metrics is None or not metrics:
        return -9999.0

    score = 0.0
    for key, w in weights.items():
        val = metrics.get(key, 0.0)
        if key == "max_drawdown_pct":
            val = abs(val)   # đã có dấu âm trong weights
        score += w * val

    # Penalty nếu quá ít trades (dấu hiệu overfit)
    if n_trades < min_trades:
        score -= overfitting_penalty * (min_trades - n_trades)

    return score


# ═══════════════════════════════════════════════════════════════════
# RANKED RESULT
# ═══════════════════════════════════════════════════════════════════
@dataclass
class RankedStrategy:
    strategy: Strategy
    metrics: Dict
    fitness: float
    wf_summary: Optional[Dict] = None
    rank: int = 0


# ═══════════════════════════════════════════════════════════════════
# PARETO FRONT (multi-objective)
# ═══════════════════════════════════════════════════════════════════
def pareto_front(candidates: List[RankedStrategy],
                 obj_keys: List[str] = None) -> List[RankedStrategy]:
    """
    Trả về Pareto-optimal set: maximize sharpe, return; minimize |drawdown|.
    """
    if obj_keys is None:
        obj_keys = ["sharpe", "total_return_pct", "max_drawdown_pct"]

    def dominates(a: RankedStrategy, b: RankedStrategy) -> bool:
        a_vals = _objectives(a, obj_keys)
        b_vals = _objectives(b, obj_keys)
        return all(av >= bv for av, bv in zip(a_vals, b_vals)) and \
               any(av >  bv for av, bv in zip(a_vals, b_vals))

    front = []
    for cand in candidates:
        if not any(dominates(other, cand) for other in candidates if other is not cand):
            front.append(cand)
    return front


def _objectives(rs: RankedStrategy, keys: List[str]) -> List[float]:
    vals = []
    for k in keys:
        v = rs.metrics.get(k, 0.0)
        if k == "max_drawdown_pct":
            v = -abs(v)   # minimize drawdown → maximize negative
        vals.append(v)
    return vals


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════
class Optimizer:
    """
    Tự động sinh, backtest, rank và trả về top-k strategy.
    """

    def __init__(
        self,
        opt_cfg: OptimizerConfig,
        signal_cfg: SignalConfig,
        backtest_cfg: BacktestConfig,
    ):
        self.opt_cfg      = opt_cfg
        self.signal_cfg   = signal_cfg
        self.backtest_cfg = backtest_cfg
        random.seed(opt_cfg.random_seed)
        np.random.seed(opt_cfg.random_seed)

    def run(
        self,
        df_feat: pd.DataFrame,     # DataFrame đã có đầy đủ indicator
        n_strategies: Optional[int] = None,
    ) -> List[RankedStrategy]:
        """
        df_feat: output của IndicatorBuilder.build()
        Trả về danh sách RankedStrategy đã sắp xếp theo fitness.
        """
        n = n_strategies or self.opt_cfg.n_strategies
        mode = self.signal_cfg.search_mode

        gen = SignalGenerator(self.signal_cfg)
        strat_gen = gen.build_generator(df_feat)

        if mode == "random":
            population = strat_gen.generate_population(n)
        else:
            population = self._evolutionary_search(strat_gen, df_feat, n)

        logger.info(f"🔬 Backtesting {len(population)} strategies...")
        ranked = []
        for i, strategy in enumerate(population):
            try:
                result = self._backtest(strategy, df_feat)
                metrics = PerformanceEvaluator.evaluate(result)
                fitness = compute_fitness(
                    metrics,
                    self.opt_cfg.fitness_weights,
                    metrics.get("n_trades", 0),
                    self.opt_cfg.min_trades,
                    self.opt_cfg.overfitting_penalty,
                )
                ranked.append(RankedStrategy(
                    strategy=strategy,
                    metrics=metrics,
                    fitness=fitness,
                ))
                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i+1}/{len(population)}")
            except Exception as e:
                logger.warning(f"Strategy {strategy.strategy_id} failed: {e}")

        # Sort by fitness descending
        ranked.sort(key=lambda r: r.fitness, reverse=True)
        for idx, r in enumerate(ranked):
            r.rank = idx + 1

        top_k = ranked[:self.opt_cfg.top_k]

        if self.opt_cfg.multi_objective:
            pareto = pareto_front(ranked)
            logger.info(f"📊 Pareto front: {len(pareto)} strategies")

        logger.info(f"🏆 Top-1 fitness={top_k[0].fitness:.3f}" if top_k else "No valid strategies.")
        return top_k

    # ── Evolutionary search ────────────────────
    def _evolutionary_search(
        self, strat_gen: StrategyGenerator, df: pd.DataFrame, budget: int
    ) -> List[Strategy]:
        pop_size = self.signal_cfg.population_size
        n_gen    = self.signal_cfg.generations
        mut_r    = self.signal_cfg.mutation_rate
        cx_r     = self.signal_cfg.crossover_rate

        population = strat_gen.generate_population(pop_size)
        best_pool: List[Strategy] = []

        for gen_idx in range(n_gen):
            # Evaluate fitness
            scored = []
            for s in population:
                try:
                    result = self._backtest(s, df)
                    m = PerformanceEvaluator.evaluate(result)
                    f = compute_fitness(m, self.opt_cfg.fitness_weights,
                                        m.get("n_trades", 0),
                                        self.opt_cfg.min_trades,
                                        self.opt_cfg.overfitting_penalty)
                    scored.append((s, f))
                except Exception:
                    scored.append((s, -9999))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_pool.extend([s for s, _ in scored[:5]])

            # Selection: top 50%
            survivors = [s for s, _ in scored[:pop_size // 2]]

            # Crossover
            children = []
            while len(children) < pop_size // 2 and len(survivors) >= 2:
                if random.random() < cx_r:
                    a, b = random.sample(survivors, 2)
                    c1, c2 = strat_gen.crossover(a, b)
                    children.extend([c1, c2])

            # Mutation
            mutants = []
            for s in survivors:
                if random.random() < mut_r:
                    mutants.append(strat_gen.mutate(s))

            population = survivors + children + mutants
            population = population[:pop_size]

            best_f = scored[0][1] if scored else -999
            logger.info(f"  Gen {gen_idx+1}/{n_gen} — best fitness: {best_f:.3f}")

        # Return all evaluated strategies (unique)
        seen = set()
        result_pool = []
        for s in best_pool + population:
            if s.strategy_id not in seen:
                seen.add(s.strategy_id)
                result_pool.append(s)
        return result_pool[:budget]

    # ── Single backtest ────────────────────────
    def _backtest(self, strategy: Strategy, df: pd.DataFrame):
        gen = SignalGenerator(self.signal_cfg)
        signals = gen.generate_signals_for(strategy, df)
        engine  = ExecutionEngine(self.backtest_cfg)
        return engine.run(df, signals)
