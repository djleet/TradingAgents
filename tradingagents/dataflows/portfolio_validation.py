"""Walk-forward validation for current and risk-balanced portfolio weights.

Every risk-balanced rebalance uses only the trailing window available before
that rebalance.  The engine is deterministic, execution-free, and compares the
same symbols, dates, costs and cash allocation against the current mix and a
buy-and-hold benchmark.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.portfolio_optimization import load_price_matrix

PORTFOLIO_VALIDATION_VERSION = "portfolio-walk-forward-v1"
TRADING_DAYS = 252


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _clean_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for raw_symbol, raw_weight in weights.items():
        symbol = normalize_symbol(str(raw_symbol).strip())
        safe_ticker_component(symbol, max_len=15)
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Weight for {symbol} must be a finite non-negative number")
        if weight:
            cleaned[symbol] = cleaned.get(symbol, 0.0) + weight
    if not 2 <= len(cleaned) <= 30:
        raise ValueError("Portfolio validation requires 2-30 positive-weight symbols")
    if sum(cleaned.values()) > 1.000001:
        raise ValueError("Portfolio weights cannot exceed 100%")
    return cleaned


def _sharpe(returns: pd.Series) -> float | None:
    volatility = float(returns.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0:
        return None
    return _number(float(returns.mean()) / volatility * math.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float | None:
    if equity.empty:
        return None
    running_peak = equity.cummax().clip(lower=1.0)
    return _number(float((equity / running_peak - 1).min()))


def _longest_underwater_period(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    underwater = equity < equity.cummax().clip(lower=1.0)
    longest = current = 0
    for is_underwater in underwater:
        current = current + 1 if bool(is_underwater) else 0
        longest = max(longest, current)
    return longest


def _risk_budget_frequency(
    returns: pd.Series,
    *,
    annual_volatility_limit: float,
    daily_es_limit: float,
    confidence: float = 0.95,
    window: int = 60,
) -> dict[str, Any]:
    volatility_breaches = 0
    tail_breaches = 0
    combined_breaches = 0
    observations = 0
    for end in range(window, len(returns) + 1):
        sample = returns.iloc[end - window : end]
        volatility = float(sample.std(ddof=1) * math.sqrt(TRADING_DAYS))
        losses = -sample.to_numpy(dtype=float)
        value_at_risk = float(np.quantile(losses, confidence))
        tail = losses[losses >= value_at_risk]
        expected_shortfall = float(tail.mean()) if len(tail) else value_at_risk
        vol_breach = volatility > annual_volatility_limit
        tail_breach = expected_shortfall > daily_es_limit
        volatility_breaches += int(vol_breach)
        tail_breaches += int(tail_breach)
        combined_breaches += int(vol_breach or tail_breach)
        observations += 1
    return {
        "observations": observations,
        "volatility_breaches": volatility_breaches,
        "expected_shortfall_breaches": tail_breaches,
        "combined_breaches": combined_breaches,
        "breach_frequency": _number(combined_breaches / observations)
        if observations
        else None,
    }


def _strategy_metrics(
    frame: pd.DataFrame,
    *,
    annual_volatility_limit: float,
    daily_es_limit: float,
) -> dict[str, Any]:
    returns = frame["net_return"]
    equity = frame["equity"]
    sessions = len(frame)
    total_return = float(equity.iloc[-1] - 1) if sessions else 0.0
    annualized_return = (
        float(equity.iloc[-1] ** (TRADING_DAYS / sessions) - 1)
        if sessions and equity.iloc[-1] > 0
        else None
    )
    rolling = equity / equity.shift(63) - 1
    risk_budget = _risk_budget_frequency(
        returns,
        annual_volatility_limit=annual_volatility_limit,
        daily_es_limit=daily_es_limit,
    )
    return {
        "total_return": _number(total_return),
        "annualized_return": _number(annualized_return),
        "annualized_volatility": _number(
            float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        ),
        "sharpe_ratio": _sharpe(returns),
        "max_drawdown": _max_drawdown(equity),
        "longest_recovery_sessions": _longest_underwater_period(equity),
        "worst_rolling_quarter": _number(float(rolling.min()))
        if rolling.notna().any()
        else None,
        "turnover": _number(float(frame["turnover"].sum())),
        "transaction_cost": _number(float(frame["transaction_cost"].sum())),
        "risk_budget": risk_budget,
    }


def _simulate(
    asset_returns: pd.DataFrame,
    evaluation_start: int,
    rebalance_sessions: int,
    target_builder: Callable[[int], pd.Series],
    *,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    symbols = list(asset_returns.columns)
    weights = pd.Series(0.0, index=symbols)
    cash_weight = 1.0
    equity = 1.0
    rows: list[dict[str, Any]] = []
    rebalances: list[dict[str, Any]] = []
    cost_rate = transaction_cost_bps / 10_000

    for oos_index, global_index in enumerate(range(evaluation_start, len(asset_returns))):
        turnover = 0.0
        transaction_cost = 0.0
        if oos_index % rebalance_sessions == 0:
            target = target_builder(global_index).reindex(symbols).fillna(0.0).astype(float)
            target_cash = max(0.0, 1.0 - float(target.sum()))
            turnover = 0.5 * (
                float((target - weights).abs().sum()) + abs(target_cash - cash_weight)
            )
            transaction_cost = turnover * cost_rate
            weights = target.copy()
            cash_weight = target_cash
            rebalances.append(
                {
                    "date": asset_returns.index[global_index].isoformat(),
                    "turnover": _number(turnover),
                    "transaction_cost": _number(transaction_cost),
                    "weights": {symbol: _number(weights[symbol]) for symbol in symbols},
                }
            )

        daily_returns = asset_returns.iloc[global_index]
        gross_return = float(weights.dot(daily_returns))
        net_return = gross_return - transaction_cost
        equity *= max(0.000001, 1.0 + net_return)
        rows.append(
            {
                "date": asset_returns.index[global_index],
                "gross_return": gross_return,
                "net_return": net_return,
                "equity": equity,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            }
        )

        denominator = 1.0 + gross_return
        if denominator > 0:
            weights = weights.mul(1.0 + daily_returns).div(denominator)
            cash_weight /= denominator

    frame = pd.DataFrame(rows).set_index("date")
    return frame, rebalances


def _regime_labels(benchmark_price: pd.Series) -> pd.Series:
    average = benchmark_price.rolling(200).mean()
    volatility = benchmark_price.pct_change(fill_method=None).rolling(20).std(ddof=1)
    volatility = volatility.mul(math.sqrt(TRADING_DAYS))
    labels = pd.Series("transition", index=benchmark_price.index, dtype="object")
    labels.loc[average.notna() & benchmark_price.ge(average)] = "bull"
    labels.loc[average.notna() & benchmark_price.lt(average)] = "bear"
    labels.loc[volatility.ge(0.30)] = "high_volatility"
    return labels


def _paired_bootstrap(
    balanced_returns: pd.Series,
    current_returns: pd.Series,
    *,
    samples: int = 500,
    block_size: int = 21,
) -> dict[str, Any]:
    difference = (balanced_returns - current_returns).to_numpy(dtype=float)
    if not len(difference):
        return {
            "annualized_excess_p05": None,
            "annualized_excess_p50": None,
            "annualized_excess_p95": None,
            "probability_balanced_outperforms": None,
        }
    rng = np.random.default_rng(42)
    block = max(1, min(block_size, len(difference)))
    estimates = []
    for _ in range(samples):
        selected: list[float] = []
        while len(selected) < len(difference):
            start = int(rng.integers(0, max(1, len(difference) - block + 1)))
            selected.extend(difference[start : start + block])
        estimates.append(float(np.mean(selected[: len(difference)]) * TRADING_DAYS))
    values = np.asarray(estimates)
    return {
        "annualized_excess_p05": _number(np.quantile(values, 0.05)),
        "annualized_excess_p50": _number(np.quantile(values, 0.50)),
        "annualized_excess_p95": _number(np.quantile(values, 0.95)),
        "probability_balanced_outperforms": _number(float((values > 0).mean())),
        "samples": samples,
        "block_size": block,
        "seed": 42,
    }


def _verdict(
    current: dict[str, Any],
    balanced: dict[str, Any],
    block_win_rate: float,
    bootstrap: dict[str, Any],
) -> tuple[str, list[str]]:
    current_sharpe = float(current.get("sharpe_ratio") or 0.0)
    balanced_sharpe = float(balanced.get("sharpe_ratio") or 0.0)
    current_breaches = float(
        current.get("risk_budget", {}).get("breach_frequency") or 0.0
    )
    balanced_breaches = float(
        balanced.get("risk_budget", {}).get("breach_frequency") or 0.0
    )
    checks = {
        "lower_volatility": float(balanced["annualized_volatility"] or 0.0)
        <= float(current["annualized_volatility"] or 0.0) * 0.98,
        "shallower_drawdown": float(balanced["max_drawdown"] or -1.0)
        >= float(current["max_drawdown"] or -1.0),
        "higher_sharpe": balanced_sharpe >= current_sharpe + 0.05,
        "return_preserved": float(balanced["total_return"] or 0.0)
        >= float(current["total_return"] or 0.0) - 0.02,
        "wins_across_blocks": block_win_rate >= 0.55,
        "fewer_budget_breaches": balanced_breaches <= current_breaches,
        "bootstrap_support": float(
            bootstrap.get("probability_balanced_outperforms") or 0.0
        )
        >= 0.55,
    }
    score = sum(checks.values())
    if checks["lower_volatility"] and checks["return_preserved"] and score >= 5:
        verdict = "improved"
    elif score <= 2 or (
        float(balanced["total_return"] or 0.0)
        < float(current["total_return"] or 0.0) - 0.05
        and balanced_sharpe <= current_sharpe
    ):
        verdict = "rejected"
    else:
        verdict = "mixed"

    reasons = [
        (
            "Risk-balanced volatility was lower."
            if checks["lower_volatility"]
            else "Risk-balanced volatility did not improve by at least 2%."
        ),
        (
            "Maximum drawdown was no worse than the current mix."
            if checks["shallower_drawdown"]
            else "Maximum drawdown was deeper than the current mix."
        ),
        (
            "Return stayed within two percentage points of the current mix."
            if checks["return_preserved"]
            else "Risk reduction sacrificed more than two percentage points of return."
        ),
        f"Risk-balanced beat the current mix in {block_win_rate:.0%} of rebalance blocks.",
        (
            "Paired bootstrap support was positive."
            if checks["bootstrap_support"]
            else "Paired bootstrap support was inconclusive."
        ),
    ]
    return verdict, reasons


def validate_portfolio_prices(
    prices: pd.DataFrame,
    weights: dict[str, float],
    benchmark_symbol: str,
    as_of_date: str,
    *,
    sessions: int = 504,
    train_sessions: int = 252,
    rebalance_sessions: int = 63,
    transaction_cost_bps: float = 7.0,
    annual_volatility_limit: float = 0.25,
    daily_es_limit: float = 0.03,
) -> dict[str, Any]:
    """Validate a current mix against trailing-only inverse-volatility weights."""
    cleaned = _clean_weights(weights)
    symbols = list(cleaned)
    benchmark = normalize_symbol(benchmark_symbol.strip())
    safe_ticker_component(benchmark, max_len=15)
    requested_sessions = max(126, min(int(sessions), 1260))
    training_size = max(126, min(int(train_sessions), 756))
    rebalance_size = int(rebalance_sessions)
    if rebalance_size not in {21, 63, 126}:
        raise ValueError("Rebalance sessions must be 21, 63 or 126")
    cost_bps = max(0.0, min(float(transaction_cost_bps), 100.0))
    required_columns = list(dict.fromkeys([*symbols, benchmark]))
    missing = [symbol for symbol in required_columns if symbol not in prices.columns]
    if missing:
        raise ValueError(f"Price matrix is missing: {', '.join(missing)}")
    clean_prices = prices[required_columns].dropna().sort_index()
    returns = clean_prices.pct_change(fill_method=None).dropna()
    minimum = training_size + max(126, rebalance_size)
    if len(returns) < minimum:
        raise ValueError(
            f"Portfolio validation needs at least {minimum + 1} common prices; "
            f"received {len(clean_prices)}"
        )
    evaluation_start = max(training_size, len(returns) - requested_sessions)
    asset_returns = returns[symbols]
    invested_weight = sum(cleaned.values())
    current_target = pd.Series(cleaned, dtype=float).reindex(symbols)

    current_frame, current_rebalances = _simulate(
        asset_returns,
        evaluation_start,
        rebalance_size,
        lambda _index: current_target,
        transaction_cost_bps=cost_bps,
    )

    balanced_training: dict[str, dict[str, Any]] = {}

    def balanced_target(index: int) -> pd.Series:
        training = asset_returns.iloc[index - training_size : index]
        volatility = training.std(ddof=1)
        if volatility.isna().any() or (volatility <= 0).any():
            raise ValueError("Risk-balanced validation found zero or unavailable volatility")
        inverse = 1.0 / volatility
        target = inverse / inverse.sum() * invested_weight
        date_key = asset_returns.index[index].isoformat()
        balanced_training[date_key] = {
            "train_start": training.index[0].isoformat(),
            "train_end": training.index[-1].isoformat(),
            "estimated_annual_volatility": {
                symbol: _number(volatility[symbol] * math.sqrt(TRADING_DAYS))
                for symbol in symbols
            },
        }
        return target

    balanced_frame, balanced_rebalances = _simulate(
        asset_returns,
        evaluation_start,
        rebalance_size,
        balanced_target,
        transaction_cost_bps=cost_bps,
    )
    benchmark_returns = returns[benchmark].iloc[evaluation_start:].copy()
    benchmark_frame = pd.DataFrame(
        {
            "net_return": benchmark_returns,
            "equity": (1.0 + benchmark_returns).cumprod(),
            "turnover": 0.0,
            "transaction_cost": 0.0,
        },
        index=benchmark_returns.index,
    )

    current_metrics = _strategy_metrics(
        current_frame,
        annual_volatility_limit=annual_volatility_limit,
        daily_es_limit=daily_es_limit,
    )
    balanced_metrics = _strategy_metrics(
        balanced_frame,
        annual_volatility_limit=annual_volatility_limit,
        daily_es_limit=daily_es_limit,
    )
    benchmark_metrics = _strategy_metrics(
        benchmark_frame,
        annual_volatility_limit=annual_volatility_limit,
        daily_es_limit=daily_es_limit,
    )

    rebalance_blocks = []
    for number, start in enumerate(range(0, len(current_frame), rebalance_size), start=1):
        end = min(len(current_frame), start + rebalance_size)
        current_window = current_frame.iloc[start:end]
        balanced_window = balanced_frame.iloc[start:end]
        benchmark_window = benchmark_frame.iloc[start:end]
        current_return = float((1.0 + current_window["net_return"]).prod() - 1)
        balanced_return = float((1.0 + balanced_window["net_return"]).prod() - 1)
        benchmark_return = float((1.0 + benchmark_window["net_return"]).prod() - 1)
        rebalance = balanced_rebalances[number - 1]
        training = balanced_training[rebalance["date"]]
        rebalance_blocks.append(
            {
                "block": number,
                "test_start": current_window.index[0].isoformat(),
                "test_end": current_window.index[-1].isoformat(),
                "train_start": training["train_start"],
                "train_end": training["train_end"],
                "current_return": _number(current_return),
                "balanced_return": _number(balanced_return),
                "benchmark_return": _number(benchmark_return),
                "balanced_minus_current": _number(balanced_return - current_return),
                "balanced_won": balanced_return > current_return,
                "balanced_weights": rebalance["weights"],
                "current_turnover": current_rebalances[number - 1]["turnover"],
                "balanced_turnover": rebalance["turnover"],
            }
        )
    block_win_rate = sum(item["balanced_won"] for item in rebalance_blocks) / len(
        rebalance_blocks
    )

    benchmark_prices = clean_prices[benchmark].reindex(returns.index)
    labels = _regime_labels(benchmark_prices).iloc[evaluation_start:]
    regimes = []
    for regime in ("bull", "bear", "high_volatility", "transition"):
        mask = labels.eq(regime)
        if not mask.any():
            continue
        regimes.append(
            {
                "regime": regime,
                "sessions": int(mask.sum()),
                "current_return": _number(
                    float((1.0 + current_frame.loc[mask, "net_return"]).prod() - 1)
                ),
                "balanced_return": _number(
                    float((1.0 + balanced_frame.loc[mask, "net_return"]).prod() - 1)
                ),
                "benchmark_return": _number(
                    float((1.0 + benchmark_frame.loc[mask, "net_return"]).prod() - 1)
                ),
                "current_volatility": _number(
                    float(current_frame.loc[mask, "net_return"].std(ddof=1) * math.sqrt(TRADING_DAYS))
                ),
                "balanced_volatility": _number(
                    float(balanced_frame.loc[mask, "net_return"].std(ddof=1) * math.sqrt(TRADING_DAYS))
                ),
            }
        )

    bootstrap = _paired_bootstrap(
        balanced_frame["net_return"], current_frame["net_return"]
    )
    verdict, reasons = _verdict(
        current_metrics,
        balanced_metrics,
        block_win_rate,
        bootstrap,
    )
    points = [
        {
            "date": index.isoformat(),
            "current_equity": _number(current_frame.loc[index, "equity"]),
            "balanced_equity": _number(balanced_frame.loc[index, "equity"]),
            "benchmark_equity": _number(benchmark_frame.loc[index, "equity"]),
            "regime": labels.loc[index],
        }
        for index in current_frame.index
    ]
    return {
        "model_version": PORTFOLIO_VALIDATION_VERSION,
        "as_of_date": as_of_date,
        "benchmark_symbol": benchmark,
        "symbols": symbols,
        "current_weights": {symbol: _number(cleaned[symbol]) for symbol in symbols},
        "invested_weight": _number(invested_weight),
        "cash_weight": _number(max(0.0, 1.0 - invested_weight)),
        "first_session": points[0]["date"],
        "last_session": points[-1]["date"],
        "oos_sessions": len(points),
        "configuration": {
            "requested_sessions": requested_sessions,
            "train_sessions": training_size,
            "rebalance_sessions": rebalance_size,
            "transaction_cost_bps": cost_bps,
            "annual_volatility_limit": annual_volatility_limit,
            "daily_es_limit": daily_es_limit,
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
        "strategies": {
            "current": current_metrics,
            "risk_balanced": balanced_metrics,
            "benchmark": benchmark_metrics,
        },
        "block_win_rate": _number(block_win_rate),
        "robustness": bootstrap,
        "rebalance_blocks": rebalance_blocks,
        "regimes": regimes,
        "points": points,
        "methodology": (
            "At each rebalance, inverse-volatility weights use only the trailing training window "
            "ending on the prior session. Current and risk-balanced mixes use identical dates, "
            "cash exposure and transaction costs; the benchmark is buy-and-hold."
        ),
        "limitations": (
            "Historical research only. The current mix is applied retrospectively, covariance can "
            "change abruptly, and taxes, intraday slippage and market impact are not modeled."
        ),
    }


def build_portfolio_validation_report(
    weights: dict[str, float],
    benchmark_symbol: str,
    as_of_date: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a common point-in-time price matrix and run portfolio validation."""
    cleaned = _clean_weights(weights)
    benchmark = normalize_symbol(benchmark_symbol.strip())
    sessions = max(126, min(int(kwargs.get("sessions", 504)), 1260))
    train_sessions = max(126, min(int(kwargs.get("train_sessions", 252)), 756))
    calendar_days = min(2000, max(730, int((sessions + train_sessions + 80) * 1.55)))
    symbols = list(dict.fromkeys([*cleaned, benchmark]))
    prices = load_price_matrix(
        symbols,
        as_of_date,
        lookback_calendar_days=calendar_days,
    )
    return validate_portfolio_prices(
        prices,
        cleaned,
        benchmark,
        as_of_date,
        **kwargs,
    )
