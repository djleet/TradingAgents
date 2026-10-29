"""Rolling out-of-sample validation for the transparent trend signal."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.quantitative import MODEL_VERSION, quantitative_score_series
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.signal_backtest import (
    SUPPORTED_BACKTEST_MODES,
    select_price_series,
)
from tradingagents.dataflows.warehouse import parse_ohlcv_payload

THRESHOLD_CANDIDATES = (10, 20, 30, 40)


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _clean_price_frame(
    frame: pd.DataFrame,
    symbol: str,
    as_of_date: str,
) -> tuple[pd.DataFrame, str]:
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    clean = frame[frame["Date"] <= as_of].drop_duplicates("Date", keep="last")
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0].sort_values("Date").copy()
    if clean.empty:
        raise NoMarketDataError(symbol, symbol, "validation could not parse price history")
    clean["Price"], basis = select_price_series(clean)
    clean = clean.dropna(subset=["Price"])
    clean = clean[clean["Price"] > 0]
    return clean[["Date", "Price"]].reset_index(drop=True), basis


def _desired_position(scores: pd.Series, threshold: int, mode: str) -> pd.Series:
    desired = pd.Series(0.0, index=scores.index)
    desired.loc[scores.ge(threshold)] = 1.0
    if mode == "long_short":
        desired.loc[scores.le(-threshold)] = -1.0
    return desired


def _apply_cost_model(
    market_return: pd.Series,
    position: pd.Series,
    *,
    commission_bps: float,
    slippage_bps: float,
    spread_bps: float,
    short_borrow_bps: float,
) -> pd.DataFrame:
    turnover = position.diff().abs().fillna(position.abs())
    one_way_bps = commission_bps + slippage_bps + spread_bps / 2
    execution_cost = turnover.mul(one_way_bps / 10_000)
    borrow_cost = position.clip(upper=0).abs().mul(short_borrow_bps / 10_000 / 252)
    gross_return = position.mul(market_return)
    return pd.DataFrame(
        {
            "position": position,
            "turnover": turnover,
            "gross_return": gross_return,
            "execution_cost": execution_cost,
            "borrow_cost": borrow_cost,
            "strategy_return": gross_return.sub(execution_cost).sub(borrow_cost),
        }
    )


def _sharpe(returns: pd.Series) -> float | None:
    volatility = returns.std(ddof=1)
    if not volatility or not math.isfinite(volatility):
        return None
    return _number(returns.mean() / volatility * math.sqrt(252))


def _maximum_drawdown(equity: pd.Series) -> float | None:
    if equity.empty:
        return None
    return _number(equity.div(equity.cummax()).sub(1).min())


def _return(equity: pd.Series) -> float | None:
    return _number(equity.iloc[-1] - 1) if not equity.empty else None


def _regime_labels(benchmark_price: pd.Series) -> pd.Series:
    average = benchmark_price.rolling(200).mean()
    volatility = benchmark_price.pct_change(fill_method=None).rolling(20).std(ddof=1).mul(
        math.sqrt(252)
    )
    regime = pd.Series("transition", index=benchmark_price.index, dtype="object")
    regime.loc[average.notna() & benchmark_price.ge(average)] = "bull"
    regime.loc[average.notna() & benchmark_price.lt(average)] = "bear"
    regime.loc[volatility.ge(0.30)] = "high_volatility"
    return regime


def _select_threshold(
    frame: pd.DataFrame,
    train_start: int,
    train_end: int,
    *,
    mode: str,
    costs: dict[str, float],
) -> tuple[int, float | None]:
    candidates: list[tuple[float, float, int, float | None]] = []
    for threshold in THRESHOLD_CANDIDATES:
        desired = _desired_position(frame["score"], threshold, mode)
        position = desired.shift(1).fillna(0.0)
        modeled = _apply_cost_model(frame["asset_return"], position, **costs)
        window = modeled.iloc[train_start:train_end]
        exposure = window["position"].abs().mean()
        sharpe = _sharpe(window["strategy_return"])
        total_return = window["strategy_return"].add(1).prod() - 1
        objective = float(sharpe) if sharpe is not None and exposure >= 0.05 else -1_000_000.0
        candidates.append((objective, float(total_return), -threshold, sharpe))
    best = max(candidates)
    return -best[2], best[3]


def _summary_metrics(frame: pd.DataFrame, folds: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_returns = frame["strategy_return"]
    strategy_equity = frame["strategy_equity"]
    asset_equity = frame["asset_equity"]
    benchmark_equity = frame["benchmark_equity"]
    annualized_volatility = strategy_returns.std(ddof=1) * math.sqrt(252)
    annualized_return = (
        strategy_equity.iloc[-1] ** (252 / max(1, len(frame) - 1)) - 1
        if strategy_equity.iloc[-1] > 0
        else None
    )
    fold_returns = [float(fold["oos_return"]) for fold in folds if fold["oos_return"] is not None]
    thresholds = [float(fold["selected_threshold"]) for fold in folds]
    profitable_fold_rate = (
        sum(value > 0 for value in fold_returns) / len(fold_returns) if fold_returns else None
    )
    threshold_stability = None
    if thresholds:
        threshold_stability = 1 - min(1.0, float(pd.Series(thresholds).std(ddof=0)) / 30)
    total_return = _return(strategy_equity)
    benchmark_return = _return(benchmark_equity)
    asset_return = _return(asset_equity)
    return {
        "oos_return": total_return,
        "asset_hold_return": asset_return,
        "benchmark_return": benchmark_return,
        "alpha_vs_asset": _number(total_return - asset_return)
        if total_return is not None and asset_return is not None
        else None,
        "alpha_vs_benchmark": _number(total_return - benchmark_return)
        if total_return is not None and benchmark_return is not None
        else None,
        "annualized_return": _number(annualized_return),
        "annualized_volatility": _number(annualized_volatility),
        "sharpe_ratio": _sharpe(strategy_returns),
        "max_drawdown": _maximum_drawdown(strategy_equity),
        "asset_max_drawdown": _maximum_drawdown(asset_equity),
        "benchmark_max_drawdown": _maximum_drawdown(benchmark_equity),
        "fold_count": len(folds),
        "profitable_fold_rate": _number(profitable_fold_rate),
        "threshold_stability": _number(threshold_stability),
        "average_threshold": _number(sum(thresholds) / len(thresholds), 2)
        if thresholds
        else None,
        "exposure": _number(frame["position"].abs().mean()),
        "turnover": _number(frame["turnover"].sum()),
        "execution_cost": _number(frame["execution_cost"].sum()),
        "borrow_cost": _number(frame["borrow_cost"].sum()),
    }


def _robustness_label(metrics: dict[str, Any]) -> str:
    positive = sum(
        (
            (metrics.get("sharpe_ratio") or 0) > 0,
            (metrics.get("alpha_vs_asset") or 0) > 0,
            (metrics.get("alpha_vs_benchmark") or 0) > 0,
            (metrics.get("profitable_fold_rate") or 0) >= 0.6,
            (metrics.get("threshold_stability") or 0) >= 0.7,
        )
    )
    return "promising" if positive >= 4 else "mixed" if positive >= 2 else "weak"


def validate_signal_frames(
    asset_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    symbol: str,
    benchmark_symbol: str,
    as_of_date: str,
    *,
    sessions: int = 504,
    train_sessions: int = 252,
    test_sessions: int = 63,
    mode: str = "long_only",
    commission_bps: float = 2.0,
    slippage_bps: float = 3.0,
    spread_bps: float = 4.0,
    short_borrow_bps: float = 300.0,
) -> dict[str, Any]:
    """Validate threshold selection on sequential unseen blocks."""
    if mode not in SUPPORTED_BACKTEST_MODES:
        raise ValueError(f"Unsupported validation mode: {mode}")
    requested_sessions = max(126, min(int(sessions), 1500))
    train_size = max(126, min(int(train_sessions), 756))
    test_size = max(21, min(int(test_sessions), 252))
    costs = {
        "commission_bps": max(0.0, min(float(commission_bps), 100.0)),
        "slippage_bps": max(0.0, min(float(slippage_bps), 100.0)),
        "spread_bps": max(0.0, min(float(spread_bps), 200.0)),
        "short_borrow_bps": max(0.0, min(float(short_borrow_bps), 5000.0)),
    }

    asset, asset_basis = _clean_price_frame(asset_frame, symbol, as_of_date)
    benchmark, benchmark_basis = _clean_price_frame(
        benchmark_frame, benchmark_symbol, as_of_date
    )
    aligned = asset.rename(columns={"Price": "asset_price"}).merge(
        benchmark.rename(columns={"Price": "benchmark_price"}), on="Date", how="inner"
    )
    minimum_bars = train_size + max(test_size, 63)
    if len(aligned) < minimum_bars:
        raise NoMarketDataError(
            symbol,
            symbol,
            f"validation needs at least {minimum_bars} aligned bars, received {len(aligned)}",
        )

    aligned["score"] = quantitative_score_series(aligned["asset_price"])["score"]
    aligned["asset_return"] = aligned["asset_price"].pct_change(fill_method=None).fillna(0.0)
    aligned["benchmark_return"] = aligned["benchmark_price"].pct_change(
        fill_method=None
    ).fillna(0.0)
    aligned["regime"] = _regime_labels(aligned["benchmark_price"])
    evaluation_start = max(train_size, len(aligned) - requested_sessions)
    oos_position = pd.Series(0.0, index=aligned.index)
    fold_id = pd.Series(0, index=aligned.index, dtype=int)
    folds: list[dict[str, Any]] = []

    for number, start in enumerate(range(evaluation_start, len(aligned), test_size), start=1):
        end = min(len(aligned), start + test_size)
        train_start = start - train_size
        threshold, train_sharpe = _select_threshold(
            aligned,
            train_start,
            start,
            mode=mode,
            costs=costs,
        )
        desired = _desired_position(aligned["score"], threshold, mode)
        position = desired.shift(1).fillna(0.0)
        oos_position.iloc[start:end] = position.iloc[start:end]
        fold_id.iloc[start:end] = number
        folds.append(
            {
                "fold": number,
                "train_start": aligned["Date"].iloc[train_start].isoformat(),
                "train_end": aligned["Date"].iloc[start - 1].isoformat(),
                "test_start": aligned["Date"].iloc[start].isoformat(),
                "test_end": aligned["Date"].iloc[end - 1].isoformat(),
                "selected_threshold": threshold,
                "train_sharpe": train_sharpe,
            }
        )

    modeled = _apply_cost_model(aligned["asset_return"], oos_position, **costs)
    for column in modeled:
        aligned[column] = modeled[column]
    aligned["fold"] = fold_id
    oos = aligned.iloc[evaluation_start:].copy().reset_index(drop=True)
    for column in (
        "asset_return",
        "benchmark_return",
        "strategy_return",
        "gross_return",
        "execution_cost",
        "borrow_cost",
        "turnover",
    ):
        oos.loc[0, column] = 0.0
    oos["strategy_equity"] = oos["strategy_return"].add(1).cumprod()
    oos["asset_equity"] = oos["asset_return"].add(1).cumprod()
    oos["benchmark_equity"] = oos["benchmark_return"].add(1).cumprod()

    for fold in folds:
        window = oos[oos["fold"].eq(fold["fold"])]
        fold_equity = window["strategy_return"].add(1).cumprod()
        fold_benchmark = window["benchmark_return"].add(1).cumprod()
        fold["oos_return"] = _return(fold_equity)
        fold["benchmark_return"] = _return(fold_benchmark)
        fold["oos_sharpe"] = _sharpe(window["strategy_return"])
        fold["max_drawdown"] = _maximum_drawdown(fold_equity)

    regimes = []
    for regime, window in oos.groupby("regime", sort=False):
        regimes.append(
            {
                "regime": regime,
                "sessions": len(window),
                "strategy_return": _number(window["strategy_return"].add(1).prod() - 1),
                "asset_return": _number(window["asset_return"].add(1).prod() - 1),
                "benchmark_return": _number(window["benchmark_return"].add(1).prod() - 1),
                "sharpe_ratio": _sharpe(window["strategy_return"]),
            }
        )

    metrics = _summary_metrics(oos, folds)
    points = [
        {
            "date": row["Date"].isoformat(),
            "strategy_equity": _number(row["strategy_equity"]),
            "asset_equity": _number(row["asset_equity"]),
            "benchmark_equity": _number(row["benchmark_equity"]),
            "position": int(row["position"]),
            "score": _number(row["score"], 2),
            "fold": int(row["fold"]),
            "regime": row["regime"],
        }
        for _, row in oos.iterrows()
    ]
    return {
        "symbol": symbol.upper(),
        "benchmark_symbol": benchmark_symbol.upper(),
        "as_of_date": as_of_date,
        "first_session": points[0]["date"],
        "last_session": points[-1]["date"],
        "oos_sessions": len(points),
        "model_version": MODEL_VERSION,
        "price_basis": {"asset": asset_basis, "benchmark": benchmark_basis},
        "configuration": {
            "requested_sessions": requested_sessions,
            "train_sessions": train_size,
            "test_sessions": test_size,
            "mode": mode,
            **costs,
            "threshold_candidates": list(THRESHOLD_CANDIDATES),
        },
        "metrics": metrics,
        "robustness": _robustness_label(metrics),
        "folds": folds,
        "regimes": regimes,
        "points": points,
        "methodology": (
            "Each fold selects a signal threshold using only its trailing training window, then "
            "locks that threshold for the next unseen test block. All signals trade one session later."
        ),
        "limitations": (
            "Research validation, not a forecast or execution record. Adjusted closes model source-"
            "provided splits/dividends; taxes, market impact and borrow availability remain excluded."
        ),
    }


def build_validation_report(
    symbol: str,
    benchmark_symbol: str,
    as_of_date: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch point-in-time histories and run rolling validation."""
    sessions = max(126, min(int(kwargs.get("sessions", 504)), 1500))
    train_sessions = max(126, min(int(kwargs.get("train_sessions", 252)), 756))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    calendar_days = min(6000, max(1000, int((sessions + train_sessions + 260) * 1.7)))
    start_date = (as_of - timedelta(days=calendar_days)).isoformat()
    asset_payload = route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
    if benchmark_symbol == symbol:
        benchmark_payload = asset_payload
    else:
        benchmark_payload = route_to_vendor(
            "get_stock_data", benchmark_symbol, start_date, as_of_date
        )
    return validate_signal_frames(
        parse_ohlcv_payload(asset_payload),
        parse_ohlcv_payload(benchmark_payload),
        symbol,
        benchmark_symbol,
        as_of_date,
        **kwargs,
    )
