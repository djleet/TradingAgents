"""Walk-forward simulation for the deterministic transparent-trend signal."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.quantitative import MODEL_VERSION, quantitative_score_series
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import parse_ohlcv_payload

SUPPORTED_BACKTEST_MODES = {"long_only", "long_short"}


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _annualized_return(final_equity: float, periods: int) -> float | None:
    if final_equity <= 0 or periods <= 0:
        return None
    return final_equity ** (252 / periods) - 1


def _maximum_drawdown(equity: pd.Series) -> float | None:
    if equity.empty:
        return None
    return _number((equity.div(equity.cummax()).sub(1)).min())


def select_price_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Prefer a complete adjusted-close series so splits/dividends are reflected."""
    raw = pd.to_numeric(frame["Close"], errors="coerce").astype(float)
    if "Adj Close" in frame:
        adjusted = pd.to_numeric(frame["Adj Close"], errors="coerce").astype(float)
        if adjusted.notna().all() and adjusted.gt(0).all():
            return adjusted, "adjusted_close"
    return raw, "raw_close"


def _performance_metrics(
    frame: pd.DataFrame,
    *,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
    short_borrow_bps: float,
    mode: str,
) -> dict[str, Any]:
    strategy_returns = frame["strategy_return"]
    strategy_equity = frame["strategy_equity"]
    benchmark_equity = frame["benchmark_equity"]
    periods = max(0, len(frame) - 1)
    annualized_volatility = strategy_returns.std(ddof=1) * math.sqrt(252)
    sharpe = (
        strategy_returns.mean() * 252 / annualized_volatility
        if annualized_volatility and math.isfinite(annualized_volatility)
        else None
    )
    active = frame["position"].ne(0)
    active_returns = frame.loc[active, "gross_strategy_return"]
    transitions = frame["position"].diff().fillna(0).ne(0)
    return {
        "total_return": _number(strategy_equity.iloc[-1] - 1),
        "benchmark_return": _number(benchmark_equity.iloc[-1] - 1),
        "annualized_return": _number(_annualized_return(strategy_equity.iloc[-1], periods)),
        "annualized_volatility": _number(annualized_volatility),
        "sharpe_ratio": _number(sharpe),
        "max_drawdown": _maximum_drawdown(strategy_equity),
        "benchmark_max_drawdown": _maximum_drawdown(benchmark_equity),
        "trade_count": int(transitions.sum()),
        "exposure": _number(active.mean()),
        "hit_rate": _number(active_returns.gt(0).mean()) if not active_returns.empty else None,
        "turnover": _number(frame["turnover"].sum()),
        "estimated_fees": _number(frame["execution_cost_return"].sum()),
        "estimated_short_borrow": _number(frame["borrow_cost_return"].sum()),
        "fee_bps": _number(fee_bps, 2),
        "slippage_bps": _number(slippage_bps, 2),
        "spread_bps": _number(spread_bps, 2),
        "short_borrow_bps": _number(short_borrow_bps, 2),
        "mode": mode,
    }


def simulate_signal_backtest(
    frame: pd.DataFrame,
    symbol: str,
    as_of_date: str,
    *,
    sessions: int = 504,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    short_borrow_bps: float = 0.0,
    mode: str = "long_only",
) -> dict[str, Any]:
    """Simulate next-session execution over an existing point-in-time frame."""
    if mode not in SUPPORTED_BACKTEST_MODES:
        raise ValueError(f"Unsupported backtest mode: {mode}")
    session_count = max(120, min(int(sessions), 1500))
    fee = max(0.0, min(float(fee_bps), 100.0))
    slippage = max(0.0, min(float(slippage_bps), 100.0))
    spread = max(0.0, min(float(spread_bps), 200.0))
    short_borrow = max(0.0, min(float(short_borrow_bps), 5000.0))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    clean = frame[frame["Date"] <= as_of].drop_duplicates("Date", keep="last")
    clean = clean.dropna(subset=["Close"])
    clean = clean[clean["Close"] > 0].sort_values("Date").copy()
    if len(clean) < 90:
        raise NoMarketDataError(
            symbol,
            symbol,
            f"walk-forward backtest needs at least 90 bars, received {len(clean)}",
        )

    clean["Price"], price_basis = select_price_series(clean)
    scores = quantitative_score_series(clean["Price"])
    clean["score"] = scores["score"].where(pd.Series(range(len(clean)), index=clean.index).ge(59))
    clean["desired_signal"] = 0.0
    clean.loc[clean["score"].ge(20), "desired_signal"] = 1.0
    if mode == "long_short":
        clean.loc[clean["score"].le(-20), "desired_signal"] = -1.0

    # Today's close can only set tomorrow's exposure. This one-row shift is the
    # key anti-lookahead boundary for the whole simulation.
    clean["position"] = clean["desired_signal"].shift(1).fillna(0.0)
    clean["market_return"] = clean["Price"].pct_change(fill_method=None).fillna(0.0)
    clean["turnover"] = clean["position"].diff().abs().fillna(clean["position"].abs())
    one_way_execution_bps = fee + slippage + spread / 2
    clean["execution_cost_return"] = clean["turnover"].mul(one_way_execution_bps / 10_000)
    clean["borrow_cost_return"] = clean["position"].clip(upper=0).abs().mul(
        short_borrow / 10_000 / 252
    )
    clean["gross_strategy_return"] = clean["position"].mul(clean["market_return"])
    clean["strategy_return"] = (
        clean["gross_strategy_return"]
        .sub(clean["execution_cost_return"])
        .sub(clean["borrow_cost_return"])
    )

    result = clean.tail(session_count).copy()
    # Normalize both curves to the same visible starting capital.
    result.iloc[0, result.columns.get_loc("market_return")] = 0.0
    result.iloc[0, result.columns.get_loc("strategy_return")] = 0.0
    result.iloc[0, result.columns.get_loc("gross_strategy_return")] = 0.0
    result.iloc[0, result.columns.get_loc("execution_cost_return")] = 0.0
    result.iloc[0, result.columns.get_loc("borrow_cost_return")] = 0.0
    result.iloc[0, result.columns.get_loc("turnover")] = 0.0
    result["strategy_equity"] = result["strategy_return"].add(1).cumprod()
    result["benchmark_equity"] = result["market_return"].add(1).cumprod()

    points = [
        {
            "date": row["Date"].isoformat(),
            "close": _number(row["Price"]),
            "score": _number(row["score"], 2),
            "desired_signal": int(row["desired_signal"]),
            "position": int(row["position"]),
            "strategy_equity": _number(row["strategy_equity"]),
            "benchmark_equity": _number(row["benchmark_equity"]),
        }
        for _, row in result.iterrows()
    ]
    return {
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "first_session": points[0]["date"],
        "last_session": points[-1]["date"],
        "sessions": len(points),
        "model_version": MODEL_VERSION,
        "price_basis": price_basis,
        "thresholds": {"bullish": 20, "bearish": -20},
        "metrics": _performance_metrics(
            result,
            fee_bps=fee,
            slippage_bps=slippage,
            spread_bps=spread,
            short_borrow_bps=short_borrow,
            mode=mode,
        ),
        "points": points,
        "methodology": (
            "Walk-forward rule simulation: each closing score is applied from the next session; "
            "commissions, half-spread and slippage are deducted when exposure changes; "
            "short borrow is charged daily."
        ),
        "limitations": (
            "Historical simulation, not a forecast or execution record. Taxes, market impact and "
            "borrow availability are not modeled; corporate actions depend on source adjustments."
        ),
    }


def build_signal_backtest(
    symbol: str,
    as_of_date: str,
    *,
    sessions: int = 504,
    fee_bps: float = 5.0,
    slippage_bps: float = 0.0,
    spread_bps: float = 0.0,
    short_borrow_bps: float = 0.0,
    mode: str = "long_only",
) -> dict[str, Any]:
    """Fetch historical prices and run the transparent signal walk-forward."""
    requested = max(120, min(int(sessions), 1500))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    calendar_days = min(4000, max(800, int((requested + 260) * 1.65)))
    start_date = (as_of - timedelta(days=calendar_days)).isoformat()
    payload = route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
    return simulate_signal_backtest(
        parse_ohlcv_payload(payload),
        symbol,
        as_of_date,
        sessions=requested,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        short_borrow_bps=short_borrow_bps,
        mode=mode,
    )
