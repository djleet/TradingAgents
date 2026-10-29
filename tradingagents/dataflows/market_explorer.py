"""Point-in-time chart and cross-asset data for the dashboard Research Cockpit."""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from tradingagents.dataflows.quantitative import (
    calculate_quantitative_snapshot,
    compute_quantitative_snapshot,
    rsi_series,
)
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import parse_ohlcv_payload

logger = logging.getLogger(__name__)


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def build_market_chart(
    symbol: str,
    as_of_date: str,
    *,
    sessions: int = 260,
) -> dict[str, Any]:
    """Return display-ready OHLCV, trend lines and the independent snapshot."""
    session_count = max(60, min(int(sessions), 500))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    calendar_days = min(2000, max(550, int(session_count * 1.8)))
    start_date = (as_of - timedelta(days=calendar_days)).isoformat()
    payload = route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
    frame = parse_ohlcv_payload(payload)
    snapshot = calculate_quantitative_snapshot(frame.copy(), symbol, as_of_date)

    frame = frame[frame["Date"] <= as_of].drop_duplicates("Date", keep="last")
    frame = frame.dropna(subset=["Close"])
    frame = frame[frame["Close"] > 0].sort_values("Date").copy()
    close = frame["Close"].astype(float)
    frame["SMA20"] = close.rolling(20).mean()
    frame["SMA50"] = close.rolling(50).mean()
    frame["SMA200"] = close.rolling(200).mean()
    frame["RSI14"] = rsi_series(close)

    points = []
    for _, row in frame.tail(session_count).iterrows():
        points.append(
            {
                "date": row["Date"].isoformat(),
                "open": _number(row["Open"]),
                "high": _number(row["High"]),
                "low": _number(row["Low"]),
                "close": _number(row["Close"]),
                "volume": _number(row["Volume"], 2),
                "sma20": _number(row["SMA20"]),
                "sma50": _number(row["SMA50"]),
                "sma200": _number(row["SMA200"]),
                "rsi14": _number(row["RSI14"], 2),
            }
        )

    first_close = points[0]["close"] if points else None
    last_close = points[-1]["close"] if points else None
    visible_return = (
        last_close / first_close - 1
        if first_close is not None and last_close is not None and first_close > 0
        else None
    )
    return {
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "visible_sessions": len(points),
        "visible_return": _number(visible_return),
        "bars": points,
        "snapshot": snapshot,
        "display_policy": "Raw point-in-time OHLCV; moving averages use prior visible history.",
    }


def _comparison_item(symbol: str, as_of_date: str, lookback_calendar_days: int) -> dict[str, Any]:
    snapshot = compute_quantitative_snapshot(
        symbol,
        as_of_date,
        lookback_calendar_days=lookback_calendar_days,
    )
    metrics = snapshot["metrics"]
    return {
        "symbol": snapshot["symbol"],
        "latest_bar_date": snapshot["latest_bar_date"],
        "latest_close": snapshot["latest_close"],
        "score": snapshot["score"],
        "label": snapshot["label"],
        "confidence": snapshot["confidence"],
        "risk_regime": snapshot["risk_regime"],
        "momentum_20": metrics["momentum_20"],
        "momentum_60": metrics["momentum_60"],
        "annualized_volatility_60": metrics["annualized_volatility_60"],
        "max_drawdown_252": metrics["max_drawdown_252"],
        "rsi_14": metrics["rsi_14"],
    }


def compare_market_snapshots(
    symbols: list[str],
    as_of_date: str,
    *,
    lookback_calendar_days: int = 550,
) -> dict[str, Any]:
    """Compare independent signals concurrently, preserving partial results."""
    days = max(120, min(int(lookback_calendar_days), 2000))

    def collect(symbol: str) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        try:
            return _comparison_item(symbol, as_of_date, days), None
        except Exception as exc:  # noqa: BLE001 - one unavailable asset must not erase the table
            logger.warning("Comparison data unavailable for %s: %s", symbol, exc)
            return None, {"symbol": symbol, "message": "Point-in-time market data unavailable"}

    with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as executor:
        collected = list(executor.map(collect, symbols))
    items = [item for item, _ in collected if item is not None]
    errors = [error for _, error in collected if error is not None]
    return {
        "as_of_date": as_of_date,
        "items": items,
        "errors": errors,
        "methodology": "All assets use the same transparent trend model and as-of date.",
    }
