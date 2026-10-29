"""Deterministic market signal independent of the LLM committee.

The score is intentionally simple and transparent. It is not a trained price
forecast: it combines medium-term momentum, moving-average trend and RSI, then
reports volatility and drawdown separately as risk. This gives the portfolio
decision an independently reproducible cross-check instead of another natural-
language opinion.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import (
    capture_quantitative_snapshot,
    parse_ohlcv_payload,
)

MODEL_VERSION = "transparent-trend-v1"

SCORE_COMPONENTS = (
    ("momentum_20", "20-session momentum", 0.25),
    ("momentum_60", "60-session momentum", 0.25),
    ("price_vs_sma_50", "price vs 50-session mean", 0.20),
    ("price_vs_sma_200", "price vs 200-session mean", 0.15),
    ("rsi_trend", "RSI trend", 0.15),
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _period_return(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    prior = close.iloc[-sessions - 1]
    return _finite(close.iloc[-1] / prior - 1) if prior else None


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Return the RSI series used by both snapshots and walk-forward tests."""
    changes = close.diff()
    gains = changes.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    losses = (-changes.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = gains / losses.replace(0, float("nan"))
    result = 100 - 100 / (1 + relative_strength)
    return result.mask(losses.eq(0), 100.0)


def _max_drawdown(close: pd.Series, sessions: int = 252) -> float | None:
    window = close.tail(sessions)
    if len(window) < 2:
        return None
    return _finite((window / window.cummax() - 1).min())


def quantitative_score_series(close: pd.Series) -> pd.DataFrame:
    """Calculate every historical score using only information available then."""
    prices = pd.to_numeric(close, errors="coerce").astype(float)
    rsi = rsi_series(prices)
    normalized = pd.DataFrame(
        {
            "momentum_20": prices.pct_change(20, fill_method=None).div(0.10).clip(-1, 1),
            "momentum_60": prices.pct_change(60, fill_method=None).div(0.20).clip(-1, 1),
            "price_vs_sma_50": (
                prices.div(prices.rolling(50).mean()).sub(1).div(0.10).clip(-1, 1)
            ),
            "price_vs_sma_200": (
                prices.div(prices.rolling(200).mean()).sub(1).div(0.20).clip(-1, 1)
            ),
            "rsi_trend": rsi.sub(50).div(25).clip(-1, 1),
        },
        index=prices.index,
    )
    weighted_sum = pd.Series(0.0, index=prices.index)
    available_weight = pd.Series(0.0, index=prices.index)
    for key, _name, weight in SCORE_COMPONENTS:
        present = normalized[key].notna()
        weighted_sum = weighted_sum.add(normalized[key].fillna(0).mul(weight))
        available_weight = available_weight.add(present.astype(float).mul(weight))
    normalized["available_weight"] = available_weight
    normalized["score"] = weighted_sum.div(available_weight.replace(0, float("nan"))).mul(100)
    normalized["rsi_14"] = rsi
    return normalized


def calculate_quantitative_snapshot(
    frame: pd.DataFrame,
    symbol: str,
    as_of_date: str,
) -> dict[str, Any]:
    """Calculate and archive a transparent signal from an existing OHLCV frame."""
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    if frame.empty:
        raise NoMarketDataError(symbol, symbol, "quantitative model could not parse OHLCV data")

    frame = frame[frame["Date"] <= as_of].drop_duplicates("Date", keep="last")
    frame = frame.dropna(subset=["Close"])
    frame = frame[frame["Close"] > 0].sort_values("Date")
    if len(frame) < 30:
        raise NoMarketDataError(
            symbol,
            symbol,
            f"quantitative model needs at least 30 bars, received {len(frame)}",
        )

    close = frame["Close"].astype(float)
    returns = close.pct_change(fill_method=None).dropna()
    latest_close = _finite(close.iloc[-1])
    momentum = {period: _period_return(close, period) for period in (20, 60, 126, 252)}
    volatility = {
        period: _finite(returns.tail(period).std(ddof=1) * math.sqrt(252))
        if len(returns.tail(period)) >= min(period, 20)
        else None
        for period in (20, 60)
    }
    averages = {
        period: _finite(close.tail(period).mean()) if len(close) >= period else None
        for period in (50, 200)
    }
    score_history = quantitative_score_series(close)
    latest_score = score_history.iloc[-1]
    rsi_14 = _finite(latest_score["rsi_14"])
    drawdown = _max_drawdown(close)

    components: list[tuple[str, float, float]] = []
    for key, name, weight in SCORE_COMPONENTS:
        value = _finite(latest_score[key])
        if value is not None:
            components.append((name, value, weight))

    available_weight = sum(weight for _, _, weight in components)
    score = (
        100 * sum(value * weight for _, value, weight in components) / available_weight
        if available_weight
        else 0.0
    )
    label = "bullish" if score >= 20 else "bearish" if score <= -20 else "neutral"
    confidence = min(1.0, len(frame) / 252) * available_weight

    vol_60 = volatility[60]
    if (vol_60 is not None and vol_60 >= 0.60) or (drawdown is not None and drawdown <= -0.35):
        risk_regime = "high"
    elif (vol_60 is not None and vol_60 >= 0.35) or (drawdown is not None and drawdown <= -0.20):
        risk_regime = "elevated"
    else:
        risk_regime = "normal"

    snapshot: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "latest_bar_date": frame["Date"].iloc[-1].isoformat(),
        "bars_used": len(frame),
        "latest_close": latest_close,
        "score": round(score, 2),
        "label": label,
        "confidence": round(confidence, 4),
        "risk_regime": risk_regime,
        "metrics": {
            "momentum_20": momentum[20],
            "momentum_60": momentum[60],
            "momentum_126": momentum[126],
            "momentum_252": momentum[252],
            "annualized_volatility_20": volatility[20],
            "annualized_volatility_60": volatility[60],
            "max_drawdown_252": drawdown,
            "sma_50": averages[50],
            "sma_200": averages[200],
            "rsi_14": rsi_14,
        },
        "components": [
            {"name": name, "normalized_value": round(value, 4), "weight": weight}
            for name, value, weight in components
        ],
        "methodology": (
            "Transparent trend composite; risk is reported separately and the score is not "
            "a return forecast or guarantee."
        ),
    }
    capture_quantitative_snapshot(snapshot)
    return snapshot


def compute_quantitative_snapshot(
    symbol: str,
    as_of_date: str,
    *,
    lookback_calendar_days: int = 550,
) -> dict[str, Any]:
    """Fetch point-in-time OHLCV and calculate a transparent signal snapshot."""
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    days = max(120, min(int(lookback_calendar_days), 2000))
    start_date = (as_of - timedelta(days=days)).isoformat()
    payload = route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
    return calculate_quantitative_snapshot(parse_ohlcv_payload(payload), symbol, as_of_date)


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2%}"


def render_quantitative_snapshot(snapshot: dict[str, Any]) -> str:
    metrics = snapshot["metrics"]
    lines = [
        f"## Independent quantitative snapshot for {snapshot['symbol']}",
        "",
        f"- Analysis date: {snapshot['as_of_date']}",
        f"- Latest bar: {snapshot['latest_bar_date']}",
        f"- Bars used: {snapshot['bars_used']}",
        f"- Composite: **{snapshot['score']:+.1f}/100 ({snapshot['label']})**",
        f"- Model confidence/coverage: {snapshot['confidence']:.0%}",
        f"- Risk regime: **{snapshot['risk_regime']}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Latest close | {snapshot['latest_close']:.2f} |",
        f"| 20-session return | {_percent(metrics['momentum_20'])} |",
        f"| 60-session return | {_percent(metrics['momentum_60'])} |",
        f"| 126-session return | {_percent(metrics['momentum_126'])} |",
        f"| 252-session return | {_percent(metrics['momentum_252'])} |",
        f"| 20-session annualized volatility | {_percent(metrics['annualized_volatility_20'])} |",
        f"| 60-session annualized volatility | {_percent(metrics['annualized_volatility_60'])} |",
        f"| 252-session maximum drawdown | {_percent(metrics['max_drawdown_252'])} |",
        f"| 50-session mean | {metrics['sma_50']:.2f} |" if metrics["sma_50"] is not None else "| 50-session mean | N/A |",
        f"| 200-session mean | {metrics['sma_200']:.2f} |" if metrics["sma_200"] is not None else "| 200-session mean | N/A |",
        f"| RSI(14) | {metrics['rsi_14']:.2f} |" if metrics["rsi_14"] is not None else "| RSI(14) | N/A |",
        "",
        "This deterministic cross-check is independent of the LLM debate. Treat it as one "
        "input, not as a prediction or trading guarantee.",
    ]
    return "\n".join(lines)


def build_quantitative_report(symbol: str, as_of_date: str, lookback_calendar_days: int = 550) -> str:
    return render_quantitative_snapshot(
        compute_quantitative_snapshot(
            symbol,
            as_of_date,
            lookback_calendar_days=lookback_calendar_days,
        )
    )
