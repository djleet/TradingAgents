"""Point-in-time portfolio weights with an optional PyPortfolioOpt backend.

The feature stays usable in the lean install through an explicit inverse-
volatility fallback. Installing the ``portfolio`` extra adds minimum-variance,
maximum-Sharpe and hierarchical risk-parity optimizers.
"""

from __future__ import annotations

import importlib.util
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.dataflows.warehouse import parse_ohlcv_payload

SUPPORTED_METHODS = {
    "inverse_volatility",
    "minimum_variance",
    "maximum_sharpe",
    "hierarchical_risk_parity",
}


def pypfopt_available() -> bool:
    return importlib.util.find_spec("pypfopt") is not None


def load_price_matrix(
    symbols: list[str],
    as_of_date: str,
    *,
    lookback_calendar_days: int = 550,
) -> pd.DataFrame:
    """Fetch a common point-in-time close-price matrix for the requested assets."""
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    days = max(120, min(int(lookback_calendar_days), 2000))
    start_date = (as_of - timedelta(days=days)).isoformat()
    series: list[pd.Series] = []
    for symbol in symbols:
        payload = route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
        frame = parse_ohlcv_payload(payload)
        frame = frame[(frame["Date"] <= as_of) & frame["Close"].notna()]
        frame = frame[frame["Close"] > 0].drop_duplicates("Date", keep="last")
        if frame.empty:
            raise ValueError(f"No usable point-in-time prices for {symbol}")
        item = frame.set_index("Date")["Close"].astype(float).rename(symbol)
        series.append(item)

    prices = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if len(prices) < 60:
        raise ValueError(
            f"Portfolio optimization needs at least 60 common sessions; received {len(prices)}"
        )
    return prices


def _inverse_volatility_weights(prices: pd.DataFrame) -> dict[str, float]:
    volatility = prices.pct_change(fill_method=None).dropna().std(ddof=1)
    if volatility.isna().any() or (volatility <= 0).any():
        raise ValueError("Cannot optimize assets with zero or unavailable return volatility")
    inverse = 1 / volatility
    weights = inverse / inverse.sum()
    return {symbol: float(weight) for symbol, weight in weights.items()}


def _pypfopt_weights(prices: pd.DataFrame, method: str) -> dict[str, float]:
    if method == "hierarchical_risk_parity":
        from pypfopt import HRPOpt

        optimizer = HRPOpt(prices.pct_change(fill_method=None).dropna())
        optimizer.optimize()
        return {key: float(value) for key, value in optimizer.clean_weights().items()}

    from pypfopt import EfficientFrontier, expected_returns, risk_models

    expected = expected_returns.mean_historical_return(prices, frequency=252)
    covariance = risk_models.sample_cov(prices, frequency=252)
    optimizer = EfficientFrontier(expected, covariance, weight_bounds=(0, 1))
    if method == "maximum_sharpe":
        optimizer.max_sharpe()
    else:
        optimizer.min_volatility()
    return {key: float(value) for key, value in optimizer.clean_weights().items()}


def _portfolio_metrics(prices: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    returns = prices.pct_change(fill_method=None).dropna()
    aligned = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    annual_return = float(returns.mean().dot(aligned) * 252)
    variance = float(aligned.dot(returns.cov() * 252).dot(aligned))
    annual_volatility = math.sqrt(max(0.0, variance))
    sharpe = (annual_return - 0.02) / annual_volatility if annual_volatility else 0.0
    return {
        "historical_annual_return": round(annual_return, 6),
        "historical_annual_volatility": round(annual_volatility, 6),
        "historical_sharpe_at_2pct": round(sharpe, 6),
    }


def optimize_portfolio(
    symbols: list[str],
    as_of_date: str,
    *,
    method: str = "minimum_variance",
    lookback_calendar_days: int = 550,
) -> dict[str, Any]:
    """Return long-only weights and transparent historical risk statistics."""
    normalized = list(
        dict.fromkeys(normalize_symbol(symbol.strip()) for symbol in symbols if symbol.strip())
    )
    if not 2 <= len(normalized) <= 30:
        raise ValueError("Portfolio optimization requires 2-30 unique symbols")
    for symbol in normalized:
        safe_ticker_component(symbol)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported optimization method: {method}")

    prices = load_price_matrix(
        normalized,
        as_of_date,
        lookback_calendar_days=lookback_calendar_days,
    )
    warning = None
    backend = "native_inverse_volatility"
    applied_method = "inverse_volatility"
    weights: dict[str, float]
    if method != "inverse_volatility" and pypfopt_available():
        try:
            weights = _pypfopt_weights(prices, method)
            backend = "pypfopt"
            applied_method = method
        except Exception as exc:  # noqa: BLE001 - optional solver failures must use fallback
            warning = f"PyPortfolioOpt could not solve this sample; used inverse volatility: {exc}"
            weights = _inverse_volatility_weights(prices)
    else:
        weights = _inverse_volatility_weights(prices)
        if method != "inverse_volatility":
            warning = (
                "PyPortfolioOpt is not installed; used the deterministic inverse-volatility fallback."
            )

    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        raise ValueError("Optimizer returned no positive portfolio weights")
    weights = {symbol: round(max(0.0, weights.get(symbol, 0.0)) / total, 6) for symbol in normalized}
    rounding_gap = round(1.0 - sum(weights.values()), 6)
    if rounding_gap:
        largest = max(weights, key=weights.get)
        weights[largest] = round(weights[largest] + rounding_gap, 6)

    result: dict[str, Any] = {
        "as_of_date": as_of_date,
        "first_bar_date": prices.index[0].isoformat(),
        "last_bar_date": prices.index[-1].isoformat(),
        "common_sessions": len(prices),
        "requested_method": method,
        "applied_method": applied_method,
        "backend": backend,
        "weights": weights,
        "metrics": _portfolio_metrics(prices, weights),
        "disclaimer": (
            "Historical estimates only; they exclude fees, taxes, slippage and future regime changes."
        ),
    }
    if warning:
        result["warning"] = warning
    return result
