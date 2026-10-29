"""Point-in-time portfolio risk diagnostics for research portfolios.

The module is deliberately execution-free.  It estimates risk from closing
prices available on or before the selected date, keeps cash at zero volatility,
and returns explainable weight changes rather than placing orders.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.portfolio_optimization import load_price_matrix

TRADING_DAYS = 252
CORRELATION_CLUSTER_THRESHOLD = 0.70


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
        raise ValueError("Portfolio risk analysis requires 2-30 positive-weight symbols")
    total = sum(cleaned.values())
    if total > 1.000001:
        raise ValueError("Portfolio weights cannot exceed 100%")
    if total <= 0:
        raise ValueError("Portfolio weights must include positive exposure")
    return cleaned


def _connected_correlation_clusters(
    correlation: pd.DataFrame,
    weights: pd.Series,
    risk_contributions: dict[str, float],
) -> list[dict[str, Any]]:
    remaining = set(correlation.columns)
    clusters: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        frontier = [seed]
        remaining.remove(seed)
        while frontier:
            current = frontier.pop()
            linked = {
                symbol
                for symbol in remaining
                if float(correlation.loc[current, symbol]) >= CORRELATION_CLUSTER_THRESHOLD
            }
            remaining -= linked
            component |= linked
            frontier.extend(sorted(linked))
        clusters.append(sorted(component))

    result = []
    for index, symbols in enumerate(
        sorted(clusters, key=lambda item: sum(float(weights[s]) for s in item), reverse=True),
        start=1,
    ):
        submatrix = correlation.loc[symbols, symbols]
        upper = [
            float(submatrix.iloc[row, column])
            for row in range(len(symbols))
            for column in range(row + 1, len(symbols))
        ]
        result.append(
            {
                "cluster": f"Cluster {index}",
                "symbols": symbols,
                "weight": round(sum(float(weights[s]) for s in symbols), 6),
                "risk_contribution": round(
                    sum(risk_contributions.get(symbol, 0.0) for symbol in symbols), 6
                ),
                "average_correlation": round(sum(upper) / len(upper), 6) if upper else None,
                "is_correlated_group": len(symbols) > 1,
            }
        )
    return result


def _risk_alerts(
    positions: list[dict[str, Any]],
    summary: dict[str, float],
    concentration: dict[str, Any],
    *,
    max_position_weight: float,
    max_component_risk: float,
    annual_volatility_limit: float,
    daily_es_limit: float,
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    def add(
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        current: float,
        limit: float,
        symbol: str | None = None,
    ) -> None:
        alerts.append(
            {
                "type": alert_type,
                "severity": severity,
                "title": title,
                "message": message,
                "symbol": symbol,
                "current": round(current, 6),
                "limit": round(limit, 6),
            }
        )

    for position in positions:
        if position["weight"] > max_position_weight:
            add(
                "position_weight",
                "high" if position["weight"] > max_position_weight * 1.25 else "warning",
                f"{position['symbol']} exceeds the position budget",
                "A single holding represents too much of current net value.",
                position["weight"],
                max_position_weight,
                position["symbol"],
            )
        if position["risk_contribution"] > max_component_risk:
            add(
                "component_risk",
                "high" if position["risk_contribution"] > max_component_risk * 1.25 else "warning",
                f"{position['symbol']} dominates estimated risk",
                "Its volatility and covariance create more risk than its capital weight suggests.",
                position["risk_contribution"],
                max_component_risk,
                position["symbol"],
            )

    if summary["annualized_volatility"] > annual_volatility_limit:
        add(
            "portfolio_volatility",
            "high" if summary["annualized_volatility"] > annual_volatility_limit * 1.25 else "warning",
            "Portfolio volatility exceeds the budget",
            "Historical annualized volatility is above the selected limit.",
            summary["annualized_volatility"],
            annual_volatility_limit,
        )
    if summary["expected_shortfall"] > daily_es_limit:
        add(
            "expected_shortfall",
            "high" if summary["expected_shortfall"] > daily_es_limit * 1.25 else "warning",
            "Tail-loss budget is exceeded",
            "Average loss on the worst historical days is above the daily budget.",
            summary["expected_shortfall"],
            daily_es_limit,
        )
    if concentration["maximum_pair_correlation"] > 0.85:
        add(
            "correlation",
            "warning",
            "Diversification may disappear under pressure",
            "At least one pair has a very high historical return correlation.",
            concentration["maximum_pair_correlation"],
            0.85,
        )
    if abs(summary["market_beta"]) > 1.30:
        add(
            "market_beta",
            "warning",
            "Market sensitivity is elevated",
            "The portfolio historically moved more than the benchmark.",
            abs(summary["market_beta"]),
            1.30,
        )
    return sorted(alerts, key=lambda item: (item["severity"] != "high", item["type"]))


def analyze_portfolio_risk(
    weights: dict[str, float],
    as_of_date: str,
    *,
    benchmark_symbol: str = "SPY",
    lookback_calendar_days: int = 550,
    confidence: float = 0.95,
    portfolio_value: float | None = None,
    market_shock: float = -0.10,
    max_position_weight: float = 0.25,
    max_component_risk: float = 0.35,
    annual_volatility_limit: float = 0.25,
    daily_es_limit: float = 0.03,
) -> dict[str, Any]:
    """Build transparent historical risk diagnostics for long-only weights."""
    cleaned = _clean_weights(weights)
    if not 0.90 <= confidence <= 0.99:
        raise ValueError("Confidence must be between 90% and 99%")
    if not -0.50 <= market_shock <= -0.01:
        raise ValueError("Market shock must be between -50% and -1%")
    benchmark = normalize_symbol(benchmark_symbol.strip())
    safe_ticker_component(benchmark, max_len=15)
    symbols = list(cleaned)
    price_symbols = list(dict.fromkeys([*symbols, benchmark]))
    prices = load_price_matrix(
        price_symbols,
        as_of_date,
        lookback_calendar_days=lookback_calendar_days,
    )
    returns = prices.pct_change(fill_method=None).dropna()
    asset_returns = returns[symbols]
    weight_series = pd.Series(cleaned, dtype=float).reindex(symbols)
    invested_weight = float(weight_series.sum())
    cash_weight = max(0.0, 1.0 - invested_weight)
    portfolio_returns = asset_returns.dot(weight_series)
    covariance = asset_returns.cov()
    correlation = asset_returns.corr().fillna(0.0)
    daily_variance = float(weight_series.dot(covariance).dot(weight_series))
    daily_volatility = math.sqrt(max(0.0, daily_variance))
    annualized_volatility = daily_volatility * math.sqrt(TRADING_DAYS)
    mean_daily_return = float(portfolio_returns.mean())

    losses = -portfolio_returns.to_numpy(dtype=float)
    historical_var = max(0.0, float(np.quantile(losses, confidence)))
    tail_losses = losses[losses >= historical_var]
    expected_shortfall = max(
        historical_var,
        float(tail_losses.mean()) if len(tail_losses) else historical_var,
    )
    z_score = NormalDist().inv_cdf(confidence)
    parametric_var = max(0.0, z_score * daily_volatility - mean_daily_return)

    cumulative = (1.0 + portfolio_returns).cumprod()
    drawdowns = cumulative / cumulative.cummax() - 1.0
    historical_max_drawdown = abs(float(drawdowns.min())) if not drawdowns.empty else 0.0

    benchmark_returns = returns[benchmark]
    benchmark_variance = float(benchmark_returns.var(ddof=1))
    betas: dict[str, float] = {}
    for symbol in symbols:
        beta = (
            float(asset_returns[symbol].cov(benchmark_returns)) / benchmark_variance
            if benchmark_variance > 0
            else 0.0
        )
        betas[symbol] = beta
    market_beta = sum(cleaned[symbol] * betas[symbol] for symbol in symbols)

    covariance_times_weights = covariance.dot(weight_series)
    positions: list[dict[str, Any]] = []
    risk_contributions: dict[str, float] = {}
    for symbol in symbols:
        component_daily_volatility = (
            float(weight_series[symbol] * covariance_times_weights[symbol] / daily_volatility)
            if daily_volatility > 0
            else 0.0
        )
        risk_contribution = (
            component_daily_volatility / daily_volatility if daily_volatility > 0 else 0.0
        )
        risk_contributions[symbol] = risk_contribution
        positions.append(
            {
                "symbol": symbol,
                "weight": round(float(weight_series[symbol]), 6),
                "annualized_volatility": round(
                    float(asset_returns[symbol].std(ddof=1) * math.sqrt(TRADING_DAYS)), 6
                ),
                "market_beta": round(betas[symbol], 6),
                "component_annualized_volatility": round(
                    component_daily_volatility * math.sqrt(TRADING_DAYS), 6
                ),
                "risk_contribution": round(risk_contribution, 6),
            }
        )

    normalized_risky_weights = weight_series / invested_weight
    hhi = float((normalized_risky_weights**2).sum())
    upper_correlations = [
        float(correlation.iloc[row, column])
        for row in range(len(symbols))
        for column in range(row + 1, len(symbols))
    ]
    concentration = {
        "largest_position": max(cleaned.items(), key=lambda item: item[1])[0],
        "largest_position_weight": round(max(cleaned.values()), 6),
        "weight_hhi": round(hhi, 6),
        "effective_positions": round(1.0 / hhi, 3) if hhi else 0.0,
        "average_pair_correlation": round(
            sum(upper_correlations) / len(upper_correlations), 6
        )
        if upper_correlations
        else 0.0,
        "maximum_pair_correlation": round(max(upper_correlations), 6)
        if upper_correlations
        else 0.0,
    }
    clusters = _connected_correlation_clusters(correlation, weight_series, risk_contributions)

    asset_volatility = asset_returns.std(ddof=1)
    inverse_volatility = 1.0 / asset_volatility.replace(0, np.nan)
    if inverse_volatility.isna().any():
        raise ValueError("Risk-balanced weights require non-zero volatility for every symbol")
    balanced_weights = inverse_volatility / inverse_volatility.sum() * invested_weight
    recommendations = []
    for symbol in symbols:
        current = float(weight_series[symbol])
        suggested = float(balanced_weights[symbol])
        delta = suggested - current
        contribution = risk_contributions[symbol]
        reason = (
            "Reduce: this holding contributes more risk than its capital weight."
            if delta < -0.005 and contribution > current / invested_weight
            else "Increase modestly: its current weight is below the inverse-volatility mix."
            if delta > 0.005
            else "Keep near current weight; already close to the risk-balanced mix."
        )
        recommendations.append(
            {
                "symbol": symbol,
                "current_weight": round(current, 6),
                "suggested_weight": round(suggested, 6),
                "change": round(delta, 6),
                "reason": reason,
            }
        )

    market_scenario_shocks = {
        symbol: min(0.0, max(-0.50, betas[symbol] * market_shock)) for symbol in symbols
    }
    broad_market_impact = sum(
        cleaned[symbol] * market_scenario_shocks[symbol] for symbol in symbols
    )
    ten_day_stress = -z_score * annualized_volatility / math.sqrt(TRADING_DAYS) * math.sqrt(10) * 1.5
    largest_symbol = concentration["largest_position"]
    largest_position_impact = cleaned[largest_symbol] * -0.20
    correlation_spike_covariance = np.outer(
        asset_returns.std(ddof=1).to_numpy(), asset_returns.std(ddof=1).to_numpy()
    ) * 0.90
    np.fill_diagonal(
        correlation_spike_covariance, asset_returns.var(ddof=1).to_numpy()
    )
    correlation_spike_daily_vol = math.sqrt(
        max(0.0, float(weight_series.to_numpy().dot(correlation_spike_covariance).dot(weight_series)))
    )
    correlation_spike_impact = -z_score * correlation_spike_daily_vol * math.sqrt(10)
    scenarios = [
        {
            "id": "broad_market_selloff",
            "name": f"Benchmark shock {market_shock:.0%}",
            "impact": round(broad_market_impact, 6),
            "impact_value": round(broad_market_impact * portfolio_value, 2)
            if portfolio_value is not None
            else None,
            "method": "Holding-level historical beta × benchmark shock",
        },
        {
            "id": "volatility_event",
            "name": "10-day volatility event",
            "impact": round(ten_day_stress, 6),
            "impact_value": round(ten_day_stress * portfolio_value, 2)
            if portfolio_value is not None
            else None,
            "method": f"{confidence:.0%} normal loss × √10 × 1.5 volatility multiplier",
        },
        {
            "id": "correlation_spike",
            "name": "Diversification breakdown",
            "impact": round(correlation_spike_impact, 6),
            "impact_value": round(correlation_spike_impact * portfolio_value, 2)
            if portfolio_value is not None
            else None,
            "method": "10-day loss with all pair correlations raised to 0.90",
        },
        {
            "id": "largest_position_shock",
            "name": f"{largest_symbol} isolated shock −20%",
            "impact": round(largest_position_impact, 6),
            "impact_value": round(largest_position_impact * portfolio_value, 2)
            if portfolio_value is not None
            else None,
            "method": "Largest capital position shocked independently",
        },
    ]

    summary = {
        "daily_volatility": round(daily_volatility, 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "historical_var": round(historical_var, 6),
        "parametric_var": round(parametric_var, 6),
        "expected_shortfall": round(expected_shortfall, 6),
        "historical_max_drawdown": round(historical_max_drawdown, 6),
        "market_beta": round(market_beta, 6),
    }
    alerts = _risk_alerts(
        positions,
        summary,
        concentration,
        max_position_weight=max_position_weight,
        max_component_risk=max_component_risk,
        annual_volatility_limit=annual_volatility_limit,
        daily_es_limit=daily_es_limit,
    )
    overall_level = (
        "high" if any(item["severity"] == "high" for item in alerts) else "watch" if alerts else "within_budget"
    )
    return {
        "as_of_date": as_of_date,
        "first_bar_date": prices.index[0].isoformat(),
        "last_bar_date": prices.index[-1].isoformat(),
        "common_sessions": len(prices),
        "benchmark_symbol": benchmark,
        "confidence": confidence,
        "portfolio_value": round(float(portfolio_value), 2)
        if portfolio_value is not None
        else None,
        "invested_weight": round(invested_weight, 6),
        "cash_weight": round(cash_weight, 6),
        "overall_level": overall_level,
        "summary": summary,
        "positions": positions,
        "correlation": {
            row: {column: round(float(correlation.loc[row, column]), 6) for column in symbols}
            for row in symbols
        },
        "concentration": concentration,
        "correlation_clusters": clusters,
        "scenarios": scenarios,
        "risk_balanced": {
            "method": "inverse_volatility",
            "recommendations": recommendations,
            "preserves_cash_weight": True,
        },
        "budgets": {
            "max_position_weight": max_position_weight,
            "max_component_risk": max_component_risk,
            "annual_volatility_limit": annual_volatility_limit,
            "daily_expected_shortfall_limit": daily_es_limit,
        },
        "alerts": alerts,
        "methodology": (
            "Historical close-to-close returns through the selected date; cash has zero volatility. "
            "Risk contribution uses the sample covariance matrix. Sector labels are intentionally "
            "replaced by point-in-time-safe correlation clusters."
        ),
        "disclaimer": (
            "Research estimates only. Historical covariance, beta and tail losses can change abruptly; "
            "suggested weights are not orders or investment advice."
        ),
    }
