"""Cross-sectional opportunity ranking built on transparent point-in-time evidence."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.quantitative import calculate_quantitative_snapshot
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.validation_lab import validate_signal_frames
from tradingagents.dataflows.warehouse import parse_ohlcv_payload

logger = logging.getLogger(__name__)


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _quality_metrics(
    frame: pd.DataFrame,
    as_of_date: str,
    required_bars: int,
) -> dict[str, Any]:
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    bounded = frame[frame["Date"] <= as_of].copy()
    duplicates = int(bounded["Date"].duplicated().sum())
    close_completeness = 1 - float(bounded["Close"].isna().mean()) if len(bounded) else 0.0
    valid = bounded.dropna(subset=["Close"])
    valid = valid[valid["Close"] > 0].drop_duplicates("Date", keep="last").sort_values("Date")
    bars = len(valid)
    coverage = min(1.0, bars / max(1, required_bars))
    latest_date = valid["Date"].iloc[-1] if bars else None
    stale_days = (as_of - latest_date).days if latest_date else None
    freshness = 1.0 if stale_days is not None and stale_days <= 7 else (
        max(0.0, 1 - (stale_days - 7) / 30) if stale_days is not None else 0.0
    )
    duplicate_penalty = min(0.25, duplicates / max(1, len(bounded)))
    score = 100 * _clip01(
        coverage * 0.60 + close_completeness * 0.20 + freshness * 0.20 - duplicate_penalty
    )
    return {
        "score": round(score, 2),
        "bars": bars,
        "required_bars": required_bars,
        "coverage": _number(coverage),
        "close_completeness": _number(close_completeness),
        "duplicate_dates": duplicates,
        "latest_bar_date": latest_date.isoformat() if latest_date else None,
        "stale_days": stale_days,
        "frame": valid,
    }


def _liquidity_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    recent = frame.tail(60)
    dollar_volume = pd.to_numeric(recent["Close"], errors="coerce").mul(
        pd.to_numeric(recent["Volume"], errors="coerce")
    )
    median = _number(dollar_volume.median(), 2)
    score = None
    if median is not None and median > 0:
        score = 100 * _clip01((math.log10(median) - 6) / 3)
    return {"median_dollar_volume_60": median, "score": _number(score, 2)}


def _scaled(value: float | None, low: float, high: float) -> float | None:
    if value is None:
        return None
    return 100 * _clip01((value - low) / (high - low))


def _opportunity_score(
    snapshot: dict[str, Any],
    validation: dict[str, Any] | None,
    quality_score: float,
    liquidity_score: float | None,
) -> tuple[float, dict[str, float]]:
    metrics = snapshot.get("metrics", {})
    drawdown = _number(metrics.get("max_drawdown_252"))
    evidence: dict[str, tuple[float | None, float]] = {
        "signal": (_scaled(_number(snapshot.get("score")), -100, 100), 0.22),
        "risk": (
            100 * (1 - min(1.0, abs(drawdown) / 0.50)) if drawdown is not None else None,
            0.10,
        ),
        "quality": (quality_score, 0.08),
        "liquidity": (liquidity_score, 0.05),
    }
    if validation:
        validation_metrics = validation["metrics"]
        evidence.update(
            {
                "oos_sharpe": (
                    _scaled(_number(validation_metrics.get("sharpe_ratio")), -1.5, 2.0),
                    0.18,
                ),
                "oos_alpha": (
                    _scaled(_number(validation_metrics.get("alpha_vs_asset")), -0.25, 0.25),
                    0.15,
                ),
                "profitable_folds": (
                    _scaled(_number(validation_metrics.get("profitable_fold_rate")), 0, 1),
                    0.12,
                ),
                "threshold_stability": (
                    _scaled(_number(validation_metrics.get("threshold_stability")), 0, 1),
                    0.10,
                ),
            }
        )
    available = {key: (value, weight) for key, (value, weight) in evidence.items() if value is not None}
    total_weight = sum(weight for _, weight in available.values())
    score = sum(value * weight for value, weight in available.values()) / total_weight
    return round(score, 2), {key: round(value, 2) for key, (value, _) in available.items()}


def _reasons(
    snapshot: dict[str, Any],
    validation: dict[str, Any] | None,
    filter_reasons: list[str],
) -> list[str]:
    if filter_reasons:
        return filter_reasons[:3]
    reasons: list[tuple[float, str]] = []
    score = float(snapshot.get("score", 0))
    reasons.append((abs(score) / 100, f"Quant signal {score:+.1f}/100"))
    volatility = _number(snapshot.get("metrics", {}).get("annualized_volatility_60"))
    if volatility is not None:
        reasons.append((min(1.0, volatility), f"60-session volatility {volatility:.1%}"))
    if validation:
        metrics = validation["metrics"]
        alpha = _number(metrics.get("alpha_vs_asset"))
        sharpe = _number(metrics.get("sharpe_ratio"))
        profitable = _number(metrics.get("profitable_fold_rate"))
        if alpha is not None:
            reasons.append((min(1.0, abs(alpha) * 3), f"OOS alpha vs asset {alpha:+.1%}"))
        if sharpe is not None:
            reasons.append((min(1.0, abs(sharpe) / 2), f"OOS Sharpe {sharpe:.2f}"))
        if profitable is not None:
            reasons.append((abs(profitable - 0.5), f"Profitable folds {profitable:.0%}"))
    return [text for _, text in sorted(reasons, reverse=True)[:3]]


def rank_opportunity(
    symbol: str,
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    as_of_date: str,
    *,
    benchmark_symbol: str = "SPY",
    validation_sessions: int = 252,
    train_sessions: int = 252,
    test_sessions: int = 63,
    min_dollar_volume: float = 5_000_000,
    max_volatility: float = 0.80,
    min_quality_score: float = 70,
) -> dict[str, Any]:
    """Rank one asset from its point-in-time market and OOS evidence."""
    required_bars = train_sessions + max(test_sessions, 63)
    quality = _quality_metrics(frame, as_of_date, required_bars)
    clean = quality.pop("frame")
    snapshot = calculate_quantitative_snapshot(clean.copy(), symbol, as_of_date)
    liquidity = _liquidity_metrics(clean)
    validation = None
    validation_error = None
    try:
        validation = validate_signal_frames(
            clean,
            benchmark_frame,
            symbol,
            benchmark_symbol,
            as_of_date,
            sessions=validation_sessions,
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            mode="long_only",
            commission_bps=2,
            slippage_bps=3,
            spread_bps=4,
            short_borrow_bps=0,
        )
    except Exception as exc:  # noqa: BLE001 - partial ranking is better than losing the symbol
        logger.info("Validation unavailable for %s: %s", symbol, exc)
        validation_error = "Insufficient aligned history for rolling validation"

    volatility = _number(snapshot["metrics"].get("annualized_volatility_60"))
    filter_reasons = []
    median_dollar_volume = liquidity["median_dollar_volume_60"]
    if median_dollar_volume is None or median_dollar_volume < min_dollar_volume:
        filter_reasons.append("Below the selected liquidity floor")
    if volatility is None or volatility > max_volatility:
        filter_reasons.append("Above the selected volatility ceiling")
    if quality["score"] < min_quality_score:
        filter_reasons.append("Below the selected data-quality floor")
    passes_filters = not filter_reasons
    score, score_components = _opportunity_score(
        snapshot,
        validation,
        quality["score"],
        liquidity["score"],
    )
    robustness = validation["robustness"] if validation else "unvalidated"
    if not passes_filters:
        bucket = "excluded"
    elif score >= 70 and robustness in {"promising", "mixed"}:
        bucket = "top_opportunity"
    elif score >= 55:
        bucket = "watch"
    elif score < 40:
        bucket = "avoid"
    else:
        bucket = "neutral"
    confidence_inputs = [float(snapshot["confidence"]), quality["score"] / 100]
    if validation:
        confidence_inputs.append(min(1.0, validation["metrics"]["fold_count"] / 4))
    validation_metrics = validation["metrics"] if validation else {}
    return {
        "symbol": symbol.upper(),
        "opportunity_score": score,
        "bucket": bucket,
        "passes_filters": passes_filters,
        "confidence": _number(sum(confidence_inputs) / len(confidence_inputs)),
        "quant_score": snapshot["score"],
        "signal": snapshot["label"],
        "risk_regime": snapshot["risk_regime"],
        "volatility_60": volatility,
        "max_drawdown_252": snapshot["metrics"].get("max_drawdown_252"),
        "momentum_60": snapshot["metrics"].get("momentum_60"),
        "median_dollar_volume_60": median_dollar_volume,
        "quality": quality,
        "validation": {
            "available": validation is not None,
            "robustness": robustness,
            "oos_return": validation_metrics.get("oos_return"),
            "alpha_vs_asset": validation_metrics.get("alpha_vs_asset"),
            "sharpe_ratio": validation_metrics.get("sharpe_ratio"),
            "profitable_fold_rate": validation_metrics.get("profitable_fold_rate"),
            "threshold_stability": validation_metrics.get("threshold_stability"),
            "fold_count": validation_metrics.get("fold_count", 0),
            "message": validation_error,
        },
        "score_components": score_components,
        "reasons": _reasons(snapshot, validation, filter_reasons),
    }


def scan_opportunities(
    symbols: list[str],
    as_of_date: str,
    *,
    benchmark_symbol: str = "SPY",
    validation_sessions: int = 252,
    train_sessions: int = 252,
    test_sessions: int = 63,
    min_dollar_volume: float = 5_000_000,
    max_volatility: float = 0.80,
    min_quality_score: float = 70,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Scan a universe concurrently while preserving partial failures."""
    unique_symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols))
    if not 2 <= len(unique_symbols) <= 500:
        raise ValueError("Opportunity scans require between 2 and 500 unique symbols")
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    calendar_days = min(
        6000,
        max(1200, int((validation_sessions + train_sessions + 260) * 1.75)),
    )
    start_date = (as_of - timedelta(days=calendar_days)).isoformat()
    benchmark_payload = route_to_vendor(
        "get_stock_data", benchmark_symbol, start_date, as_of_date
    )
    benchmark_frame = parse_ohlcv_payload(benchmark_payload)
    if benchmark_frame.empty:
        raise ValueError(f"Benchmark data unavailable for {benchmark_symbol}")

    def collect(symbol: str) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
        try:
            if symbol == benchmark_symbol:
                payload = benchmark_payload
            else:
                for attempt in range(2):
                    try:
                        payload = route_to_vendor(
                            "get_stock_data", symbol, start_date, as_of_date
                        )
                        break
                    except VendorUnavailableError:
                        if attempt == 1:
                            raise
                        time.sleep(0.75)
            item = rank_opportunity(
                symbol,
                parse_ohlcv_payload(payload),
                benchmark_frame,
                as_of_date,
                benchmark_symbol=benchmark_symbol,
                validation_sessions=validation_sessions,
                train_sessions=train_sessions,
                test_sessions=test_sessions,
                min_dollar_volume=min_dollar_volume,
                max_volatility=max_volatility,
                min_quality_score=min_quality_score,
            )
            return item, None
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not erase the scan
            logger.warning("Opportunity scan failed for %s: %s", symbol, exc)
            return None, {"symbol": symbol, "message": "Point-in-time evidence unavailable"}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(unique_symbols))) as executor:
        collected = list(executor.map(collect, unique_symbols))
    items = [item for item, _ in collected if item is not None]
    errors = [error for _, error in collected if error is not None]
    items.sort(
        key=lambda item: (
            item["passes_filters"],
            item["opportunity_score"],
            item["confidence"],
        ),
        reverse=True,
    )
    rank = 0
    for item in items:
        if item["passes_filters"]:
            rank += 1
            item["rank"] = rank
        else:
            item["rank"] = None
    bucket_counts = {
        bucket: sum(item["bucket"] == bucket for item in items)
        for bucket in ("top_opportunity", "watch", "neutral", "avoid", "excluded")
    }
    return {
        "as_of_date": as_of_date,
        "benchmark_symbol": benchmark_symbol,
        "symbols": unique_symbols,
        "requested_symbols": len(unique_symbols),
        "scanned_symbols": len(items),
        "eligible_symbols": rank,
        "errors": errors,
        "bucket_counts": bucket_counts,
        "filters": {
            "min_dollar_volume": min_dollar_volume,
            "max_volatility": max_volatility,
            "min_quality_score": min_quality_score,
        },
        "validation_configuration": {
            "sessions": validation_sessions,
            "train_sessions": train_sessions,
            "test_sessions": test_sessions,
            "benchmark_symbol": benchmark_symbol,
        },
        "ranking_methodology": (
            "Transparent 0-100 composite of current signal, rolling OOS Sharpe/alpha/fold "
            "consistency, drawdown, data quality and liquidity. Missing evidence is not imputed."
        ),
        "items": items,
    }
