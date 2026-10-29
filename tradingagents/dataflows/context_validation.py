"""Historical validation for the independent fundamental and event overlay."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.opportunity_scanner import rank_opportunity
from tradingagents.dataflows.research_context import build_research_context
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import (
    dependency_available,
    parse_ohlcv_payload,
    warehouse_path,
)

_LOCK = threading.RLock()


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if math.isfinite(result) else None


def _safe_json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)


def _clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Date", "Close", "OutcomePrice"])
    result = frame.copy().drop_duplicates("Date", keep="last").sort_values("Date")
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    adjusted = (
        pd.to_numeric(result["Adj Close"], errors="coerce")
        if "Adj Close" in result
        else pd.Series(index=result.index, dtype=float)
    )
    result["OutcomePrice"] = adjusted.where(adjusted.gt(0), result["Close"])
    return result.dropna(subset=["Date", "Close", "OutcomePrice"])


def _checkpoint_dates(
    benchmark: pd.DataFrame,
    as_of_date: str,
    checkpoints: int,
    forward_sessions: int,
    *,
    minimum_history: int = 315,
) -> list[tuple[str, str]]:
    clean = _clean_prices(benchmark)
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    dates = [value for value in clean["Date"].tolist() if value <= as_of]
    latest_index = len(dates) - forward_sessions - 1
    selected = []
    for offset in range(checkpoints):
        index = latest_index - offset * forward_sessions
        if index < minimum_history or index + forward_sessions >= len(dates):
            break
        selected.append(
            (dates[index].isoformat(), dates[index + forward_sessions].isoformat())
        )
    return list(reversed(selected))


def _forward_outcome(
    frame: pd.DataFrame, checkpoint_date: str, target_date: str
) -> dict[str, Any] | None:
    clean = _clean_prices(frame)
    checkpoint = datetime.strptime(checkpoint_date, "%Y-%m-%d").date()
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    entry = clean[clean["Date"] <= checkpoint].tail(1)
    exit_rows = clean[clean["Date"] >= target].head(1)
    if entry.empty or exit_rows.empty:
        return None
    entry_price = float(entry.iloc[0]["OutcomePrice"])
    exit_price = float(exit_rows.iloc[0]["OutcomePrice"])
    raw_entry_price = float(entry.iloc[0]["Close"])
    if min(entry_price, exit_price, raw_entry_price) <= 0:
        return None
    return {
        "entry_date": entry.iloc[0]["Date"].isoformat(),
        "exit_date": exit_rows.iloc[0]["Date"].isoformat(),
        "entry_close": raw_entry_price,
        "forward_return": exit_price / entry_price - 1,
    }


def _cohort(
    observations: list[dict[str, Any]],
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [item for item in observations if predicate(item)]
    returns = [float(item["forward_return"]) for item in selected]
    excess = [float(item["excess_return"]) for item in selected]
    confidence = [
        float(item["decision_confidence"])
        for item in selected
        if item.get("decision_confidence") is not None
    ]
    return {
        "cohort": name,
        "observations": len(selected),
        "symbols": len({item["symbol"] for item in selected}),
        "average_return": _number(statistics.mean(returns)) if returns else None,
        "median_return": _number(statistics.median(returns)) if returns else None,
        "average_excess_return": _number(statistics.mean(excess)) if excess else None,
        "hit_rate": _number(sum(value > 0 for value in returns) / len(returns))
        if returns
        else None,
        "excess_hit_rate": _number(sum(value > 0 for value in excess) / len(excess))
        if excess
        else None,
        "worst_return": _number(min(returns)) if returns else None,
        "average_decision_confidence": _number(statistics.mean(confidence), 2)
        if confidence
        else None,
    }


def summarize_context_validation(
    observations: list[dict[str, Any]],
    *,
    min_opportunity_score: float = 55.0,
    min_fundamental_score: float = 55.0,
    max_event_risk: float = 40.0,
    min_context_coverage: float = 0.50,
    min_decision_confidence: float = 60.0,
) -> dict[str, Any]:
    """Compare unchanged scanner selections with context-aware cohorts."""

    def eligible(item: dict[str, Any]) -> bool:
        return bool(item.get("passes_filters"))

    def scanner(item: dict[str, Any]) -> bool:
        return eligible(item) and float(item["opportunity_score"]) >= min_opportunity_score

    def context_qualified(item: dict[str, Any]) -> bool:
        return (
            scanner(item)
            and item.get("fundamental_score") is not None
            and float(item["fundamental_score"]) >= min_fundamental_score
            and float(item.get("event_risk") or 0) <= max_event_risk
            and float(item.get("context_coverage") or 0) >= min_context_coverage
        )

    def decision_supported(item: dict[str, Any]) -> bool:
        return (
            eligible(item)
            and item.get("decision_confidence") is not None
            and float(item["decision_confidence"]) >= min_decision_confidence
            and item.get("decision_label") != "caution"
        )

    cohorts = [
        _cohort(observations, "eligible_universe", eligible),
        _cohort(observations, "scanner_selected", scanner),
        _cohort(observations, "context_qualified", context_qualified),
        _cohort(observations, "decision_supported", decision_supported),
    ]
    by_name = {item["cohort"]: item for item in cohorts}
    baseline = by_name["scanner_selected"]
    context = by_name["context_qualified"]
    decision = by_name["decision_supported"]

    def lift(item: dict[str, Any]) -> float | None:
        if (
            item["average_excess_return"] is None
            or baseline["average_excess_return"] is None
        ):
            return None
        return _number(
            float(item["average_excess_return"])
            - float(baseline["average_excess_return"])
        )

    context_lift = lift(context)
    decision_lift = lift(decision)
    if baseline["observations"] < 6 or decision["observations"] < 3:
        verdict = "insufficient_sample"
    elif decision_lift is not None and decision_lift >= 0.02:
        verdict = "promising"
    elif decision_lift is not None and decision_lift > -0.01:
        verdict = "mixed"
    else:
        verdict = "not_supported"
    return {
        "cohorts": cohorts,
        "lift": {
            "context_vs_scanner_excess_return": context_lift,
            "decision_vs_scanner_excess_return": decision_lift,
        },
        "verdict": verdict,
        "rules": {
            "min_opportunity_score": min_opportunity_score,
            "min_fundamental_score": min_fundamental_score,
            "max_event_risk": max_event_risk,
            "min_context_coverage": min_context_coverage,
            "min_decision_confidence": min_decision_confidence,
        },
    }


def _connect(config: dict[str, Any] | None = None):
    if not dependency_available():
        return None
    import duckdb

    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False):
        return None
    path = warehouse_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS context_validation_experiments (
            experiment_id VARCHAR PRIMARY KEY,
            as_of_date DATE NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            symbol_count INTEGER NOT NULL,
            checkpoint_count INTEGER NOT NULL,
            forward_sessions INTEGER NOT NULL,
            verdict VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_context_validation_created
        ON context_validation_experiments(created_at)
        """
    )
    return connection


def capture_context_validation(
    report: dict[str, Any], config: dict[str, Any] | None = None
) -> str | None:
    connection = _connect(config)
    if connection is None:
        return None
    identity = {key: value for key, value in report.items() if key != "registry"}
    experiment_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    try:
        with _LOCK:
            connection.execute(
                """
                INSERT OR IGNORE INTO context_validation_experiments
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    experiment_id,
                    report["as_of_date"],
                    report["benchmark_symbol"],
                    len(report["symbols"]),
                    len(report["checkpoints"]),
                    report["configuration"]["forward_sessions"],
                    report["summary"]["verdict"],
                    _safe_json(identity),
                    datetime.now(UTC),
                ],
            )
        return experiment_id
    finally:
        connection.close()


def list_context_validations(
    config: dict[str, Any] | None = None, *, limit: int = 12
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    try:
        rows = connection.execute(
            """
            SELECT experiment_id, as_of_date, benchmark_symbol, symbol_count,
                   checkpoint_count, forward_sessions, verdict, created_at
            FROM context_validation_experiments
            ORDER BY created_at DESC LIMIT ?
            """,
            [max(1, min(int(limit), 100))],
        ).fetchall()
        columns = [column[0] for column in connection.description]
        return [
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in zip(columns, row, strict=True)
            }
            for row in rows
        ]
    finally:
        connection.close()


def build_context_validation(
    symbols: list[str],
    benchmark_symbol: str,
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    checkpoints: int = 4,
    forward_sessions: int = 21,
    min_opportunity_score: float = 55.0,
    min_fundamental_score: float = 55.0,
    max_event_risk: float = 40.0,
    min_context_coverage: float = 0.50,
    min_decision_confidence: float = 60.0,
) -> dict[str, Any]:
    """Replay scanner decisions at historical cutoffs and measure future outcomes."""
    cleaned_symbols = list(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    if not 2 <= len(cleaned_symbols) <= 8:
        raise ValueError("Context validation requires between 2 and 8 symbols")
    bounded_checkpoints = max(2, min(int(checkpoints), 6))
    bounded_forward = max(21, min(int(forward_sessions), 63))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    required_sessions = 315 + bounded_checkpoints * bounded_forward + 260
    start_date = (as_of - timedelta(days=int(required_sessions * 1.8))).isoformat()
    benchmark_payload = route_to_vendor(
        "get_stock_data", benchmark_symbol, start_date, as_of_date
    )
    benchmark_frame = parse_ohlcv_payload(benchmark_payload)
    selected_checkpoints = _checkpoint_dates(
        benchmark_frame,
        as_of_date,
        bounded_checkpoints,
        bounded_forward,
    )
    if len(selected_checkpoints) < 2:
        raise ValueError("Not enough mature benchmark history for context validation")

    frames: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []
    for symbol in cleaned_symbols:
        try:
            payload = (
                benchmark_payload
                if symbol == benchmark_symbol
                else route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
            )
            frame = parse_ohlcv_payload(payload)
            if frame.empty:
                raise ValueError("No usable price history")
            frames[symbol] = frame
        except Exception:  # noqa: BLE001 - partial cross-sectional replay remains useful
            errors.append({"symbol": symbol, "message": "Price history unavailable"})

    observations: list[dict[str, Any]] = []
    for checkpoint_date, target_date in selected_checkpoints:
        benchmark_outcome = _forward_outcome(
            benchmark_frame, checkpoint_date, target_date
        )
        if benchmark_outcome is None:
            continue
        for symbol, frame in frames.items():
            try:
                outcome = _forward_outcome(frame, checkpoint_date, target_date)
                if outcome is None:
                    raise ValueError("Mature forward outcome unavailable")
                opportunity = rank_opportunity(
                    symbol,
                    frame,
                    benchmark_frame,
                    checkpoint_date,
                    benchmark_symbol=benchmark_symbol,
                    validation_sessions=252,
                    train_sessions=252,
                    test_sessions=63,
                )
                context = build_research_context(
                    symbol,
                    checkpoint_date,
                    config,
                    opportunity=opportunity,
                    closing_price=outcome["entry_close"],
                )
                decision = context["decision_confidence"]
                observations.append(
                    {
                        "symbol": symbol,
                        "checkpoint_date": checkpoint_date,
                        "outcome_date": outcome["exit_date"],
                        "forward_sessions": bounded_forward,
                        "forward_return": _number(outcome["forward_return"]),
                        "benchmark_return": _number(
                            benchmark_outcome["forward_return"]
                        ),
                        "excess_return": _number(
                            outcome["forward_return"]
                            - benchmark_outcome["forward_return"]
                        ),
                        "passes_filters": opportunity["passes_filters"],
                        "opportunity_score": opportunity["opportunity_score"],
                        "opportunity_bucket": opportunity["bucket"],
                        "fundamental_score": context["fundamentals"]["score"],
                        "event_risk": context["events"]["risk_score"],
                        "context_coverage": context["overlay"]["data_coverage"],
                        "decision_confidence": decision["score"],
                        "decision_label": decision["label"],
                    }
                )
            except Exception:  # noqa: BLE001 - preserve successful observations
                errors.append(
                    {
                        "symbol": symbol,
                        "message": f"Replay unavailable at {checkpoint_date}",
                    }
                )

    if not observations:
        raise ValueError("No complete historical observations could be evaluated")
    summary = summarize_context_validation(
        observations,
        min_opportunity_score=min_opportunity_score,
        min_fundamental_score=min_fundamental_score,
        max_event_risk=max_event_risk,
        min_context_coverage=min_context_coverage,
        min_decision_confidence=min_decision_confidence,
    )
    report = {
        "as_of_date": as_of_date,
        "benchmark_symbol": benchmark_symbol.upper(),
        "symbols": cleaned_symbols,
        "checkpoints": [
            {"decision_date": start, "outcome_date": end}
            for start, end in selected_checkpoints
        ],
        "configuration": {
            "forward_sessions": bounded_forward,
            "requested_checkpoints": bounded_checkpoints,
            "opportunity_validation_sessions": 252,
            "opportunity_train_sessions": 252,
            "opportunity_test_sessions": 63,
        },
        "summary": summary,
        "observations": observations,
        "errors": errors,
        "methodology": (
            "Each historical checkpoint rebuilds the unchanged scanner and SEC overlay using "
            "only evidence available on that date. Outcome returns begin at that checkpoint "
            "and end after the requested benchmark sessions. Checkpoints do not overlap."
        ),
        "limitations": (
            "Small samples are descriptive, not proof of future performance. Survivorship bias "
            "can remain when the supplied universe contains only today's constituents."
        ),
    }
    experiment_id = capture_context_validation(report, config)
    report["registry"] = {
        "saved": experiment_id is not None,
        "experiment_id": experiment_id,
    }
    return report
