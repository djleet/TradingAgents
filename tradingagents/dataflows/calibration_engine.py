"""Outcome calibration for scanner scores and research decisions.

The engine is deliberately observational.  It measures what happened after a
point-in-time score or candidate decision and produces shadow recommendations;
it never rewrites production weights, promotes candidates, or places orders.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from statistics import median
from typing import Any

import pandas as pd

from tradingagents.dataflows.candidate_pipeline import _connect as _candidate_connect
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import parse_ohlcv_payload

CALIBRATION_VERSION = "outcome-calibration-v1"
DEFAULT_HORIZONS = (5, 21, 63)
CURRENT_SCORE_THRESHOLD = 70
CURRENT_COMPONENT_WEIGHTS = {
    "signal": 0.22,
    "risk": 0.10,
    "quality": 0.08,
    "liquidity": 0.05,
    "oos_sharpe": 0.18,
    "oos_alpha": 0.15,
    "profitable_folds": 0.12,
    "threshold_stability": 0.10,
}

PriceLoader = Callable[[list[str], str, str], tuple[dict[str, pd.DataFrame], list[dict[str, str]]]]


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        result = float(value)
        return round(result, digits) if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_runs (
            run_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            horizons_json VARCHAR NOT NULL,
            source_observations INTEGER NOT NULL,
            matured_outcomes INTEGER NOT NULL,
            verdict VARCHAR NOT NULL,
            configuration_json VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_calibration_runs_created
        ON calibration_runs(as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_outcomes (
            outcome_id VARCHAR PRIMARY KEY,
            run_id VARCHAR NOT NULL,
            observation_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            horizon_sessions INTEGER NOT NULL,
            entry_date DATE NOT NULL,
            exit_date DATE NOT NULL,
            net_return DOUBLE NOT NULL,
            benchmark_return DOUBLE NOT NULL,
            excess_return DOUBLE NOT NULL,
            success BOOLEAN NOT NULL,
            metadata_json VARCHAR NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_calibration_outcomes_lookup
        ON calibration_outcomes(symbol, decision_date, horizon_sessions)
        """
    )


def _quality_score(item: dict[str, Any]) -> float | None:
    quality = item.get("quality") or {}
    return _number(quality.get("score", item.get("quality_score")))


def _decision_confidence(item: dict[str, Any]) -> float | None:
    decision = item.get("decision_confidence") or {}
    if isinstance(decision, dict):
        return _number(decision.get("score"))
    return _number(decision)


def _load_observations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load immutable scan vintages and enrich them with candidate decisions."""
    connection = _candidate_connect(config)
    try:
        scan_rows = connection.execute(
            """
            SELECT scan_id, as_of_date, benchmark_symbol, result_json, created_at
            FROM opportunity_scans ORDER BY as_of_date, created_at
            """
        ).fetchall()
        event_rows = connection.execute(
            """
            SELECT candidate_id, event_type, to_stage, snapshot_json, created_at
            FROM research_candidate_events ORDER BY created_at
            """
        ).fetchall()
    finally:
        connection.close()

    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for scan_id, as_of, benchmark, result_json, created_at in scan_rows:
        report = json.loads(result_json)
        decision_date = as_of.isoformat()
        for item in report.get("items", []):
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            observation_id = hashlib.sha256(f"scan|{scan_id}|{symbol}".encode()).hexdigest()
            observations[(scan_id, symbol)] = {
                "observation_id": observation_id,
                "observation_type": "scanner",
                "symbol": symbol,
                "benchmark_symbol": str(benchmark).upper(),
                "decision_date": decision_date,
                "recorded_at": created_at.isoformat(),
                "scan_id": scan_id,
                "score": _number(item.get("opportunity_score")),
                "rank": item.get("rank"),
                "bucket": str(item.get("bucket") or "neutral"),
                "passes_filters": bool(item.get("passes_filters", True)),
                "decision_confidence": _decision_confidence(item),
                "quality_score": _quality_score(item),
                "validation_verdict": str(
                    (item.get("validation") or {}).get("robustness") or "unvalidated"
                ),
                "regime": str(item.get("risk_regime") or "unknown"),
                "score_components": {
                    key: value
                    for key, raw in (item.get("score_components") or {}).items()
                    if (value := _number(raw)) is not None
                },
                "candidate_stage": None,
            }

    candidate_observations: dict[str, dict[str, Any]] = {}
    for candidate_id, event_type, to_stage, snapshot_json, created_at in event_rows:
        snapshot = json.loads(snapshot_json)
        symbol = str(snapshot.get("symbol") or "").upper()
        scan_id = snapshot.get("scan_id")
        matching = observations.get((scan_id, symbol)) if scan_id and symbol else None
        if matching:
            if snapshot.get("decision_confidence") is not None:
                matching["decision_confidence"] = _number(
                    snapshot.get("decision_confidence")
                )
            if snapshot.get("quality_score") is not None:
                matching["quality_score"] = _number(snapshot.get("quality_score"))
            if to_stage:
                matching["candidate_stage"] = to_stage

        if event_type not in {"stage_changed", "paper_synced"} or to_stage not in {
            "qualified",
            "paper_position",
            "closed",
            "rejected",
        }:
            continue
        decision_day = created_at.date().isoformat()
        key = f"{candidate_id}|{to_stage}|{decision_day}"
        candidate_observations[key] = {
            "observation_id": hashlib.sha256(f"candidate|{key}".encode()).hexdigest(),
            "observation_type": "candidate_decision",
            "symbol": symbol,
            "benchmark_symbol": "SPY",
            "decision_date": decision_day,
            "recorded_at": created_at.isoformat(),
            "scan_id": scan_id,
            "score": _number(snapshot.get("latest_score")),
            "rank": snapshot.get("latest_rank"),
            "bucket": str(snapshot.get("latest_bucket") or "neutral"),
            "passes_filters": True,
            "decision_confidence": _number(snapshot.get("decision_confidence")),
            "quality_score": _number(snapshot.get("quality_score")),
            "validation_verdict": str(snapshot.get("validation_verdict") or "unvalidated"),
            "regime": str(snapshot.get("risk_level") or "unknown"),
            "score_components": {},
            "candidate_stage": to_stage,
        }

    combined = [*observations.values(), *candidate_observations.values()]
    return [item for item in combined if item["symbol"]]


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Date", "Close"])
    cleaned = frame.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce").dt.date
    cleaned["Close"] = pd.to_numeric(cleaned["Close"], errors="coerce")
    return (
        cleaned.dropna(subset=["Date", "Close"])
        .loc[lambda value: value["Close"] > 0, ["Date", "Close"]]
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _default_price_loader(
    symbols: list[str], start_date: str, end_date: str
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    def collect(symbol: str) -> tuple[str, pd.DataFrame | None, dict[str, str] | None]:
        try:
            payload = route_to_vendor("get_stock_data", symbol, start_date, end_date)
            frame = _clean_frame(parse_ohlcv_payload(payload))
            if frame.empty:
                raise ValueError("empty price history")
            return symbol, frame, None
        except Exception:  # noqa: BLE001 - partial outcome coverage is reported
            return symbol, None, {
                "symbol": symbol,
                "message": "Forward price history unavailable",
            }

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(symbols)))) as executor:
        results = list(executor.map(collect, symbols))
    return (
        {symbol: frame for symbol, frame, _ in results if frame is not None},
        [error for _, _, error in results if error is not None],
    )


def _forward_outcome(
    observation: dict[str, Any],
    horizon: int,
    frames: dict[str, pd.DataFrame],
    transaction_cost_bps: float,
) -> dict[str, Any] | None:
    asset = frames.get(observation["symbol"])
    benchmark = frames.get(observation["benchmark_symbol"])
    if asset is None or benchmark is None:
        return None
    decision_day = date.fromisoformat(observation["decision_date"])
    future = asset.loc[asset["Date"] > decision_day].reset_index(drop=True)
    if len(future) <= horizon:
        return None
    entry = future.iloc[0]
    exit_row = future.iloc[horizon]
    entry_date = entry["Date"]
    exit_date = exit_row["Date"]
    benchmark_entry = benchmark.loc[benchmark["Date"] >= entry_date]
    benchmark_exit = benchmark.loc[benchmark["Date"] <= exit_date]
    if benchmark_entry.empty or benchmark_exit.empty:
        return None
    benchmark_entry_row = benchmark_entry.iloc[0]
    benchmark_exit_row = benchmark_exit.iloc[-1]
    if benchmark_entry_row["Date"] > exit_date:
        return None
    gross_return = float(exit_row["Close"] / entry["Close"] - 1)
    net_return = gross_return - transaction_cost_bps / 10_000
    benchmark_return = float(
        benchmark_exit_row["Close"] / benchmark_entry_row["Close"] - 1
    )
    excess_return = net_return - benchmark_return
    outcome = {
        **observation,
        "horizon_sessions": horizon,
        "entry_date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "gross_return": _number(gross_return),
        "net_return": _number(net_return),
        "benchmark_return": _number(benchmark_return),
        "excess_return": _number(excess_return),
        "success": excess_return > 0,
    }
    outcome["outcome_id"] = hashlib.sha256(
        f"{observation['observation_id']}|{horizon}|{entry_date}|{exit_date}".encode()
    ).hexdigest()
    return outcome


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "observations": 0,
            "hit_rate": None,
            "average_return": None,
            "average_benchmark_return": None,
            "average_excess_return": None,
            "median_excess_return": None,
            "worst_return": None,
        }
    returns = [float(row["net_return"]) for row in rows]
    benchmark = [float(row["benchmark_return"]) for row in rows]
    excess = [float(row["excess_return"]) for row in rows]
    return {
        "observations": len(rows),
        "hit_rate": _number(sum(row["success"] for row in rows) / len(rows)),
        "average_return": _number(sum(returns) / len(returns)),
        "average_benchmark_return": _number(sum(benchmark) / len(benchmark)),
        "average_excess_return": _number(sum(excess) / len(excess)),
        "median_excess_return": _number(median(excess)),
        "worst_return": _number(min(returns)),
    }


def _brier(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row for row in rows if row.get(field) is not None]
    if not values:
        return None
    return _number(
        sum((float(row[field]) / 100 - float(row["success"])) ** 2 for row in values)
        / len(values)
    )


def _reliability(
    rows: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    bins = ((0, 40), (40, 55), (55, 70), (70, 85), (85, 101))
    result = []
    for low, high in bins:
        selected = [
            row
            for row in rows
            if row.get(field) is not None and low <= float(row[field]) < high
        ]
        summary = _metric_summary(selected)
        result.append(
            {
                "label": f"{low}–{high - 1}",
                "minimum": low,
                "maximum": high - 1,
                "average_value": _number(
                    sum(float(row[field]) for row in selected) / len(selected)
                )
                if selected
                else None,
                **summary,
            }
        )
    return result


def _cohort_rows(outcomes: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    rows = [item for item in outcomes if item["horizon_sessions"] == horizon]
    selectors = (
        ("scanner_all", "All scanner observations", lambda item: item["observation_type"] == "scanner"),
        ("eligible", "Passed scanner filters", lambda item: item["observation_type"] == "scanner" and item["passes_filters"]),
        ("top_opportunity", "Top opportunities", lambda item: item["bucket"] == "top_opportunity"),
        ("context_reviewed", "Context reviewed", lambda item: item.get("decision_confidence") is not None),
        ("decision_supported", "Decision confidence ≥ 65", lambda item: (item.get("decision_confidence") or -1) >= 65),
        ("qualified", "Qualified candidates", lambda item: item.get("candidate_stage") in {"qualified", "paper_position", "closed"}),
        ("rejected", "Rejected candidates", lambda item: item.get("candidate_stage") == "rejected"),
    )
    return [
        {"cohort": key, "label": label, **_metric_summary([row for row in rows if selector(row)])}
        for key, label, selector in selectors
    ]


def _regime_rows(outcomes: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        if row["horizon_sessions"] == horizon:
            grouped[str(row.get("regime") or "unknown")].append(row)
    return [
        {"regime": regime, **_metric_summary(rows)}
        for regime, rows in sorted(grouped.items())
    ]


def _drift(outcomes: list[dict[str, Any]], minimum_observations: int) -> dict[str, Any]:
    rows = sorted(
        [item for item in outcomes if item["horizon_sessions"] == 21],
        key=lambda item: (item["decision_date"], item["symbol"]),
    )
    minimum_half = max(3, math.ceil(minimum_observations / 2))
    if len(rows) < minimum_half * 2:
        return {
            "status": "insufficient_data",
            "message": "More independent 21-session outcomes are required before drift can be assessed.",
            "previous": _metric_summary([]),
            "recent": _metric_summary(rows),
            "excess_delta": None,
            "hit_rate_delta": None,
        }
    midpoint = len(rows) // 2
    previous = _metric_summary(rows[:midpoint])
    recent = _metric_summary(rows[midpoint:])
    excess_delta = float(recent["average_excess_return"]) - float(
        previous["average_excess_return"]
    )
    hit_delta = float(recent["hit_rate"]) - float(previous["hit_rate"])
    if excess_delta <= -0.02 or hit_delta <= -0.10:
        status = "degrading"
    elif excess_delta >= 0.02 and hit_delta >= 0:
        status = "improving"
    else:
        status = "stable"
    return {
        "status": status,
        "message": "Recent outcomes are compared with the earlier half of the same 21-session sample.",
        "previous": previous,
        "recent": recent,
        "excess_delta": _number(excess_delta),
        "hit_rate_delta": _number(hit_delta),
    }


def _shadow_recommendations(
    outcomes: list[dict[str, Any]], minimum_observations: int
) -> dict[str, Any]:
    rows = [item for item in outcomes if item["horizon_sessions"] == 21]
    thresholds = []
    for threshold in (50, 55, 60, 65, 70, 75, 80):
        selected = [row for row in rows if row.get("score") is not None and row["score"] >= threshold]
        summary = _metric_summary(selected)
        support = summary["observations"] >= minimum_observations
        objective = (
            float(summary["average_excess_return"])
            + 0.05 * (float(summary["hit_rate"]) - 0.5)
            if support
            else None
        )
        thresholds.append(
            {"threshold": threshold, "supported": support, "objective": _number(objective), **summary}
        )
    supported = [item for item in thresholds if item["supported"]]
    best = max(supported, key=lambda item: item["objective"]) if supported else None

    weights = []
    raw_shadow: dict[str, float] = {}
    for component, current_weight in CURRENT_COMPONENT_WEIGHTS.items():
        paired = [
            (float(row["score_components"][component]), float(row["excess_return"]))
            for row in rows
            if component in row.get("score_components", {})
        ]
        correlation = None
        supported_component = len(paired) >= minimum_observations
        if supported_component:
            component_series = pd.Series([item[0] for item in paired]).rank()
            outcome_series = pd.Series([item[1] for item in paired]).rank()
            calculated = component_series.corr(outcome_series)
            correlation = _number(calculated) if pd.notna(calculated) else None
        multiplier = 1 + max(-0.5, min(0.5, correlation or 0.0))
        raw_shadow[component] = max(0.02, current_weight * multiplier)
        weights.append(
            {
                "component": component,
                "current_weight": current_weight,
                "shadow_weight": None,
                "rank_correlation": correlation,
                "observations": len(paired),
                "supported": supported_component,
            }
        )
    total = sum(raw_shadow.values())
    for item in weights:
        item["shadow_weight"] = _number(raw_shadow[item["component"]] / total, 4)
    return {
        "status": "supported" if best else "insufficient_data",
        "current_qualification_threshold": CURRENT_SCORE_THRESHOLD,
        "recommended_qualification_threshold": best["threshold"] if best else None,
        "thresholds": thresholds,
        "weights": weights,
        "notice": (
            "Exploratory shadow settings only. They are not applied automatically and require prospective validation."
        ),
    }


def build_calibration_report(
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
    transaction_cost_bps: float = 10.0,
    minimum_observations: int = 8,
    price_loader: PriceLoader | None = None,
) -> dict[str, Any]:
    """Measure forward outcomes with strict post-decision market data."""
    cfg = config or get_config()
    report_date = date.fromisoformat(as_of_date)
    resolved_horizons = tuple(sorted({int(value) for value in horizons}))
    if not resolved_horizons or any(value < 1 or value > 252 for value in resolved_horizons):
        raise ValueError("Calibration horizons must be between 1 and 252 sessions")
    if not 0 <= transaction_cost_bps <= 500:
        raise ValueError("Transaction costs must be between 0 and 500 bps")
    if not 3 <= minimum_observations <= 500:
        raise ValueError("Minimum observations must be between 3 and 500")

    observations = [
        item
        for item in _load_observations(cfg)
        if date.fromisoformat(item["decision_date"]) < report_date
    ]
    if observations:
        earliest = min(date.fromisoformat(item["decision_date"]) for item in observations)
        symbols = sorted(
            {
                symbol
                for item in observations
                for symbol in (item["symbol"], item["benchmark_symbol"])
            }
        )
        frames, errors = (price_loader or _default_price_loader)(
            symbols,
            (earliest - timedelta(days=7)).isoformat(),
            as_of_date,
        )
        frames = {symbol: _clean_frame(frame) for symbol, frame in frames.items()}
    else:
        earliest = None
        frames, errors = {}, []

    outcomes = [
        outcome
        for observation in observations
        for horizon in resolved_horizons
        if (
            outcome := _forward_outcome(
                observation, horizon, frames, transaction_cost_bps
            )
        )
        is not None
    ]
    horizon_summaries = []
    for horizon in resolved_horizons:
        selected = [item for item in outcomes if item["horizon_sessions"] == horizon]
        horizon_summaries.append(
            {
                "horizon_sessions": horizon,
                **_metric_summary(selected),
                "score_brier": _brier(selected, "score"),
                "decision_brier": _brier(selected, "decision_confidence"),
            }
        )

    primary_horizon = 21 if 21 in resolved_horizons else resolved_horizons[0]
    primary = [item for item in outcomes if item["horizon_sessions"] == primary_horizon]
    drift = _drift(outcomes, minimum_observations)
    if len(primary) < minimum_observations:
        verdict = "insufficient_data"
    elif drift["status"] == "degrading":
        verdict = "degrading"
    elif (_metric_summary(primary)["average_excess_return"] or 0) > 0:
        verdict = "positive"
    else:
        verdict = "mixed"

    report = {
        "model_version": CALIBRATION_VERSION,
        "as_of_date": as_of_date,
        "verdict": verdict,
        "configuration": {
            "horizons": list(resolved_horizons),
            "transaction_cost_bps": transaction_cost_bps,
            "minimum_observations": minimum_observations,
            "entry_policy": "First available close strictly after the decision date",
            "benchmark_policy": "Same-period close-to-close benchmark return",
        },
        "coverage": {
            "source_observations": len(observations),
            "matured_outcomes": len(outcomes),
            "pending_outcomes": len(observations) * len(resolved_horizons) - len(outcomes),
            "symbols": len({item["symbol"] for item in observations}),
            "earliest_decision_date": earliest.isoformat() if earliest else None,
            "latest_decision_date": max(
                (item["decision_date"] for item in observations), default=None
            ),
            "data_errors": errors,
        },
        "horizons": horizon_summaries,
        "reliability": {
            "horizon_sessions": primary_horizon,
            "score": _reliability(primary, "score"),
            "decision_confidence": _reliability(primary, "decision_confidence"),
            "notice": "Scores are treated as ranking intensities, not literal probabilities; Brier values are diagnostic only.",
        },
        "cohorts": _cohort_rows(outcomes, primary_horizon),
        "regimes": _regime_rows(outcomes, primary_horizon),
        "drift": drift,
        "shadow_recommendations": _shadow_recommendations(
            outcomes, minimum_observations
        ),
        "outcomes": outcomes,
        "limitations": (
            "Overlapping horizons and repeated symbols are not independent observations. "
            "Current-index universes may contain survivorship bias. Shadow recommendations "
            "must be tested prospectively before any model change."
        ),
    }
    return report


def capture_calibration_run(
    report: dict[str, Any], config: dict[str, Any] | None = None
) -> str:
    """Persist the complete report and its observation-level outcomes."""
    cfg = config or get_config()
    result_json = _safe_json(report)
    identity = {
        "model_version": report["model_version"],
        "as_of_date": report["as_of_date"],
        "configuration": report["configuration"],
        "outcome_ids": sorted(item["outcome_id"] for item in report["outcomes"]),
    }
    run_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    now = datetime.now(UTC)
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            INSERT OR REPLACE INTO calibration_runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                run_id,
                report["model_version"],
                report["as_of_date"],
                _safe_json(report["configuration"]["horizons"]),
                report["coverage"]["source_observations"],
                report["coverage"]["matured_outcomes"],
                report["verdict"],
                _safe_json(report["configuration"]),
                result_json,
                now,
            ],
        )
        for outcome in report["outcomes"]:
            metadata = {
                key: outcome.get(key)
                for key in (
                    "observation_type",
                    "scan_id",
                    "score",
                    "rank",
                    "bucket",
                    "decision_confidence",
                    "quality_score",
                    "validation_verdict",
                    "regime",
                    "candidate_stage",
                    "score_components",
                )
            }
            connection.execute(
                """
                INSERT OR REPLACE INTO calibration_outcomes VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    outcome["outcome_id"],
                    run_id,
                    outcome["observation_id"],
                    outcome["symbol"],
                    outcome["benchmark_symbol"],
                    outcome["decision_date"],
                    outcome["horizon_sessions"],
                    outcome["entry_date"],
                    outcome["exit_date"],
                    outcome["net_return"],
                    outcome["benchmark_return"],
                    outcome["excess_return"],
                    outcome["success"],
                    _safe_json(metadata),
                    now,
                ],
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return run_id


def list_calibration_runs(
    config: dict[str, Any] | None = None, *, limit: int = 20
) -> list[dict[str, Any]]:
    """Return compact persisted calibration runs."""
    cfg = config or get_config()
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT run_id, model_version, as_of_date, horizons_json,
                   source_observations, matured_outcomes, verdict, created_at
            FROM calibration_runs ORDER BY created_at DESC LIMIT ?
            """,
            [max(1, min(int(limit), 100))],
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "run_id": row[0],
            "model_version": row[1],
            "as_of_date": row[2].isoformat(),
            "horizons": json.loads(row[3]),
            "source_observations": row[4],
            "matured_outcomes": row[5],
            "verdict": row[6],
            "created_at": row[7].isoformat(),
        }
        for row in rows
    ]


def get_latest_calibration_run(
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the newest complete report for dashboard restoration."""
    cfg = config or get_config()
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT run_id, result_json, created_at FROM calibration_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    report = json.loads(row[1])
    report["registry"] = {
        "saved": True,
        "run_id": row[0],
        "created_at": row[2].isoformat(),
    }
    return report
