"""Prospective A/B shadow experiments for scanner decision thresholds.

Experiments are observational and immutable in intent: they begin when saved,
consume only later persisted scans, and never change production configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from statistics import median
from typing import Any

import pandas as pd

from tradingagents.dataflows.candidate_pipeline import _connect as _candidate_connect
from tradingagents.dataflows.config import get_config

SHADOW_LAB_VERSION = "prospective-shadow-ab-v1"
METRICS = {
    "opportunity_score": "Opportunity score",
    "decision_confidence": "Decision Confidence",
    "quality_score": "Data quality",
}

PriceLoader = Callable[
    [list[str], str, str], tuple[dict[str, pd.DataFrame], list[dict[str, str]]]
]


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if math.isfinite(result) else None


def _utc_iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_experiments (
            experiment_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            hypothesis VARCHAR NOT NULL,
            metric VARCHAR NOT NULL,
            control_threshold DOUBLE NOT NULL,
            challenger_threshold DOUBLE NOT NULL,
            horizon_sessions INTEGER NOT NULL,
            transaction_cost_bps DOUBLE NOT NULL,
            minimum_observations INTEGER NOT NULL,
            status VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            closed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shadow_experiments_status
        ON shadow_experiments(status, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_experiment_evaluations (
            evaluation_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            verdict VARCHAR NOT NULL,
            control_outcomes INTEGER NOT NULL,
            challenger_outcomes INTEGER NOT NULL,
            excess_delta DOUBLE,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            reviewed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_experiment
        ON shadow_experiment_evaluations(experiment_id, as_of_date, created_at)
        """
    )


def _experiment_from_row(row) -> dict[str, Any]:
    return {
        "experiment_id": row[0],
        "model_version": row[1],
        "name": row[2],
        "hypothesis": row[3],
        "metric": row[4],
        "metric_label": METRICS.get(row[4], row[4]),
        "control_threshold": _number(row[5]),
        "challenger_threshold": _number(row[6]),
        "horizon_sessions": row[7],
        "transaction_cost_bps": _number(row[8]),
        "minimum_observations": row[9],
        "status": row[10],
        "created_at": _utc_iso(row[11]),
        "updated_at": _utc_iso(row[12]),
        "closed_at": _utc_iso(row[13]),
    }


def _validate_configuration(
    metric: str,
    control_threshold: float,
    challenger_threshold: float,
    horizon_sessions: int,
    transaction_cost_bps: float,
    minimum_observations: int,
) -> None:
    if metric not in METRICS:
        raise ValueError(f"Metric must be one of: {', '.join(METRICS)}")
    if not 0 <= float(control_threshold) <= 100:
        raise ValueError("Control threshold must be between 0 and 100")
    if not 0 <= float(challenger_threshold) <= 100:
        raise ValueError("Challenger threshold must be between 0 and 100")
    if float(control_threshold) == float(challenger_threshold):
        raise ValueError("Control and challenger thresholds must differ")
    if not 1 <= int(horizon_sessions) <= 252:
        raise ValueError("Horizon must be between 1 and 252 sessions")
    if not 0 <= float(transaction_cost_bps) <= 500:
        raise ValueError("Transaction costs must be between 0 and 500 bps")
    if not 3 <= int(minimum_observations) <= 500:
        raise ValueError("Minimum observations must be between 3 and 500")


def create_shadow_experiment(
    name: str,
    hypothesis: str,
    metric: str,
    control_threshold: float,
    challenger_threshold: float,
    config: dict[str, Any] | None = None,
    *,
    horizon_sessions: int = 21,
    transaction_cost_bps: float = 10.0,
    minimum_observations: int = 10,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Register a prospective experiment starting at its durable creation time."""
    clean_name = name.strip()
    clean_hypothesis = hypothesis.strip()
    if not 3 <= len(clean_name) <= 100:
        raise ValueError("Experiment name must contain 3 to 100 characters")
    if not 10 <= len(clean_hypothesis) <= 500:
        raise ValueError("Hypothesis must contain 10 to 500 characters")
    _validate_configuration(
        metric,
        control_threshold,
        challenger_threshold,
        horizon_sessions,
        transaction_cost_bps,
        minimum_observations,
    )
    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    identity = {
        "version": SHADOW_LAB_VERSION,
        "name": clean_name,
        "hypothesis": clean_hypothesis,
        "metric": metric,
        "control_threshold": float(control_threshold),
        "challenger_threshold": float(challenger_threshold),
        "horizon_sessions": int(horizon_sessions),
        "transaction_cost_bps": float(transaction_cost_bps),
        "minimum_observations": int(minimum_observations),
        "created_at": created_at.isoformat(),
    }
    experiment_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO shadow_experiments VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL
            )
            """,
            [
                experiment_id,
                SHADOW_LAB_VERSION,
                clean_name,
                clean_hypothesis,
                metric,
                float(control_threshold),
                float(challenger_threshold),
                int(horizon_sessions),
                float(transaction_cost_bps),
                int(minimum_observations),
                created_at,
                created_at,
            ],
        )
    finally:
        connection.close()
    return get_shadow_experiment(experiment_id, config)


def get_shadow_experiment(
    experiment_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT * FROM shadow_experiments WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
    finally:
        connection.close()
    return _experiment_from_row(row) if row else None


def close_shadow_experiment(
    experiment_id: str,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    closed_at = (now or datetime.now(UTC)).astimezone(UTC)
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT status, closed_at FROM shadow_experiments WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
        if not row:
            return None
        resolved_close = row[1] or closed_at
        connection.execute(
            """
            UPDATE shadow_experiments
            SET status = 'closed', closed_at = ?, updated_at = ?
            WHERE experiment_id = ?
            """,
            [resolved_close, closed_at, experiment_id],
        )
    finally:
        connection.close()
    return get_shadow_experiment(experiment_id, config)


def _latest_evaluation_row(connection, experiment_id: str):
    return connection.execute(
        """
        SELECT evaluation_id, verdict, control_outcomes, challenger_outcomes,
               excess_delta, as_of_date, created_at, reviewed_at
        FROM shadow_experiment_evaluations
        WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1
        """,
        [experiment_id],
    ).fetchone()


def list_shadow_experiments(
    config: dict[str, Any] | None = None,
    *,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if status not in {None, "active", "closed"}:
        raise ValueError("Status must be active or closed")
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        query = "SELECT * FROM shadow_experiments"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 100)))
        rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            experiment = _experiment_from_row(row)
            latest = _latest_evaluation_row(connection, experiment["experiment_id"])
            experiment["latest_evaluation"] = (
                {
                    "evaluation_id": latest[0],
                    "verdict": latest[1],
                    "control_outcomes": latest[2],
                    "challenger_outcomes": latest[3],
                    "excess_delta": _number(latest[4]),
                    "as_of_date": latest[5].isoformat(),
                    "created_at": _utc_iso(latest[6]),
                    "reviewed_at": _utc_iso(latest[7]),
                }
                if latest
                else None
            )
            result.append(experiment)
    finally:
        connection.close()
    return result


def _metric_value(item: dict[str, Any], metric: str) -> float | None:
    if metric == "opportunity_score":
        return _number(item.get("opportunity_score"))
    if metric == "decision_confidence":
        value = item.get("decision_confidence")
        return _number(value.get("score")) if isinstance(value, dict) else _number(value)
    quality = item.get("quality") or {}
    return _number(quality.get("score", item.get("quality_score")))


def _load_prospective_observations(
    config: dict[str, Any], experiment: dict[str, Any], as_of_date: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cutoff = datetime.combine(
        date.fromisoformat(as_of_date) + timedelta(days=1), time.min, tzinfo=UTC
    )
    end_at = min(
        cutoff,
        datetime.fromisoformat(experiment["closed_at"])
        if experiment.get("closed_at")
        else cutoff,
    )
    connection = _candidate_connect(config)
    try:
        rows = connection.execute(
            """
            SELECT scan_id, benchmark_symbol, result_json, created_at
            FROM opportunity_scans
            WHERE created_at >= ? AND created_at < ?
            ORDER BY created_at, scan_id
            """,
            [datetime.fromisoformat(experiment["created_at"]), end_at],
        ).fetchall()
    finally:
        connection.close()
    observations = []
    unavailable = 0
    scored = 0
    for scan_id, benchmark_symbol, result_json, created_at in rows:
        report = json.loads(result_json)
        for item in report.get("items", []):
            if not bool(item.get("passes_filters", True)):
                continue
            symbol = str(item.get("symbol") or "").upper()
            metric_value = _metric_value(item, experiment["metric"])
            if not symbol or metric_value is None:
                unavailable += 1
                continue
            scored += 1
            control_selected = metric_value >= float(experiment["control_threshold"])
            challenger_selected = metric_value >= float(
                experiment["challenger_threshold"]
            )
            if not control_selected and not challenger_selected:
                continue
            observations.append(
                {
                    "observation_id": hashlib.sha256(
                        f"{experiment['experiment_id']}|{scan_id}|{symbol}".encode()
                    ).hexdigest(),
                    "scan_id": scan_id,
                    "symbol": symbol,
                    "benchmark_symbol": str(benchmark_symbol or "SPY").upper(),
                    "decision_date": created_at.date().isoformat(),
                    "recorded_at": _utc_iso(created_at),
                    "metric": experiment["metric"],
                    "metric_value": metric_value,
                    "control_selected": control_selected,
                    "challenger_selected": challenger_selected,
                    "score": _number(item.get("opportunity_score")),
                    "rank": item.get("rank"),
                }
            )
    return observations, {
        "source_scans": len(rows),
        "scored_observations": scored,
        "unavailable_metric_count": unavailable,
    }


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "Date" not in frame or "Close" not in frame:
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


def _load_archived_prices(
    config: dict[str, Any], symbols: list[str], start_date: str, end_date: str
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    if not symbols:
        return {}, []
    connection = _candidate_connect(config)
    placeholders = ",".join("?" for _ in symbols)
    try:
        rows = connection.execute(
            f"""
            SELECT symbol, bar_date, close FROM (
                SELECT symbol, bar_date, close,
                       row_number() OVER (
                           PARTITION BY symbol, bar_date
                           ORDER BY as_of_date DESC NULLS LAST, ingested_at DESC
                       ) AS recency
                FROM market_bars
                WHERE symbol IN ({placeholders})
                  AND bar_date BETWEEN ? AND ?
                  AND as_of_date <= ?
            ) WHERE recency = 1 ORDER BY symbol, bar_date
            """,
            [*symbols, start_date, end_date, end_date],
        ).fetchall()
    except Exception:
        rows = []
    finally:
        connection.close()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for symbol, bar_date, close in rows:
        grouped[symbol].append({"Date": bar_date, "Close": close})
    frames = {
        symbol: _clean_frame(pd.DataFrame(grouped[symbol]))
        for symbol in symbols
        if grouped.get(symbol)
    }
    errors = [
        {"symbol": symbol, "message": "Archived forward price history unavailable"}
        for symbol in symbols
        if symbol not in frames
    ]
    return frames, errors


def _forward_outcome(
    observation: dict[str, Any],
    experiment: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    asset = frames.get(observation["symbol"])
    benchmark = frames.get(observation["benchmark_symbol"])
    if asset is None or benchmark is None:
        return None
    decision_day = date.fromisoformat(observation["decision_date"])
    future = asset.loc[asset["Date"] > decision_day].reset_index(drop=True)
    horizon = int(experiment["horizon_sessions"])
    if len(future) <= horizon:
        return None
    entry = future.iloc[0]
    exit_row = future.iloc[horizon]
    entry_date = entry["Date"]
    exit_date = exit_row["Date"]
    benchmark_window = benchmark.loc[
        (benchmark["Date"] >= entry_date) & (benchmark["Date"] <= exit_date)
    ].reset_index(drop=True)
    if len(benchmark_window) < 2:
        return None
    gross_return = float(exit_row["Close"] / entry["Close"] - 1)
    net_return = gross_return - float(experiment["transaction_cost_bps"]) / 10_000
    benchmark_return = float(
        benchmark_window.iloc[-1]["Close"] / benchmark_window.iloc[0]["Close"] - 1
    )
    excess_return = net_return - benchmark_return
    outcome = {
        **observation,
        "horizon_sessions": horizon,
        "entry_date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "net_return": _number(net_return),
        "benchmark_return": _number(benchmark_return),
        "excess_return": _number(excess_return),
        "success": excess_return > 0,
    }
    outcome["outcome_id"] = hashlib.sha256(
        (
            f"{SHADOW_LAB_VERSION}|{observation['observation_id']}|{horizon}|"
            f"{entry_date}|{exit_date}|{experiment['transaction_cost_bps']}"
        ).encode()
    ).hexdigest()
    return outcome


def _arm_summary(
    observations: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    selection_key: str,
) -> dict[str, Any]:
    selected = [item for item in observations if item[selection_key]]
    matured = [item for item in outcomes if item[selection_key]]
    if not matured:
        return {
            "selected_observations": len(selected),
            "matured_outcomes": 0,
            "pending_outcomes": len(selected),
            "hit_rate": None,
            "average_return": None,
            "average_excess_return": None,
            "median_excess_return": None,
            "worst_return": None,
        }
    returns = [float(item["net_return"]) for item in matured]
    excess = [float(item["excess_return"]) for item in matured]
    return {
        "selected_observations": len(selected),
        "matured_outcomes": len(matured),
        "pending_outcomes": len(selected) - len(matured),
        "hit_rate": _number(sum(item["success"] for item in matured) / len(matured)),
        "average_return": _number(sum(returns) / len(returns)),
        "average_excess_return": _number(sum(excess) / len(excess)),
        "median_excess_return": _number(median(excess)),
        "worst_return": _number(min(returns)),
    }


def build_shadow_evaluation(
    experiment_id: str,
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    observations: list[dict[str, Any]] | None = None,
    price_loader: PriceLoader | None = None,
    observation_start_at: str | datetime | None = None,
) -> dict[str, Any]:
    """Evaluate one registered experiment using only post-registration scans."""
    cfg = config or get_config()
    report_date = date.fromisoformat(as_of_date)
    experiment = get_shadow_experiment(experiment_id, cfg)
    if not experiment:
        raise ValueError("Shadow experiment not found")
    if report_date < date.fromisoformat(experiment["created_at"][:10]):
        raise ValueError("Evaluation date cannot precede experiment creation")
    if observations is None:
        source, source_stats = _load_prospective_observations(
            cfg, experiment, as_of_date
        )
    else:
        source = [
            item
            for item in observations
            if date.fromisoformat(item["decision_date"]) <= report_date
        ]
        source_stats = {
            "source_scans": len({item.get("scan_id") for item in source}),
            "scored_observations": len(source),
            "unavailable_metric_count": 0,
        }
    window_start = (
        datetime.fromisoformat(observation_start_at)
        if isinstance(observation_start_at, str)
        else observation_start_at
    )
    if window_start is not None:
        window_start = window_start.astimezone(UTC)
        source = [
            item
            for item in source
            if datetime.fromisoformat(item["recorded_at"]).astimezone(UTC)
            >= window_start
        ]
    if source:
        earliest = min(date.fromisoformat(item["decision_date"]) for item in source)
        symbols = sorted(
            {
                symbol
                for item in source
                for symbol in (item["symbol"], item.get("benchmark_symbol") or "SPY")
            }
        )
        loader = price_loader or (
            lambda selected, start, end: _load_archived_prices(
                cfg, selected, start, end
            )
        )
        frames, errors = loader(
            symbols, (earliest - timedelta(days=7)).isoformat(), as_of_date
        )
        frames = {
            symbol: _clean_frame(frame)
            .loc[lambda value: value["Date"] <= report_date]
            .reset_index(drop=True)
            for symbol, frame in frames.items()
        }
    else:
        earliest = None
        frames, errors = {}, []
    outcomes = [
        outcome
        for observation in source
        if (outcome := _forward_outcome(observation, experiment, frames)) is not None
    ]
    control = _arm_summary(source, outcomes, "control_selected")
    challenger = _arm_summary(source, outcomes, "challenger_selected")
    minimum = int(experiment["minimum_observations"])
    supported = (
        control["matured_outcomes"] >= minimum
        and challenger["matured_outcomes"] >= minimum
    )
    if supported:
        excess_delta = float(challenger["average_excess_return"]) - float(
            control["average_excess_return"]
        )
        hit_rate_delta = float(challenger["hit_rate"]) - float(control["hit_rate"])
    else:
        excess_delta = None
        hit_rate_delta = None
    if not supported:
        verdict = "insufficient_data"
    elif excess_delta >= 0.005 and hit_rate_delta >= 0:
        verdict = "challenger_promising"
    elif excess_delta <= -0.005:
        verdict = "control_leads"
    else:
        verdict = "inconclusive"
    overlap = sum(
        item["control_selected"] and item["challenger_selected"] for item in outcomes
    )
    control_only = sum(
        item["control_selected"] and not item["challenger_selected"]
        for item in outcomes
    )
    challenger_only = sum(
        item["challenger_selected"] and not item["control_selected"]
        for item in outcomes
    )
    report = {
        "model_version": SHADOW_LAB_VERSION,
        "experiment": experiment,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "coverage": {
            **source_stats,
            "selected_observations": len(source),
            "matured_unique_outcomes": len(outcomes),
            "pending_unique_outcomes": len(source) - len(outcomes),
            "earliest_decision_date": earliest.isoformat() if earliest else None,
            "observation_window_start": _utc_iso(window_start),
            "data_errors": errors,
        },
        "control": {
            "label": "Control A",
            "threshold": experiment["control_threshold"],
            **control,
        },
        "challenger": {
            "label": "Challenger B",
            "threshold": experiment["challenger_threshold"],
            **challenger,
        },
        "comparison": {
            "supported": supported,
            "minimum_observations_per_arm": minimum,
            "excess_delta": _number(excess_delta),
            "hit_rate_delta": _number(hit_rate_delta),
            "overlap_outcomes": overlap,
            "control_only_outcomes": control_only,
            "challenger_only_outcomes": challenger_only,
            "evidence_strength": (
                "reinforced"
                if supported
                and min(
                    control["matured_outcomes"], challenger["matured_outcomes"]
                )
                >= minimum * 2
                else "minimum_supported"
                if supported
                else "collecting"
            ),
        },
        "timeline": sorted(
            outcomes,
            key=lambda item: (item["decision_date"], item["symbol"]),
            reverse=True,
        ),
        "policy": {
            "prospective_only": True,
            "network_fetches": False,
            "llm_calls": False,
            "production_changes": False,
            "orders": False,
        },
        "limitations": (
            "This is a non-randomized shadow comparison. Overlapping arms, repeated symbols "
            "and market regimes are not independent. A promising challenger must be reviewed "
            "and validated again before any production change."
        ),
    }
    return report


def capture_shadow_evaluation(
    report: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    identity = {
        "model_version": report["model_version"],
        "experiment_id": report["experiment"]["experiment_id"],
        "as_of_date": report["as_of_date"],
        "outcome_ids": sorted(item["outcome_id"] for item in report["timeline"]),
    }
    evaluation_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    now = datetime.now(UTC)
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO shadow_experiment_evaluations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
            )
            """,
            [
                evaluation_id,
                report["experiment"]["experiment_id"],
                report["as_of_date"],
                report["verdict"],
                report["control"]["matured_outcomes"],
                report["challenger"]["matured_outcomes"],
                report["comparison"]["excess_delta"],
                _safe_json(report),
                now,
            ],
        )
        row = connection.execute(
            """
            SELECT created_at, reviewed_at FROM shadow_experiment_evaluations
            WHERE evaluation_id = ?
            """,
            [evaluation_id],
        ).fetchone()
    finally:
        connection.close()
    return {
        "saved": True,
        "evaluation_id": evaluation_id,
        "created_at": _utc_iso(row[0]),
        "reviewed_at": _utc_iso(row[1]),
    }


def get_latest_shadow_evaluation(
    experiment_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT evaluation_id, payload_json, created_at, reviewed_at
            FROM shadow_experiment_evaluations
            WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            [experiment_id],
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    report = json.loads(row[1])
    report["registry"] = {
        "saved": True,
        "evaluation_id": row[0],
        "created_at": _utc_iso(row[2]),
        "reviewed_at": _utc_iso(row[3]),
    }
    return report


def get_shadow_evaluation(
    evaluation_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT evaluation_id, payload_json, created_at, reviewed_at
            FROM shadow_experiment_evaluations WHERE evaluation_id = ?
            """,
            [evaluation_id],
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    report = json.loads(row[1])
    report["registry"] = {
        "saved": True,
        "evaluation_id": row[0],
        "created_at": _utc_iso(row[2]),
        "reviewed_at": _utc_iso(row[3]),
    }
    return report


def list_shadow_evaluations(
    experiment_id: str,
    config: dict[str, Any] | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT evaluation_id, as_of_date, verdict, control_outcomes,
                   challenger_outcomes, excess_delta, created_at, reviewed_at
            FROM shadow_experiment_evaluations
            WHERE experiment_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            [experiment_id, max(1, min(int(limit), 100))],
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "evaluation_id": row[0],
            "as_of_date": row[1].isoformat(),
            "verdict": row[2],
            "control_outcomes": row[3],
            "challenger_outcomes": row[4],
            "excess_delta": _number(row[5]),
            "created_at": _utc_iso(row[6]),
            "reviewed_at": _utc_iso(row[7]),
        }
        for row in rows
    ]


def mark_shadow_evaluation_reviewed(
    evaluation_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT reviewed_at FROM shadow_experiment_evaluations WHERE evaluation_id = ?",
            [evaluation_id],
        ).fetchone()
        if not row:
            return None
        reviewed_at = row[0] or datetime.now(UTC)
        connection.execute(
            """
            UPDATE shadow_experiment_evaluations SET reviewed_at = ?
            WHERE evaluation_id = ?
            """,
            [reviewed_at, evaluation_id],
        )
    finally:
        connection.close()
    return {"evaluation_id": evaluation_id, "reviewed_at": _utc_iso(reviewed_at)}


def get_shadow_lab_status(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experiments = list_shadow_experiments(config, limit=100)
    active = [item for item in experiments if item["status"] == "active"]
    latest = [item["latest_evaluation"] for item in experiments if item["latest_evaluation"]]
    unreviewed = [item for item in latest if not item.get("reviewed_at")]
    actionable = [
        item for item in unreviewed if item.get("verdict") != "insufficient_data"
    ]
    return {
        "experiment_count": len(experiments),
        "active_count": len(active),
        "closed_count": len(experiments) - len(active),
        "evaluated_count": len(latest),
        "unreviewed_count": len(unreviewed),
        "actionable_unreviewed_count": len(actionable),
        "challenger_promising_count": sum(
            item.get("verdict") == "challenger_promising" for item in actionable
        ),
        "control_leads_count": sum(
            item.get("verdict") == "control_leads" for item in actionable
        ),
        "experiments": experiments,
    }
