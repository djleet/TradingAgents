"""Review candidate decisions and thesis alerts against mature forward outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from statistics import median
from typing import Any

import pandas as pd

from tradingagents.dataflows.candidate_pipeline import _connect as _candidate_connect
from tradingagents.dataflows.config import get_config

DECISION_REVIEW_VERSION = "decision-counterfactual-v1"
DEFAULT_HORIZONS = (5, 21, 63)
DECISION_STAGES = {"qualified", "paper_position", "rejected"}

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


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_review_runs (
            run_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            horizons_json VARCHAR NOT NULL,
            observation_count INTEGER NOT NULL,
            matured_outcome_count INTEGER NOT NULL,
            verdict VARCHAR NOT NULL,
            configuration_json VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            reviewed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_decision_review_runs
        ON decision_review_runs(as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_review_outcomes (
            outcome_id VARCHAR PRIMARY KEY,
            run_id VARCHAR NOT NULL,
            observation_id VARCHAR NOT NULL,
            observation_type VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            horizon_sessions INTEGER NOT NULL,
            entry_date DATE NOT NULL,
            exit_date DATE NOT NULL,
            net_return DOUBLE NOT NULL,
            benchmark_return DOUBLE NOT NULL,
            excess_return DOUBLE NOT NULL,
            decision_success BOOLEAN NOT NULL,
            counterfactual_delta DOUBLE,
            classification VARCHAR NOT NULL,
            metadata_json VARCHAR NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_decision_review_outcomes
        ON decision_review_outcomes(symbol, decision_date, horizon_sessions)
        """
    )


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


def _load_observations(config: dict[str, Any]) -> list[dict[str, Any]]:
    connection = _candidate_connect(config)
    try:
        event_rows = connection.execute(
            """
            SELECT event_id, candidate_id, event_type, from_stage, to_stage,
                   snapshot_json, note, created_at
            FROM research_candidate_events ORDER BY created_at
            """
        ).fetchall()
        try:
            thesis_rows = connection.execute(
                """
                SELECT evaluation_id, candidate_id, symbol, analysis_date,
                       verdict, payload_json, created_at
                FROM thesis_invalidation_evaluations
                WHERE verdict IN ('invalidated', 'warning')
                ORDER BY created_at
                """
            ).fetchall()
        except Exception:
            thesis_rows = []
    finally:
        connection.close()

    observations = []
    for (
        event_id,
        candidate_id,
        event_type,
        from_stage,
        to_stage,
        snapshot_json,
        note,
        created_at,
    ) in event_rows:
        if event_type not in {"stage_changed", "paper_synced"}:
            continue
        if to_stage not in DECISION_STAGES:
            continue
        snapshot = json.loads(snapshot_json)
        symbol = str(snapshot.get("symbol") or "").upper()
        if not symbol:
            continue
        observations.append(
            {
                "observation_id": event_id,
                "observation_type": "candidate_decision",
                "candidate_id": candidate_id,
                "symbol": symbol,
                "benchmark_symbol": "SPY",
                "decision_date": created_at.date().isoformat(),
                "recorded_at": created_at.astimezone(UTC).isoformat(),
                "evidence_date": None,
                "decision": to_stage,
                "from_stage": from_stage,
                "note": note,
                "score": _number(snapshot.get("latest_score")),
                "decision_confidence": _number(snapshot.get("decision_confidence")),
                "triggered_rules": [],
            }
        )

    for (
        evaluation_id,
        candidate_id,
        symbol,
        analysis_date,
        verdict,
        payload_json,
        created_at,
    ) in thesis_rows:
        payload = json.loads(payload_json)
        triggered = [item for item in payload.get("rules", []) if item.get("triggered")]
        observations.append(
            {
                "observation_id": evaluation_id,
                "observation_type": "thesis_alert",
                "candidate_id": candidate_id,
                "symbol": str(symbol).upper(),
                "benchmark_symbol": "SPY",
                "decision_date": created_at.date().isoformat(),
                "recorded_at": created_at.astimezone(UTC).isoformat(),
                "evidence_date": analysis_date.isoformat(),
                "decision": verdict,
                "from_stage": None,
                "note": payload.get("message"),
                "score": None,
                "decision_confidence": None,
                "triggered_rules": triggered,
                "version_number": (payload.get("version") or {}).get("version_number"),
            }
        )
    return observations


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
    benchmark_window = benchmark.loc[
        (benchmark["Date"] >= entry_date) & (benchmark["Date"] <= exit_date)
    ].reset_index(drop=True)
    if len(benchmark_window) < 2:
        return None
    gross_return = float(exit_row["Close"] / entry["Close"] - 1)
    net_return = gross_return - transaction_cost_bps / 10_000
    benchmark_return = float(
        benchmark_window.iloc[-1]["Close"] / benchmark_window.iloc[0]["Close"] - 1
    )
    excess_return = net_return - benchmark_return
    decision = observation["decision"]
    if observation["observation_type"] == "candidate_decision":
        proceed = decision in {"qualified", "paper_position"}
        decision_success = excess_return > 0 if proceed else excess_return <= 0
        counterfactual_return = benchmark_return
        counterfactual_delta = excess_return if proceed else -excess_return
        classification = "helpful_decision" if decision_success else "decision_regret"
        assumed_exposure = 1.0 if proceed else 0.0
    else:
        assumed_exposure = 0.0 if decision == "invalidated" else 0.5
        counterfactual_return = net_return * assumed_exposure
        counterfactual_delta = counterfactual_return - net_return
        decision_success = counterfactual_delta > 0
        if counterfactual_delta > 0.005:
            classification = "helpful_alert"
        elif counterfactual_delta < -0.005:
            classification = "false_alert"
        else:
            classification = "neutral_alert"
    outcome = {
        **observation,
        "horizon_sessions": horizon,
        "entry_date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "entry_price": _number(entry["Close"]),
        "exit_price": _number(exit_row["Close"]),
        "gross_return": _number(gross_return),
        "net_return": _number(net_return),
        "benchmark_return": _number(benchmark_return),
        "excess_return": _number(excess_return),
        "decision_success": decision_success,
        "assumed_exposure": assumed_exposure,
        "counterfactual_return": _number(counterfactual_return),
        "counterfactual_delta": _number(counterfactual_delta),
        "classification": classification,
    }
    outcome["outcome_id"] = hashlib.sha256(
        (
            f"{DECISION_REVIEW_VERSION}|{observation['observation_id']}|{horizon}|"
            f"{entry_date}|{exit_date}|{transaction_cost_bps:.6f}|{assumed_exposure}"
        ).encode()
    ).hexdigest()
    return outcome


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "observations": 0,
            "success_rate": None,
            "average_return": None,
            "average_excess_return": None,
            "median_excess_return": None,
            "average_counterfactual_delta": None,
        }
    returns = [float(item["net_return"]) for item in rows]
    excess = [float(item["excess_return"]) for item in rows]
    deltas = [
        float(item["counterfactual_delta"])
        for item in rows
        if item.get("counterfactual_delta") is not None
    ]
    return {
        "observations": len(rows),
        "success_rate": _number(
            sum(bool(item["decision_success"]) for item in rows) / len(rows)
        ),
        "average_return": _number(sum(returns) / len(returns)),
        "average_excess_return": _number(sum(excess) / len(excess)),
        "median_excess_return": _number(median(excess)),
        "average_counterfactual_delta": (
            _number(sum(deltas) / len(deltas)) if deltas else None
        ),
    }


def _decision_attribution(
    outcomes: list[dict[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in outcomes:
        if item["horizon_sessions"] == horizon and item["observation_type"] == "candidate_decision":
            grouped[item["decision"]].append(item)
    return [
        {"decision": decision, **_summary(rows)}
        for decision, rows in sorted(grouped.items())
    ]


def _rule_attribution(
    outcomes: list[dict[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    rule_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for outcome in outcomes:
        if outcome["horizon_sessions"] != horizon or outcome["observation_type"] != "thesis_alert":
            continue
        for rule in outcome.get("triggered_rules", []):
            key = (
                rule.get("metric"),
                rule.get("operator"),
                rule.get("threshold"),
                rule.get("severity"),
            )
            grouped[key].append(outcome)
            rule_meta[key] = {
                "metric": rule.get("metric"),
                "label": rule.get("label") or str(rule.get("metric", "rule")).replace("_", " "),
                "operator": rule.get("operator"),
                "threshold": rule.get("threshold"),
                "severity": rule.get("severity"),
            }
    result = []
    for key, rows in grouped.items():
        helpful = sum(item["classification"] == "helpful_alert" for item in rows)
        false = sum(item["classification"] == "false_alert" for item in rows)
        result.append(
            {
                **rule_meta[key],
                **_summary(rows),
                "helpful_count": helpful,
                "false_alert_count": false,
                "helpful_rate": _number(helpful / len(rows)),
            }
        )
    return sorted(
        result,
        key=lambda item: (
            -(item["observations"] or 0),
            -(item["helpful_rate"] or 0),
            item["metric"],
        ),
    )


def build_decision_review(
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
    transaction_cost_bps: float = 10.0,
    minimum_observations: int = 3,
    observations: list[dict[str, Any]] | None = None,
    price_loader: PriceLoader | None = None,
) -> dict[str, Any]:
    """Build a no-look-ahead decision and alert counterfactual review."""
    cfg = config or get_config()
    report_date = date.fromisoformat(as_of_date)
    resolved_horizons = tuple(sorted({int(value) for value in horizons}))
    if not resolved_horizons or any(value < 1 or value > 252 for value in resolved_horizons):
        raise ValueError("Decision review horizons must be between 1 and 252 sessions")
    if not 0 <= transaction_cost_bps <= 500:
        raise ValueError("Transaction costs must be between 0 and 500 bps")
    if not 1 <= minimum_observations <= 500:
        raise ValueError("Minimum observations must be between 1 and 500")
    source = observations if observations is not None else _load_observations(cfg)
    eligible = [
        item
        for item in source
        if date.fromisoformat(item["decision_date"]) < report_date
    ]
    if eligible:
        earliest = min(date.fromisoformat(item["decision_date"]) for item in eligible)
        symbols = sorted(
            {
                symbol
                for item in eligible
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
        for observation in eligible
        for horizon in resolved_horizons
        if (
            outcome := _forward_outcome(
                observation, horizon, frames, transaction_cost_bps
            )
        )
        is not None
    ]
    primary_horizon = 21 if 21 in resolved_horizons else resolved_horizons[0]
    primary = [item for item in outcomes if item["horizon_sessions"] == primary_horizon]
    decisions = [item for item in primary if item["observation_type"] == "candidate_decision"]
    alerts = [item for item in primary if item["observation_type"] == "thesis_alert"]
    helpful_alerts = sum(item["classification"] == "helpful_alert" for item in alerts)
    false_alerts = sum(item["classification"] == "false_alert" for item in alerts)
    if len(primary) < minimum_observations:
        verdict = "insufficient_data"
    elif false_alerts > helpful_alerts:
        verdict = "learning_required"
    elif helpful_alerts and helpful_alerts >= false_alerts * 2:
        verdict = "effective"
    else:
        verdict = "mixed"
    horizon_rows = []
    for horizon in resolved_horizons:
        selected = [item for item in outcomes if item["horizon_sessions"] == horizon]
        alert_rows = [item for item in selected if item["observation_type"] == "thesis_alert"]
        horizon_rows.append(
            {
                "horizon_sessions": horizon,
                **_summary(selected),
                "decision_count": sum(
                    item["observation_type"] == "candidate_decision" for item in selected
                ),
                "alert_count": len(alert_rows),
                "helpful_alert_count": sum(
                    item["classification"] == "helpful_alert" for item in alert_rows
                ),
                "false_alert_count": sum(
                    item["classification"] == "false_alert" for item in alert_rows
                ),
            }
        )
    report = {
        "model_version": DECISION_REVIEW_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "configuration": {
            "horizons": list(resolved_horizons),
            "primary_horizon": primary_horizon,
            "transaction_cost_bps": transaction_cost_bps,
            "minimum_observations": minimum_observations,
            "entry_policy": "First archived close strictly after the recorded decision timestamp",
            "counterfactual_policy": {
                "invalidated": "0% exposure after the next close",
                "warning": "50% exposure after the next close",
                "candidate_decision": "Proceed/avoid compared with the benchmark alternative",
            },
        },
        "coverage": {
            "source_observations": len(eligible),
            "candidate_decisions": sum(
                item["observation_type"] == "candidate_decision" for item in eligible
            ),
            "thesis_alerts": sum(
                item["observation_type"] == "thesis_alert" for item in eligible
            ),
            "matured_outcomes": len(outcomes),
            "pending_outcomes": len(eligible) * len(resolved_horizons) - len(outcomes),
            "earliest_decision_date": earliest.isoformat() if earliest else None,
            "data_errors": errors,
        },
        "summary": {
            "primary_horizon": primary_horizon,
            "decisions_reviewed": len(decisions),
            "decision_accuracy": _summary(decisions)["success_rate"],
            "alerts_reviewed": len(alerts),
            "helpful_alert_count": helpful_alerts,
            "false_alert_count": false_alerts,
            "alert_helpful_rate": _number(helpful_alerts / len(alerts)) if alerts else None,
            "average_counterfactual_delta": _summary(alerts)[
                "average_counterfactual_delta"
            ],
        },
        "horizons": horizon_rows,
        "decision_attribution": _decision_attribution(outcomes, primary_horizon),
        "rule_attribution": _rule_attribution(outcomes, primary_horizon),
        "timeline": sorted(
            primary,
            key=lambda item: (item["decision_date"], item["symbol"]),
            reverse=True,
        ),
        "outcomes": outcomes,
        "policy": {
            "llm_calls": False,
            "network_fetches": False,
            "orders": False,
            "automatic_model_changes": False,
        },
        "limitations": (
            "Counterfactual returns are standardized exposure proxies, not realized portfolio P&L. "
            "Repeated symbols and overlapping horizons are not independent. Missing archived closes "
            "remain pending and are never imputed. Rule changes require prospective validation."
        ),
    }
    return report


def capture_decision_review(
    report: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = config or get_config()
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
            INSERT OR IGNORE INTO decision_review_runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
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
                _safe_json(report),
                now,
            ],
        )
        for outcome in report["outcomes"]:
            metadata = {
                key: outcome.get(key)
                for key in (
                    "candidate_id",
                    "decision",
                    "from_stage",
                    "recorded_at",
                    "evidence_date",
                    "note",
                    "score",
                    "decision_confidence",
                    "triggered_rules",
                    "assumed_exposure",
                    "counterfactual_return",
                )
            }
            connection.execute(
                """
                INSERT OR IGNORE INTO decision_review_outcomes VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    outcome["outcome_id"],
                    run_id,
                    outcome["observation_id"],
                    outcome["observation_type"],
                    outcome["symbol"],
                    outcome["decision_date"],
                    outcome["horizon_sessions"],
                    outcome["entry_date"],
                    outcome["exit_date"],
                    outcome["net_return"],
                    outcome["benchmark_return"],
                    outcome["excess_return"],
                    outcome["decision_success"],
                    outcome.get("counterfactual_delta"),
                    outcome["classification"],
                    _safe_json(metadata),
                    now,
                ],
            )
        connection.execute("COMMIT")
        row = connection.execute(
            "SELECT created_at, reviewed_at FROM decision_review_runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return {
        "saved": True,
        "run_id": run_id,
        "created_at": row[0].astimezone(UTC).isoformat(),
        "reviewed_at": row[1].astimezone(UTC).isoformat() if row[1] else None,
    }


def get_latest_decision_review(
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT run_id, result_json, created_at, reviewed_at
            FROM decision_review_runs ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    report = json.loads(row[1])
    report["registry"] = {
        "saved": True,
        "run_id": row[0],
        "created_at": row[2].astimezone(UTC).isoformat(),
        "reviewed_at": row[3].astimezone(UTC).isoformat() if row[3] else None,
    }
    return report


def list_decision_reviews(
    config: dict[str, Any] | None = None, *, limit: int = 20
) -> list[dict[str, Any]]:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT run_id, model_version, as_of_date, horizons_json,
                   observation_count, matured_outcome_count, verdict,
                   created_at, reviewed_at
            FROM decision_review_runs ORDER BY created_at DESC LIMIT ?
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
            "observation_count": row[4],
            "matured_outcome_count": row[5],
            "verdict": row[6],
            "created_at": row[7].astimezone(UTC).isoformat(),
            "reviewed_at": row[8].astimezone(UTC).isoformat() if row[8] else None,
        }
        for row in rows
    ]


def mark_decision_review_reviewed(
    run_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT reviewed_at FROM decision_review_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if not row:
            return None
        reviewed_at = row[0] or datetime.now(UTC)
        connection.execute(
            "UPDATE decision_review_runs SET reviewed_at = ? WHERE run_id = ?",
            [reviewed_at, run_id],
        )
    finally:
        connection.close()
    return {
        "run_id": run_id,
        "reviewed_at": reviewed_at.astimezone(UTC).isoformat(),
    }
