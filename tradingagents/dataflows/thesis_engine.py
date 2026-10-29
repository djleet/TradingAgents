"""Version research theses and test explicit invalidation conditions."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.dataflows.candidate_pipeline import get_candidate, list_candidates
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.warehouse import (
    _connect as _warehouse_connect,
    dependency_available,
    warehouse_path,
)

THESIS_ENGINE_VERSION = "thesis-invalidation-v1"
OPERATORS = {"lt", "lte", "gt", "gte", "eq", "neq"}
SEVERITIES = {"warning", "critical"}
METRICS = {
    "opportunity_score": {"label": "Opportunity score", "unit": "score"},
    "decision_confidence": {"label": "Decision confidence", "unit": "score"},
    "data_quality_score": {"label": "Data quality", "unit": "score"},
    "event_risk": {"label": "Event risk", "unit": "score"},
    "consensus_score": {"label": "Consensus score", "unit": "score"},
    "agreement_rate": {"label": "Source agreement", "unit": "percent"},
    "unresolved_critical_conflicts": {
        "label": "Unresolved critical conflicts",
        "unit": "count",
    },
    "validation_sharpe": {"label": "Validation Sharpe", "unit": "ratio"},
    "validation_oos_return_pct": {
        "label": "Out-of-sample return",
        "unit": "percent",
    },
    "analyst_target_upside_pct": {
        "label": "Analyst target upside",
        "unit": "percent",
    },
    "put_call_open_interest": {"label": "Put/call open interest", "unit": "ratio"},
    "fundamental_score": {"label": "Filed fundamental score", "unit": "score"},
    "expected_move_pct": {"label": "Options expected move", "unit": "percent"},
}

_LOCK = threading.RLock()


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _number(value: Any, digits: int = 4) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if math.isfinite(result) else None


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_thesis_versions (
            version_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            candidate_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            version_number INTEGER NOT NULL,
            thesis_text VARCHAR NOT NULL,
            rules_json VARCHAR NOT NULL,
            note VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            UNIQUE(candidate_id, version_number)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_candidate_thesis_versions
        ON candidate_thesis_versions(candidate_id, version_number, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS thesis_invalidation_evaluations (
            evaluation_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            version_id VARCHAR NOT NULL,
            candidate_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            analysis_date DATE NOT NULL,
            verdict VARCHAR NOT NULL,
            triggered_count INTEGER NOT NULL,
            critical_count INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_thesis_invalidation_evaluations
        ON thesis_invalidation_evaluations(candidate_id, analysis_date, created_at)
        """
    )


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    connection = _warehouse_connect(warehouse_path(cfg))
    _ensure_schema(connection)
    return connection


def metric_catalog() -> list[dict[str, str]]:
    """Return the stable metric catalog for API and UI controls."""
    return [
        {"metric": metric, "label": details["label"], "unit": details["unit"]}
        for metric, details in METRICS.items()
    ]


def normalize_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and canonicalize explicit invalidation rules."""
    if not 1 <= len(rules) <= 12:
        raise ValueError("A thesis version requires between 1 and 12 invalidation rules")
    normalized = []
    for raw in rules:
        metric = str(raw.get("metric") or "").strip()
        operator = str(raw.get("operator") or "").strip().lower()
        severity = str(raw.get("severity") or "warning").strip().lower()
        threshold = _number(raw.get("threshold"))
        rationale = str(raw.get("rationale") or "").strip()[:300]
        if metric not in METRICS:
            raise ValueError(f"Unknown thesis metric: {metric}")
        if operator not in OPERATORS:
            raise ValueError(f"Unknown thesis operator: {operator}")
        if severity not in SEVERITIES:
            raise ValueError("Rule severity must be warning or critical")
        if threshold is None:
            raise ValueError("Every thesis rule requires a finite numeric threshold")
        if len(rationale) < 3:
            raise ValueError("Every thesis rule requires a short rationale")
        identity = {
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "severity": severity,
            "rationale": rationale,
        }
        normalized.append(
            {
                "rule_id": hashlib.sha256(_safe_json(identity).encode()).hexdigest(),
                **identity,
                "label": METRICS[metric]["label"],
                "unit": METRICS[metric]["unit"],
            }
        )
    if len({item["rule_id"] for item in normalized}) != len(normalized):
        raise ValueError("Duplicate thesis invalidation rules are not allowed")
    return normalized


def _version_from_row(row) -> dict[str, Any]:
    return {
        "version_id": row[0],
        "model_version": row[1],
        "candidate_id": row[2],
        "symbol": row[3],
        "version_number": row[4],
        "thesis_text": row[5],
        "rules": json.loads(row[6]),
        "note": row[7],
        "created_at": row[8].astimezone(UTC).isoformat(),
    }


def get_latest_thesis_version(
    candidate_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _connect(config)
    if connection is None:
        return None
    try:
        row = connection.execute(
            """
            SELECT version_id, model_version, candidate_id, symbol, version_number,
                   thesis_text, rules_json, note, created_at
            FROM candidate_thesis_versions WHERE candidate_id = ?
            ORDER BY version_number DESC LIMIT 1
            """,
            [candidate_id],
        ).fetchone()
    finally:
        connection.close()
    return _version_from_row(row) if row else None


def create_thesis_version(
    candidate_id: str,
    thesis_text: str,
    rules: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    note: str = "Research thesis version saved",
) -> dict[str, Any]:
    """Append an immutable thesis version, deduplicating unchanged saves."""
    cfg = config or get_config()
    thesis = str(thesis_text or "").strip()[:2000]
    if len(thesis) < 10:
        raise ValueError("Research thesis must contain at least 10 characters")
    normalized = normalize_rules(rules)
    candidate = get_candidate(candidate_id, cfg)
    if candidate is None:
        raise ValueError("Candidate not found")
    with _LOCK:
        connection = _connect(cfg)
        if connection is None:
            raise RuntimeError("Thesis Engine requires the local data warehouse")
        try:
            latest_row = connection.execute(
                """
                SELECT version_id, model_version, candidate_id, symbol, version_number,
                       thesis_text, rules_json, note, created_at
                FROM candidate_thesis_versions WHERE candidate_id = ?
                ORDER BY version_number DESC LIMIT 1
                """,
                [candidate_id],
            ).fetchone()
            if latest_row:
                latest = _version_from_row(latest_row)
                if latest["thesis_text"] == thesis and latest["rules"] == normalized:
                    return {**latest, "created": False}
                version_number = int(latest["version_number"]) + 1
            else:
                version_number = 1
            identity = {
                "candidate_id": candidate_id,
                "symbol": candidate["symbol"],
                "version_number": version_number,
                "thesis_text": thesis,
                "rules": normalized,
            }
            version_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
            now = datetime.now(UTC)
            connection.execute(
                """
                INSERT INTO candidate_thesis_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    version_id,
                    THESIS_ENGINE_VERSION,
                    candidate_id,
                    candidate["symbol"],
                    version_number,
                    thesis,
                    _safe_json(normalized),
                    str(note or "Research thesis version saved").strip()[:1000],
                    now,
                ],
            )
        finally:
            connection.close()
    return {
        "version_id": version_id,
        "model_version": THESIS_ENGINE_VERSION,
        "candidate_id": candidate_id,
        "symbol": candidate["symbol"],
        "version_number": version_number,
        "thesis_text": thesis,
        "rules": normalized,
        "note": str(note or "Research thesis version saved").strip()[:1000],
        "created_at": now.isoformat(),
        "created": True,
    }


def _set_metric(
    values: dict[str, dict[str, Any]], metric: str, value: Any, source: str
) -> None:
    number = _number(value)
    if number is not None:
        values[metric] = {
            "value": number,
            "label": METRICS[metric]["label"],
            "unit": METRICS[metric]["unit"],
            "source": source,
        }


def _metric_values(
    candidate: dict[str, Any],
    bundle: dict[str, Any],
    consensus: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    _set_metric(values, "opportunity_score", candidate.get("latest_score"), "candidate")
    _set_metric(
        values, "decision_confidence", candidate.get("decision_confidence"), "candidate"
    )
    _set_metric(values, "data_quality_score", candidate.get("quality_score"), "candidate")
    _set_metric(values, "event_risk", candidate.get("event_risk"), "candidate")
    by_dataset = {item.get("dataset"): item for item in bundle.get("items", [])}
    opportunity = (by_dataset.get("opportunity_scan") or {}).get("facts") or {}
    _set_metric(
        values, "opportunity_score", opportunity.get("opportunity_score"), "opportunity_scan"
    )
    validation = (by_dataset.get("validation") or {}).get("facts") or {}
    _set_metric(values, "validation_sharpe", validation.get("sharpe_ratio"), "validation")
    if validation.get("oos_return") is not None:
        _set_metric(
            values,
            "validation_oos_return_pct",
            float(validation["oos_return"]) * 100,
            "validation",
        )
    earnings = (by_dataset.get("earnings_estimates") or {}).get("facts") or {}
    if earnings.get("target_upside") is not None:
        _set_metric(
            values,
            "analyst_target_upside_pct",
            float(earnings["target_upside"]) * 100,
            "earnings_estimates",
        )
    options = (by_dataset.get("options_implied_volatility") or {}).get("facts") or {}
    _set_metric(
        values,
        "put_call_open_interest",
        options.get("put_call_open_interest"),
        "options_implied_volatility",
    )
    if options.get("expected_move_pct") is not None:
        _set_metric(
            values,
            "expected_move_pct",
            float(options["expected_move_pct"]) * 100,
            "options_implied_volatility",
        )
    context = (by_dataset.get("research_context") or {}).get("facts") or {}
    _set_metric(
        values, "fundamental_score", context.get("fundamental_score"), "research_context"
    )
    _set_metric(values, "event_risk", context.get("event_risk_score"), "research_context")
    _set_metric(
        values,
        "decision_confidence",
        context.get("decision_confidence"),
        "research_context",
    )
    if consensus:
        gate = consensus.get("qualification_gate") or {}
        _set_metric(values, "consensus_score", consensus.get("consensus_score"), "consensus")
        _set_metric(values, "agreement_rate", consensus.get("agreement_rate"), "consensus")
        _set_metric(
            values,
            "unresolved_critical_conflicts",
            gate.get("unresolved_critical_count", 0),
            "consensus",
        )
    return values


def _triggered(value: float, operator: str, threshold: float) -> bool:
    if operator == "lt":
        return value < threshold
    if operator == "lte":
        return value <= threshold
    if operator == "gt":
        return value > threshold
    if operator == "gte":
        return value >= threshold
    if operator == "eq":
        return math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-9)
    return not math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-9)


def evaluate_thesis(
    candidate_id: str,
    analysis_date: str,
    config: dict[str, Any] | None = None,
    *,
    bundle: dict[str, Any] | None = None,
    consensus: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Evaluate the active thesis without mutating the candidate decision."""
    date.fromisoformat(analysis_date)
    cfg = config or get_config()
    current = candidate or get_candidate(candidate_id, cfg)
    if current is None:
        raise ValueError("Candidate not found")
    version = get_latest_thesis_version(candidate_id, cfg)
    if version is None:
        raise ValueError("Candidate has no versioned research thesis")
    routed = bundle or current.get("evidence", {}).get("evidence_router") or {}
    resolved_consensus = (
        consensus or current.get("evidence", {}).get("consensus_engine") or None
    )
    values = _metric_values(current, routed, resolved_consensus)
    results = []
    for rule in version["rules"]:
        observed = values.get(rule["metric"])
        available = observed is not None
        is_triggered = (
            _triggered(observed["value"], rule["operator"], rule["threshold"])
            if observed
            else False
        )
        results.append(
            {
                **rule,
                "available": available,
                "observed_value": observed["value"] if observed else None,
                "source": observed["source"] if observed else None,
                "triggered": is_triggered,
            }
        )
    triggered = [item for item in results if item["triggered"]]
    critical = [item for item in triggered if item["severity"] == "critical"]
    evaluated = [item for item in results if item["available"]]
    if critical:
        verdict = "invalidated"
    elif triggered:
        verdict = "warning"
    elif evaluated:
        verdict = "intact"
    else:
        verdict = "insufficient"
    report = {
        "model_version": THESIS_ENGINE_VERSION,
        "candidate_id": candidate_id,
        "symbol": current["symbol"],
        "analysis_date": analysis_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "version": version,
        "verdict": verdict,
        "triggered_count": len(triggered),
        "critical_count": len(critical),
        "evaluated_rule_count": len(evaluated),
        "missing_rule_count": len(results) - len(evaluated),
        "rules": results,
        "metrics": values,
        "message": (
            "One or more critical invalidation conditions are true. Review the thesis explicitly."
            if critical
            else "One or more warning conditions are true; the thesis remains active pending review."
            if triggered
            else "No available invalidation condition is currently true."
            if evaluated
            else "None of the configured invalidation metrics is currently available."
        ),
        "policy": {
            "decision_effect": "alert_only",
            "stage_changes": False,
            "orders": False,
            "missing_data": "Missing metrics are visible but never treated as triggered rules.",
        },
        "evidence": {
            "bundle_id": routed.get("bundle_id"),
            "consensus_report_id": (
                (resolved_consensus.get("registry") or {}).get("report_id")
                or resolved_consensus.get("report_id")
                if resolved_consensus
                else None
            ),
        },
    }
    identity = {key: value for key, value in report.items() if key != "generated_at"}
    evaluation_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    report["evaluation_id"] = evaluation_id
    if not persist:
        return report
    now = datetime.now(UTC)
    with _LOCK:
        connection = _connect(cfg)
        if connection is None:
            raise RuntimeError("Thesis Engine requires the local data warehouse")
        try:
            connection.execute(
                """
                INSERT OR IGNORE INTO thesis_invalidation_evaluations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    evaluation_id,
                    THESIS_ENGINE_VERSION,
                    version["version_id"],
                    candidate_id,
                    current["symbol"],
                    analysis_date,
                    verdict,
                    len(triggered),
                    len(critical),
                    _safe_json(report),
                    now,
                ],
            )
            created_at = connection.execute(
                """SELECT created_at FROM thesis_invalidation_evaluations
                   WHERE evaluation_id = ?""",
                [evaluation_id],
            ).fetchone()[0]
        finally:
            connection.close()
    report["registry"] = {
        "saved": True,
        "evaluation_id": evaluation_id,
        "created_at": created_at.astimezone(UTC).isoformat(),
    }
    return report


def _evaluation_from_row(row) -> dict[str, Any]:
    report = json.loads(row[1])
    report["registry"] = {
        "saved": True,
        "evaluation_id": row[0],
        "created_at": row[2].astimezone(UTC).isoformat(),
    }
    return report


def get_latest_thesis_evaluation(
    candidate_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _connect(config)
    if connection is None:
        return None
    try:
        row = connection.execute(
            """
            SELECT evaluation_id, payload_json, created_at
            FROM thesis_invalidation_evaluations WHERE candidate_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            [candidate_id],
        ).fetchone()
    finally:
        connection.close()
    return _evaluation_from_row(row) if row else None


def get_thesis_state(
    candidate_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    connection = _connect(config)
    if connection is None:
        return {"available": False, "versions": [], "active_version": None, "evaluation": None}
    try:
        version_rows = connection.execute(
            """
            SELECT version_id, model_version, candidate_id, symbol, version_number,
                   thesis_text, rules_json, note, created_at
            FROM candidate_thesis_versions WHERE candidate_id = ?
            ORDER BY version_number DESC LIMIT 50
            """,
            [candidate_id],
        ).fetchall()
        evaluation_rows = connection.execute(
            """
            SELECT evaluation_id, payload_json, created_at
            FROM thesis_invalidation_evaluations WHERE candidate_id = ?
            ORDER BY created_at DESC LIMIT 30
            """,
            [candidate_id],
        ).fetchall()
    finally:
        connection.close()
    versions = [_version_from_row(row) for row in version_rows]
    evaluations = [_evaluation_from_row(row) for row in evaluation_rows]
    active = versions[0] if versions else None
    latest_for_active = next(
        (
            item
            for item in evaluations
            if active and item.get("version", {}).get("version_id") == active["version_id"]
        ),
        None,
    )
    return {
        "available": active is not None,
        "active_version": active,
        "evaluation": latest_for_active,
        "versions": versions,
        "evaluations": evaluations,
    }


def get_thesis_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or get_config()
    candidates = [
        item
        for item in list_candidates(cfg).get("items", [])
        if item.get("stage") not in {"closed", "rejected"}
    ]
    reports = []
    for candidate in candidates:
        state = get_thesis_state(candidate["candidate_id"], cfg)
        evaluation = state.get("evaluation") or {}
        active = state.get("active_version") or {}
        reports.append(
            {
                "candidate_id": candidate["candidate_id"],
                "symbol": candidate["symbol"],
                "version_number": active.get("version_number"),
                "rule_count": len(active.get("rules") or []),
                "verdict": evaluation.get("verdict") if active else "unversioned",
                "triggered_count": evaluation.get("triggered_count", 0),
                "critical_count": evaluation.get("critical_count", 0),
                "evaluation_id": evaluation.get("evaluation_id"),
            }
        )
    return {
        "reports": reports,
        "active_candidate_count": len(candidates),
        "versioned_candidate_count": sum(item["version_number"] is not None for item in reports),
        "invalidated_count": sum(item["verdict"] == "invalidated" for item in reports),
        "warning_count": sum(item["verdict"] == "warning" for item in reports),
        "intact_count": sum(item["verdict"] == "intact" for item in reports),
        "unevaluated_count": sum(
            item["verdict"] in {"unversioned", None} for item in reports
        ),
    }
