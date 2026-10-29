"""Persistent research workflow from discovery to paper-trade outcome.

The pipeline deliberately stores decisions separately from market evidence.  A
new scan may refresh a score, but it never silently promotes or rejects a
candidate.  Stage changes remain explicit and are recorded in an append-only
event log, except for the objective paper-position state which is synchronized
from the simulated portfolio.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import UTC, date, datetime, timedelta
from typing import Any

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.warehouse import (
    _connect as _warehouse_connect,
    dependency_available,
    warehouse_path,
)

_LOCK = threading.RLock()

STAGES = (
    "discovered",
    "validating",
    "qualified",
    "paper_position",
    "closed",
    "rejected",
)
PRIORITIES = ("high", "medium", "low")
ALLOWED_TRANSITIONS = {
    "discovered": {"validating", "rejected"},
    "validating": {"discovered", "qualified", "rejected"},
    "qualified": {"validating", "rejected"},
    "paper_position": {"qualified"},
    "closed": {"discovered"},
    "rejected": {"discovered"},
}


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    connection = _warehouse_connect(warehouse_path(cfg))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS research_candidates (
            candidate_id VARCHAR PRIMARY KEY,
            symbol VARCHAR UNIQUE NOT NULL,
            stage VARCHAR NOT NULL,
            priority VARCHAR NOT NULL,
            thesis VARCHAR NOT NULL,
            next_action VARCHAR NOT NULL,
            review_date DATE,
            source VARCHAR NOT NULL,
            latest_score DOUBLE,
            latest_rank INTEGER,
            latest_bucket VARCHAR,
            decision_confidence DOUBLE,
            decision_label VARCHAR,
            risk_level VARCHAR,
            event_risk DOUBLE,
            quality_score DOUBLE,
            validation_verdict VARCHAR,
            scan_id VARCHAR,
            position_id VARCHAR,
            evidence_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            last_refreshed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_research_candidates_workflow
        ON research_candidates(stage, priority, review_date, updated_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS research_candidate_events (
            event_id VARCHAR PRIMARY KEY,
            candidate_id VARCHAR NOT NULL,
            event_type VARCHAR NOT NULL,
            from_stage VARCHAR,
            to_stage VARCHAR,
            note VARCHAR NOT NULL,
            snapshot_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_research_candidate_events
        ON research_candidate_events(candidate_id, created_at)
        """
    )
    return connection


def _enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("data_warehouse_enabled", False)) and dependency_available()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _validation_verdict(evidence: dict[str, Any]) -> str:
    validation = evidence.get("validation") or {}
    explicit = validation.get("robustness") or validation.get("verdict")
    if explicit:
        return str(explicit).lower().replace(" ", "_")[:40]
    alpha = _number(validation.get("alpha_vs_asset"))
    profitable = _number(validation.get("profitable_fold_rate"))
    sharpe = _number(validation.get("sharpe_ratio"))
    if alpha is None and profitable is None and sharpe is None:
        return "not_run"
    positives = sum(
        (
            alpha is not None and alpha > 0,
            profitable is not None and profitable >= 0.5,
            sharpe is not None and sharpe > 0,
        )
    )
    if positives >= 3:
        return "supported"
    if positives >= 1:
        return "mixed"
    return "weak"


def _evidence_fields(evidence: dict[str, Any]) -> dict[str, Any]:
    context = evidence.get("research_context") or {}
    events = context.get("events") or {}
    decision = evidence.get("decision_confidence") or context.get("decision_confidence") or {}
    quality = evidence.get("quality") or {}
    score = _number(evidence.get("opportunity_score"))
    rank = evidence.get("rank")
    try:
        rank = int(rank) if rank is not None else None
    except (TypeError, ValueError):
        rank = None
    event_risk = _number(events.get("risk_score"))
    volatility = _number(evidence.get("volatility_60"))
    risk_level = events.get("risk_level")
    if not risk_level:
        if volatility is None:
            risk_level = "unknown"
        elif volatility >= 0.45:
            risk_level = "high"
        elif volatility >= 0.25:
            risk_level = "elevated"
        else:
            risk_level = "normal"
    return {
        "latest_score": score,
        "latest_rank": rank,
        "latest_bucket": str(evidence.get("bucket") or "neutral")[:40],
        "decision_confidence": _number(decision.get("score")),
        "decision_label": str(decision.get("label") or "unavailable")[:40],
        "risk_level": str(risk_level).lower().replace(" ", "_")[:40],
        "event_risk": event_risk,
        "quality_score": _number(quality.get("score", evidence.get("quality_score"))),
        "validation_verdict": _validation_verdict(evidence),
    }


def _priority_for(score: float | None) -> str:
    if score is not None and score >= 70:
        return "high"
    if score is not None and score >= 55:
        return "medium"
    return "low"


def _next_action_for(fields: dict[str, Any]) -> str:
    if fields["validation_verdict"] == "not_run":
        return "Run out-of-sample validation"
    if fields["decision_confidence"] is None:
        return "Review SEC context and event risk"
    if fields["risk_level"] in {"high", "critical"}:
        return "Review event risk before qualification"
    if (
        fields["decision_confidence"] >= 65
        and fields["validation_verdict"] == "supported"
    ):
        return "Review for qualification"
    return "Gather more evidence and reassess"


def _candidate_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "symbol",
            "stage",
            "priority",
            "latest_score",
            "latest_rank",
            "latest_bucket",
            "decision_confidence",
            "decision_label",
            "risk_level",
            "event_risk",
            "quality_score",
            "validation_verdict",
            "scan_id",
            "position_id",
            "review_date",
            "next_action",
        )
    }


def _record_event(
    connection,
    candidate_id: str,
    event_type: str,
    *,
    from_stage: str | None,
    to_stage: str | None,
    note: str,
    snapshot: dict[str, Any],
) -> None:
    created_at = datetime.now(UTC)
    event_id = hashlib.sha256(
        f"{candidate_id}|{event_type}|{created_at.isoformat()}|{_safe_json(snapshot)}".encode()
    ).hexdigest()
    connection.execute(
        """
        INSERT INTO research_candidate_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            event_id,
            candidate_id,
            event_type,
            from_stage,
            to_stage,
            note.strip()[:1000],
            _safe_json(snapshot),
            created_at,
        ],
    )


_CANDIDATE_COLUMNS = (
    "candidate_id",
    "symbol",
    "stage",
    "priority",
    "thesis",
    "next_action",
    "review_date",
    "source",
    "latest_score",
    "latest_rank",
    "latest_bucket",
    "decision_confidence",
    "decision_label",
    "risk_level",
    "event_risk",
    "quality_score",
    "validation_verdict",
    "scan_id",
    "position_id",
    "evidence_json",
    "created_at",
    "updated_at",
    "last_refreshed_at",
)


def _row_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    item = dict(zip(_CANDIDATE_COLUMNS, row, strict=True))
    item["evidence"] = json.loads(item.pop("evidence_json") or "{}")
    for key in ("review_date", "created_at", "updated_at", "last_refreshed_at"):
        value = item.get(key)
        item[key] = value.isoformat() if value is not None and hasattr(value, "isoformat") else value
    review_date = item.get("review_date")
    item["review_due"] = bool(
        review_date
        and review_date <= date.today().isoformat()
        and item["stage"] not in {"closed", "rejected"}
    )
    return item


def upsert_candidate(
    symbol: str,
    evidence: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    priority: str | None = None,
    thesis: str = "",
    next_action: str | None = None,
    review_date: str | None = None,
    source: str = "opportunity_scanner",
) -> dict[str, Any]:
    """Create a candidate, or refresh the evidence of its existing record."""
    cfg = config or get_config()
    if not _enabled(cfg):
        raise RuntimeError("Candidate pipeline requires the local data warehouse")
    canonical = normalize_symbol(symbol.strip().upper())
    fields = _evidence_fields(evidence)
    selected_priority = priority or _priority_for(fields["latest_score"])
    if selected_priority not in PRIORITIES:
        raise ValueError("Priority must be high, medium or low")
    if review_date:
        date.fromisoformat(review_date)
    candidate_id = hashlib.sha256(f"candidate|{canonical}".encode()).hexdigest()[:32]
    now = datetime.now(UTC)
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            existing_row = connection.execute(
                "SELECT * FROM research_candidates WHERE symbol = ?", [canonical]
            ).fetchone()
            if existing_row:
                existing = _row_dict(existing_row)
                evidence = {**existing["evidence"], **evidence}
                fields = _evidence_fields(evidence)
                connection.execute(
                    """
                    UPDATE research_candidates SET latest_score = ?, latest_rank = ?,
                        latest_bucket = ?, decision_confidence = ?, decision_label = ?,
                        risk_level = ?, event_risk = ?, quality_score = ?,
                        validation_verdict = ?, scan_id = coalesce(?, scan_id),
                        evidence_json = ?, updated_at = ?, last_refreshed_at = ?
                    WHERE candidate_id = ?
                    """,
                    [
                        fields["latest_score"], fields["latest_rank"], fields["latest_bucket"],
                        fields["decision_confidence"], fields["decision_label"],
                        fields["risk_level"], fields["event_risk"], fields["quality_score"],
                        fields["validation_verdict"], evidence.get("scan_id"),
                        _safe_json(evidence), now, now, existing["candidate_id"],
                    ],
                )
                candidate_id = existing["candidate_id"]
                event_type = "evidence_refreshed"
                note = "Evidence refreshed from the opportunity workflow"
            else:
                resolved_review = review_date or (date.today() + timedelta(days=7)).isoformat()
                resolved_action = next_action or _next_action_for(fields)
                connection.execute(
                    """
                    INSERT INTO research_candidates VALUES (
                        ?, ?, 'discovered', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        NULL, ?, ?, ?, ?
                    )
                    """,
                    [
                        candidate_id, canonical, selected_priority, thesis.strip()[:2000],
                        resolved_action.strip()[:500], resolved_review, source[:60],
                        fields["latest_score"], fields["latest_rank"], fields["latest_bucket"],
                        fields["decision_confidence"], fields["decision_label"],
                        fields["risk_level"], fields["event_risk"], fields["quality_score"],
                        fields["validation_verdict"], evidence.get("scan_id"),
                        _safe_json(evidence), now, now, now,
                    ],
                )
                event_type = "created"
                note = "Promoted from Opportunity Scanner"
            row = connection.execute(
                "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
            ).fetchone()
            result = _row_dict(row)
            _record_event(
                connection,
                candidate_id,
                event_type,
                from_stage=result["stage"] if existing_row else None,
                to_stage=result["stage"],
                note=note,
                snapshot=_candidate_snapshot(result),
            )
            connection.execute("COMMIT")
            return result
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def list_candidates(
    config: dict[str, Any] | None = None,
    *,
    stage: str | None = None,
    priority: str | None = None,
    risk: str | None = None,
) -> dict[str, Any]:
    """Return pipeline cards and stage totals."""
    cfg = config or get_config()
    if not _enabled(cfg):
        return {
            "items": [],
            "count": 0,
            "stage_counts": dict.fromkeys(STAGES, 0),
            "due_count": 0,
        }
    if stage and stage not in STAGES:
        raise ValueError("Unknown candidate stage")
    if priority and priority not in PRIORITIES:
        raise ValueError("Unknown candidate priority")
    clauses: list[str] = []
    params: list[Any] = []
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    if priority:
        clauses.append("priority = ?")
        params.append(priority)
    if risk:
        clauses.append("risk_level = ?")
        params.append(risk)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _LOCK:
        connection = _connect(cfg)
        try:
            rows = connection.execute(
                f"""SELECT * FROM research_candidates {where}
                    ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                             coalesce(review_date, DATE '9999-12-31'), updated_at DESC""",
                params,
            ).fetchall()
            counts = dict(
                connection.execute(
                    "SELECT stage, count(*) FROM research_candidates GROUP BY stage"
                ).fetchall()
            )
        finally:
            connection.close()
    items = [_row_dict(row) for row in rows]
    return {
        "items": items,
        "count": len(items),
        "stage_counts": {item: int(counts.get(item, 0)) for item in STAGES},
        "due_count": sum(item["review_due"] for item in items),
    }


def get_candidate(
    candidate_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return one candidate without mutating its workflow state."""
    cfg = config or get_config()
    if not _enabled(cfg):
        return None
    with _LOCK:
        connection = _connect(cfg)
        try:
            row = connection.execute(
                "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
            ).fetchone()
        finally:
            connection.close()
    return _row_dict(row) if row else None


def attach_candidate_evidence(
    candidate_id: str,
    bundle: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a routed evidence gate to a candidate and audit material changes."""
    cfg = config or get_config()
    if not _enabled(cfg):
        raise RuntimeError("Candidate pipeline requires the local data warehouse")
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
            ).fetchone()
            if not row:
                raise ValueError("Candidate not found")
            current = _row_dict(row)
            prior = current["evidence"].get("evidence_router") or {}
            if prior.get("bundle_id") != bundle.get("bundle_id"):
                evidence = {**current["evidence"], "evidence_router": bundle}
                now = datetime.now(UTC)
                connection.execute(
                    """UPDATE research_candidates SET evidence_json = ?, updated_at = ?,
                              last_refreshed_at = ? WHERE candidate_id = ?""",
                    [_safe_json(evidence), now, now, candidate_id],
                )
                updated = _row_dict(
                    connection.execute(
                        "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
                    ).fetchone()
                )
                _record_event(
                    connection,
                    candidate_id,
                    "evidence_routed",
                    from_stage=current["stage"],
                    to_stage=current["stage"],
                    note=(
                        f"Evidence gate {bundle.get('verdict', 'unknown')}: "
                        f"{bundle.get('eligible_count', 0)} eligible, "
                        f"{bundle.get('blocked_count', 0)} blocked"
                    ),
                    snapshot=_candidate_snapshot(updated),
                )
                current = updated
            connection.execute("COMMIT")
            return current
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def attach_candidate_consensus(
    candidate_id: str,
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach an immutable consensus result without changing workflow stage."""
    cfg = config or get_config()
    if not _enabled(cfg):
        raise RuntimeError("Candidate pipeline requires the local data warehouse")
    report_id = (report.get("registry") or {}).get("report_id") or report.get(
        "report_id"
    )
    if not report_id:
        raise ValueError("Consensus report is missing its report id")
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
            ).fetchone()
            if not row:
                raise ValueError("Candidate not found")
            current = _row_dict(row)
            prior = current["evidence"].get("consensus_engine") or {}
            prior_id = (prior.get("registry") or {}).get("report_id") or prior.get(
                "report_id"
            )
            if prior_id != report_id:
                evidence = {**current["evidence"], "consensus_engine": report}
                now = datetime.now(UTC)
                connection.execute(
                    """UPDATE research_candidates SET evidence_json = ?, updated_at = ?,
                              last_refreshed_at = ? WHERE candidate_id = ?""",
                    [_safe_json(evidence), now, now, candidate_id],
                )
                updated = _row_dict(
                    connection.execute(
                        "SELECT * FROM research_candidates WHERE candidate_id = ?",
                        [candidate_id],
                    ).fetchone()
                )
                gate = report.get("qualification_gate") or {}
                _record_event(
                    connection,
                    candidate_id,
                    "consensus_evaluated",
                    from_stage=current["stage"],
                    to_stage=current["stage"],
                    note=(
                        f"Consensus {report.get('verdict', 'unknown')}: "
                        f"{len(report.get('conflicts', []))} conflicts, "
                        f"{gate.get('unresolved_critical_count', 0)} unresolved critical"
                    ),
                    snapshot=_candidate_snapshot(updated),
                )
                current = updated
            connection.execute("COMMIT")
            return current
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def attach_candidate_thesis_evaluation(
    candidate_id: str,
    evaluation: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a thesis evaluation and record alerts without changing stage."""
    cfg = config or get_config()
    if not _enabled(cfg):
        raise RuntimeError("Candidate pipeline requires the local data warehouse")
    evaluation_id = (evaluation.get("registry") or {}).get(
        "evaluation_id"
    ) or evaluation.get("evaluation_id")
    if not evaluation_id:
        raise ValueError("Thesis evaluation is missing its evaluation id")
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
            ).fetchone()
            if not row:
                raise ValueError("Candidate not found")
            current = _row_dict(row)
            prior = current["evidence"].get("thesis_engine") or {}
            prior_id = (prior.get("registry") or {}).get(
                "evaluation_id"
            ) or prior.get("evaluation_id")
            if prior_id != evaluation_id:
                evidence = {**current["evidence"], "thesis_engine": evaluation}
                now = datetime.now(UTC)
                connection.execute(
                    """UPDATE research_candidates SET evidence_json = ?, updated_at = ?,
                              last_refreshed_at = ? WHERE candidate_id = ?""",
                    [_safe_json(evidence), now, now, candidate_id],
                )
                updated = _row_dict(
                    connection.execute(
                        "SELECT * FROM research_candidates WHERE candidate_id = ?",
                        [candidate_id],
                    ).fetchone()
                )
                verdict = evaluation.get("verdict", "unknown")
                _record_event(
                    connection,
                    candidate_id,
                    "thesis_invalidated"
                    if verdict == "invalidated"
                    else "thesis_evaluated",
                    from_stage=current["stage"],
                    to_stage=current["stage"],
                    note=(
                        f"Thesis {verdict}: "
                        f"{evaluation.get('triggered_count', 0)} triggered rules, "
                        f"{evaluation.get('critical_count', 0)} critical; stage unchanged"
                    ),
                    snapshot=_candidate_snapshot(updated),
                )
                current = updated
            connection.execute("COMMIT")
            return current
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def update_candidate(
    candidate_id: str,
    updates: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply an explicit workflow decision and append it to the audit trail."""
    cfg = config or get_config()
    if not _enabled(cfg):
        raise RuntimeError("Candidate pipeline requires the local data warehouse")
    allowed = {"stage", "priority", "thesis", "next_action", "review_date", "note"}
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"Unsupported candidate fields: {', '.join(sorted(unknown))}")
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
            ).fetchone()
            if not row:
                raise ValueError("Candidate not found")
            current = _row_dict(row)
            target_stage = updates.get("stage", current["stage"])
            if target_stage not in STAGES:
                raise ValueError("Unknown candidate stage")
            if target_stage != current["stage"] and target_stage not in ALLOWED_TRANSITIONS[current["stage"]]:
                raise ValueError(f"Cannot move candidate from {current['stage']} to {target_stage}")
            selected_priority = updates.get("priority", current["priority"])
            if selected_priority not in PRIORITIES:
                raise ValueError("Priority must be high, medium or low")
            selected_review = updates.get("review_date", current["review_date"])
            if selected_review:
                date.fromisoformat(str(selected_review))
            thesis = str(updates.get("thesis", current["thesis"]) or "").strip()[:2000]
            next_action = str(updates.get("next_action", current["next_action"]) or "").strip()[:500]
            if not next_action:
                raise ValueError("Next action cannot be empty")
            now = datetime.now(UTC)
            connection.execute(
                """
                UPDATE research_candidates SET stage = ?, priority = ?, thesis = ?,
                    next_action = ?, review_date = ?, updated_at = ?
                WHERE candidate_id = ?
                """,
                [target_stage, selected_priority, thesis, next_action, selected_review, now, candidate_id],
            )
            result = _row_dict(
                connection.execute(
                    "SELECT * FROM research_candidates WHERE candidate_id = ?", [candidate_id]
                ).fetchone()
            )
            _record_event(
                connection,
                candidate_id,
                "stage_changed" if target_stage != current["stage"] else "details_updated",
                from_stage=current["stage"],
                to_stage=target_stage,
                note=str(updates.get("note") or "Candidate review updated"),
                snapshot=_candidate_snapshot(result),
            )
            connection.execute("COMMIT")
            return result
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def list_candidate_history(
    candidate_id: str,
    config: dict[str, Any] | None = None,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return newest-first audit events for one candidate."""
    cfg = config or get_config()
    if not _enabled(cfg):
        return []
    with _LOCK:
        connection = _connect(cfg)
        try:
            rows = connection.execute(
                """
                SELECT event_id, event_type, from_stage, to_stage, note, snapshot_json, created_at
                FROM research_candidate_events WHERE candidate_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                [candidate_id, max(1, min(int(limit), 200))],
            ).fetchall()
        finally:
            connection.close()
    return [
        {
            "event_id": row[0],
            "event_type": row[1],
            "from_stage": row[2],
            "to_stage": row[3],
            "note": row[4],
            "snapshot": json.loads(row[5]),
            "created_at": row[6].isoformat(),
        }
        for row in rows
    ]


def refresh_candidates_from_scan(
    report: dict[str, Any],
    scan_id: str | None,
    config: dict[str, Any] | None = None,
) -> int:
    """Refresh existing candidates when a manual or scheduled scan finishes."""
    cfg = config or get_config()
    if not _enabled(cfg):
        return 0
    items = {str(item.get("symbol", "")).upper(): item for item in report.get("items", [])}
    refreshed = 0
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            rows = connection.execute(
                "SELECT * FROM research_candidates WHERE stage NOT IN ('closed', 'rejected')"
            ).fetchall()
            for raw in rows:
                current = _row_dict(raw)
                evidence = items.get(current["symbol"])
                if not evidence:
                    continue
                evidence = {**current["evidence"], **evidence, "scan_id": scan_id}
                fields = _evidence_fields(evidence)
                previous_snapshot = _candidate_snapshot(current)
                now = datetime.now(UTC)
                connection.execute(
                    """
                    UPDATE research_candidates SET latest_score = ?, latest_rank = ?,
                        latest_bucket = ?, decision_confidence = ?, decision_label = ?,
                        risk_level = ?, event_risk = ?, quality_score = ?,
                        validation_verdict = ?, scan_id = ?, evidence_json = ?,
                        updated_at = ?, last_refreshed_at = ? WHERE candidate_id = ?
                    """,
                    [
                        fields["latest_score"], fields["latest_rank"], fields["latest_bucket"],
                        fields["decision_confidence"], fields["decision_label"],
                        fields["risk_level"], fields["event_risk"], fields["quality_score"],
                        fields["validation_verdict"], scan_id, _safe_json(evidence),
                        now, now, current["candidate_id"],
                    ],
                )
                updated = _row_dict(
                    connection.execute(
                        "SELECT * FROM research_candidates WHERE candidate_id = ?",
                        [current["candidate_id"]],
                    ).fetchone()
                )
                snapshot = _candidate_snapshot(updated)
                if snapshot != previous_snapshot:
                    _record_event(
                        connection,
                        current["candidate_id"],
                        "evidence_refreshed",
                        from_stage=current["stage"],
                        to_stage=current["stage"],
                        note="Score, rank or evidence refreshed by a completed scan",
                        snapshot=snapshot,
                    )
                refreshed += 1
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return refreshed


def sync_candidates_with_paper(
    portfolio: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> int:
    """Reflect objective paper-position state without creating live orders."""
    cfg = config or get_config()
    if not _enabled(cfg):
        return 0
    open_by_symbol = {item["symbol"]: item for item in portfolio.get("open_positions", [])}
    closed_by_symbol: dict[str, dict[str, Any]] = {}
    for item in portfolio.get("closed_positions", []):
        closed_by_symbol[item["symbol"]] = item
    changed = 0
    with _LOCK:
        connection = _connect(cfg)
        try:
            connection.execute("BEGIN TRANSACTION")
            rows = connection.execute("SELECT * FROM research_candidates").fetchall()
            for raw in rows:
                current = _row_dict(raw)
                position = open_by_symbol.get(current["symbol"])
                target = current["stage"]
                note = ""
                position_id = current["position_id"]
                if position and current["stage"] not in {"closed", "rejected"}:
                    target = "paper_position"
                    position_id = position.get("position_id")
                    note = "Linked to an open simulated position"
                elif current["stage"] == "paper_position" and current["symbol"] in closed_by_symbol:
                    target = "closed"
                    position_id = closed_by_symbol[current["symbol"]].get("position_id")
                    note = "Simulated position closed; outcome retained in the journal"
                if target == current["stage"] and position_id == current["position_id"]:
                    continue
                now = datetime.now(UTC)
                connection.execute(
                    "UPDATE research_candidates SET stage = ?, position_id = ?, updated_at = ? WHERE candidate_id = ?",
                    [target, position_id, now, current["candidate_id"]],
                )
                updated = {**current, "stage": target, "position_id": position_id}
                _record_event(
                    connection,
                    current["candidate_id"],
                    "paper_synced",
                    from_stage=current["stage"],
                    to_stage=target,
                    note=note,
                    snapshot=_candidate_snapshot(updated),
                )
                changed += 1
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return changed
