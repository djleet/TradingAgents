"""Build compact, auditable research packs from saved local evidence.

The pack is deliberately deterministic.  It compares persisted enrichment
snapshots, reads candidate/scanner changes, routes point-in-time evidence and
records the resulting review surface without calling an LLM or placing orders.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.dataflows.candidate_pipeline import (
    attach_candidate_consensus,
    attach_candidate_evidence,
    attach_candidate_thesis_evaluation,
    list_candidates,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.consensus_engine import build_consensus_report
from tradingagents.dataflows.evidence_router import build_evidence_bundle
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.thesis_engine import evaluate_thesis
from tradingagents.dataflows.warehouse import (
    _connect as _warehouse_connect,
    dependency_available,
    warehouse_path,
)

RESEARCH_PACK_VERSION = "automated-research-pack-v1"
ACTIVE_STAGES = {"discovered", "validating", "qualified", "paper_position"}
_SEVERITY_WEIGHT = {"high": 30, "medium": 18, "info": 8}


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
        CREATE TABLE IF NOT EXISTS automated_research_packs (
            pack_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            trigger VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            candidate_count INTEGER NOT NULL,
            material_change_count INTEGER NOT NULL,
            high_risk_count INTEGER NOT NULL,
            ready_evidence_count INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            reviewed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_automated_research_packs
        ON automated_research_packs(as_of_date, created_at)
        """
    )


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    connection = _warehouse_connect(warehouse_path(cfg))
    _ensure_schema(connection)
    return connection


def _change(
    change_type: str,
    label: str,
    message: str,
    *,
    severity: str = "medium",
    direction: str = "neutral",
    previous: Any = None,
    current: Any = None,
) -> dict[str, Any]:
    return {
        "type": change_type,
        "label": label,
        "message": message,
        "severity": severity,
        "direction": direction,
        "previous": previous,
        "current": current,
    }


def _load_enrichment_snapshots(
    connection, symbol: str, as_of_date: str
) -> list[dict[str, Any]]:
    try:
        rows = connection.execute(
            """
            SELECT payload_json FROM enrichment_snapshots
            WHERE symbol = ? AND as_of_date <= ?
            ORDER BY created_at DESC LIMIT 2
            """,
            [symbol, as_of_date],
        ).fetchall()
    except Exception:  # older warehouses may not have enrichment yet
        return []
    snapshots = []
    for row in rows:
        try:
            snapshots.append(json.loads(row[0]))
        except (TypeError, json.JSONDecodeError):
            continue
    return snapshots


def _metric_change(
    changes: list[dict[str, Any]],
    *,
    change_type: str,
    label: str,
    previous: Any,
    current: Any,
    threshold: float,
    high_threshold: float | None = None,
) -> None:
    old = _number(previous)
    new = _number(current)
    if old is None or new is None:
        return
    delta = round(new - old, 4)
    if abs(delta) < threshold:
        return
    severity = "high" if high_threshold is not None and abs(delta) >= high_threshold else "medium"
    direction = "positive" if delta > 0 else "negative"
    changes.append(
        _change(
            change_type,
            label,
            f"{label} changed from {old:g} to {new:g} ({delta:+g}).",
            severity=severity,
            direction=direction,
            previous=old,
            current=new,
        )
    )


def _enrichment_changes(
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not current:
        return []
    if not previous:
        coverage = _number((current.get("coverage") or {}).get("score"))
        return [
            _change(
                "enrichment_baseline",
                "New enrichment baseline",
                f"The first saved enrichment baseline is available at {coverage or 0:g}/100 coverage.",
                severity="medium",
                direction="new",
                current=coverage,
            )
        ]

    changes: list[dict[str, Any]] = []
    current_coverage = (current.get("coverage") or {}).get("score")
    previous_coverage = (previous.get("coverage") or {}).get("score")
    _metric_change(
        changes,
        change_type="coverage_change",
        label="Evidence coverage",
        previous=previous_coverage,
        current=current_coverage,
        threshold=5,
        high_threshold=15,
    )

    old_earnings = (previous.get("earnings") or {}).get("earnings_date")
    new_earnings = (current.get("earnings") or {}).get("earnings_date")
    if old_earnings and new_earnings and str(old_earnings) != str(new_earnings):
        changes.append(
            _change(
                "earnings_date_changed",
                "Earnings date changed",
                f"The saved earnings date moved from {old_earnings} to {new_earnings}.",
                severity="high",
                direction="risk",
                previous=old_earnings,
                current=new_earnings,
            )
        )

    current_earnings = current.get("earnings") or {}
    previous_earnings = previous.get("earnings") or {}
    _metric_change(
        changes,
        change_type="target_upside_change",
        label="Consensus target upside",
        previous=previous_earnings.get("target_upside"),
        current=current_earnings.get("target_upside"),
        threshold=0.03,
        high_threshold=0.1,
    )
    current_options = current.get("options") or {}
    previous_options = previous.get("options") or {}
    _metric_change(
        changes,
        change_type="expected_move_change",
        label="Option-implied move",
        previous=previous_options.get("expected_move_pct"),
        current=current_options.get("expected_move_pct"),
        threshold=0.02,
        high_threshold=0.06,
    )
    _metric_change(
        changes,
        change_type="implied_volatility_change",
        label="ATM implied volatility",
        previous=previous_options.get("atm_iv"),
        current=current_options.get("atm_iv"),
        threshold=0.04,
        high_threshold=0.1,
    )

    old_sources = {
        item.get("dataset"): item.get("status")
        for item in previous.get("sources", [])
        if item.get("dataset")
    }
    new_sources = {
        item.get("dataset"): item.get("status")
        for item in current.get("sources", [])
        if item.get("dataset")
    }
    for dataset, new_status in new_sources.items():
        old_status = old_sources.get(dataset)
        if not old_status or old_status == new_status:
            continue
        unavailable = new_status == "unavailable"
        changes.append(
            _change(
                "source_status_changed",
                dataset.replace("_", " ").title(),
                f"Source status changed from {old_status} to {new_status}.",
                severity="high" if unavailable else "info",
                direction="negative" if unavailable else "positive",
                previous=old_status,
                current=new_status,
            )
        )

    old_flags = {
        (str(item.get("type") or "flag"), str(item.get("title") or "Evidence flag")): item
        for item in previous.get("decision_flags", [])
    }
    new_flags = {
        (str(item.get("type") or "flag"), str(item.get("title") or "Evidence flag")): item
        for item in current.get("decision_flags", [])
    }
    for key, flag in new_flags.items():
        if key in old_flags:
            continue
        high = flag.get("severity") == "high"
        changes.append(
            _change(
                "decision_flag_added",
                str(flag.get("title") or "New decision flag"),
                str(flag.get("message") or "A new enrichment decision flag appeared."),
                severity="high" if high else "medium",
                direction="risk",
                current=flag.get("type"),
            )
        )
    for key, flag in old_flags.items():
        if key not in new_flags:
            changes.append(
                _change(
                    "decision_flag_cleared",
                    str(flag.get("title") or "Decision flag cleared"),
                    "A previously saved enrichment flag is no longer present.",
                    severity="info",
                    direction="positive",
                    previous=flag.get("type"),
                )
            )
    return changes


def _candidate_changes(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    scan_change = (candidate.get("evidence") or {}).get("change") or {}
    state = scan_change.get("state")
    if state in {"new", "entered_top", "improved", "deteriorated"}:
        score_delta = _number(scan_change.get("score_delta"), 2)
        direction = "negative" if state == "deteriorated" else "positive" if state != "new" else "new"
        severity = "high" if state in {"entered_top", "deteriorated"} else "medium"
        message = f"Scanner state is {str(state).replace('_', ' ')}"
        if score_delta is not None:
            message += f" with a {score_delta:+g} point score move"
        changes.append(
            _change(
                "scanner_change",
                "Opportunity score changed",
                message + ".",
                severity=severity,
                direction=direction,
                previous=scan_change.get("previous_score"),
                current=candidate.get("latest_score"),
            )
        )
    risk = str(candidate.get("risk_level") or "unknown").lower()
    if risk in {"elevated", "high", "critical"}:
        changes.append(
            _change(
                "current_risk",
                "Current risk needs review",
                f"Candidate risk is currently classified as {risk}.",
                severity="high" if risk in {"high", "critical"} else "medium",
                direction="risk",
                current=risk,
            )
        )
    if candidate.get("review_due"):
        changes.append(
            _change(
                "review_due",
                "Candidate review is due",
                f"The saved review date is {candidate.get('review_date') or 'due'}.",
                severity="medium",
                direction="workflow",
                current=candidate.get("review_date"),
            )
        )
    return changes


def _pack_item(
    candidate: dict[str, Any],
    changes: list[dict[str, Any]],
    bundle: dict[str, Any],
    consensus: dict[str, Any] | None = None,
    thesis_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    impact = min(
        100,
        sum(_SEVERITY_WEIGHT.get(item["severity"], 8) for item in changes)
        + (8 if candidate.get("priority") == "high" else 0),
    )
    severity = (
        "high"
        if any(item["severity"] == "high" for item in changes)
        else "medium"
        if changes
        else "info"
    )
    top_change = max(
        changes,
        key=lambda item: _SEVERITY_WEIGHT.get(item["severity"], 0),
        default=None,
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "symbol": candidate.get("symbol"),
        "stage": candidate.get("stage"),
        "priority": candidate.get("priority"),
        "severity": severity,
        "impact_score": impact,
        "headline": (
            top_change.get("label")
            if top_change
            else candidate.get("next_action") or "Candidate evidence review"
        ),
        "next_action": candidate.get("next_action") or "Review routed evidence",
        "latest_score": candidate.get("latest_score"),
        "latest_rank": candidate.get("latest_rank"),
        "risk_level": candidate.get("risk_level"),
        "decision_confidence": candidate.get("decision_confidence"),
        "changes": changes,
        "material_change_count": sum(item["severity"] != "info" for item in changes),
        "negative_change_count": sum(
            item["direction"] in {"negative", "risk"} for item in changes
        ),
        "evidence": {
            "bundle_id": bundle.get("bundle_id"),
            "verdict": bundle.get("verdict", "insufficient"),
            "coverage_score": bundle.get("coverage_score", 0),
            "eligible_count": bundle.get("eligible_count", 0),
            "blocked_count": bundle.get("blocked_count", 0),
            "missing_count": bundle.get("missing_count", 0),
        },
        "consensus": {
            "report_id": (
                (consensus.get("registry") or {}).get("report_id")
                or consensus.get("report_id")
                if consensus
                else None
            ),
            "verdict": consensus.get("verdict") if consensus else "unavailable",
            "consensus_score": consensus.get("consensus_score") if consensus else None,
            "agreement_rate": consensus.get("agreement_rate") if consensus else None,
            "conflict_count": len(consensus.get("conflicts", [])) if consensus else 0,
            "unresolved_critical_count": (
                (consensus.get("qualification_gate") or {}).get(
                    "unresolved_critical_count", 0
                )
                if consensus
                else 0
            ),
            "qualification_allowed": (
                (consensus.get("qualification_gate") or {}).get("allowed", True)
                if consensus
                else True
            ),
        },
        "thesis_engine": {
            "evaluation_id": (
                (thesis_evaluation.get("registry") or {}).get("evaluation_id")
                or thesis_evaluation.get("evaluation_id")
                if thesis_evaluation
                else None
            ),
            "version_number": (
                (thesis_evaluation.get("version") or {}).get("version_number")
                if thesis_evaluation
                else None
            ),
            "verdict": (
                thesis_evaluation.get("verdict")
                if thesis_evaluation
                else "unversioned"
            ),
            "triggered_count": (
                thesis_evaluation.get("triggered_count", 0)
                if thesis_evaluation
                else 0
            ),
            "critical_count": (
                thesis_evaluation.get("critical_count", 0)
                if thesis_evaluation
                else 0
            ),
        },
    }


def build_research_pack(
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    symbols: list[str] | None = None,
    max_candidates: int = 8,
    trigger: str = "manual",
    persist: bool = True,
    candidate_payload: dict[str, Any] | None = None,
    enrichment_snapshots: dict[str, list[dict[str, Any]]] | None = None,
    router=build_evidence_bundle,
    attach=attach_candidate_evidence,
    consensus_builder=build_consensus_report,
    consensus_attach=attach_candidate_consensus,
    thesis_evaluator=evaluate_thesis,
    thesis_attach=attach_candidate_thesis_evaluation,
) -> dict[str, Any]:
    """Build one dated pack from candidate deltas and routed saved evidence."""
    date.fromisoformat(as_of_date)
    bounded_max = max(1, min(int(max_candidates), 20))
    selected_symbols = (
        {normalize_symbol(symbol) for symbol in symbols} if symbols is not None else None
    )
    cfg = config or get_config()
    candidates = (
        candidate_payload if candidate_payload is not None else list_candidates(cfg)
    ).get("items", [])
    active = [
        item
        for item in candidates
        if item.get("stage") in ACTIVE_STAGES
        and (selected_symbols is None or item.get("symbol") in selected_symbols)
    ]
    connection = _connect(cfg)
    if connection is None:
        raise RuntimeError("Automated Research Pack requires the local data warehouse")

    prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    try:
        for candidate in active:
            symbol = candidate["symbol"]
            snapshots = (
                enrichment_snapshots.get(symbol, [])
                if enrichment_snapshots is not None
                else _load_enrichment_snapshots(connection, symbol, as_of_date)
            )
            changes = _candidate_changes(candidate)
            changes.extend(
                _enrichment_changes(
                    snapshots[0] if snapshots else None,
                    snapshots[1] if len(snapshots) > 1 else None,
                )
            )
            if changes:
                prepared.append((candidate, changes))
    finally:
        connection.close()

    prepared.sort(
        key=lambda pair: (
            -sum(_SEVERITY_WEIGHT.get(item["severity"], 0) for item in pair[1]),
            0 if pair[0].get("priority") == "high" else 1,
            pair[0].get("symbol", ""),
        )
    )
    items = []
    routing_failures = 0
    consensus_failures = 0
    thesis_failures = 0
    unversioned_theses = 0
    for candidate, changes in prepared[:bounded_max]:
        try:
            bundle = router(
                candidate["symbol"],
                as_of_date,
                cfg,
                purpose="candidate",
            )
            if candidate.get("candidate_id") and bundle.get("bundle_id"):
                attach(candidate["candidate_id"], bundle, cfg)
        except Exception as exc:  # a single evidence gate must not erase the pack
            routing_failures += 1
            bundle = {
                "verdict": "insufficient",
                "coverage_score": 0,
                "eligible_count": 0,
                "blocked_count": 0,
                "missing_count": 8,
                "error_type": type(exc).__name__,
            }
        consensus = None
        try:
            consensus = consensus_builder(
                candidate["symbol"],
                as_of_date,
                cfg,
                bundle=bundle,
                candidate=candidate,
            )
            if candidate.get("candidate_id") and consensus.get("report_id"):
                consensus_attach(candidate["candidate_id"], consensus, cfg)
            gate = consensus.get("qualification_gate") or {}
            if gate.get("unresolved_critical_count", 0):
                changes.append(
                    _change(
                        "critical_contradiction",
                        "Critical evidence contradiction",
                        (
                            f"{gate['unresolved_critical_count']} critical contradiction"
                            f"{'s' if gate['unresolved_critical_count'] != 1 else ''} block qualification."
                        ),
                        severity="high",
                        direction="risk",
                        current=gate["unresolved_critical_count"],
                    )
                )
        except Exception:  # routed evidence remains useful without consensus synthesis
            consensus_failures += 1
        thesis_evaluation = None
        try:
            thesis_evaluation = thesis_evaluator(
                candidate["candidate_id"],
                as_of_date,
                cfg,
                bundle=bundle,
                consensus=consensus,
                candidate=candidate,
            )
            thesis_attach(candidate["candidate_id"], thesis_evaluation, cfg)
            if thesis_evaluation.get("verdict") in {"invalidated", "warning"}:
                changes.append(
                    _change(
                        "thesis_invalidation",
                        "Research thesis requires review",
                        thesis_evaluation.get(
                            "message", "An invalidation rule triggered."
                        ),
                        severity=(
                            "high"
                            if thesis_evaluation.get("verdict") == "invalidated"
                            else "medium"
                        ),
                        direction="risk",
                        current=thesis_evaluation.get("triggered_count", 0),
                    )
                )
        except ValueError as exc:
            if "no versioned research thesis" in str(exc).lower():
                unversioned_theses += 1
            else:
                thesis_failures += 1
        except Exception:  # thesis alerts must not erase routed research evidence
            thesis_failures += 1
        items.append(
            _pack_item(candidate, changes, bundle, consensus, thesis_evaluation)
        )

    material_changes = sum(item["material_change_count"] for item in items)
    high_risk = sum(
        item["severity"] == "high"
        or str(item.get("risk_level") or "").lower() in {"high", "critical"}
        for item in items
    )
    ready_evidence = sum(item["evidence"]["verdict"] == "ready" for item in items)
    status = "attention_required" if high_risk else "changes_detected" if items else "quiet"
    headline = (
        f"{high_risk} high-priority research update{'s' if high_risk != 1 else ''} require review"
        if high_risk
        else f"{len(items)} candidate update{'s' if len(items) != 1 else ''} collected"
        if items
        else "No material candidate changes in the selected evidence"
    )
    pack = {
        "model_version": RESEARCH_PACK_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "trigger": trigger[:60],
        "status": status,
        "headline": headline,
        "summary": {
            "candidate_count": len(items),
            "material_change_count": material_changes,
            "high_risk_count": high_risk,
            "ready_evidence_count": ready_evidence,
            "partial_evidence_count": sum(
                item["evidence"]["verdict"] == "partial" for item in items
            ),
            "routing_failure_count": routing_failures,
            "consensus_failure_count": consensus_failures,
            "thesis_failure_count": thesis_failures,
            "unversioned_thesis_count": unversioned_theses,
            "invalidated_thesis_count": sum(
                item["thesis_engine"]["verdict"] == "invalidated" for item in items
            ),
            "blocked_by_consensus_count": sum(
                not item["consensus"]["qualification_allowed"] for item in items
            ),
            "considered_candidate_count": len(active),
        },
        "items": items,
        "policy": {
            "network_fetches": False,
            "llm_calls": False,
            "orders": False,
            "selection_rule": "Only active candidates with saved changes, current risk or due reviews are included.",
        },
        "notice": "Research review only. This pack does not make or execute an investment decision.",
    }
    identity = {
        "as_of_date": pack["as_of_date"],
        "trigger": pack["trigger"],
        "status": pack["status"],
        "summary": pack["summary"],
        "items": pack["items"],
        "policy": pack["policy"],
    }
    pack_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    pack["pack_id"] = pack_id
    if not persist:
        return pack

    now = datetime.now(UTC)
    connection = _connect(cfg)
    if connection is None:
        raise RuntimeError("Automated Research Pack requires the local data warehouse")
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO automated_research_packs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
            )
            """,
            [
                pack_id,
                RESEARCH_PACK_VERSION,
                as_of_date,
                pack["trigger"],
                status,
                len(items),
                material_changes,
                high_risk,
                ready_evidence,
                _safe_json(pack),
                now,
            ],
        )
        row = connection.execute(
            "SELECT created_at, reviewed_at FROM automated_research_packs WHERE pack_id = ?",
            [pack_id],
        ).fetchone()
    finally:
        connection.close()
    pack["registry"] = {
        "pack_id": pack_id,
        "saved": True,
        "created_at": row[0].astimezone(UTC).isoformat(),
        "reviewed_at": row[1].astimezone(UTC).isoformat() if row[1] else None,
    }
    return pack


def get_latest_research_pack(
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    connection = _connect(config)
    if connection is None:
        return None
    try:
        row = connection.execute(
            """
            SELECT pack_id, payload_json, created_at, reviewed_at
            FROM automated_research_packs ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    pack = json.loads(row[1])
    pack["registry"] = {
        "pack_id": row[0],
        "saved": True,
        "created_at": row[2].astimezone(UTC).isoformat(),
        "reviewed_at": row[3].astimezone(UTC).isoformat() if row[3] else None,
    }
    return pack


def list_research_packs(
    config: dict[str, Any] | None = None, *, limit: int = 20
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    try:
        rows = connection.execute(
            """
            SELECT pack_id, as_of_date, trigger, status, candidate_count,
                   material_change_count, high_risk_count, ready_evidence_count,
                   created_at, reviewed_at
            FROM automated_research_packs ORDER BY created_at DESC LIMIT ?
            """,
            [max(1, min(int(limit), 100))],
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "pack_id": row[0],
            "as_of_date": row[1].isoformat(),
            "trigger": row[2],
            "status": row[3],
            "candidate_count": row[4],
            "material_change_count": row[5],
            "high_risk_count": row[6],
            "ready_evidence_count": row[7],
            "created_at": row[8].astimezone(UTC).isoformat(),
            "reviewed_at": row[9].astimezone(UTC).isoformat() if row[9] else None,
        }
        for row in rows
    ]


def mark_research_pack_reviewed(
    pack_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _connect(config)
    if connection is None:
        return None
    reviewed_at = datetime.now(UTC)
    try:
        row = connection.execute(
            """
            UPDATE automated_research_packs SET reviewed_at = ?
            WHERE pack_id = ? RETURNING pack_id
            """,
            [reviewed_at, pack_id],
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    return {"pack_id": pack_id, "reviewed_at": reviewed_at.isoformat()}
