"""Deterministic daily research briefing across the local decision workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.dataflows.calibration_engine import get_latest_calibration_run
from tradingagents.dataflows.candidate_pipeline import (
    _connect as _candidate_connect,
    list_candidates,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.consensus_engine import get_consensus_status
from tradingagents.dataflows.decision_review import get_latest_decision_review
from tradingagents.dataflows.enrichment_hub import get_latest_enrichment_snapshot
from tradingagents.dataflows.enrichment_scheduler import get_enrichment_operations_status
from tradingagents.dataflows.model_governance import get_model_governance_status
from tradingagents.dataflows.paper_portfolio import get_paper_portfolio
from tradingagents.dataflows.research_pack import get_latest_research_pack
from tradingagents.dataflows.shadow_lab import get_shadow_lab_status
from tradingagents.dataflows.thesis_engine import get_thesis_status
from tradingagents.dataflows.warehouse import (
    get_warehouse_status,
    list_opportunity_alerts,
    list_scan_schedules,
)

BRIEFING_VERSION = "daily-research-briefing-v1"
_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "info": 1}


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat()


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_research_briefings (
            briefing_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            posture VARCHAR NOT NULL,
            priority_count INTEGER NOT NULL,
            critical_count INTEGER NOT NULL,
            high_count INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            reviewed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_daily_research_briefings
        ON daily_research_briefings(as_of_date, created_at)
        """
    )


def _priority(
    category: str,
    severity: str,
    score: int,
    title: str,
    summary: str,
    rationale: str,
    action_label: str,
    target_id: str,
    *,
    symbol: str | None = None,
    source_id: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = f"{category}|{source_id or ''}|{symbol or ''}|{title}"
    return {
        "priority_id": hashlib.sha256(identity.encode()).hexdigest(),
        "category": category,
        "severity": severity,
        "priority_score": score,
        "symbol": symbol,
        "title": title,
        "summary": summary,
        "rationale": rationale,
        "action_label": action_label,
        "target_id": target_id,
        "source_id": source_id,
        "evidence": evidence or {},
    }


def _latest_scan_state(config: dict[str, Any]) -> dict[str, Any] | None:
    connection = _candidate_connect(config)
    try:
        row = connection.execute(
            """
            SELECT scan_id, as_of_date, universe_name, result_json, created_at
            FROM opportunity_scans ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    report = json.loads(row[3])
    return {
        "scan_id": row[0],
        "as_of_date": row[1].isoformat(),
        "universe_name": row[2],
        "created_at": row[4].isoformat(),
        "requested_symbols": report.get("requested_symbols", 0),
        "eligible_symbols": report.get("eligible_symbols", 0),
        "bucket_counts": report.get("bucket_counts", {}),
        "changes": report.get("changes", {}),
        "errors": report.get("errors", []),
        "top_items": [
            {
                "symbol": item.get("symbol"),
                "score": item.get("opportunity_score"),
                "rank": item.get("rank"),
                "bucket": item.get("bucket"),
                "change": item.get("change") or {},
            }
            for item in report.get("items", [])
            if item.get("bucket") == "top_opportunity"
        ][:10],
    }


def _candidate_priorities(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for candidate in candidates:
        symbol = candidate["symbol"]
        if candidate.get("review_due"):
            base = {"high": 88, "medium": 76, "low": 64}.get(
                candidate.get("priority"), 70
            )
            severity = "high" if candidate.get("priority") == "high" else "medium"
            result.append(
                _priority(
                    "candidate_review",
                    severity,
                    base,
                    f"{symbol} review is due",
                    candidate.get("next_action") or "Review the candidate evidence.",
                    f"The saved review date was {candidate.get('review_date')}; stage is {candidate.get('stage', 'unknown').replace('_', ' ')}.",
                    "Review candidate",
                    "candidate-pipeline",
                    symbol=symbol,
                    source_id=candidate.get("candidate_id"),
                    evidence={
                        "candidate_id": candidate.get("candidate_id"),
                        "stage": candidate.get("stage"),
                        "score": candidate.get("latest_score"),
                        "decision_confidence": candidate.get("decision_confidence"),
                        "review_date": candidate.get("review_date"),
                    },
                )
            )
        if candidate.get("stage") == "qualified" and not candidate.get("position_id"):
            result.append(
                _priority(
                    "qualified_candidate",
                    "high" if candidate.get("priority") == "high" else "medium",
                    82 if candidate.get("priority") == "high" else 72,
                    f"Decide the next step for qualified {symbol}",
                    candidate.get("next_action") or "Review for paper tracking.",
                    "The research gate is complete, but no simulated position is linked.",
                    "Open candidate",
                    "candidate-pipeline",
                    symbol=symbol,
                    source_id=candidate.get("candidate_id"),
                    evidence={
                        "candidate_id": candidate.get("candidate_id"),
                        "score": candidate.get("latest_score"),
                        "validation_verdict": candidate.get("validation_verdict"),
                    },
                )
            )
    return result


def _opportunity_priorities(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for alert in alerts:
        severity = "high" if alert.get("severity") == "high" else "medium"
        score = 90 if alert.get("alert_type") in {"entered_top", "top_opportunity"} else 74
        if alert.get("alert_type") == "deteriorated":
            score = 78
        result.append(
            _priority(
                "opportunity_change",
                severity,
                score,
                alert.get("title") or "Opportunity evidence changed",
                alert.get("message") or "Review the latest scanner evidence.",
                "This is an unacknowledged scanner change from the latest persisted workflow.",
                "Open scanner",
                "opportunity-scanner",
                symbol=alert.get("symbol"),
                source_id=alert.get("alert_id"),
                evidence={
                    "alert_id": alert.get("alert_id"),
                    "alert_type": alert.get("alert_type"),
                    "current_score": alert.get("current_score"),
                    "previous_score": alert.get("previous_score"),
                },
            )
        )
    return result


def _paper_priorities(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for alert in portfolio.get("risk_alerts", []):
        high = alert.get("severity") == "high"
        result.append(
            _priority(
                "paper_risk",
                "critical" if high else "high",
                98 if high else 86,
                alert.get("title") or f"Review {alert.get('symbol', 'paper position')}",
                alert.get("message") or "A simulated-position risk rule needs review.",
                "Paper risk rules are explicit decision checkpoints; no order will be sent automatically.",
                "Review paper position",
                "paper-portfolio",
                symbol=alert.get("symbol"),
                source_id=alert.get("alert_id"),
                evidence={
                    "alert_id": alert.get("alert_id"),
                    "alert_type": alert.get("alert_type"),
                    "as_of_date": str(alert.get("as_of_date") or ""),
                },
            )
        )
    return result


def _calibration_priorities(calibration: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not calibration:
        return [
            _priority(
                "model_health",
                "info",
                45,
                "Establish a calibration baseline",
                "No saved outcome-calibration experiment is available yet.",
                "A baseline is needed before model drift or shadow thresholds can be assessed.",
                "Open calibration",
                "calibration-engine",
            )
        ]
    drift = calibration.get("drift") or {}
    verdict = calibration.get("verdict")
    if drift.get("status") == "degrading" or verdict == "degrading":
        return [
            _priority(
                "model_health",
                "high",
                92,
                "Calibration indicates model degradation",
                f"Recent 21-session excess changed by {float(drift.get('excess_delta') or 0):+.1%}.",
                "Recent mature outcomes underperformed the earlier half of the stored sample.",
                "Review calibration",
                "calibration-engine",
                source_id=(calibration.get("registry") or {}).get("run_id"),
                evidence={
                    "verdict": verdict,
                    "drift_status": drift.get("status"),
                    "excess_delta": drift.get("excess_delta"),
                    "hit_rate_delta": drift.get("hit_rate_delta"),
                },
            )
        ]
    if verdict == "insufficient_data":
        return [
            _priority(
                "model_health",
                "info",
                42,
                "Calibration sample is still immature",
                "More forward outcomes are required before changing thresholds or score weights.",
                "Shadow recommendations remain disabled until the configured minimum sample is met.",
                "Open calibration",
                "calibration-engine",
                source_id=(calibration.get("registry") or {}).get("run_id"),
            )
        ]
    return []


def _decision_review_priorities(
    review: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not review or (review.get("registry") or {}).get("reviewed_at"):
        return []
    coverage = review.get("coverage") or {}
    if not coverage.get("matured_outcomes"):
        return []
    summary = review.get("summary") or {}
    false_alerts = int(summary.get("false_alert_count") or 0)
    verdict = review.get("verdict")
    if verdict not in {"learning_required", "mixed"} and not false_alerts:
        return []
    high = verdict == "learning_required"
    return [
        _priority(
            "decision_review",
            "high" if high else "medium",
            90 if high else 75,
            (
                "Decision review found a learning gap"
                if high
                else "Mature decision outcomes are ready for review"
            ),
            (
                f"{false_alerts} thesis alert{'s' if false_alerts != 1 else ''} "
                "would have reduced the standardized forward result."
            ),
            "This is an exposure counterfactual, not realized portfolio P&L; rule changes remain manual.",
            "Open decision review",
            "decision-review-engine",
            source_id=(review.get("registry") or {}).get("run_id"),
            evidence={
                "verdict": verdict,
                "false_alert_count": false_alerts,
                "helpful_alert_count": summary.get("helpful_alert_count", 0),
                "decision_accuracy": summary.get("decision_accuracy"),
            },
        )
    ]


def _shadow_lab_priorities(status: dict[str, Any]) -> list[dict[str, Any]]:
    promising = int(status.get("challenger_promising_count") or 0)
    control_leads = int(status.get("control_leads_count") or 0)
    if not promising and not control_leads:
        return []
    if promising:
        return [
            _priority(
                "shadow_experiment",
                "high",
                86,
                f"{promising} shadow challenger{'s' if promising != 1 else ''} look promising",
                "Prospective outcomes favor at least one alternative threshold.",
                "The result is non-randomized and requires explicit review; no production setting was changed.",
                "Open Shadow Lab",
                "shadow-lab",
                source_id="shadow-challenger-results",
                evidence={
                    "challenger_promising_count": promising,
                    "active_count": status.get("active_count", 0),
                },
            )
        ]
    return [
        _priority(
            "shadow_experiment",
            "medium",
            72,
            "Control settings lead in a shadow experiment",
            f"{control_leads} prospective comparison{'s' if control_leads != 1 else ''} currently favor the control.",
            "Review the evidence before closing or redesigning the challenger.",
            "Open Shadow Lab",
            "shadow-lab",
            source_id="shadow-control-results",
            evidence={"control_leads_count": control_leads},
        )
    ]


def _governance_priorities(status: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = int(status.get("promotion_eligible_count") or 0)
    pending = int(status.get("pending_approval_count") or 0)
    if eligible:
        return [
            _priority(
                "model_governance",
                "high",
                89,
                f"{eligible} governed challenger{'s are' if eligible != 1 else ' is'} promotion-eligible",
                "Prospective probation and both independent approvals have been recorded.",
                "Promotion remains an explicit operator decision and retains a one-click rollback path.",
                "Open Model Governance",
                "model-governance",
                source_id="governance-promotion-eligible",
                evidence={
                    "promotion_eligible_count": eligible,
                    "active_policy_count": len(status.get("active_policies", [])),
                },
            )
        ]
    if pending:
        return [
            _priority(
                "model_governance",
                "medium",
                74,
                f"{pending} promotion proposal{'s need' if pending != 1 else ' needs'} independent approval",
                "Research and Risk must each record a documented decision.",
                "A proposal cannot enter probation until both roles approve it.",
                "Open Model Governance",
                "model-governance",
                source_id="governance-pending-approval",
                evidence={"pending_approval_count": pending},
            )
        ]
    return []


def _workflow_priorities(
    as_of_date: str,
    scan: dict[str, Any] | None,
    schedules: list[dict[str, Any]],
    warehouse: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    if not scan:
        result.append(
            _priority(
                "workflow",
                "medium",
                68,
                "Run the first Opportunity Scanner universe",
                "No persisted cross-sectional scan is available for this briefing.",
                "Discovery and change tracking need a saved point-in-time scan.",
                "Open scanner",
                "opportunity-scanner",
            )
        )
    else:
        age = (date.fromisoformat(as_of_date) - date.fromisoformat(scan["as_of_date"])).days
        if age > 3:
            result.append(
                _priority(
                    "workflow",
                    "medium",
                    66,
                    "Opportunity evidence is stale",
                    f"The latest persisted scan is {age} calendar days old.",
                    "Daily prioritization is less useful when discovery evidence is not current.",
                    "Refresh scanner",
                    "opportunity-scanner",
                    source_id=scan.get("scan_id"),
                    evidence={"latest_scan_date": scan.get("as_of_date"), "age_days": age},
                )
            )
        errors = len(scan.get("errors") or [])
        if errors:
            result.append(
                _priority(
                    "data_quality",
                    "medium",
                    min(79, 62 + errors),
                    f"{errors} scanner dataset{'s' if errors != 1 else ''} unavailable",
                    "The latest scan kept partial results and did not impute missing instruments.",
                    "Coverage gaps can change ranks and should be reviewed before qualification.",
                    "Inspect scanner",
                    "opportunity-scanner",
                    source_id=scan.get("scan_id"),
                    evidence={"errors": scan.get("errors")},
                )
            )
    failed = [item for item in schedules if item.get("last_status") == "failed"]
    if failed:
        result.append(
            _priority(
                "workflow",
                "high",
                84,
                f"{len(failed)} daily scan schedule{'s' if len(failed) != 1 else ''} failed",
                "At least one saved universe did not complete its latest scheduled run.",
                "The briefing may be based on stale evidence until the scheduled scan succeeds.",
                "Review schedules",
                "opportunity-scanner",
                evidence={"schedule_ids": [item.get("schedule_id") for item in failed]},
            )
        )
    if not warehouse.get("ready"):
        result.append(
            _priority(
                "data_quality",
                "high",
                90,
                "Local research warehouse is unavailable",
                "Persistent evidence and audit history could not be verified.",
                "Do not rely on an incomplete briefing until local storage is healthy.",
                "Inspect dashboard status",
                "opportunity-scanner",
            )
        )
    return result


def _enrichment_priorities(enrichment: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not enrichment:
        return []
    result = []
    for flag in enrichment.get("decision_flags", []):
        if flag.get("severity") not in {"high", "medium"}:
            continue
        severity = flag["severity"]
        result.append(
            _priority(
                "enrichment",
                severity,
                89 if severity == "high" else 73,
                flag.get("title") or "Review enriched evidence",
                flag.get("message") or "A saved enrichment source needs review.",
                "The signal comes from the latest persisted multi-source snapshot and retains its observation-time policy.",
                "Open enrichment",
                "data-enrichment-hub",
                symbol=enrichment.get("symbol"),
                source_id=(enrichment.get("registry") or {}).get("snapshot_id"),
                evidence={
                    "flag_type": flag.get("type"),
                    "coverage_score": (enrichment.get("coverage") or {}).get("score"),
                    "observed_at": enrichment.get("observed_at"),
                },
            )
        )
    return result


def _enrichment_operations_priorities(operations: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for alert in operations.get("alerts", []):
        high = alert.get("severity") == "high"
        result.append(
            _priority(
                "data_quality",
                "high" if high else "medium",
                87 if high else 69,
                alert.get("title") or "Enrichment source needs attention",
                alert.get("message") or "An automated enrichment check needs review.",
                "The scheduler preserved partial results and did not fill missing evidence.",
                "Review enrichment health",
                "data-enrichment-hub",
                symbol=alert.get("symbol"),
                source_id=alert.get("alert_id"),
                evidence={
                    "alert_id": alert.get("alert_id"),
                    "alert_type": alert.get("alert_type"),
                    "dataset": alert.get("dataset"),
                    "schedule_id": alert.get("schedule_id"),
                },
            )
        )
    failed = [
        item for item in operations.get("schedules", []) if item.get("last_status") == "failed"
    ]
    if failed:
        result.append(
            _priority(
                "workflow",
                "high",
                85,
                f"{len(failed)} enrichment schedule{'s' if len(failed) != 1 else ''} failed",
                "Every symbol in at least one bounded enrichment cycle failed.",
                "The scheduler will retry after cooldown; current evidence may be stale.",
                "Review enrichment schedules",
                "data-enrichment-hub",
                evidence={"schedule_ids": [item.get("schedule_id") for item in failed]},
            )
        )
    return result


def _research_pack_priorities(pack: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not pack or not (pack.get("summary") or {}).get("candidate_count"):
        return []
    summary = pack["summary"]
    high = bool(summary.get("high_risk_count"))
    registry = pack.get("registry") or {}
    return [
        _priority(
            "research_pack",
            "high" if high else "medium",
            88 if high else 73,
            pack.get("headline") or "Research Pack changes are ready",
            (
                f"{summary.get('material_change_count', 0)} material changes across "
                f"{summary.get('candidate_count', 0)} selected candidates."
            ),
            "Every selected candidate has a dated Evidence Router result; review the deltas before changing a research decision.",
            "Open research pack",
            "research-pack",
            source_id=registry.get("pack_id") or pack.get("pack_id"),
            evidence={
                "pack_id": registry.get("pack_id") or pack.get("pack_id"),
                "high_risk_count": summary.get("high_risk_count", 0),
                "ready_evidence_count": summary.get("ready_evidence_count", 0),
                "reviewed_at": registry.get("reviewed_at"),
            },
        )
    ]


def _consensus_priorities(status: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = status.get("blocked_symbol_count", 0)
    if not blocked:
        return []
    reports = [
        item for item in status.get("reports", []) if not item.get("qualification_allowed")
    ]
    symbols = [item.get("symbol") for item in reports if item.get("symbol")]
    return [
        _priority(
            "evidence_conflict",
            "high",
            91,
            f"{blocked} candidate{'s' if blocked != 1 else ''} blocked by evidence conflicts",
            (
                f"{status.get('critical_conflict_count', 0)} unresolved critical contradictions require an explicit review."
            ),
            "Only unresolved critical contradictions block qualification; missing data alone does not.",
            "Open consensus engine",
            "consensus-engine",
            source_id="consensus-critical-conflicts",
            evidence={"symbols": symbols, "blocked_symbol_count": blocked},
        )
    ]


def _thesis_priorities(status: dict[str, Any]) -> list[dict[str, Any]]:
    invalidated = status.get("invalidated_count", 0)
    warnings = status.get("warning_count", 0)
    if not invalidated and not warnings:
        return []
    affected = [
        item
        for item in status.get("reports", [])
        if item.get("verdict") in {"invalidated", "warning"}
    ]
    symbols = [item.get("symbol") for item in affected if item.get("symbol")]
    return [
        _priority(
            "thesis_invalidation",
            "high" if invalidated else "medium",
            94 if invalidated else 76,
            (
                f"{invalidated} research thesis"
                f"{'es are' if invalidated != 1 else ' is'} invalidated"
                if invalidated
                else f"{warnings} research thesis warning{'s' if warnings != 1 else ''}"
            ),
            (
                f"{invalidated} critical and {warnings} warning thesis states require explicit review."
            ),
            "The engine only raises an audit alert; it does not change candidate stages or decisions.",
            "Open thesis engine",
            "candidate-pipeline",
            source_id="thesis-invalidation-alerts",
            evidence={"symbols": symbols, "invalidated_count": invalidated},
        )
    ]
def _deduplicate(priorities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest item for the same category and symbol/source."""
    retained: dict[tuple[str, str], dict[str, Any]] = {}
    for item in priorities:
        identity = item.get("symbol") or item.get("source_id") or item["title"]
        key = (item["category"], str(identity))
        previous = retained.get(key)
        if previous is None or item["priority_score"] > previous["priority_score"]:
            retained[key] = item
    return sorted(
        retained.values(),
        key=lambda item: (
            -_SEVERITY_ORDER[item["severity"]],
            -int(item["priority_score"]),
            item["title"],
        ),
    )


def build_daily_briefing(
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    candidate_payload: dict[str, Any] | None = None,
    opportunity_alerts: list[dict[str, Any]] | None = None,
    portfolio: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    scan_state: dict[str, Any] | None = None,
    schedules: list[dict[str, Any]] | None = None,
    warehouse_status: dict[str, Any] | None = None,
    enrichment: dict[str, Any] | None = None,
    enrichment_operations: dict[str, Any] | None = None,
    research_pack: dict[str, Any] | None = None,
    consensus_status: dict[str, Any] | None = None,
    thesis_status: dict[str, Any] | None = None,
    decision_review: dict[str, Any] | None = None,
    shadow_lab_status: dict[str, Any] | None = None,
    governance_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one explainable, action-oriented briefing without external calls."""
    cfg = config or get_config()
    date.fromisoformat(as_of_date)
    candidates_data = candidate_payload if candidate_payload is not None else list_candidates(cfg)
    alerts = (
        opportunity_alerts
        if opportunity_alerts is not None
        else list_opportunity_alerts(cfg, limit=50, unacknowledged_only=True)
    )
    paper = portfolio if portfolio is not None else get_paper_portfolio(cfg)
    latest_calibration = (
        calibration if calibration is not None else get_latest_calibration_run(cfg)
    )
    latest_scan = scan_state if scan_state is not None else _latest_scan_state(cfg)
    scan_schedules = schedules if schedules is not None else list_scan_schedules(cfg)
    warehouse = warehouse_status if warehouse_status is not None else get_warehouse_status(cfg)
    latest_enrichment = (
        enrichment
        if enrichment is not None
        else get_latest_enrichment_snapshot(cfg)
    )
    enrichment_ops = (
        enrichment_operations
        if enrichment_operations is not None
        else get_enrichment_operations_status(cfg)
    )
    latest_research_pack = (
        research_pack if research_pack is not None else get_latest_research_pack(cfg)
    )
    consensus = (
        consensus_status if consensus_status is not None else get_consensus_status(cfg)
    )
    theses = thesis_status if thesis_status is not None else get_thesis_status(cfg)
    latest_decision_review = (
        decision_review
        if decision_review is not None
        else get_latest_decision_review(cfg)
    )
    shadow_status = (
        shadow_lab_status
        if shadow_lab_status is not None
        else get_shadow_lab_status(cfg)
    )
    governance = (
        governance_status
        if governance_status is not None
        else get_model_governance_status(cfg)
    )

    priorities = _deduplicate(
        [
            *_paper_priorities(paper),
            *_candidate_priorities(candidates_data.get("items", [])),
            *_opportunity_priorities(alerts),
            *_calibration_priorities(latest_calibration),
            *_enrichment_priorities(latest_enrichment),
            *_enrichment_operations_priorities(enrichment_ops),
            *_research_pack_priorities(latest_research_pack),
            *_consensus_priorities(consensus),
            *_thesis_priorities(theses),
            *_decision_review_priorities(latest_decision_review),
            *_shadow_lab_priorities(shadow_status),
            *_governance_priorities(governance),
            *_workflow_priorities(as_of_date, latest_scan, scan_schedules, warehouse),
        ]
    )
    counts = {
        severity: sum(item["severity"] == severity for item in priorities)
        for severity in _SEVERITY_ORDER
    }
    urgent = counts["critical"] + counts["high"]
    if counts["critical"]:
        posture = "immediate_review"
        headline = "Immediate simulated-risk review required"
    elif counts["high"]:
        posture = "focused_review"
        headline = "High-priority research decisions are ready"
    elif counts["medium"]:
        posture = "routine_review"
        headline = "Routine evidence review is due"
    else:
        posture = "clear"
        headline = "No urgent research exceptions"

    candidates = candidates_data.get("items", [])
    due_candidates = [item for item in candidates if item.get("review_due")]
    qualified = [item for item in candidates if item.get("stage") == "qualified"]
    open_positions = paper.get("open_positions", [])
    checklist = [
        {
            "check_id": "urgent_priorities",
            "label": "Review critical and high priorities",
            "count": urgent,
            "complete": urgent == 0,
            "target_id": "daily-briefing",
        },
        {
            "check_id": "candidate_reviews",
            "label": "Complete due candidate reviews",
            "count": len(due_candidates),
            "complete": not due_candidates,
            "target_id": "candidate-pipeline",
        },
        {
            "check_id": "opportunity_alerts",
            "label": "Resolve new scanner changes",
            "count": len(alerts),
            "complete": not alerts,
            "target_id": "opportunity-scanner",
        },
        {
            "check_id": "paper_alerts",
            "label": "Review paper risk alerts",
            "count": len(paper.get("risk_alerts", [])),
            "complete": not paper.get("risk_alerts"),
            "target_id": "paper-portfolio",
        },
        {
            "check_id": "enrichment_health",
            "label": "Resolve enrichment freshness and source alerts",
            "count": enrichment_ops.get("unacknowledged_alert_count", 0),
            "complete": not enrichment_ops.get("unacknowledged_alert_count", 0),
            "target_id": "data-enrichment-hub",
        },
        {
            "check_id": "research_pack_review",
            "label": "Review material Research Pack changes",
            "count": (
                1
                if latest_research_pack
                and (latest_research_pack.get("summary") or {}).get("candidate_count")
                and not (latest_research_pack.get("registry") or {}).get("reviewed_at")
                else 0
            ),
            "complete": bool(
                not latest_research_pack
                or not (latest_research_pack.get("summary") or {}).get("candidate_count")
                or (latest_research_pack.get("registry") or {}).get("reviewed_at")
            ),
            "target_id": "research-pack",
        },
        {
            "check_id": "consensus_conflicts",
            "label": "Resolve critical evidence contradictions",
            "count": consensus.get("critical_conflict_count", 0),
            "complete": not consensus.get("critical_conflict_count", 0),
            "target_id": "consensus-engine",
        },
        {
            "check_id": "thesis_invalidations",
            "label": "Review thesis invalidations and warnings",
            "count": theses.get("invalidated_count", 0) + theses.get("warning_count", 0),
            "complete": not (
                theses.get("invalidated_count", 0) + theses.get("warning_count", 0)
            ),
            "target_id": "candidate-pipeline",
        },
        {
            "check_id": "decision_review",
            "label": "Review mature decision and alert outcomes",
            "count": (
                1
                if latest_decision_review
                and (latest_decision_review.get("coverage") or {}).get(
                    "matured_outcomes"
                )
                and not (latest_decision_review.get("registry") or {}).get(
                    "reviewed_at"
                )
                else 0
            ),
            "complete": bool(
                not latest_decision_review
                or not (latest_decision_review.get("coverage") or {}).get(
                    "matured_outcomes"
                )
                or (latest_decision_review.get("registry") or {}).get("reviewed_at")
            ),
            "target_id": "decision-review-engine",
        },
        {
            "check_id": "shadow_experiments",
            "label": "Review mature Shadow Lab comparisons",
            "count": shadow_status.get("actionable_unreviewed_count", 0),
            "complete": not shadow_status.get("actionable_unreviewed_count", 0),
            "target_id": "shadow-lab",
        },
        {
            "check_id": "model_governance",
            "label": "Resolve governed approvals and eligible promotions",
            "count": governance.get("pending_approval_count", 0)
            + governance.get("promotion_eligible_count", 0),
            "complete": not (
                governance.get("pending_approval_count", 0)
                + governance.get("promotion_eligible_count", 0)
            ),
            "target_id": "model-governance",
        },
    ]
    return {
        "model_version": BRIEFING_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "headline": headline,
        "posture": posture,
        "counts": {**counts, "urgent": urgent, "total": len(priorities)},
        "summary": {
            "candidate_count": len(candidates),
            "due_candidate_count": len(due_candidates),
            "qualified_count": len(qualified),
            "open_position_count": len(open_positions),
            "opportunity_alert_count": len(alerts),
            "paper_alert_count": len(paper.get("risk_alerts", [])),
        },
        "priorities": priorities[:15],
        "checklist": checklist,
        "latest_scan": latest_scan,
        "model_health": {
            "available": latest_calibration is not None,
            "verdict": latest_calibration.get("verdict") if latest_calibration else None,
            "drift": (latest_calibration.get("drift") or {}).get("status")
            if latest_calibration
            else None,
            "as_of_date": latest_calibration.get("as_of_date")
            if latest_calibration
            else None,
        },
        "enrichment": {
            "available": latest_enrichment is not None,
            "symbol": latest_enrichment.get("symbol") if latest_enrichment else None,
            "coverage_score": (latest_enrichment.get("coverage") or {}).get("score")
            if latest_enrichment
            else None,
            "coverage_label": (latest_enrichment.get("coverage") or {}).get("label")
            if latest_enrichment
            else None,
            "flag_count": len(latest_enrichment.get("decision_flags", []))
            if latest_enrichment
            else 0,
            "observed_at": latest_enrichment.get("observed_at")
            if latest_enrichment
            else None,
        },
        "enrichment_operations": {
            "active_schedule_count": enrichment_ops.get("active_schedule_count", 0),
            "healthy_source_count": enrichment_ops.get("healthy_source_count", 0),
            "degraded_source_count": enrichment_ops.get("degraded_source_count", 0),
            "unacknowledged_alert_count": enrichment_ops.get(
                "unacknowledged_alert_count", 0
            ),
            "stale_symbol_count": enrichment_ops.get("stale_symbol_count", 0),
        },
        "research_pack": {
            "available": latest_research_pack is not None,
            "pack_id": (
                (latest_research_pack.get("registry") or {}).get("pack_id")
                or latest_research_pack.get("pack_id")
                if latest_research_pack
                else None
            ),
            "status": latest_research_pack.get("status") if latest_research_pack else None,
            "candidate_count": (
                (latest_research_pack.get("summary") or {}).get("candidate_count", 0)
                if latest_research_pack
                else 0
            ),
            "material_change_count": (
                (latest_research_pack.get("summary") or {}).get(
                    "material_change_count", 0
                )
                if latest_research_pack
                else 0
            ),
            "high_risk_count": (
                (latest_research_pack.get("summary") or {}).get("high_risk_count", 0)
                if latest_research_pack
                else 0
            ),
            "reviewed_at": (
                (latest_research_pack.get("registry") or {}).get("reviewed_at")
                if latest_research_pack
                else None
            ),
        },
        "consensus": {
            "evaluated_symbol_count": consensus.get("evaluated_symbol_count", 0),
            "blocked_symbol_count": consensus.get("blocked_symbol_count", 0),
            "critical_conflict_count": consensus.get("critical_conflict_count", 0),
            "conflict_count": consensus.get("conflict_count", 0),
            "aligned_symbol_count": consensus.get("aligned_symbol_count", 0),
        },
        "thesis_engine": {
            "active_candidate_count": theses.get("active_candidate_count", 0),
            "versioned_candidate_count": theses.get("versioned_candidate_count", 0),
            "invalidated_count": theses.get("invalidated_count", 0),
            "warning_count": theses.get("warning_count", 0),
            "intact_count": theses.get("intact_count", 0),
            "unevaluated_count": theses.get("unevaluated_count", 0),
        },
        "decision_review": {
            "available": latest_decision_review is not None,
            "run_id": (
                (latest_decision_review.get("registry") or {}).get("run_id")
                if latest_decision_review
                else None
            ),
            "verdict": (
                latest_decision_review.get("verdict")
                if latest_decision_review
                else None
            ),
            "matured_outcomes": (
                (latest_decision_review.get("coverage") or {}).get(
                    "matured_outcomes", 0
                )
                if latest_decision_review
                else 0
            ),
            "decision_accuracy": (
                (latest_decision_review.get("summary") or {}).get(
                    "decision_accuracy"
                )
                if latest_decision_review
                else None
            ),
            "helpful_alert_count": (
                (latest_decision_review.get("summary") or {}).get(
                    "helpful_alert_count", 0
                )
                if latest_decision_review
                else 0
            ),
            "false_alert_count": (
                (latest_decision_review.get("summary") or {}).get(
                    "false_alert_count", 0
                )
                if latest_decision_review
                else 0
            ),
            "reviewed_at": (
                (latest_decision_review.get("registry") or {}).get("reviewed_at")
                if latest_decision_review
                else None
            ),
        },
        "shadow_lab": {
            "experiment_count": shadow_status.get("experiment_count", 0),
            "active_count": shadow_status.get("active_count", 0),
            "evaluated_count": shadow_status.get("evaluated_count", 0),
            "actionable_unreviewed_count": shadow_status.get(
                "actionable_unreviewed_count", 0
            ),
            "challenger_promising_count": shadow_status.get(
                "challenger_promising_count", 0
            ),
            "control_leads_count": shadow_status.get("control_leads_count", 0),
        },
        "model_governance": {
            "proposal_count": governance.get("proposal_count", 0),
            "pending_approval_count": governance.get("pending_approval_count", 0),
            "probation_count": governance.get("probation_count", 0),
            "promotion_eligible_count": governance.get(
                "promotion_eligible_count", 0
            ),
            "promoted_count": governance.get("promoted_count", 0),
            "rollback_available_count": governance.get(
                "rollback_available_count", 0
            ),
            "active_policy_count": len(governance.get("active_policies", [])),
        },
        "portfolio": {
            "nav": paper.get("nav"),
            "total_return": paper.get("total_return"),
            "alpha_vs_benchmark": paper.get("alpha_vs_benchmark"),
            "max_drawdown": paper.get("max_drawdown"),
            "open_positions": len(open_positions),
            "alerts": len(paper.get("risk_alerts", [])),
        },
        "data_health": {
            "warehouse_ready": warehouse.get("ready", False),
            "market_bars": warehouse.get("market_bar_count", 0),
            "quality_failures": warehouse.get("quality_failures", 0),
            "latest_scan_date": latest_scan.get("as_of_date") if latest_scan else None,
            "latest_scan_errors": len(latest_scan.get("errors", [])) if latest_scan else 0,
            "enrichment_alerts": enrichment_ops.get("unacknowledged_alert_count", 0),
            "stale_enrichment_symbols": enrichment_ops.get("stale_symbol_count", 0),
        },
        "notice": "Research workflow only. The briefing cannot place or route an order.",
    }


def capture_daily_briefing(
    briefing: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Persist a content-addressed briefing snapshot and retain review state."""
    cfg = config or get_config()
    identity = {
        "as_of_date": briefing["as_of_date"],
        "posture": briefing["posture"],
        "priorities": briefing["priorities"],
        "checklist": briefing["checklist"],
        "model_health": briefing["model_health"],
        "enrichment": briefing.get("enrichment"),
        "enrichment_operations": briefing.get("enrichment_operations"),
        "research_pack": briefing.get("research_pack"),
        "consensus": briefing.get("consensus"),
        "decision_review": briefing.get("decision_review"),
        "shadow_lab": briefing.get("shadow_lab"),
        "model_governance": briefing.get("model_governance"),
        "portfolio": briefing["portfolio"],
        "data_health": briefing["data_health"],
    }
    briefing_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    now = datetime.now(UTC)
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        connection.execute(
            """
            INSERT OR IGNORE INTO daily_research_briefings VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
            )
            """,
            [
                briefing_id,
                briefing["model_version"],
                briefing["as_of_date"],
                briefing["posture"],
                briefing["counts"]["total"],
                briefing["counts"]["critical"],
                briefing["counts"]["high"],
                _safe_json(briefing),
                now,
            ],
        )
        row = connection.execute(
            "SELECT created_at, reviewed_at FROM daily_research_briefings WHERE briefing_id = ?",
            [briefing_id],
        ).fetchone()
    finally:
        connection.close()
    return {
        "briefing_id": briefing_id,
        "saved": True,
        "created_at": _utc_iso(row[0]),
        "reviewed_at": _utc_iso(row[1]),
    }


def mark_briefing_reviewed(
    briefing_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Mark a briefing snapshot reviewed without deleting its priorities."""
    cfg = config or get_config()
    reviewed_at = datetime.now(UTC)
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        row = connection.execute(
            """
            UPDATE daily_research_briefings SET reviewed_at = ?
            WHERE briefing_id = ? RETURNING briefing_id
            """,
            [reviewed_at, briefing_id],
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    return {"briefing_id": briefing_id, "reviewed_at": _utc_iso(reviewed_at)}


def get_latest_daily_briefing(
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the newest saved briefing with its registry metadata."""
    cfg = config or get_config()
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT briefing_id, payload_json, created_at, reviewed_at
            FROM daily_research_briefings ORDER BY created_at DESC LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    briefing = json.loads(row[1])
    briefing["registry"] = {
        "briefing_id": row[0],
        "saved": True,
        "created_at": _utc_iso(row[2]),
        "reviewed_at": _utc_iso(row[3]),
    }
    return briefing


def list_daily_briefings(
    config: dict[str, Any] | None = None, *, limit: int = 10
) -> list[dict[str, Any]]:
    """List compact briefing snapshots for the dashboard history."""
    cfg = config or get_config()
    connection = _candidate_connect(cfg)
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT briefing_id, as_of_date, posture, priority_count, critical_count,
                   high_count, created_at, reviewed_at
            FROM daily_research_briefings ORDER BY created_at DESC LIMIT ?
            """,
            [max(1, min(int(limit), 100))],
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "briefing_id": row[0],
            "as_of_date": row[1].isoformat(),
            "posture": row[2],
            "priority_count": row[3],
            "critical_count": row[4],
            "high_count": row[5],
            "created_at": _utc_iso(row[6]),
            "reviewed_at": _utc_iso(row[7]),
        }
        for row in rows
    ]
