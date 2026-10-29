"""Explain agreement and contradictions across routed research evidence."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.dataflows.candidate_pipeline import list_candidates
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.evidence_router import build_evidence_bundle
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.warehouse import (
    _connect as _warehouse_connect,
    dependency_available,
    warehouse_path,
)

CONSENSUS_VERSION = "evidence-consensus-v1"
RESOLUTIONS = {"resolved", "accepted_risk"}
_SEVERITY_PENALTY = {"critical": 18, "high": 9, "medium": 4}


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
        CREATE TABLE IF NOT EXISTS evidence_consensus_reports (
            report_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            analysis_date DATE NOT NULL,
            verdict VARCHAR NOT NULL,
            consensus_score DOUBLE NOT NULL,
            agreement_rate DOUBLE NOT NULL,
            conflict_count INTEGER NOT NULL,
            critical_count INTEGER NOT NULL,
            qualification_allowed BOOLEAN NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_evidence_consensus_lookup
        ON evidence_consensus_reports(symbol, analysis_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_consensus_conflicts (
            report_id VARCHAR NOT NULL,
            conflict_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            severity VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (report_id, conflict_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_conflict_resolutions (
            conflict_id VARCHAR PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            resolution VARCHAR NOT NULL,
            note VARCHAR NOT NULL,
            resolved_at TIMESTAMPTZ NOT NULL
        )
        """
    )


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    connection = _warehouse_connect(warehouse_path(cfg))
    _ensure_schema(connection)
    return connection


def _signal(
    dataset: str,
    label: str,
    stance: int,
    confidence: float,
    value: Any,
    rationale: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "label": label,
        "stance": max(-1, min(int(stance), 1)),
        "stance_label": "supportive" if stance > 0 else "cautionary" if stance < 0 else "neutral",
        "confidence": round(max(0.0, min(float(confidence), 100.0)), 1),
        "value": value,
        "rationale": rationale,
    }


def _stance_for_score(value: Any, *, high: float, low: float) -> int:
    number = _number(value)
    if number is None:
        return 0
    return 1 if number >= high else -1 if number <= low else 0


def _signals(bundle: dict[str, Any], candidate: dict[str, Any] | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_dataset = {item.get("dataset"): item for item in bundle.get("items", [])}

    opportunity = by_dataset.get("opportunity_scan") or {}
    opportunity_facts = opportunity.get("facts") or {}
    opportunity_score = _number(opportunity_facts.get("opportunity_score"))
    if opportunity_score is not None:
        stance = _stance_for_score(opportunity_score, high=65, low=45)
        result.append(
            _signal(
                "opportunity_scan",
                "Opportunity score",
                stance,
                opportunity.get("confidence_score") or 65,
                opportunity_score,
                f"Scanner score {opportunity_score:g}/100 is {('supportive' if stance > 0 else 'cautionary' if stance < 0 else 'neutral')}.",
            )
        )

    validation = by_dataset.get("validation") or {}
    validation_facts = validation.get("facts") or {}
    if validation_facts:
        robustness = str(validation_facts.get("robustness") or "unknown").lower()
        sharpe = _number(validation_facts.get("sharpe_ratio"))
        oos_return = _number(validation_facts.get("oos_return"))
        profitable = _number(validation_facts.get("profitable_fold_rate"))
        positive = robustness in {"robust", "supported"} and (sharpe is None or sharpe > 0)
        negative = (
            robustness in {"weak", "fragile", "rejected", "unsupported"}
            or (sharpe is not None and sharpe < 0)
            or (oos_return is not None and oos_return < 0 and (profitable or 0) < 0.5)
        )
        stance = 1 if positive else -1 if negative else 0
        result.append(
            _signal(
                "validation",
                "Walk-forward validation",
                stance,
                validation.get("confidence_score") or 60,
                robustness,
                f"Validation is {robustness}; Sharpe is {sharpe if sharpe is not None else 'unknown'}.",
            )
        )

    earnings = by_dataset.get("earnings_estimates") or {}
    earnings_facts = earnings.get("facts") or {}
    target_upside = _number(earnings_facts.get("target_upside"))
    if target_upside is not None:
        stance = 1 if target_upside >= 0.1 else -1 if target_upside <= -0.05 else 0
        result.append(
            _signal(
                "earnings_estimates",
                "Analyst target range",
                stance,
                earnings.get("confidence_score") or 55,
                target_upside,
                f"Consensus target upside is {target_upside * 100:+.1f}%.",
            )
        )

    options = by_dataset.get("options_implied_volatility") or {}
    options_facts = options.get("facts") or {}
    put_call = _number(options_facts.get("put_call_open_interest"))
    if put_call is not None:
        stance = 1 if put_call <= 0.75 else -1 if put_call >= 1.25 else 0
        result.append(
            _signal(
                "options_implied_volatility",
                "Options positioning",
                stance,
                options.get("confidence_score") or 50,
                put_call,
                f"Put/call open-interest ratio is {put_call:.2f}.",
            )
        )

    insiders = by_dataset.get("sec_insider_filings") or {}
    insider_facts = insiders.get("facts") or {}
    insider_summary = insider_facts.get("transaction_summary")
    if insider_summary:
        summary_text = _safe_json(insider_summary).lower()
        positive_words = ("purchase", "bought", "acquisition", "buy")
        negative_words = ("sale", "sold", "disposition", "sell")
        positive = any(word in summary_text for word in positive_words)
        negative = any(word in summary_text for word in negative_words)
        stance = 1 if positive and not negative else -1 if negative and not positive else 0
        result.append(
            _signal(
                "sec_insider_filings",
                "Insider filings",
                stance,
                insiders.get("confidence_score") or 75,
                insider_summary,
                "SEC ownership filings were classified from their archived transaction summary.",
            )
        )

    context = by_dataset.get("research_context") or {}
    context_facts = context.get("facts") or {}
    fundamental_score = _number(context_facts.get("fundamental_score"))
    if fundamental_score is not None:
        stance = _stance_for_score(fundamental_score, high=65, low=40)
        result.append(
            _signal(
                "research_context",
                "Filed fundamentals",
                stance,
                context.get("confidence_score") or 60,
                fundamental_score,
                f"Point-in-time fundamental score is {fundamental_score:g}/100.",
            )
        )

    if candidate and not result:
        score = _number(candidate.get("latest_score"))
        if score is not None:
            result.append(
                _signal(
                    "candidate_pipeline",
                    "Saved candidate score",
                    _stance_for_score(score, high=65, low=45),
                    candidate.get("decision_confidence") or 45,
                    score,
                    "The candidate score is available, but its underlying routed sources are missing.",
                )
            )
    return result


def _conflict(
    symbol: str,
    conflict_type: str,
    severity: str,
    title: str,
    message: str,
    left: dict[str, Any],
    right: dict[str, Any],
    rule: str,
) -> dict[str, Any]:
    identity = {
        "symbol": symbol,
        "type": conflict_type,
        "left": [left.get("dataset"), left.get("value")],
        "right": [right.get("dataset"), right.get("value")],
        "rule": rule,
    }
    return {
        "conflict_id": hashlib.sha256(_safe_json(identity).encode()).hexdigest(),
        "type": conflict_type,
        "severity": severity,
        "title": title,
        "message": message,
        "left": {
            "dataset": left.get("dataset"),
            "label": left.get("label"),
            "value": left.get("value"),
            "stance": left.get("stance_label"),
        },
        "right": {
            "dataset": right.get("dataset"),
            "label": right.get("label"),
            "value": right.get("value"),
            "stance": right.get("stance_label"),
        },
        "rule": rule,
        "qualification_impact": severity == "critical",
        "resolution": None,
    }


def _find_signal(signals: list[dict[str, Any]], dataset: str) -> dict[str, Any] | None:
    return next((item for item in signals if item["dataset"] == dataset), None)


def _detect_conflicts(
    symbol: str,
    signals: list[dict[str, Any]],
    bundle: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    opportunity = _find_signal(signals, "opportunity_scan")
    validation = _find_signal(signals, "validation")
    fundamentals = _find_signal(signals, "research_context")
    earnings = _find_signal(signals, "earnings_estimates")
    options = _find_signal(signals, "options_implied_volatility")

    if opportunity and validation and opportunity["stance"] * validation["stance"] == -1:
        critical = min(opportunity["confidence"], validation["confidence"]) >= 60
        conflicts.append(
            _conflict(
                symbol,
                "scanner_vs_validation",
                "critical" if critical else "high",
                "Scanner and walk-forward evidence disagree",
                "The current cross-sectional signal points in the opposite direction from out-of-sample validation.",
                opportunity,
                validation,
                "Opposite directional stances; critical when both source confidences are at least 60.",
            )
        )
    if opportunity and fundamentals and opportunity["stance"] * fundamentals["stance"] == -1:
        conflicts.append(
            _conflict(
                symbol,
                "scanner_vs_fundamentals",
                "critical" if min(opportunity["confidence"], fundamentals["confidence"]) >= 65 else "high",
                "Market signal and filed fundamentals disagree",
                "The opportunity score and point-in-time fundamental evidence have opposite directional stances.",
                opportunity,
                fundamentals,
                "Opposite scanner and filed-fundamental stances.",
            )
        )
    if earnings and opportunity and earnings["stance"] * opportunity["stance"] == -1:
        large_target = abs(float(earnings.get("value") or 0)) >= 0.25
        conflicts.append(
            _conflict(
                symbol,
                "analyst_vs_scanner",
                "critical" if large_target and opportunity["confidence"] >= 60 else "high",
                "Analyst consensus and Scanner disagree",
                "The analyst target range points in the opposite direction from the deterministic opportunity signal.",
                earnings,
                opportunity,
                "Opposite stances; critical for target gaps of at least 25% with credible scanner evidence.",
            )
        )
    if earnings and options and earnings["stance"] * options["stance"] == -1:
        extreme_options = float(options.get("value") or 0) >= 1.8
        conflicts.append(
            _conflict(
                symbol,
                "analyst_vs_options",
                "critical" if extreme_options else "high",
                "Analyst targets and options positioning disagree",
                "Consensus targets and listed-options positioning have opposite directional implications.",
                earnings,
                options,
                "Opposite target and options stances; critical when put/call open interest is at least 1.8.",
            )
        )

    if candidate and opportunity:
        risk_level = str(candidate.get("risk_level") or "unknown").lower()
        event_risk = _number(candidate.get("event_risk")) or 0
        if opportunity["stance"] > 0 and risk_level in {"high", "critical"}:
            risk_signal = _signal(
                "candidate_risk",
                "Candidate risk",
                -1,
                min(100, max(60, event_risk)),
                risk_level,
                f"Current candidate risk is {risk_level} with event risk {event_risk:g}/100.",
            )
            conflicts.append(
                _conflict(
                    symbol,
                    "score_vs_event_risk",
                    "critical" if event_risk >= 70 else "high",
                    "High opportunity score conflicts with current risk",
                    "The supportive opportunity score coexists with a high saved risk classification.",
                    opportunity,
                    risk_signal,
                    "Critical only when event risk is at least 70/100.",
                )
            )
        saved_validation = str(candidate.get("validation_verdict") or "").lower()
        if validation and validation["stance"] < 0 and saved_validation in {
            "supported",
            "robust",
            "promising",
        }:
            saved_signal = _signal(
                "candidate_pipeline",
                "Saved validation verdict",
                1,
                candidate.get("decision_confidence") or 60,
                saved_validation,
                "The workflow still carries a supportive validation label.",
            )
            conflicts.append(
                _conflict(
                    symbol,
                    "stale_validation_state",
                    "critical",
                    "Candidate validation state is stale",
                    "The saved workflow verdict is supportive while the latest routed validation is cautionary.",
                    saved_signal,
                    validation,
                    "A supportive workflow label cannot coexist with cautionary current validation at qualification.",
                )
            )
    return conflicts


def _apply_resolutions(connection, conflicts: list[dict[str, Any]]) -> None:
    if not conflicts:
        return
    identifiers = [item["conflict_id"] for item in conflicts]
    placeholders = ",".join("?" for _ in identifiers)
    rows = connection.execute(
        f"""SELECT conflict_id, resolution, note, resolved_at
            FROM evidence_conflict_resolutions WHERE conflict_id IN ({placeholders})""",
        identifiers,
    ).fetchall()
    resolutions = {
        row[0]: {
            "resolution": row[1],
            "note": row[2],
            "resolved_at": row[3].astimezone(UTC).isoformat(),
        }
        for row in rows
    }
    for conflict in conflicts:
        conflict["resolution"] = resolutions.get(conflict["conflict_id"])


def build_consensus_report(
    symbol: str,
    analysis_date: str,
    config: dict[str, Any] | None = None,
    *,
    bundle: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Evaluate directional agreement and explicit cross-source conflicts."""
    canonical = normalize_symbol(symbol)
    date.fromisoformat(analysis_date)
    cfg = config or get_config()
    routed = bundle or build_evidence_bundle(
        canonical, analysis_date, cfg, purpose="candidate"
    )
    if candidate is None:
        candidate = next(
            (
                item
                for item in list_candidates(cfg).get("items", [])
                if item.get("symbol") == canonical
            ),
            None,
        )
    signals = _signals(routed, candidate)
    conflicts = _detect_conflicts(canonical, signals, routed, candidate)
    connection = _connect(cfg)
    if connection is None:
        raise RuntimeError("Consensus Engine requires the local data warehouse")
    try:
        _apply_resolutions(connection, conflicts)
    finally:
        connection.close()

    directional = [item for item in signals if item["stance"]]
    confidence_total = sum(item["confidence"] for item in directional)
    weighted_stance = (
        sum(item["stance"] * item["confidence"] for item in directional)
        / confidence_total
        if confidence_total
        else 0
    )
    raw_consensus_score = round(50 + weighted_stance * 40, 1)
    majority = 1 if weighted_stance > 0 else -1 if weighted_stance < 0 else 0
    agreement_rate = round(
        100
        * (
            sum(item["stance"] == majority for item in directional) / len(directional)
            if directional and majority
            else 0
        ),
        1,
    )
    unresolved = [item for item in conflicts if not item.get("resolution")]
    critical = [item for item in unresolved if item["severity"] == "critical"]
    penalty = min(
        55,
        sum(_SEVERITY_PENALTY.get(item["severity"], 0) for item in unresolved),
    )
    starting_confidence = _number(
        candidate.get("decision_confidence") if candidate else None, 1
    )
    if starting_confidence is None:
        starting_confidence = (
            round(sum(item["confidence"] for item in signals) / len(signals), 1)
            if signals
            else 0
        )
    adjusted_confidence = round(max(0, starting_confidence - penalty), 1)
    qualification_allowed = not critical
    if critical:
        verdict = "blocked"
    elif unresolved:
        verdict = "contested"
    elif conflicts:
        verdict = "resolved"
    elif len(directional) < 2:
        verdict = "insufficient"
    elif agreement_rate >= 67:
        verdict = "aligned"
    else:
        verdict = "mixed"
    gaps = list(routed.get("missing") or [])
    if routed.get("blocked"):
        gaps.extend(
            f"{item.get('dataset')}:blocked" for item in routed.get("blocked", [])
        )
    report = {
        "model_version": CONSENSUS_VERSION,
        "symbol": canonical,
        "analysis_date": analysis_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "consensus_score": raw_consensus_score,
        "agreement_rate": agreement_rate,
        "signal_count": len(signals),
        "directional_signal_count": len(directional),
        "signals": signals,
        "conflicts": conflicts,
        "gaps": sorted(set(gaps)),
        "confidence": {
            "starting_score": starting_confidence,
            "contradiction_penalty": penalty,
            "adjusted_score": adjusted_confidence,
        },
        "qualification_gate": {
            "allowed": qualification_allowed,
            "unresolved_critical_count": len(critical),
            "unresolved_conflict_count": len(unresolved),
            "reason": (
                "Resolve or explicitly accept every critical contradiction before qualification."
                if critical
                else "No unresolved critical contradiction blocks qualification."
            ),
        },
        "evidence": {
            "bundle_id": routed.get("bundle_id"),
            "verdict": routed.get("verdict"),
            "coverage_score": routed.get("coverage_score", 0),
            "eligible_count": routed.get("eligible_count", 0),
            "blocked_count": routed.get("blocked_count", 0),
            "missing_count": routed.get("missing_count", 0),
        },
        "policy": {
            "qualification_block": "Only unresolved critical contradictions block Qualified.",
            "missing_data": "Missing evidence reduces visibility but does not create a contradiction.",
            "llm_calls": False,
            "orders": False,
        },
    }
    identity = {key: value for key, value in report.items() if key != "generated_at"}
    report_id = hashlib.sha256(_safe_json(identity).encode()).hexdigest()
    report["report_id"] = report_id
    if not persist:
        return report

    now = datetime.now(UTC)
    connection = _connect(cfg)
    if connection is None:
        raise RuntimeError("Consensus Engine requires the local data warehouse")
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO evidence_consensus_reports VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                report_id,
                CONSENSUS_VERSION,
                canonical,
                analysis_date,
                verdict,
                raw_consensus_score,
                agreement_rate,
                len(conflicts),
                len(critical),
                qualification_allowed,
                _safe_json(report),
                now,
            ],
        )
        for conflict in conflicts:
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence_consensus_conflicts
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    report_id,
                    conflict["conflict_id"],
                    canonical,
                    conflict["severity"],
                    _safe_json(conflict),
                    now,
                ],
            )
        created_at = connection.execute(
            "SELECT created_at FROM evidence_consensus_reports WHERE report_id = ?",
            [report_id],
        ).fetchone()[0]
    finally:
        connection.close()
    report["registry"] = {
        "saved": True,
        "report_id": report_id,
        "created_at": created_at.astimezone(UTC).isoformat(),
    }
    return report


def get_latest_consensus_report(
    config: dict[str, Any] | None = None, *, symbol: str | None = None
) -> dict[str, Any] | None:
    connection = _connect(config)
    if connection is None:
        return None
    try:
        if symbol:
            row = connection.execute(
                """
                SELECT report_id, payload_json, created_at
                FROM evidence_consensus_reports WHERE symbol = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                [normalize_symbol(symbol)],
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT report_id, payload_json, created_at
                FROM evidence_consensus_reports ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    report = json.loads(row[1])
    report["registry"] = {
        "saved": True,
        "report_id": row[0],
        "created_at": row[2].astimezone(UTC).isoformat(),
    }
    return report


def list_consensus_reports(
    config: dict[str, Any] | None = None,
    *,
    symbol: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    where = "WHERE symbol = ?" if symbol else ""
    params: list[Any] = [normalize_symbol(symbol)] if symbol else []
    params.append(max(1, min(int(limit), 200)))
    try:
        rows = connection.execute(
            f"""
            SELECT report_id, symbol, analysis_date, verdict, consensus_score,
                   agreement_rate, conflict_count, critical_count,
                   qualification_allowed, created_at
            FROM evidence_consensus_reports {where}
            ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "report_id": row[0],
            "symbol": row[1],
            "analysis_date": row[2].isoformat(),
            "verdict": row[3],
            "consensus_score": row[4],
            "agreement_rate": row[5],
            "conflict_count": row[6],
            "critical_count": row[7],
            "qualification_allowed": row[8],
            "created_at": row[9].astimezone(UTC).isoformat(),
        }
        for row in rows
    ]


def get_consensus_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    history = list_consensus_reports(config, limit=200)
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for item in history:
        latest_by_symbol.setdefault(item["symbol"], item)
    reports = list(latest_by_symbol.values())
    return {
        "reports": reports,
        "evaluated_symbol_count": len(reports),
        "blocked_symbol_count": sum(not item["qualification_allowed"] for item in reports),
        "critical_conflict_count": sum(item["critical_count"] for item in reports),
        "conflict_count": sum(item["conflict_count"] for item in reports),
        "aligned_symbol_count": sum(item["verdict"] == "aligned" for item in reports),
        "latest_report": get_latest_consensus_report(config),
    }


def resolve_consensus_conflict(
    conflict_id: str,
    resolution: str,
    note: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if resolution not in RESOLUTIONS:
        raise ValueError("Resolution must be resolved or accepted_risk")
    cleaned_note = note.strip()
    if len(cleaned_note) < 3:
        raise ValueError("Resolution note must explain the decision")
    connection = _connect(config)
    if connection is None:
        raise RuntimeError("Consensus Engine requires the local data warehouse")
    try:
        row = connection.execute(
            """
            SELECT symbol FROM evidence_consensus_conflicts
            WHERE conflict_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            [conflict_id],
        ).fetchone()
        if not row:
            return None
        resolved_at = datetime.now(UTC)
        connection.execute(
            """
            INSERT INTO evidence_conflict_resolutions VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (conflict_id) DO UPDATE SET
                resolution = excluded.resolution, note = excluded.note,
                resolved_at = excluded.resolved_at
            """,
            [conflict_id, row[0], resolution, cleaned_note[:1000], resolved_at],
        )
    finally:
        connection.close()
    return {
        "conflict_id": conflict_id,
        "symbol": row[0],
        "resolution": resolution,
        "note": cleaned_note[:1000],
        "resolved_at": resolved_at.isoformat(),
    }
