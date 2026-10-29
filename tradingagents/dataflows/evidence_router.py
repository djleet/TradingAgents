"""Route persisted research evidence through a strict point-in-time gate.

The router never fetches vendors.  It only reads locally archived snapshots,
selects evidence that was eligible on the requested analysis date, and records
both accepted and blocked sources for audit.  Live-only observations made after
the analysis date are deliberately excluded from the agent-facing context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import UTC, date, datetime
from typing import Any

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.enrichment_hub import DATASETS
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.warehouse import (
    _connect as _warehouse_connect,
    dependency_available,
    warehouse_path,
)

logger = logging.getLogger(__name__)

EVIDENCE_ROUTER_VERSION = "evidence-router-v1"
PURPOSES = ("analysis", "candidate", "backtest")

_DATA_KEYS = {
    "earnings_estimates": "earnings",
    "options_implied_volatility": "options",
    "institutional_ownership": "ownership",
    "sec_insider_filings": "sec_insiders",
    "macro_calendar": "macro",
}
_EXPECTED = (*DATASETS, "opportunity_scan", "validation", "research_context")


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _number(value: Any, digits: int = 4) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if math.isfinite(result) else None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _date_part(value: Any) -> str | None:
    text = _iso(value)
    return text[:10] if text else None


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_bundles (
            bundle_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            analysis_date DATE NOT NULL,
            purpose VARCHAR NOT NULL,
            verdict VARCHAR NOT NULL,
            coverage_score DOUBLE NOT NULL,
            eligible_count INTEGER NOT NULL,
            blocked_count INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_evidence_bundles_lookup
        ON evidence_bundles(symbol, purpose, analysis_date, created_at)
        """
    )


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    connection = _warehouse_connect(warehouse_path(cfg))
    _ensure_schema(connection)
    return connection


def _source_facts(dataset: str, data: dict[str, Any]) -> dict[str, Any]:
    """Keep the routed payload compact and stable for prompts and APIs."""
    if dataset == "earnings_estimates":
        targets = data.get("price_targets") or {}
        first_eps = (data.get("earnings_estimates") or [{}])[0]
        return {
            "earnings_date": data.get("earnings_date"),
            "target_upside": _number(data.get("target_upside")),
            "target_mean": _number(targets.get("mean")),
            "next_eps_estimate": _number(first_eps.get("avg")),
            "next_eps_growth": _number(first_eps.get("growth")),
        }
    if dataset == "options_implied_volatility":
        return {
            "nearest_expiry": data.get("nearest_expiry"),
            "atm_iv": _number(data.get("atm_iv")),
            "expected_move_pct": _number(data.get("expected_move_pct")),
            "put_call_open_interest": _number(data.get("put_call_open_interest")),
        }
    if dataset == "institutional_ownership":
        return {
            "institutional_ownership_pct": _number(data.get("institutional_ownership_pct")),
            "insider_ownership_pct": _number(data.get("insider_ownership_pct")),
            "institution_count": len(data.get("institutions") or data.get("major_holders") or []),
        }
    if dataset == "sec_insider_filings":
        filings = data.get("filings") or data.get("recent_filings") or []
        return {
            "filing_count": len(filings),
            "latest_filing_date": (filings[0].get("filing_date") or filings[0].get("filedAt")) if filings else None,
            "transaction_summary": data.get("transaction_summary") or data.get("summary"),
        }
    if dataset == "macro_calendar":
        releases = data.get("upcoming_releases") or data.get("releases") or []
        indicators = data.get("indicators") or {}
        return {
            "upcoming_release_count": len(releases),
            "next_release": releases[0] if releases else None,
            "indicators": indicators,
        }
    return {}


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (dict, list)):
        return _safe_json(value)[:300]
    return str(value)


def _summary(title: str, facts: dict[str, Any]) -> str:
    populated = [(key.replace("_", " "), value) for key, value in facts.items() if value not in (None, "", [], {})]
    if not populated:
        return f"{title}: archived evidence is available."
    return f"{title}: " + "; ".join(f"{key}={_fmt(value)}" for key, value in populated[:6]) + "."


def _load_enrichment(connection, symbol: str, analysis_date: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    eligible: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    missing: set[str] = set(DATASETS)
    try:
        rows = connection.execute(
            """
            SELECT snapshot_id, payload_json, created_at
            FROM enrichment_snapshots
            WHERE symbol = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT 50
            """,
            [symbol, analysis_date],
        ).fetchall()
    except Exception:  # table is optional in older warehouses
        return eligible, blocked, missing

    selected: set[str] = set()
    blocked_seen: set[str] = set()
    for snapshot_id, payload_json, created_at in rows:
        try:
            snapshot = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        source_map = {item.get("dataset"): item for item in snapshot.get("sources", [])}
        for dataset in DATASETS:
            if dataset in selected:
                continue
            source = source_map.get(dataset) or {}
            status = source.get("status", "unavailable")
            if status == "unavailable":
                continue
            missing.discard(dataset)
            observed_at = source.get("observed_at") or snapshot.get("observed_at")
            point_in_time = bool(source.get("point_in_time"))
            if not point_in_time and _date_part(observed_at) > analysis_date:
                if dataset not in blocked_seen:
                    blocked.append(
                        {
                            "dataset": dataset,
                            "title": dataset.replace("_", " ").title(),
                            "provider": source.get("provider", "Unknown"),
                            "point_in_time": False,
                            "evidence_date": snapshot.get("as_of_date"),
                            "observed_at": observed_at,
                            "status": "blocked",
                            "reason": "observed_after_analysis_date",
                            "message": "Live snapshot was observed after the analysis date.",
                            "snapshot_id": snapshot_id,
                        }
                    )
                    blocked_seen.add(dataset)
                continue
            data = snapshot.get(_DATA_KEYS[dataset]) or {}
            facts = _source_facts(dataset, data)
            title = dataset.replace("_", " ").title()
            eligible.append(
                {
                    "kind": "enrichment",
                    "dataset": dataset,
                    "title": title,
                    "provider": source.get("provider", "Unknown"),
                    "authority": source.get("authority", "unknown"),
                    "point_in_time": point_in_time,
                    "evidence_date": snapshot.get("as_of_date"),
                    "observed_at": observed_at,
                    "confidence_score": _number(source.get("confidence_score"), 1) or 0.0,
                    "status": "eligible",
                    "summary": _summary(title, facts),
                    "facts": facts,
                    "source_id": snapshot_id,
                    "created_at": _iso(created_at),
                }
            )
            selected.add(dataset)
            missing.discard(dataset)
    return eligible, blocked, missing


def _load_opportunity(connection, symbol: str, analysis_date: str) -> dict[str, Any] | None:
    rows = connection.execute(
        """SELECT scan_id, as_of_date, result_json, created_at FROM opportunity_scans
           WHERE as_of_date <= ? ORDER BY as_of_date DESC, created_at DESC LIMIT 30""",
        [analysis_date],
    ).fetchall()
    for scan_id, as_of_date, payload, created_at in rows:
        report = json.loads(payload)
        item = next((candidate for candidate in report.get("items", []) if candidate.get("symbol") == symbol), None)
        if not item:
            continue
        validation = item.get("validation") or {}
        facts = {
            "opportunity_score": _number(item.get("opportunity_score"), 2),
            "rank": item.get("rank"),
            "bucket": item.get("bucket"),
            "signal": item.get("signal"),
            "risk_regime": item.get("risk_regime"),
            "quality_score": _number((item.get("quality") or {}).get("score"), 1),
            "validation_robustness": validation.get("robustness"),
        }
        return {
            "kind": "local_model",
            "dataset": "opportunity_scan",
            "title": "Opportunity Scanner",
            "provider": "Local deterministic model",
            "authority": "derived",
            "point_in_time": True,
            "evidence_date": _iso(as_of_date),
            "observed_at": _iso(created_at),
            "confidence_score": _number(float(item.get("confidence", 0)) * 100, 1) or 0.0,
            "status": "eligible",
            "summary": _summary("Opportunity Scanner", facts),
            "facts": facts,
            "source_id": scan_id,
            "created_at": _iso(created_at),
        }
    return None


def _load_validation(connection, symbol: str, analysis_date: str) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT experiment_id, as_of_date, robustness, result_json, created_at
        FROM validation_experiments WHERE symbol = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC, created_at DESC LIMIT 1
        """,
        [symbol, analysis_date],
    ).fetchone()
    if not row:
        return None
    experiment_id, as_of_date, robustness, payload, created_at = row
    report = json.loads(payload)
    metrics = report.get("metrics") or {}
    facts = {
        "robustness": robustness,
        "oos_return": _number(metrics.get("oos_return")),
        "sharpe_ratio": _number(metrics.get("sharpe_ratio")),
        "max_drawdown": _number(metrics.get("max_drawdown")),
        "profitable_fold_rate": _number(metrics.get("profitable_fold_rate")),
        "fold_count": metrics.get("fold_count"),
    }
    confidence = 86 if robustness == "robust" else 72 if robustness in {"supported", "mixed"} else 55
    return {
        "kind": "local_model",
        "dataset": "validation",
        "title": "Walk-forward validation",
        "provider": "Local deterministic model",
        "authority": "derived",
        "point_in_time": True,
        "evidence_date": _iso(as_of_date),
        "observed_at": _iso(created_at),
        "confidence_score": confidence,
        "status": "eligible",
        "summary": _summary("Walk-forward validation", facts),
        "facts": facts,
        "source_id": experiment_id,
        "created_at": _iso(created_at),
    }


def _load_context(connection, symbol: str, analysis_date: str) -> dict[str, Any] | None:
    try:
        row = connection.execute(
            """
            SELECT context_id, as_of_date, result_json, created_at
            FROM research_context_snapshots WHERE symbol = ? AND as_of_date <= ?
            ORDER BY as_of_date DESC, created_at DESC LIMIT 1
            """,
            [symbol, analysis_date],
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    context_id, as_of_date, payload, created_at = row
    report = json.loads(payload)
    fundamentals = report.get("fundamentals") or {}
    events = report.get("events") or {}
    decision = report.get("decision_confidence") or {}
    facts = {
        "fundamental_score": _number(fundamentals.get("score"), 1),
        "event_risk_score": _number(events.get("risk_score"), 1),
        "event_risk_level": events.get("risk_level"),
        "decision_confidence": _number(decision.get("score"), 1),
        "decision_label": decision.get("label"),
    }
    return {
        "kind": "research_context",
        "dataset": "research_context",
        "title": "Research Context",
        "provider": "Point-in-time local context",
        "authority": "derived",
        "point_in_time": True,
        "evidence_date": _iso(as_of_date),
        "observed_at": _iso(created_at),
        "confidence_score": _number(decision.get("score"), 1) or 60.0,
        "status": "eligible",
        "summary": _summary("Research Context", facts),
        "facts": facts,
        "source_id": context_id,
        "created_at": _iso(created_at),
    }


def _empty_bundle(symbol: str, analysis_date: str, purpose: str, message: str) -> dict[str, Any]:
    return {
        "model_version": EVIDENCE_ROUTER_VERSION,
        "symbol": symbol,
        "analysis_date": analysis_date,
        "purpose": purpose,
        "verdict": "insufficient",
        "coverage_score": 0.0,
        "eligible_count": 0,
        "blocked_count": 0,
        "missing_count": len(_EXPECTED),
        "items": [],
        "blocked": [],
        "missing": list(_EXPECTED),
        "message": message,
        "policy": {
            "live_rule": "Live-only evidence is eligible only when observed on or before the analysis date.",
            "point_in_time_rule": "Pinned evidence may be reconstructed later when its source query is bounded by the analysis date.",
            "network_fetches": False,
        },
    }


def build_evidence_bundle(
    symbol: str,
    analysis_date: str,
    config: dict[str, Any] | None = None,
    *,
    purpose: str = "analysis",
    persist: bool = True,
) -> dict[str, Any]:
    """Select and optionally persist all evidence eligible at ``analysis_date``."""
    canonical = normalize_symbol(symbol)
    date.fromisoformat(analysis_date)
    if purpose not in PURPOSES:
        raise ValueError(f"Purpose must be one of: {', '.join(PURPOSES)}")
    connection = _connect(config)
    if connection is None:
        return _empty_bundle(canonical, analysis_date, purpose, "Local evidence warehouse is unavailable.")
    try:
        eligible, blocked, missing = _load_enrichment(connection, canonical, analysis_date)
        loaders = {
            "opportunity_scan": _load_opportunity,
            "validation": _load_validation,
            "research_context": _load_context,
        }
        for dataset, loader in loaders.items():
            try:
                item = loader(connection, canonical, analysis_date)
            except Exception as exc:  # one optional evidence family must not break a run
                logger.warning("Evidence Router could not read %s for %s: %s", dataset, canonical, exc)
                item = None
            if item:
                eligible.append(item)
            else:
                missing.add(dataset)

        confidence_total = sum(float(item.get("confidence_score") or 0) for item in eligible)
        coverage_score = round(100 * confidence_total / (100 * len(_EXPECTED)), 1)
        datasets = {item["dataset"] for item in eligible}
        core_ready = {"opportunity_scan", "validation"}.issubset(datasets)
        enrichment_ready = bool(datasets.intersection(DATASETS))
        verdict = "ready" if core_ready and enrichment_ready and coverage_score >= 28 else "partial" if eligible else "insufficient"
        message = (
            "Evidence availability gate passed with local validation and dated enrichment; this is not an investment endorsement."
            if verdict == "ready"
            else "Some evidence families are missing or ineligible; treat the conclusion as provisional."
            if verdict == "partial"
            else "No eligible archived evidence was found for this date."
        )
        bundle = {
            "model_version": EVIDENCE_ROUTER_VERSION,
            "symbol": canonical,
            "analysis_date": analysis_date,
            "purpose": purpose,
            "verdict": verdict,
            "coverage_score": coverage_score,
            "eligible_count": len(eligible),
            "blocked_count": len(blocked),
            "missing_count": len(missing),
            "items": sorted(eligible, key=lambda item: (item["kind"], item["dataset"])),
            "blocked": sorted(blocked, key=lambda item: item["dataset"]),
            "missing": sorted(missing),
            "message": message,
            "policy": {
                "live_rule": "Live-only evidence is eligible only when observed on or before the analysis date.",
                "point_in_time_rule": "Pinned evidence may be reconstructed later when its source query is bounded by the analysis date.",
                "network_fetches": False,
                "blocked_payloads_exposed_to_agents": False,
            },
        }
        semantic = {key: value for key, value in bundle.items() if key not in {"registry"}}
        bundle_id = hashlib.sha256(_safe_json(semantic).encode()).hexdigest()
        bundle["bundle_id"] = bundle_id
        if persist:
            now = datetime.now(UTC)
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence_bundles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    bundle_id,
                    EVIDENCE_ROUTER_VERSION,
                    canonical,
                    analysis_date,
                    purpose,
                    verdict,
                    coverage_score,
                    len(eligible),
                    len(blocked),
                    _safe_json(bundle),
                    now,
                ],
            )
            created = connection.execute(
                "SELECT created_at FROM evidence_bundles WHERE bundle_id = ?", [bundle_id]
            ).fetchone()[0]
            bundle["registry"] = {
                "saved": True,
                "bundle_id": bundle_id,
                "created_at": created.astimezone(UTC).isoformat(),
            }
        return bundle
    finally:
        connection.close()


def render_evidence_context(bundle: dict[str, Any]) -> str:
    """Render only eligible facts for agent prompts; blocked payloads stay out."""
    lines = [
        "EVIDENCE ROUTER (archived, policy-filtered context)",
        f"Analysis date: {bundle.get('analysis_date')} | verdict: {bundle.get('verdict')} | coverage: {bundle.get('coverage_score', 0)}/100",
        "Use this as supporting evidence, cite its dates, and explicitly preserve uncertainty. Do not infer values from blocked or missing sources.",
    ]
    items = bundle.get("items") or []
    if items:
        for item in items:
            label = "point-in-time" if item.get("point_in_time") else "live snapshot eligible by observation date"
            lines.append(
                f"- [{item.get('dataset')}] {item.get('summary')} "
                f"Evidence date {item.get('evidence_date')}; {label}; confidence {item.get('confidence_score', 0)}/100."
            )
    else:
        lines.append("- No eligible archived evidence was available.")
    if bundle.get("blocked"):
        names = ", ".join(item.get("dataset", "unknown") for item in bundle["blocked"])
        lines.append(f"Blocked by the temporal gate (facts not exposed): {names}.")
    if bundle.get("missing"):
        lines.append("Missing evidence families: " + ", ".join(bundle["missing"]) + ".")
    return "\n".join(lines)


def get_latest_evidence_bundle(
    config: dict[str, Any] | None = None,
    *,
    symbol: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any] | None:
    if purpose and purpose not in PURPOSES:
        raise ValueError(f"Purpose must be one of: {', '.join(PURPOSES)}")
    connection = _connect(config)
    if connection is None:
        return None
    clauses, params = [], []
    if symbol:
        clauses.append("symbol = ?")
        params.append(normalize_symbol(symbol))
    if purpose:
        clauses.append("purpose = ?")
        params.append(purpose)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    try:
        row = connection.execute(
            f"SELECT bundle_id, payload_json, created_at FROM evidence_bundles {where} ORDER BY created_at DESC LIMIT 1",
            params,
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    bundle = json.loads(row[1])
    bundle["registry"] = {"saved": True, "bundle_id": row[0], "created_at": row[2].astimezone(UTC).isoformat()}
    return bundle


def list_evidence_bundles(
    config: dict[str, Any] | None = None,
    *,
    symbol: str | None = None,
    purpose: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if purpose and purpose not in PURPOSES:
        raise ValueError(f"Purpose must be one of: {', '.join(PURPOSES)}")
    connection = _connect(config)
    if connection is None:
        return []
    clauses, params = [], []
    if symbol:
        clauses.append("symbol = ?")
        params.append(normalize_symbol(symbol))
    if purpose:
        clauses.append("purpose = ?")
        params.append(purpose)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 100)))
    try:
        rows = connection.execute(
            f"""SELECT bundle_id, symbol, analysis_date, purpose, verdict, coverage_score,
                       eligible_count, blocked_count, created_at
                FROM evidence_bundles {where} ORDER BY created_at DESC LIMIT ?""",
            params,
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "bundle_id": row[0],
            "symbol": row[1],
            "analysis_date": row[2].isoformat(),
            "purpose": row[3],
            "verdict": row[4],
            "coverage_score": row[5],
            "eligible_count": row[6],
            "blocked_count": row[7],
            "created_at": row[8].astimezone(UTC).isoformat(),
        }
        for row in rows
    ]
