"""Explicit promotion governance for prospective shadow experiments."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from tradingagents.dataflows.candidate_pipeline import _connect as _candidate_connect
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.shadow_lab import (
    METRICS,
    build_shadow_evaluation,
    get_shadow_evaluation,
    list_shadow_experiments,
)

MODEL_GOVERNANCE_VERSION = "model-governance-v1"
APPROVAL_ROLES = ("research", "risk")
APPROVAL_DECISIONS = ("approved", "rejected")
PROPOSAL_STATUSES = (
    "pending_approval",
    "approved",
    "probation",
    "promoted",
    "rejected",
    "rolled_back",
)

_LOCK = threading.RLock()


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
        CREATE TABLE IF NOT EXISTS model_governance_proposals (
            proposal_id VARCHAR PRIMARY KEY,
            proposal_number INTEGER UNIQUE NOT NULL,
            model_version VARCHAR NOT NULL,
            experiment_id VARCHAR NOT NULL,
            evaluation_id VARCHAR UNIQUE NOT NULL,
            metric VARCHAR NOT NULL,
            champion_threshold DOUBLE NOT NULL,
            challenger_threshold DOUBLE NOT NULL,
            rationale VARCHAR NOT NULL,
            minimum_probation_observations INTEGER NOT NULL,
            status VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            probation_started_at TIMESTAMPTZ,
            eligible_at TIMESTAMPTZ,
            promoted_at TIMESTAMPTZ,
            rejected_at TIMESTAMPTZ,
            rolled_back_at TIMESTAMPTZ,
            active_policy_version_id VARCHAR,
            previous_policy_version_id VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_governance_proposals_status
        ON model_governance_proposals(status, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_governance_approvals (
            approval_id VARCHAR PRIMARY KEY,
            proposal_id VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            decision VARCHAR NOT NULL,
            note VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_governance_approvals_proposal
        ON model_governance_approvals(proposal_id, role, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_governance_probation_reviews (
            review_id VARCHAR PRIMARY KEY,
            proposal_id VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            verdict VARCHAR NOT NULL,
            control_outcomes INTEGER NOT NULL,
            challenger_outcomes INTEGER NOT NULL,
            excess_delta DOUBLE,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_governance_probation_proposal
        ON model_governance_probation_reviews(proposal_id, as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS governed_policy_versions (
            policy_version_id VARCHAR PRIMARY KEY,
            metric VARCHAR NOT NULL,
            version_number INTEGER NOT NULL,
            threshold DOUBLE NOT NULL,
            source_proposal_id VARCHAR,
            previous_policy_version_id VARCHAR,
            status VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            activated_at TIMESTAMPTZ NOT NULL,
            deactivated_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_governed_policy_active
        ON governed_policy_versions(metric, status, version_number)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_governance_events (
            event_id VARCHAR PRIMARY KEY,
            proposal_id VARCHAR NOT NULL,
            event_type VARCHAR NOT NULL,
            from_status VARCHAR,
            to_status VARCHAR NOT NULL,
            note VARCHAR,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_governance_events_proposal
        ON model_governance_events(proposal_id, created_at)
        """
    )


_PROPOSAL_COLUMNS = (
    "proposal_id",
    "proposal_number",
    "model_version",
    "experiment_id",
    "evaluation_id",
    "metric",
    "champion_threshold",
    "challenger_threshold",
    "rationale",
    "minimum_probation_observations",
    "status",
    "created_at",
    "updated_at",
    "probation_started_at",
    "eligible_at",
    "promoted_at",
    "rejected_at",
    "rolled_back_at",
    "active_policy_version_id",
    "previous_policy_version_id",
)


def _proposal_from_row(row) -> dict[str, Any]:
    item = dict(zip(_PROPOSAL_COLUMNS, row, strict=True))
    item["metric_label"] = METRICS.get(item["metric"], item["metric"])
    for key in (
        "created_at",
        "updated_at",
        "probation_started_at",
        "eligible_at",
        "promoted_at",
        "rejected_at",
        "rolled_back_at",
    ):
        item[key] = _utc_iso(item[key])
    item["champion_threshold"] = _number(item["champion_threshold"])
    item["challenger_threshold"] = _number(item["challenger_threshold"])
    return item


def _policy_from_row(row) -> dict[str, Any]:
    return {
        "policy_version_id": row[0],
        "metric": row[1],
        "metric_label": METRICS.get(row[1], row[1]),
        "version_number": row[2],
        "threshold": _number(row[3]),
        "source_proposal_id": row[4],
        "previous_policy_version_id": row[5],
        "status": row[6],
        "created_at": _utc_iso(row[7]),
        "activated_at": _utc_iso(row[8]),
        "deactivated_at": _utc_iso(row[9]),
    }


def _latest_approvals(connection, proposal_id: str) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT role, decision, note, created_at, approval_id FROM (
            SELECT *, row_number() OVER (
                PARTITION BY role ORDER BY created_at DESC, approval_id DESC
            ) AS recency
            FROM model_governance_approvals WHERE proposal_id = ?
        ) WHERE recency = 1
        """,
        [proposal_id],
    ).fetchall()
    return {
        row[0]: {
            "role": row[0],
            "decision": row[1],
            "note": row[2],
            "created_at": _utc_iso(row[3]),
            "approval_id": row[4],
        }
        for row in rows
    }


def _latest_probation(connection, proposal_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT review_id, as_of_date, verdict, control_outcomes,
               challenger_outcomes, excess_delta, payload_json, created_at
        FROM model_governance_probation_reviews
        WHERE proposal_id = ? ORDER BY created_at DESC LIMIT 1
        """,
        [proposal_id],
    ).fetchone()
    if not row:
        return None
    return {
        "review_id": row[0],
        "as_of_date": row[1].isoformat(),
        "verdict": row[2],
        "control_outcomes": row[3],
        "challenger_outcomes": row[4],
        "excess_delta": _number(row[5]),
        "report": json.loads(row[6]),
        "created_at": _utc_iso(row[7]),
    }


def _record_event(
    connection,
    proposal_id: str,
    event_type: str,
    from_status: str | None,
    to_status: str,
    *,
    note: str | None = None,
    payload: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> None:
    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    identity = (
        f"{proposal_id}|{event_type}|{from_status}|{to_status}|"
        f"{created_at.isoformat()}|{note or ''}"
    )
    connection.execute(
        """
        INSERT INTO model_governance_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            hashlib.sha256(identity.encode()).hexdigest(),
            proposal_id,
            event_type,
            from_status,
            to_status,
            note,
            _safe_json(payload or {}),
            created_at,
        ],
    )


def get_active_governed_policies(
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT * FROM governed_policy_versions
            WHERE status = 'active' ORDER BY metric, version_number DESC
            """
        ).fetchall()
    finally:
        connection.close()
    return [_policy_from_row(row) for row in rows]


def _active_policy(connection, metric: str):
    return connection.execute(
        """
        SELECT * FROM governed_policy_versions
        WHERE metric = ? AND status = 'active'
        ORDER BY version_number DESC LIMIT 1
        """,
        [metric],
    ).fetchone()


def create_promotion_proposal(
    evaluation_id: str,
    rationale: str,
    config: dict[str, Any] | None = None,
    *,
    minimum_probation_observations: int = 10,
    now: datetime | None = None,
) -> dict[str, Any]:
    clean_rationale = rationale.strip()
    if not 10 <= len(clean_rationale) <= 1000:
        raise ValueError("Promotion rationale must contain 10 to 1000 characters")
    if not 3 <= int(minimum_probation_observations) <= 500:
        raise ValueError("Probation minimum must be between 3 and 500 observations")
    cfg = config or get_config()
    evaluation = get_shadow_evaluation(evaluation_id, cfg)
    if not evaluation:
        raise ValueError("Shadow evaluation not found")
    if evaluation.get("verdict") != "challenger_promising":
        raise ValueError("Only a promising challenger can enter promotion governance")
    if not (evaluation.get("registry") or {}).get("reviewed_at"):
        raise ValueError("Shadow evaluation must be reviewed before promotion governance")
    if not (evaluation.get("comparison") or {}).get("supported"):
        raise ValueError("Shadow comparison has not reached its minimum sample")
    experiment = evaluation["experiment"]
    minimum = int(minimum_probation_observations)
    if minimum < int(experiment["minimum_observations"]):
        raise ValueError("Probation minimum cannot be lower than the shadow experiment minimum")
    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    proposal_id = hashlib.sha256(f"promotion|{evaluation_id}".encode()).hexdigest()
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            existing = connection.execute(
                "SELECT * FROM model_governance_proposals WHERE evaluation_id = ?",
                [evaluation_id],
            ).fetchone()
            if existing:
                return _proposal_from_row(existing)
            active = _active_policy(connection, experiment["metric"])
            if active and abs(float(active[3]) - float(experiment["control_threshold"])) > 1e-9:
                raise ValueError(
                    "Shadow control no longer matches the active governed champion"
                )
            proposal_number = connection.execute(
                "SELECT coalesce(max(proposal_number), 0) + 1 FROM model_governance_proposals"
            ).fetchone()[0]
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT INTO model_governance_proposals VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending_approval', ?, ?,
                    NULL, NULL, NULL, NULL, NULL, NULL, NULL
                )
                """,
                [
                    proposal_id,
                    proposal_number,
                    MODEL_GOVERNANCE_VERSION,
                    experiment["experiment_id"],
                    evaluation_id,
                    experiment["metric"],
                    experiment["control_threshold"],
                    experiment["challenger_threshold"],
                    clean_rationale,
                    minimum,
                    created_at,
                    created_at,
                ],
            )
            _record_event(
                connection,
                proposal_id,
                "proposal_created",
                None,
                "pending_approval",
                note=clean_rationale,
                payload={"evaluation_id": evaluation_id},
                now=created_at,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def record_governance_approval(
    proposal_id: str,
    role: str,
    decision: str,
    note: str,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if role not in APPROVAL_ROLES:
        raise ValueError("Approval role must be research or risk")
    if decision not in APPROVAL_DECISIONS:
        raise ValueError("Approval decision must be approved or rejected")
    clean_note = note.strip()
    if not 10 <= len(clean_note) <= 1000:
        raise ValueError("Approval note must contain 10 to 1000 characters")
    cfg = config or get_config()
    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_governance_proposals WHERE proposal_id = ?",
                [proposal_id],
            ).fetchone()
            if not row:
                raise ValueError("Governance proposal not found")
            proposal = _proposal_from_row(row)
            if proposal["status"] != "pending_approval":
                raise ValueError("Approvals are closed for this proposal")
            approval_id = hashlib.sha256(
                f"{proposal_id}|{role}|{decision}|{created_at.isoformat()}".encode()
            ).hexdigest()
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                "INSERT INTO model_governance_approvals VALUES (?, ?, ?, ?, ?, ?)",
                [approval_id, proposal_id, role, decision, clean_note, created_at],
            )
            if decision == "rejected":
                new_status = "rejected"
                connection.execute(
                    """
                    UPDATE model_governance_proposals
                    SET status = 'rejected', rejected_at = ?, updated_at = ?
                    WHERE proposal_id = ?
                    """,
                    [created_at, created_at, proposal_id],
                )
            else:
                approvals = _latest_approvals(connection, proposal_id)
                fully_approved = all(
                    approvals.get(required, {}).get("decision") == "approved"
                    for required in APPROVAL_ROLES
                )
                new_status = "approved" if fully_approved else "pending_approval"
                connection.execute(
                    """
                    UPDATE model_governance_proposals SET status = ?, updated_at = ?
                    WHERE proposal_id = ?
                    """,
                    [new_status, created_at, proposal_id],
                )
            _record_event(
                connection,
                proposal_id,
                f"{role}_{decision}",
                proposal["status"],
                new_status,
                note=clean_note,
                payload={"role": role, "decision": decision},
                now=created_at,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def start_governance_probation(
    proposal_id: str,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    cfg = config or get_config()
    started_at = (now or datetime.now(UTC)).astimezone(UTC)
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_governance_proposals WHERE proposal_id = ?",
                [proposal_id],
            ).fetchone()
            if not row:
                raise ValueError("Governance proposal not found")
            proposal = _proposal_from_row(row)
            if proposal["status"] != "approved":
                raise ValueError("Both Research and Risk approvals are required")
            active = _active_policy(connection, proposal["metric"])
            if active and abs(float(active[3]) - float(proposal["champion_threshold"])) > 1e-9:
                raise ValueError("Active champion changed before probation could start")
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                UPDATE model_governance_proposals
                SET status = 'probation', probation_started_at = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                [started_at, started_at, proposal_id],
            )
            _record_event(
                connection,
                proposal_id,
                "probation_started",
                "approved",
                "probation",
                note="Prospective probation window started",
                payload={"minimum_observations": proposal["minimum_probation_observations"]},
                now=started_at,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def refresh_governance_probation(
    proposal_id: str,
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    observations: list[dict[str, Any]] | None = None,
    price_loader=None,
) -> dict[str, Any]:
    cfg = config or get_config()
    proposal = get_governance_proposal(proposal_id, cfg)
    if not proposal:
        raise ValueError("Governance proposal not found")
    if proposal["status"] != "probation":
        raise ValueError("Proposal is not in probation")
    report = build_shadow_evaluation(
        proposal["experiment_id"],
        as_of_date,
        cfg,
        observations=observations,
        price_loader=price_loader,
        observation_start_at=proposal["probation_started_at"],
    )
    minimum = int(proposal["minimum_probation_observations"])
    control_n = int(report["control"]["matured_outcomes"])
    challenger_n = int(report["challenger"]["matured_outcomes"])
    supported = control_n >= minimum and challenger_n >= minimum
    if not supported:
        verdict = "collecting"
    elif report["verdict"] == "challenger_promising":
        verdict = "eligible"
    elif report["verdict"] == "control_leads":
        verdict = "failed"
    else:
        verdict = "inconclusive"
    now = datetime.now(UTC)
    review_id = hashlib.sha256(
        (
            f"{proposal_id}|{as_of_date}|{verdict}|"
            f"{','.join(sorted(item['outcome_id'] for item in report['timeline']))}"
        ).encode()
    ).hexdigest()
    report["governance_probation"] = {
        "proposal_id": proposal_id,
        "verdict": verdict,
        "minimum_observations_per_arm": minimum,
        "promotion_eligible": verdict == "eligible",
    }
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                INSERT OR IGNORE INTO model_governance_probation_reviews VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    review_id,
                    proposal_id,
                    as_of_date,
                    verdict,
                    control_n,
                    challenger_n,
                    report["comparison"]["excess_delta"],
                    _safe_json(report),
                    now,
                ],
            )
            if verdict == "eligible":
                connection.execute(
                    """
                    UPDATE model_governance_proposals
                    SET eligible_at = coalesce(eligible_at, ?), updated_at = ?
                    WHERE proposal_id = ?
                    """,
                    [now, now, proposal_id],
                )
            _record_event(
                connection,
                proposal_id,
                "probation_refreshed",
                "probation",
                "probation",
                note=f"Probation result: {verdict}",
                payload={
                    "review_id": review_id,
                    "control_outcomes": control_n,
                    "challenger_outcomes": challenger_n,
                },
                now=now,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def _validate_action_note(note: str, label: str) -> str:
    cleaned = note.strip()
    if not 10 <= len(cleaned) <= 1000:
        raise ValueError(f"{label} note must contain 10 to 1000 characters")
    return cleaned


def promote_governance_proposal(
    proposal_id: str,
    note: str,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    clean_note = _validate_action_note(note, "Promotion")
    cfg = config or get_config()
    promoted_at = (now or datetime.now(UTC)).astimezone(UTC)
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_governance_proposals WHERE proposal_id = ?",
                [proposal_id],
            ).fetchone()
            if not row:
                raise ValueError("Governance proposal not found")
            proposal = _proposal_from_row(row)
            if proposal["status"] != "probation":
                raise ValueError("Only a proposal in probation can be promoted")
            probation = _latest_probation(connection, proposal_id)
            if not probation or probation["verdict"] != "eligible":
                raise ValueError("Latest probation review is not promotion-eligible")
            approvals = _latest_approvals(connection, proposal_id)
            if not all(
                approvals.get(role, {}).get("decision") == "approved"
                for role in APPROVAL_ROLES
            ):
                raise ValueError("Both approvals must remain valid")
            active = _active_policy(connection, proposal["metric"])
            if active and abs(float(active[3]) - float(proposal["champion_threshold"])) > 1e-9:
                raise ValueError("Active champion no longer matches the proposal")
            connection.execute("BEGIN TRANSACTION")
            if active:
                previous_policy_id = active[0]
                connection.execute(
                    """
                    UPDATE governed_policy_versions
                    SET status = 'superseded', deactivated_at = ?
                    WHERE policy_version_id = ?
                    """,
                    [promoted_at, previous_policy_id],
                )
            else:
                previous_policy_id = hashlib.sha256(
                    f"baseline|{proposal['metric']}|{proposal['champion_threshold']}".encode()
                ).hexdigest()
                connection.execute(
                    """
                    INSERT OR IGNORE INTO governed_policy_versions VALUES (
                        ?, ?, 1, ?, NULL, NULL, 'superseded', ?, ?, ?
                    )
                    """,
                    [
                        previous_policy_id,
                        proposal["metric"],
                        proposal["champion_threshold"],
                        promoted_at,
                        promoted_at,
                        promoted_at,
                    ],
                )
            next_version = connection.execute(
                """
                SELECT coalesce(max(version_number), 0) + 1
                FROM governed_policy_versions WHERE metric = ?
                """,
                [proposal["metric"]],
            ).fetchone()[0]
            policy_id = hashlib.sha256(
                f"policy|{proposal_id}|{next_version}".encode()
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO governed_policy_versions VALUES (
                    ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL
                )
                """,
                [
                    policy_id,
                    proposal["metric"],
                    next_version,
                    proposal["challenger_threshold"],
                    proposal_id,
                    previous_policy_id,
                    promoted_at,
                    promoted_at,
                ],
            )
            connection.execute(
                """
                UPDATE model_governance_proposals
                SET status = 'promoted', promoted_at = ?, updated_at = ?,
                    active_policy_version_id = ?, previous_policy_version_id = ?
                WHERE proposal_id = ?
                """,
                [promoted_at, promoted_at, policy_id, previous_policy_id, proposal_id],
            )
            _record_event(
                connection,
                proposal_id,
                "challenger_promoted",
                "probation",
                "promoted",
                note=clean_note,
                payload={
                    "policy_version_id": policy_id,
                    "previous_policy_version_id": previous_policy_id,
                },
                now=promoted_at,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def reject_governance_proposal(
    proposal_id: str,
    note: str,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    clean_note = _validate_action_note(note, "Rejection")
    cfg = config or get_config()
    rejected_at = (now or datetime.now(UTC)).astimezone(UTC)
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_governance_proposals WHERE proposal_id = ?",
                [proposal_id],
            ).fetchone()
            if not row:
                raise ValueError("Governance proposal not found")
            proposal = _proposal_from_row(row)
            if proposal["status"] not in {"pending_approval", "approved", "probation"}:
                raise ValueError("Proposal can no longer be rejected")
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                UPDATE model_governance_proposals
                SET status = 'rejected', rejected_at = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                [rejected_at, rejected_at, proposal_id],
            )
            _record_event(
                connection,
                proposal_id,
                "proposal_rejected",
                proposal["status"],
                "rejected",
                note=clean_note,
                now=rejected_at,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def rollback_governed_policy(
    proposal_id: str,
    note: str,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    clean_note = _validate_action_note(note, "Rollback")
    cfg = config or get_config()
    rolled_back_at = (now or datetime.now(UTC)).astimezone(UTC)
    with _LOCK:
        connection = _candidate_connect(cfg)
        try:
            _ensure_schema(connection)
            row = connection.execute(
                "SELECT * FROM model_governance_proposals WHERE proposal_id = ?",
                [proposal_id],
            ).fetchone()
            if not row:
                raise ValueError("Governance proposal not found")
            proposal = _proposal_from_row(row)
            if proposal["status"] != "promoted":
                raise ValueError("Only a promoted proposal can be rolled back")
            active = connection.execute(
                "SELECT status FROM governed_policy_versions WHERE policy_version_id = ?",
                [proposal["active_policy_version_id"]],
            ).fetchone()
            if not active or active[0] != "active":
                raise ValueError("Promoted policy is no longer the active champion")
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                UPDATE governed_policy_versions
                SET status = 'rolled_back', deactivated_at = ?
                WHERE policy_version_id = ?
                """,
                [rolled_back_at, proposal["active_policy_version_id"]],
            )
            connection.execute(
                """
                UPDATE governed_policy_versions
                SET status = 'active', deactivated_at = NULL
                WHERE policy_version_id = ?
                """,
                [proposal["previous_policy_version_id"]],
            )
            connection.execute(
                """
                UPDATE model_governance_proposals
                SET status = 'rolled_back', rolled_back_at = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                [rolled_back_at, rolled_back_at, proposal_id],
            )
            _record_event(
                connection,
                proposal_id,
                "policy_rolled_back",
                "promoted",
                "rolled_back",
                note=clean_note,
                payload={
                    "restored_policy_version_id": proposal[
                        "previous_policy_version_id"
                    ]
                },
                now=rolled_back_at,
            )
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
    return get_governance_proposal(proposal_id, cfg)


def get_governance_proposal(
    proposal_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT * FROM model_governance_proposals WHERE proposal_id = ?",
            [proposal_id],
        ).fetchone()
        if not row:
            return None
        proposal = _proposal_from_row(row)
        proposal["approvals"] = _latest_approvals(connection, proposal_id)
        proposal["probation"] = _latest_probation(connection, proposal_id)
        event_rows = connection.execute(
            """
            SELECT event_id, event_type, from_status, to_status, note,
                   payload_json, created_at
            FROM model_governance_events
            WHERE proposal_id = ? ORDER BY created_at DESC LIMIT 50
            """,
            [proposal_id],
        ).fetchall()
        proposal["events"] = [
            {
                "event_id": item[0],
                "event_type": item[1],
                "from_status": item[2],
                "to_status": item[3],
                "note": item[4],
                "payload": json.loads(item[5]),
                "created_at": _utc_iso(item[6]),
            }
            for item in event_rows
        ]
        if proposal.get("active_policy_version_id"):
            policy_row = connection.execute(
                "SELECT * FROM governed_policy_versions WHERE policy_version_id = ?",
                [proposal["active_policy_version_id"]],
            ).fetchone()
            proposal["policy_version"] = (
                _policy_from_row(policy_row) if policy_row else None
            )
        else:
            proposal["policy_version"] = None
    finally:
        connection.close()
    return proposal


def list_governance_proposals(
    config: dict[str, Any] | None = None, *, limit: int = 50
) -> list[dict[str, Any]]:
    connection = _candidate_connect(config or get_config())
    try:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT * FROM model_governance_proposals
            ORDER BY proposal_number DESC LIMIT ?
            """,
            [max(1, min(int(limit), 100))],
        ).fetchall()
        items = []
        for row in rows:
            proposal = _proposal_from_row(row)
            proposal["approvals"] = _latest_approvals(
                connection, proposal["proposal_id"]
            )
            proposal["probation"] = _latest_probation(
                connection, proposal["proposal_id"]
            )
            items.append(proposal)
    finally:
        connection.close()
    return items


def evaluate_governed_policy(
    candidate: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    policies = get_active_governed_policies(config)
    field_map = {
        "opportunity_score": "latest_score",
        "decision_confidence": "decision_confidence",
        "quality_score": "quality_score",
    }
    checks = []
    for policy in policies:
        value = _number(candidate.get(field_map[policy["metric"]]))
        available = value is not None
        passed = not available or value >= float(policy["threshold"])
        checks.append(
            {
                "metric": policy["metric"],
                "metric_label": policy["metric_label"],
                "value": value,
                "threshold": policy["threshold"],
                "available": available,
                "passed": passed,
                "policy_version_id": policy["policy_version_id"],
                "version_number": policy["version_number"],
            }
        )
    violations = [item for item in checks if item["available"] and not item["passed"]]
    return {
        "allowed": not violations,
        "active_policy_count": len(policies),
        "checks": checks,
        "violations": violations,
        "missing_metrics": [item["metric"] for item in checks if not item["available"]],
        "policy": "Missing metrics lower visibility but do not block qualification.",
    }


def get_model_governance_status(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or get_config()
    proposals = list_governance_proposals(cfg, limit=100)
    proposal_evaluations = {item["evaluation_id"] for item in proposals}
    eligible = []
    for experiment in list_shadow_experiments(cfg, limit=100):
        evaluation = experiment.get("latest_evaluation") or {}
        if (
            evaluation.get("verdict") == "challenger_promising"
            and evaluation.get("reviewed_at")
            and evaluation.get("evaluation_id") not in proposal_evaluations
        ):
            eligible.append(
                {
                    "evaluation_id": evaluation["evaluation_id"],
                    "experiment_id": experiment["experiment_id"],
                    "experiment_name": experiment["name"],
                    "metric": experiment["metric"],
                    "metric_label": experiment["metric_label"],
                    "control_threshold": experiment["control_threshold"],
                    "challenger_threshold": experiment["challenger_threshold"],
                    "excess_delta": evaluation.get("excess_delta"),
                    "as_of_date": evaluation.get("as_of_date"),
                }
            )
    return {
        "proposal_count": len(proposals),
        "pending_approval_count": sum(
            item["status"] == "pending_approval" for item in proposals
        ),
        "approved_count": sum(item["status"] == "approved" for item in proposals),
        "probation_count": sum(item["status"] == "probation" for item in proposals),
        "promotion_eligible_count": sum(
            item.get("probation", {}).get("verdict") == "eligible"
            for item in proposals
            if item.get("probation")
        ),
        "promoted_count": sum(item["status"] == "promoted" for item in proposals),
        "rollback_available_count": sum(
            item["status"] == "promoted" for item in proposals
        ),
        "active_policies": get_active_governed_policies(cfg),
        "eligible_shadow_evaluations": eligible,
        "proposals": proposals,
    }
