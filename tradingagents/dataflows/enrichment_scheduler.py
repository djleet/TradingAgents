"""Durable, rate-limited scheduler for the Data Enrichment Hub."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.enrichment_hub import (
    build_enrichment_snapshot,
    capture_enrichment_snapshot,
)
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.warehouse import (
    _connect as _warehouse_connect,
    dependency_available,
    warehouse_path,
)

SCHEDULER_VERSION = "enrichment-scheduler-v2"
_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_BASE_COOLDOWN_SECONDS = 15 * 60
_CIRCUIT_MAX_COOLDOWN_SECONDS = 6 * 60 * 60


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_schedules (
            schedule_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            universe_id VARCHAR,
            symbols_json VARCHAR NOT NULL,
            symbol_count INTEGER NOT NULL,
            local_time VARCHAR NOT NULL,
            timezone VARCHAR NOT NULL,
            weekdays_json VARCHAR NOT NULL,
            max_symbols_per_run INTEGER NOT NULL,
            min_interval_seconds DOUBLE NOT NULL,
            max_attempts INTEGER NOT NULL,
            stale_after_hours INTEGER NOT NULL,
            active BOOLEAN NOT NULL,
            cursor_index INTEGER NOT NULL,
            last_run_date DATE,
            last_run_at TIMESTAMPTZ,
            last_status VARCHAR,
            last_error VARCHAR,
            last_run_id VARCHAR,
            last_summary_json VARCHAR,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_enrichment_schedules_active
           ON enrichment_schedules(active, updated_at)"""
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_schedule_runs (
            run_id VARCHAR PRIMARY KEY,
            schedule_id VARCHAR NOT NULL,
            model_version VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            status VARCHAR NOT NULL,
            requested_count INTEGER NOT NULL,
            attempted_count INTEGER NOT NULL,
            completed_count INTEGER NOT NULL,
            failed_count INTEGER NOT NULL,
            source_success_count INTEGER NOT NULL,
            source_failure_count INTEGER NOT NULL,
            duration_seconds DOUBLE NOT NULL,
            result_json VARCHAR NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_enrichment_runs_schedule
           ON enrichment_schedule_runs(schedule_id, started_at)"""
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_source_health (
            dataset VARCHAR PRIMARY KEY,
            provider VARCHAR NOT NULL,
            total_checks INTEGER NOT NULL,
            successful_checks INTEGER NOT NULL,
            partial_checks INTEGER NOT NULL,
            failed_checks INTEGER NOT NULL,
            consecutive_failures INTEGER NOT NULL,
            average_confidence DOUBLE NOT NULL,
            last_status VARCHAR NOT NULL,
            last_checked_at TIMESTAMPTZ NOT NULL,
            last_success_at TIMESTAMPTZ,
            last_error VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_scheduler_alerts (
            alert_id VARCHAR PRIMARY KEY,
            schedule_id VARCHAR NOT NULL,
            run_id VARCHAR,
            symbol VARCHAR,
            dataset VARCHAR,
            alert_type VARCHAR NOT NULL,
            severity VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            message VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            acknowledged_at TIMESTAMPTZ,
            affected_symbols_json VARCHAR NOT NULL DEFAULT '[]',
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            next_retry_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        "ALTER TABLE enrichment_scheduler_alerts ADD COLUMN IF NOT EXISTS "
        "affected_symbols_json VARCHAR DEFAULT '[]'"
    )
    connection.execute(
        "ALTER TABLE enrichment_scheduler_alerts ADD COLUMN IF NOT EXISTS "
        "occurrence_count INTEGER DEFAULT 1"
    )
    connection.execute(
        "ALTER TABLE enrichment_scheduler_alerts ADD COLUMN IF NOT EXISTS "
        "next_retry_at TIMESTAMPTZ"
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_enrichment_alerts_status
           ON enrichment_scheduler_alerts(acknowledged_at, created_at)"""
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_source_circuits (
            dataset VARCHAR PRIMARY KEY,
            provider VARCHAR NOT NULL,
            state VARCHAR NOT NULL,
            opened_at TIMESTAMPTZ,
            retry_at TIMESTAMPTZ,
            open_count INTEGER NOT NULL,
            last_probe_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    bootstrap_time = datetime.now(UTC)
    connection.execute(
        """INSERT INTO enrichment_source_circuits
           SELECT h.dataset, h.provider, 'open', ?, ?, 1, NULL, ?
           FROM enrichment_source_health h
           LEFT JOIN enrichment_source_circuits c USING (dataset)
           WHERE c.dataset IS NULL
             AND h.last_status = 'unavailable'
             AND h.consecutive_failures >= ?""",
        [
            bootstrap_time,
            bootstrap_time + timedelta(seconds=_CIRCUIT_BASE_COOLDOWN_SECONDS),
            bootstrap_time,
            _CIRCUIT_FAILURE_THRESHOLD,
        ],
    )


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    connection = _warehouse_connect(warehouse_path(cfg))
    _ensure_schema(connection)
    return connection


def _schedule_row(row: tuple[Any, ...]) -> dict[str, Any]:
    summary = json.loads(row[19]) if row[19] else None
    return {
        "schedule_id": row[0],
        "name": row[1],
        "universe_id": row[2],
        "symbols": json.loads(row[3]),
        "symbol_count": row[4],
        "local_time": row[5],
        "timezone": row[6],
        "weekdays": json.loads(row[7]),
        "max_symbols_per_run": row[8],
        "min_interval_seconds": row[9],
        "max_attempts": row[10],
        "stale_after_hours": row[11],
        "active": row[12],
        "cursor_index": row[13],
        "last_run_date": row[14].isoformat() if row[14] else None,
        "last_run_at": row[15].astimezone(UTC).isoformat() if row[15] else None,
        "last_status": row[16],
        "last_error": row[17],
        "last_run_id": row[18],
        "last_summary": summary,
        "created_at": row[20].astimezone(UTC).isoformat(),
        "updated_at": row[21].astimezone(UTC).isoformat(),
    }


def save_enrichment_schedule(
    name: str,
    symbols: list[str],
    config: dict[str, Any] | None = None,
    *,
    local_time: str = "16:45",
    timezone: str = "America/New_York",
    weekdays: list[int] | None = None,
    max_symbols_per_run: int = 10,
    min_interval_seconds: float = 2.0,
    max_attempts: int = 2,
    stale_after_hours: int = 36,
    universe_id: str | None = None,
    schedule_id: str | None = None,
    active: bool = True,
) -> dict[str, Any]:
    """Create or update one rotating enrichment schedule."""
    cleaned = list(dict.fromkeys(normalize_symbol(symbol) for symbol in symbols))
    if not name.strip():
        raise ValueError("Schedule name cannot be empty")
    if not cleaned or len(cleaned) > 500:
        raise ValueError("An enrichment schedule needs between 1 and 500 symbols")
    if not _TIME_PATTERN.fullmatch(local_time):
        raise ValueError("Schedule time must use HH:MM")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Unknown schedule timezone") from exc
    run_days = weekdays if weekdays is not None else [0, 1, 2, 3, 4]
    if not run_days or any(day not in range(7) for day in run_days):
        raise ValueError("Weekdays must contain values from 0 to 6")
    if not 1 <= int(max_symbols_per_run) <= 100:
        raise ValueError("Symbols per run must be between 1 and 100")
    if not 0 <= float(min_interval_seconds) <= 60:
        raise ValueError("Pacing interval must be between 0 and 60 seconds")
    if not 1 <= int(max_attempts) <= 4:
        raise ValueError("Retry attempts must be between 1 and 4")
    if not 1 <= int(stale_after_hours) <= 720:
        raise ValueError("Freshness threshold must be between 1 and 720 hours")
    connection = _connect(config)
    if connection is None:
        raise RuntimeError("Enrichment scheduler requires the local data warehouse")
    now = datetime.now(UTC)
    resolved_id = schedule_id or hashlib.sha256(
        f"enrichment|{name}|{now.isoformat()}".encode()
    ).hexdigest()[:16]
    try:
        connection.execute(
            """
            INSERT INTO enrichment_schedules VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0,
                NULL, NULL, NULL, NULL, NULL, NULL, ?, ?
            )
            ON CONFLICT (schedule_id) DO UPDATE SET
                name = excluded.name, universe_id = excluded.universe_id,
                symbols_json = excluded.symbols_json, symbol_count = excluded.symbol_count,
                local_time = excluded.local_time, timezone = excluded.timezone,
                weekdays_json = excluded.weekdays_json,
                max_symbols_per_run = excluded.max_symbols_per_run,
                min_interval_seconds = excluded.min_interval_seconds,
                max_attempts = excluded.max_attempts,
                stale_after_hours = excluded.stale_after_hours,
                active = excluded.active, updated_at = excluded.updated_at
            """,
            [
                resolved_id,
                name.strip()[:100],
                universe_id,
                _safe_json(cleaned),
                len(cleaned),
                local_time,
                timezone,
                _safe_json(sorted(set(run_days))),
                int(max_symbols_per_run),
                float(min_interval_seconds),
                int(max_attempts),
                int(stale_after_hours),
                bool(active),
                now,
                now,
            ],
        )
        row = connection.execute(
            "SELECT * FROM enrichment_schedules WHERE schedule_id = ?", [resolved_id]
        ).fetchone()
        return _schedule_row(row)
    finally:
        connection.close()


def list_enrichment_schedules(
    config: dict[str, Any] | None = None, *, active_only: bool = False
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    where = "WHERE active" if active_only else ""
    try:
        rows = connection.execute(
            f"SELECT * FROM enrichment_schedules {where} ORDER BY updated_at DESC"
        ).fetchall()
    finally:
        connection.close()
    return [_schedule_row(row) for row in rows]


def get_enrichment_schedule(
    schedule_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    connection = _connect(config)
    if connection is None:
        return None
    try:
        row = connection.execute(
            "SELECT * FROM enrichment_schedules WHERE schedule_id = ?", [schedule_id]
        ).fetchone()
    finally:
        connection.close()
    return _schedule_row(row) if row else None


def set_enrichment_schedule_active(
    schedule_id: str, active: bool, config: dict[str, Any] | None = None
) -> bool:
    connection = _connect(config)
    if connection is None:
        return False
    try:
        row = connection.execute(
            """UPDATE enrichment_schedules SET active = ?, updated_at = ?
               WHERE schedule_id = ? RETURNING schedule_id""",
            [active, datetime.now(UTC), schedule_id],
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def delete_enrichment_schedule(
    schedule_id: str, config: dict[str, Any] | None = None
) -> bool:
    connection = _connect(config)
    if connection is None:
        return False
    try:
        row = connection.execute(
            "DELETE FROM enrichment_schedules WHERE schedule_id = ? RETURNING schedule_id",
            [schedule_id],
        ).fetchone()
        return row is not None
    finally:
        connection.close()


def enrichment_schedule_is_due(
    schedule: dict[str, Any], now: datetime | None = None
) -> bool:
    if not schedule.get("active"):
        return False
    try:
        zone = ZoneInfo(schedule["timezone"])
        local_now = (now or datetime.now(zone)).astimezone(zone)
        hour, minute = (int(value) for value in schedule["local_time"].split(":"))
    except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
        return False
    if local_now.weekday() not in schedule.get("weekdays", []):
        return False
    retry_at = (schedule.get("last_summary") or {}).get("next_retry_at")
    if retry_at:
        try:
            retry_time = datetime.fromisoformat(str(retry_at).replace("Z", "+00:00"))
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=UTC)
            if local_now.astimezone(UTC) >= retry_time.astimezone(UTC):
                return True
        except (TypeError, ValueError):
            pass
    if (local_now.hour, local_now.minute) < (hour, minute):
        return False
    return schedule.get("last_run_date") != local_now.date().isoformat()


def _select_batch(schedule: dict[str, Any]) -> tuple[list[str], int]:
    symbols = schedule["symbols"]
    if not symbols:
        return [], 0
    start = int(schedule.get("cursor_index") or 0) % len(symbols)
    size = min(int(schedule["max_symbols_per_run"]), len(symbols))
    batch = [symbols[(start + offset) % len(symbols)] for offset in range(size)]
    return batch, (start + size) % len(symbols)


def _update_source_health(connection, source: dict[str, Any], checked_at: datetime) -> int:
    dataset = str(source.get("dataset") or "unknown")
    provider = str(source.get("provider") or "Unknown")[:100]
    status = str(source.get("status") or "unavailable")
    confidence = float(source.get("confidence_score") or 0)
    current = connection.execute(
        """SELECT total_checks, successful_checks, partial_checks, failed_checks,
                  consecutive_failures, average_confidence
           FROM enrichment_source_health WHERE dataset = ?""",
        [dataset],
    ).fetchone()
    total, successful, partial, failed, consecutive, average = current or (0, 0, 0, 0, 0, 0)
    total += 1
    successful += int(status == "available")
    partial += int(status == "partial")
    failed += int(status == "unavailable")
    consecutive = consecutive + 1 if status == "unavailable" else 0
    average = ((float(average) * (total - 1)) + confidence) / total
    last_success = checked_at if status in {"available", "partial"} else None
    error = str(source.get("error") or "")[:500] or None
    connection.execute(
        """
        INSERT INTO enrichment_source_health VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (dataset) DO UPDATE SET
            provider = excluded.provider, total_checks = excluded.total_checks,
            successful_checks = excluded.successful_checks,
            partial_checks = excluded.partial_checks, failed_checks = excluded.failed_checks,
            consecutive_failures = excluded.consecutive_failures,
            average_confidence = excluded.average_confidence,
            last_status = excluded.last_status, last_checked_at = excluded.last_checked_at,
            last_success_at = coalesce(excluded.last_success_at, enrichment_source_health.last_success_at),
            last_error = excluded.last_error
        """,
        [
            dataset, provider, total, successful, partial, failed, consecutive,
            round(average, 1), status, checked_at, last_success, error,
        ],
    )
    return consecutive


def _circuit_snapshot(connection, checked_at: datetime) -> dict[str, str | None]:
    """Return open circuits and move expired ones to a single half-open probe."""
    rows = connection.execute(
        """SELECT dataset, state, retry_at FROM enrichment_source_circuits
           WHERE state IN ('open', 'half_open')"""
    ).fetchall()
    disabled: dict[str, str | None] = {}
    for dataset, state, retry_at in rows:
        if state == "open" and retry_at and checked_at < retry_at.astimezone(UTC):
            disabled[str(dataset)] = retry_at.astimezone(UTC).isoformat()
            continue
        if state == "open":
            connection.execute(
                """UPDATE enrichment_source_circuits
                   SET state = 'half_open', last_probe_at = ?, updated_at = ?
                   WHERE dataset = ?""",
                [checked_at, checked_at, dataset],
            )
    return disabled


def _update_source_circuit(
    connection,
    *,
    dataset: str,
    provider: str,
    status: str,
    consecutive_failures: int,
    checked_at: datetime,
) -> dict[str, Any]:
    current = connection.execute(
        """SELECT state, open_count, retry_at FROM enrichment_source_circuits
           WHERE dataset = ?""",
        [dataset],
    ).fetchone()
    state, open_count, previous_retry = current or ("closed", 0, None)
    if status in {"available", "partial"}:
        connection.execute(
            """INSERT INTO enrichment_source_circuits VALUES (
                   ?, ?, 'closed', NULL, NULL, ?, NULL, ?
               ) ON CONFLICT (dataset) DO UPDATE SET
                   provider = excluded.provider, state = 'closed', opened_at = NULL,
                   retry_at = NULL, last_probe_at = NULL, updated_at = excluded.updated_at""",
            [dataset, provider, int(open_count), checked_at],
        )
        return {"state": "closed", "retry_at": None, "open_count": int(open_count)}

    should_open = (
        consecutive_failures >= _CIRCUIT_FAILURE_THRESHOLD or state == "half_open"
    )
    if not should_open:
        return {
            "state": state,
            "retry_at": previous_retry.astimezone(UTC).isoformat()
            if previous_retry
            else None,
            "open_count": int(open_count),
        }
    next_open_count = int(open_count) + 1 if state != "open" else int(open_count)
    cooldown = min(
        _CIRCUIT_BASE_COOLDOWN_SECONDS * (2 ** max(next_open_count - 1, 0)),
        _CIRCUIT_MAX_COOLDOWN_SECONDS,
    )
    retry_at = checked_at + timedelta(seconds=cooldown)
    connection.execute(
        """INSERT INTO enrichment_source_circuits VALUES (
               ?, ?, 'open', ?, ?, ?, NULL, ?
           ) ON CONFLICT (dataset) DO UPDATE SET
               provider = excluded.provider, state = 'open',
               opened_at = excluded.opened_at, retry_at = excluded.retry_at,
               open_count = excluded.open_count, updated_at = excluded.updated_at""",
        [dataset, provider, checked_at, retry_at, next_open_count, checked_at],
    )
    return {
        "state": "open",
        "retry_at": retry_at.isoformat(),
        "open_count": next_open_count,
    }


def _create_alert(
    connection,
    *,
    schedule_id: str,
    run_id: str | None,
    as_of_date: str,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    symbol: str | None = None,
    dataset: str | None = None,
    affected_symbols: list[str] | None = None,
    occurrence_count: int = 1,
    next_retry_at: str | None = None,
) -> dict[str, Any]:
    identity = f"{schedule_id}|{as_of_date}|{alert_type}|{symbol or ''}|{dataset or ''}"
    alert_id = hashlib.sha256(identity.encode()).hexdigest()
    created_at = datetime.now(UTC)
    symbols = list(dict.fromkeys(affected_symbols or ([symbol] if symbol else [])))
    retry_at = (
        datetime.fromisoformat(next_retry_at.replace("Z", "+00:00"))
        if next_retry_at
        else None
    )
    connection.execute(
        """INSERT INTO enrichment_scheduler_alerts (
               alert_id, schedule_id, run_id, symbol, dataset, alert_type,
               severity, title, message, created_at, acknowledged_at,
               affected_symbols_json, occurrence_count, next_retry_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
           ON CONFLICT (alert_id) DO UPDATE SET
               run_id = excluded.run_id, severity = excluded.severity,
               title = excluded.title, message = excluded.message,
               created_at = excluded.created_at, acknowledged_at = NULL,
               affected_symbols_json = excluded.affected_symbols_json,
               occurrence_count = excluded.occurrence_count,
               next_retry_at = excluded.next_retry_at""",
        [
            alert_id,
            schedule_id,
            run_id,
            symbol,
            dataset,
            alert_type,
            severity,
            title[:200],
            message[:1000],
            created_at,
            _safe_json(symbols),
            max(1, int(occurrence_count)),
            retry_at,
        ],
    )
    return {
        "alert_id": alert_id,
        "schedule_id": schedule_id,
        "run_id": run_id,
        "symbol": symbol,
        "dataset": dataset,
        "alert_type": alert_type,
        "severity": severity,
        "title": title,
        "message": message,
        "created_at": created_at.isoformat(),
        "acknowledged_at": None,
        "affected_symbols": symbols,
        "affected_symbol_count": len(symbols),
        "occurrence_count": max(1, int(occurrence_count)),
        "next_retry_at": retry_at.isoformat() if retry_at else None,
    }


def _resolve_alert_group(
    connection,
    *,
    schedule_id: str,
    alert_type: str,
    dataset: str | None = None,
    resolved_at: datetime | None = None,
) -> None:
    dataset_clause = "AND dataset = ?" if dataset else "AND dataset IS NULL"
    parameters: list[Any] = [resolved_at or datetime.now(UTC), schedule_id, alert_type]
    if dataset:
        parameters.append(dataset)
    connection.execute(
        f"""UPDATE enrichment_scheduler_alerts SET acknowledged_at = ?
            WHERE acknowledged_at IS NULL AND schedule_id = ? AND alert_type = ?
            {dataset_clause}""",
        parameters,
    )


def _stale_alerts(
    connection,
    schedule: dict[str, Any],
    run_id: str,
    as_of_date: str,
    now: datetime,
) -> list[dict[str, Any]]:
    stale_symbols: list[str] = []
    max_age_hours = 0.0
    threshold = float(schedule["stale_after_hours"])
    for symbol in schedule["symbols"]:
        try:
            row = connection.execute(
                """SELECT observed_at FROM enrichment_snapshots
                   WHERE symbol = ? ORDER BY observed_at DESC LIMIT 1""",
                [symbol],
            ).fetchone()
        except Exception:
            row = None
        age_hours = None
        if row:
            observed = row[0]
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            age_hours = max(0.0, (now - observed.astimezone(UTC)).total_seconds() / 3600)
        if age_hours is None or age_hours > threshold:
            stale_symbols.append(symbol)
            max_age_hours = max(max_age_hours, age_hours or threshold * 2 + 1)
    _resolve_alert_group(
        connection,
        schedule_id=schedule["schedule_id"],
        alert_type="stale_symbol",
        resolved_at=now,
    )
    if not stale_symbols:
        return []
    severity = "high" if max_age_hours > threshold * 2 else "medium"
    return [
        _create_alert(
            connection,
            schedule_id=schedule["schedule_id"],
            run_id=run_id,
            as_of_date=as_of_date,
            alert_type="stale_symbol",
            severity=severity,
            title=f"{len(stale_symbols)} enrichment snapshots are stale",
            message=(
                f"{len(stale_symbols)} symbols exceed the {int(threshold)} hour freshness "
                "target; review or run the controlled schedule."
            ),
            affected_symbols=stale_symbols,
            occurrence_count=len(stale_symbols),
        )
    ]


def execute_enrichment_schedule(
    schedule: dict[str, Any],
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    collector: Callable[..., dict[str, Any]] = build_enrichment_snapshot,
    capture: Callable[..., dict[str, Any]] = capture_enrichment_snapshot,
    sleep_fn: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one bounded batch and persist snapshots, health, alerts and audit state."""
    datetime.strptime(as_of_date, "%Y-%m-%d")
    connection = _connect(config)
    if connection is None:
        raise RuntimeError("Enrichment scheduler requires the local data warehouse")
    started = now or datetime.now(UTC)
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    batch, next_cursor = _select_batch(schedule)
    run_id = hashlib.sha256(
        f"{schedule['schedule_id']}|{as_of_date}|{started.isoformat()}".encode()
    ).hexdigest()
    connection.execute(
        """INSERT INTO enrichment_schedule_runs VALUES (
               ?, ?, ?, ?, 'running', ?, 0, 0, 0, 0, 0, 0, '{}', ?, NULL
           )""",
        [run_id, schedule["schedule_id"], SCHEDULER_VERSION, as_of_date, len(batch), started],
    )
    connection.close()

    results: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    failed_symbols: list[str] = []
    low_coverage_symbols: list[str] = []
    source_issue_groups: dict[str, dict[str, Any]] = {}
    healthy_datasets: set[str] = set()
    source_successes = 0
    source_failures = 0
    for index, symbol in enumerate(batch):
        attempts = 0
        snapshot = None
        error_type = None
        while attempts < int(schedule["max_attempts"]):
            attempts += 1
            try:
                connection = _connect(config)
                try:
                    disabled_datasets = _circuit_snapshot(
                        connection, datetime.now(UTC)
                    )
                finally:
                    connection.close()
                collector_config = dict(config or {})
                collector_config["enrichment_disabled_datasets"] = disabled_datasets
                snapshot = collector(symbol, as_of_date, collector_config)
                snapshot["registry"] = capture(snapshot, collector_config)
                break
            except Exception as exc:  # bounded retry preserves the rest of the batch
                error_type = type(exc).__name__
                if attempts < int(schedule["max_attempts"]):
                    sleep_fn(min(2 ** (attempts - 1), 8))
        connection = _connect(config)
        try:
            checked_at = datetime.now(UTC)
            if snapshot is None:
                failed_symbols.append(symbol)
                results.append(
                    {"symbol": symbol, "status": "failed", "attempts": attempts, "error_type": error_type}
                )
            else:
                unavailable = []
                for source in snapshot.get("sources", []):
                    dataset = str(source.get("dataset") or "unknown")
                    provider = str(source.get("provider") or "Unknown")
                    if source.get("circuit_open"):
                        group = source_issue_groups.setdefault(
                            dataset,
                            {
                                "provider": provider,
                                "symbols": [],
                                "max_consecutive": 0,
                                "circuit_state": "open",
                                "retry_at": source.get("retry_at"),
                                "skipped_count": 0,
                            },
                        )
                        group["symbols"].append(symbol)
                        group["skipped_count"] += 1
                        unavailable.append(dataset)
                        continue
                    consecutive = _update_source_health(connection, source, checked_at)
                    circuit = _update_source_circuit(
                        connection,
                        dataset=dataset,
                        provider=provider,
                        status=str(source.get("status") or "unavailable"),
                        consecutive_failures=consecutive,
                        checked_at=checked_at,
                    )
                    if source.get("status") == "unavailable":
                        source_failures += 1
                        unavailable.append(dataset)
                        group = source_issue_groups.setdefault(
                            dataset,
                            {
                                "provider": provider,
                                "symbols": [],
                                "max_consecutive": 0,
                                "circuit_state": circuit["state"],
                                "retry_at": circuit.get("retry_at"),
                                "skipped_count": 0,
                            },
                        )
                        group["symbols"].append(symbol)
                        group["max_consecutive"] = max(
                            int(group["max_consecutive"]), consecutive
                        )
                        if circuit["state"] == "open":
                            group["circuit_state"] = "open"
                            group["retry_at"] = circuit.get("retry_at")
                    else:
                        source_successes += 1
                        healthy_datasets.add(dataset)
                coverage = float((snapshot.get("coverage") or {}).get("score") or 0)
                if coverage < 58:
                    low_coverage_symbols.append(symbol)
                results.append(
                    {
                        "symbol": symbol,
                        "status": "completed_with_warnings" if unavailable else "completed",
                        "attempts": attempts,
                        "coverage_score": coverage,
                        "coverage_label": (snapshot.get("coverage") or {}).get("label"),
                        "unavailable_sources": unavailable,
                        "snapshot_id": (snapshot.get("registry") or {}).get("snapshot_id"),
                    }
                )
        finally:
            connection.close()
        if index + 1 < len(batch) and float(schedule["min_interval_seconds"]) > 0:
            sleep_fn(float(schedule["min_interval_seconds"]))

    completed = sum(item["status"] != "failed" for item in results)
    failed = len(results) - completed
    status = "failed" if results and completed == 0 else "completed_with_warnings" if failed or alerts else "completed"
    finished = datetime.now(UTC)
    connection = _connect(config)
    try:
        _resolve_alert_group(
            connection,
            schedule_id=schedule["schedule_id"],
            alert_type="refresh_failed",
            resolved_at=finished,
        )
        if failed_symbols:
            alerts.append(
                _create_alert(
                    connection,
                    schedule_id=schedule["schedule_id"],
                    run_id=run_id,
                    as_of_date=as_of_date,
                    alert_type="refresh_failed",
                    severity="high",
                    title=f"{len(failed_symbols)} enrichment refreshes failed",
                    message="All bounded attempts failed for the affected symbols.",
                    affected_symbols=failed_symbols,
                    occurrence_count=len(failed_symbols),
                )
            )
        for dataset in healthy_datasets - set(source_issue_groups):
            _resolve_alert_group(
                connection,
                schedule_id=schedule["schedule_id"],
                alert_type="source_unavailable",
                dataset=dataset,
                resolved_at=finished,
            )
        for dataset, group in source_issue_groups.items():
            _resolve_alert_group(
                connection,
                schedule_id=schedule["schedule_id"],
                alert_type="source_unavailable",
                dataset=dataset,
                resolved_at=finished,
            )
            affected = list(dict.fromkeys(group["symbols"]))
            circuit_open = group["circuit_state"] == "open"
            label = dataset.replace("_", " ").title()
            retry_note = (
                f" Automatic probe scheduled for {group['retry_at']}."
                if group.get("retry_at")
                else ""
            )
            alerts.append(
                _create_alert(
                    connection,
                    schedule_id=schedule["schedule_id"],
                    run_id=run_id,
                    as_of_date=as_of_date,
                    alert_type="source_unavailable",
                    severity=(
                        "high"
                        if circuit_open or int(group["max_consecutive"]) >= 2
                        else "medium"
                    ),
                    title=(
                        f"{label} paused after repeated failures"
                        if circuit_open
                        else f"{label} unavailable for {len(affected)} symbol"
                        f"{'s' if len(affected) != 1 else ''}"
                    ),
                    message=(
                        f"{len(affected)} symbols were affected by {group['provider']}; "
                        f"{int(group['skipped_count'])} redundant calls were skipped."
                        f"{retry_note}"
                    ),
                    dataset=dataset,
                    affected_symbols=affected,
                    occurrence_count=len(affected),
                    next_retry_at=group.get("retry_at"),
                )
            )
        _resolve_alert_group(
            connection,
            schedule_id=schedule["schedule_id"],
            alert_type="low_coverage",
            resolved_at=finished,
        )
        if low_coverage_symbols:
            affected = list(dict.fromkeys(low_coverage_symbols))
            alerts.append(
                _create_alert(
                    connection,
                    schedule_id=schedule["schedule_id"],
                    run_id=run_id,
                    as_of_date=as_of_date,
                    alert_type="low_coverage",
                    severity="medium",
                    title=f"{len(affected)} enrichment snapshots have limited coverage",
                    message="Review missing sources before qualification.",
                    affected_symbols=affected,
                    occurrence_count=len(affected),
                )
            )
        alerts.extend(_stale_alerts(connection, schedule, run_id, as_of_date, finished))
        if status == "completed" and alerts:
            status = "completed_with_warnings"
        open_circuits = connection.execute(
            """SELECT dataset, retry_at FROM enrichment_source_circuits
               WHERE state = 'open' ORDER BY retry_at"""
        ).fetchall()
        next_retry_at = (
            min(row[1].astimezone(UTC) for row in open_circuits if row[1]).isoformat()
            if any(row[1] for row in open_circuits)
            else None
        )
        summary = {
            "model_version": SCHEDULER_VERSION,
            "run_id": run_id,
            "schedule_id": schedule["schedule_id"],
            "as_of_date": as_of_date,
            "status": status,
            "requested_count": len(batch),
            "attempted_count": len(results),
            "completed_count": completed,
            "failed_count": failed,
            "source_success_count": source_successes,
            "source_failure_count": source_failures,
            "source_circuit_skip_count": sum(
                int(group["skipped_count"]) for group in source_issue_groups.values()
            ),
            "open_circuit_count": len(open_circuits),
            "next_retry_at": next_retry_at,
            "alert_count": len({item["alert_id"] for item in alerts}),
            "next_cursor": next_cursor,
            "results": results,
            "duration_seconds": round((finished - started).total_seconds(), 2),
        }
        error = "Every symbol failed after bounded retries." if status == "failed" else None
        connection.execute(
            """UPDATE enrichment_schedule_runs SET status = ?, attempted_count = ?,
                      completed_count = ?, failed_count = ?, source_success_count = ?,
                      source_failure_count = ?, duration_seconds = ?, result_json = ?,
                      completed_at = ? WHERE run_id = ?""",
            [
                status, len(results), completed, failed, source_successes, source_failures,
                summary["duration_seconds"], _safe_json(summary), finished, run_id,
            ],
        )
        connection.execute(
            """UPDATE enrichment_schedules SET cursor_index = ?, last_run_date = ?,
                      last_run_at = ?, last_status = ?, last_error = ?, last_run_id = ?,
                      last_summary_json = ?, updated_at = ? WHERE schedule_id = ?""",
            [
                next_cursor, None if status == "failed" else as_of_date, finished, status, error, run_id,
                _safe_json(summary), finished, schedule["schedule_id"],
            ],
        )
        return summary
    finally:
        connection.close()


def list_enrichment_runs(
    config: dict[str, Any] | None = None, *, schedule_id: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    where = "WHERE schedule_id = ?" if schedule_id else ""
    params: list[Any] = [schedule_id] if schedule_id else []
    params.append(max(1, min(int(limit), 100)))
    try:
        rows = connection.execute(
            f"""SELECT run_id, schedule_id, as_of_date, status, requested_count,
                       completed_count, failed_count, source_success_count,
                       source_failure_count, duration_seconds, started_at, completed_at
                FROM enrichment_schedule_runs {where} ORDER BY started_at DESC LIMIT ?""",
            params,
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "run_id": row[0], "schedule_id": row[1], "as_of_date": row[2].isoformat(),
            "status": row[3], "requested_count": row[4], "completed_count": row[5],
            "failed_count": row[6], "source_success_count": row[7],
            "source_failure_count": row[8], "duration_seconds": row[9],
            "started_at": row[10].astimezone(UTC).isoformat(),
            "completed_at": row[11].astimezone(UTC).isoformat() if row[11] else None,
        }
        for row in rows
    ]


def list_enrichment_source_health(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    try:
        rows = connection.execute(
            """SELECT h.*, c.state, c.retry_at, c.open_count
               FROM enrichment_source_health h
               LEFT JOIN enrichment_source_circuits c USING (dataset)
               ORDER BY h.dataset"""
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "dataset": row[0], "provider": row[1], "total_checks": row[2],
            "successful_checks": row[3], "partial_checks": row[4], "failed_checks": row[5],
            "consecutive_failures": row[6], "average_confidence": row[7],
            "last_status": row[8], "last_checked_at": row[9].astimezone(UTC).isoformat(),
            "last_success_at": row[10].astimezone(UTC).isoformat() if row[10] else None,
            "last_error": row[11],
            "availability_rate": round(100 * (row[3] + row[4]) / row[2], 1) if row[2] else 0,
            "circuit_state": row[12] or "closed",
            "retry_at": row[13].astimezone(UTC).isoformat() if row[13] else None,
            "circuit_open_count": int(row[14] or 0),
        }
        for row in rows
    ]


def _group_alert_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in items:
        key = (
            str(item["schedule_id"]),
            str(item["alert_type"]),
            str(item.get("dataset") or "")
            if item["alert_type"] == "source_unavailable"
            else "",
        )
        grouped.setdefault(key, []).append(item)
    result = []
    for key, members in grouped.items():
        members.sort(key=lambda item: item["created_at"], reverse=True)
        latest = dict(members[0])
        symbols = list(
            dict.fromkeys(
                symbol
                for item in members
                for symbol in (
                    item.get("affected_symbols")
                    or ([item["symbol"]] if item.get("symbol") else [])
                )
            )
        )
        occurrences = sum(int(item.get("occurrence_count") or 1) for item in members)
        latest["alert_id"] = hashlib.sha256(
            f"enrichment-alert-group|{'|'.join(key)}".encode()
        ).hexdigest()
        latest["member_alert_ids"] = [item["alert_id"] for item in members]
        latest["grouped_alert_count"] = len(members)
        latest["occurrence_count"] = occurrences
        latest["affected_symbols"] = symbols
        latest["affected_symbol_count"] = len(symbols)
        if len(symbols) != 1:
            latest["symbol"] = None
        latest["next_retry_at"] = min(
            (
                item["next_retry_at"]
                for item in members
                if item.get("next_retry_at")
            ),
            default=None,
        )
        if latest["alert_type"] == "source_unavailable" and symbols:
            count = len(symbols)
            label = str(latest.get("dataset") or "Source").replace("_", " ").title()
            latest["title"] = (
                f"{label} unavailable across {count} symbols"
                if count != 1
                else f"{label} unavailable for 1 symbol"
            )
            latest["message"] = (
                f"{count} symbol{'s are' if count != 1 else ' is'} affected; "
                f"{occurrences} source event{'s were' if occurrences != 1 else ' was'} "
                "consolidated into this alert."
            )
        elif latest["alert_type"] == "stale_symbol" and symbols:
            count = len(symbols)
            latest["title"] = (
                f"{count} enrichment snapshot{'s are' if count != 1 else ' is'} stale"
            )
            latest["message"] = (
                f"{count} symbol{'s exceed' if count != 1 else ' exceeds'} the configured "
                f"freshness target; {occurrences} event{'s were' if occurrences != 1 else ' was'} "
                "consolidated."
            )
        elif latest["alert_type"] == "low_coverage" and symbols:
            count = len(symbols)
            latest["title"] = (
                f"{count} snapshot{'s' if count != 1 else ''} "
                f"{'have' if count != 1 else 'has'} limited coverage"
            )
            latest["message"] = (
                f"{count} symbol{'s' if count != 1 else ''} "
                f"{'need' if count != 1 else 'needs'} evidence review before qualification."
            )
        elif latest["alert_type"] == "refresh_failed" and symbols:
            count = len(symbols)
            latest["title"] = (
                f"{count} enrichment refresh{'es' if count != 1 else ''} failed"
            )
            latest["message"] = (
                f"All bounded attempts failed for {count} symbol{'s' if count != 1 else ''}."
            )
        result.append(latest)
    return sorted(result, key=lambda item: item["created_at"], reverse=True)


def list_enrichment_alerts(
    config: dict[str, Any] | None = None, *, unacknowledged_only: bool = False, limit: int = 100
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    where = "WHERE acknowledged_at IS NULL" if unacknowledged_only else ""
    try:
        rows = connection.execute(
            f"""SELECT alert_id, schedule_id, run_id, symbol, dataset, alert_type,
                       severity, title, message, created_at, acknowledged_at,
                       coalesce(affected_symbols_json, '[]'),
                       coalesce(occurrence_count, 1), next_retry_at
                FROM enrichment_scheduler_alerts {where}
                ORDER BY created_at DESC LIMIT ?""",
            [500],
        ).fetchall()
    finally:
        connection.close()
    raw = [
        {
            "alert_id": row[0], "schedule_id": row[1], "run_id": row[2],
            "symbol": row[3], "dataset": row[4], "alert_type": row[5],
            "severity": row[6], "title": row[7], "message": row[8],
            "created_at": row[9].astimezone(UTC).isoformat(),
            "acknowledged_at": row[10].astimezone(UTC).isoformat() if row[10] else None,
            "affected_symbols": json.loads(row[11]) if row[11] else [],
            "occurrence_count": int(row[12] or 1),
            "next_retry_at": row[13].astimezone(UTC).isoformat() if row[13] else None,
        }
        for row in rows
    ]
    return _group_alert_items(raw)[: max(1, min(int(limit), 500))]


def acknowledge_enrichment_alert(
    alert_id: str, config: dict[str, Any] | None = None
) -> bool:
    connection = _connect(config)
    if connection is None:
        return False
    try:
        row = connection.execute(
            """UPDATE enrichment_scheduler_alerts SET acknowledged_at = ?
               WHERE alert_id = ? AND acknowledged_at IS NULL RETURNING alert_id""",
            [datetime.now(UTC), alert_id],
        ).fetchone()
        if row is not None:
            return True
        rows = connection.execute(
            """SELECT alert_id, schedule_id, run_id, symbol, dataset, alert_type,
                      severity, title, message, created_at, acknowledged_at,
                      coalesce(affected_symbols_json, '[]'),
                      coalesce(occurrence_count, 1), next_retry_at
               FROM enrichment_scheduler_alerts WHERE acknowledged_at IS NULL"""
        ).fetchall()
        raw = [
            {
                "alert_id": item[0],
                "schedule_id": item[1],
                "run_id": item[2],
                "symbol": item[3],
                "dataset": item[4],
                "alert_type": item[5],
                "severity": item[6],
                "title": item[7],
                "message": item[8],
                "created_at": item[9].astimezone(UTC).isoformat(),
                "acknowledged_at": None,
                "affected_symbols": json.loads(item[11]) if item[11] else [],
                "occurrence_count": int(item[12] or 1),
                "next_retry_at": item[13].astimezone(UTC).isoformat()
                if item[13]
                else None,
            }
            for item in rows
        ]
        group = next(
            (item for item in _group_alert_items(raw) if item["alert_id"] == alert_id),
            None,
        )
        if not group:
            return False
        member_ids = group["member_alert_ids"]
        placeholders = ", ".join("?" for _ in member_ids)
        connection.execute(
            f"""UPDATE enrichment_scheduler_alerts SET acknowledged_at = ?
                WHERE alert_id IN ({placeholders}) AND acknowledged_at IS NULL""",
            [datetime.now(UTC), *member_ids],
        )
        return True
    finally:
        connection.close()


def get_enrichment_operations_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    schedules = list_enrichment_schedules(config)
    health = list_enrichment_source_health(config)
    alerts = list_enrichment_alerts(config, unacknowledged_only=True)
    next_retries = [item["retry_at"] for item in health if item.get("retry_at")]
    return {
        "schedules": schedules,
        "schedule_count": len(schedules),
        "active_schedule_count": sum(bool(item["active"]) for item in schedules),
        "source_health": health,
        "healthy_source_count": sum(item["last_status"] != "unavailable" for item in health),
        "degraded_source_count": sum(item["last_status"] == "unavailable" for item in health),
        "open_circuit_count": sum(item["circuit_state"] == "open" for item in health),
        "next_retry_at": min(next_retries) if next_retries else None,
        "alerts": alerts,
        "unacknowledged_alert_count": len(alerts),
        "affected_symbol_count": len(
            {
                symbol
                for item in alerts
                for symbol in item.get("affected_symbols", [])
            }
        ),
        "stale_symbol_count": len(
            {
                symbol
                for item in alerts
                if item["alert_type"] == "stale_symbol"
                for symbol in item.get("affected_symbols", [])
            }
        ),
        "recent_runs": list_enrichment_runs(config, limit=10),
    }
