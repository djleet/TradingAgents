"""Durable, rate-limited scheduler for the Data Enrichment Hub."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
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

SCHEDULER_VERSION = "enrichment-scheduler-v1"
_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


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
            acknowledged_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_enrichment_alerts_status
           ON enrichment_scheduler_alerts(acknowledged_at, created_at)"""
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
) -> dict[str, Any]:
    identity = f"{schedule_id}|{as_of_date}|{alert_type}|{symbol or ''}|{dataset or ''}"
    alert_id = hashlib.sha256(identity.encode()).hexdigest()
    created_at = datetime.now(UTC)
    connection.execute(
        """INSERT OR IGNORE INTO enrichment_scheduler_alerts
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
        [
            alert_id, schedule_id, run_id, symbol, dataset, alert_type, severity,
            title[:200], message[:1000], created_at,
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
    }


def _stale_alerts(
    connection,
    schedule: dict[str, Any],
    run_id: str,
    as_of_date: str,
    now: datetime,
) -> list[dict[str, Any]]:
    alerts = []
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
            age_label = "no saved snapshot" if age_hours is None else f"{age_hours:.0f} hours old"
            alerts.append(
                _create_alert(
                    connection,
                    schedule_id=schedule["schedule_id"],
                    run_id=run_id,
                    as_of_date=as_of_date,
                    alert_type="stale_symbol",
                    severity="high" if age_hours is None or age_hours > threshold * 2 else "medium",
                    title=f"{symbol} enrichment is stale",
                    message=f"Latest enrichment is {age_label}; freshness target is {int(threshold)} hours.",
                    symbol=symbol,
                )
            )
    return alerts


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
    source_successes = 0
    source_failures = 0
    for index, symbol in enumerate(batch):
        attempts = 0
        snapshot = None
        error_type = None
        while attempts < int(schedule["max_attempts"]):
            attempts += 1
            try:
                snapshot = collector(symbol, as_of_date, config)
                snapshot["registry"] = capture(snapshot, config)
                break
            except Exception as exc:  # bounded retry preserves the rest of the batch
                error_type = type(exc).__name__
                if attempts < int(schedule["max_attempts"]):
                    sleep_fn(min(2 ** (attempts - 1), 8))
        connection = _connect(config)
        try:
            checked_at = datetime.now(UTC)
            if snapshot is None:
                alerts.append(
                    _create_alert(
                        connection,
                        schedule_id=schedule["schedule_id"],
                        run_id=run_id,
                        as_of_date=as_of_date,
                        alert_type="refresh_failed",
                        severity="high",
                        title=f"{symbol} enrichment refresh failed",
                        message=f"All {attempts} bounded attempts failed ({error_type or 'provider error'}).",
                        symbol=symbol,
                    )
                )
                results.append(
                    {"symbol": symbol, "status": "failed", "attempts": attempts, "error_type": error_type}
                )
            else:
                unavailable = []
                for source in snapshot.get("sources", []):
                    consecutive = _update_source_health(connection, source, checked_at)
                    if source.get("status") == "unavailable":
                        source_failures += 1
                        unavailable.append(source.get("dataset"))
                        alerts.append(
                            _create_alert(
                                connection,
                                schedule_id=schedule["schedule_id"],
                                run_id=run_id,
                                as_of_date=as_of_date,
                                alert_type="source_unavailable",
                                severity="high" if consecutive >= 2 else "medium",
                                title=f"{source.get('dataset', 'Source').replace('_', ' ').title()} unavailable",
                                message=(
                                    f"{symbol} returned no usable data from {source.get('provider', 'the provider')}. "
                                    f"Consecutive source failures: {consecutive}."
                                ),
                                symbol=symbol,
                                dataset=source.get("dataset"),
                            )
                        )
                    else:
                        source_successes += 1
                coverage = float((snapshot.get("coverage") or {}).get("score") or 0)
                if coverage < 58:
                    alerts.append(
                        _create_alert(
                            connection,
                            schedule_id=schedule["schedule_id"],
                            run_id=run_id,
                            as_of_date=as_of_date,
                            alert_type="low_coverage",
                            severity="medium",
                            title=f"{symbol} enrichment coverage is limited",
                            message=f"Coverage is {coverage:.0f}/100; review missing sources before qualification.",
                            symbol=symbol,
                        )
                    )
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
        alerts.extend(_stale_alerts(connection, schedule, run_id, as_of_date, finished))
        if status == "completed" and alerts:
            status = "completed_with_warnings"
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
            "SELECT * FROM enrichment_source_health ORDER BY dataset"
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
        }
        for row in rows
    ]


def list_enrichment_alerts(
    config: dict[str, Any] | None = None, *, unacknowledged_only: bool = False, limit: int = 100
) -> list[dict[str, Any]]:
    connection = _connect(config)
    if connection is None:
        return []
    where = "WHERE acknowledged_at IS NULL" if unacknowledged_only else ""
    try:
        rows = connection.execute(
            f"""SELECT * FROM enrichment_scheduler_alerts {where}
                ORDER BY created_at DESC LIMIT ?""",
            [max(1, min(int(limit), 500))],
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "alert_id": row[0], "schedule_id": row[1], "run_id": row[2],
            "symbol": row[3], "dataset": row[4], "alert_type": row[5],
            "severity": row[6], "title": row[7], "message": row[8],
            "created_at": row[9].astimezone(UTC).isoformat(),
            "acknowledged_at": row[10].astimezone(UTC).isoformat() if row[10] else None,
        }
        for row in rows
    ]


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
        return row is not None
    finally:
        connection.close()


def get_enrichment_operations_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    schedules = list_enrichment_schedules(config)
    health = list_enrichment_source_health(config)
    alerts = list_enrichment_alerts(config, unacknowledged_only=True)
    return {
        "schedules": schedules,
        "schedule_count": len(schedules),
        "active_schedule_count": sum(bool(item["active"]) for item in schedules),
        "source_health": health,
        "healthy_source_count": sum(item["last_status"] != "unavailable" for item in health),
        "degraded_source_count": sum(item["last_status"] == "unavailable" for item in health),
        "alerts": alerts,
        "unacknowledged_alert_count": len(alerts),
        "stale_symbol_count": len({item["symbol"] for item in alerts if item["alert_type"] == "stale_symbol"}),
        "recent_runs": list_enrichment_runs(config, limit=10),
    }
