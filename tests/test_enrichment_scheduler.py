from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingagents.dataflows import enrichment_scheduler

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _snapshot(symbol: str, as_of_date: str, *, macro_available: bool = True):
    observed = datetime.now(UTC).isoformat()
    names = (
        "earnings_estimates",
        "options_implied_volatility",
        "institutional_ownership",
        "sec_insider_filings",
        "macro_calendar",
    )
    sources = []
    for name in names:
        available = name != "macro_calendar" or macro_available
        sources.append(
            {
                "dataset": name,
                "provider": "Test provider",
                "status": "available" if available else "unavailable",
                "confidence_score": 90 if available else 0,
                "point_in_time": name in {"sec_insider_filings", "macro_calendar"},
                "observed_at": observed,
                "error": None if available else "Temporarily unavailable",
            }
        )
    available_count = sum(item["status"] == "available" for item in sources)
    return {
        "model_version": "test-enrichment-v1",
        "symbol": symbol,
        "as_of_date": as_of_date,
        "observed_at": observed,
        "coverage": {
            "score": 82 if macro_available else 65,
            "label": "robust" if macro_available else "usable",
            "available_sources": available_count,
        },
        "sources": sources,
        "decision_flags": [],
    }


@pytest.mark.unit
def test_schedule_crud_and_due_policy(warehouse_config):
    schedule = enrichment_scheduler.save_enrichment_schedule(
        "Core refresh",
        ["aapl", "msft"],
        warehouse_config,
        local_time="16:45",
        max_symbols_per_run=1,
    )

    assert schedule["symbols"] == ["AAPL", "MSFT"]
    assert enrichment_scheduler.enrichment_schedule_is_due(
        schedule, datetime(2026, 10, 23, 22, 0, tzinfo=UTC)
    )
    assert not enrichment_scheduler.enrichment_schedule_is_due(
        schedule, datetime(2026, 10, 24, 22, 0, tzinfo=UTC)
    )
    assert enrichment_scheduler.set_enrichment_schedule_active(
        schedule["schedule_id"], False, warehouse_config
    )
    paused = enrichment_scheduler.get_enrichment_schedule(
        schedule["schedule_id"], warehouse_config
    )
    assert paused["active"] is False
    assert enrichment_scheduler.delete_enrichment_schedule(
        schedule["schedule_id"], warehouse_config
    )
    assert enrichment_scheduler.list_enrichment_schedules(warehouse_config) == []


@pytest.mark.unit
def test_scheduler_rotates_retries_and_records_source_health(warehouse_config):
    schedule = enrichment_scheduler.save_enrichment_schedule(
        "Rotating refresh",
        ["AAPL", "MSFT", "NVDA"],
        warehouse_config,
        max_symbols_per_run=2,
        min_interval_seconds=0,
        max_attempts=2,
        stale_after_hours=48,
    )
    attempts = {}

    def collector(symbol, as_of_date, _config):
        attempts[symbol] = attempts.get(symbol, 0) + 1
        if symbol == "AAPL" and attempts[symbol] == 1:
            raise TimeoutError("transient")
        return _snapshot(symbol, as_of_date, macro_available=symbol != "MSFT")

    report = enrichment_scheduler.execute_enrichment_schedule(
        schedule,
        "2026-10-23",
        warehouse_config,
        collector=collector,
        sleep_fn=lambda _seconds: None,
    )

    assert attempts == {"AAPL": 2, "MSFT": 1}
    assert report["completed_count"] == 2
    assert report["failed_count"] == 0
    assert report["next_cursor"] == 2
    assert report["status"] == "completed_with_warnings"
    health = {
        item["dataset"]: item
        for item in enrichment_scheduler.list_enrichment_source_health(warehouse_config)
    }
    assert health["earnings_estimates"]["availability_rate"] == 100
    assert health["macro_calendar"]["availability_rate"] == 50
    alerts = enrichment_scheduler.list_enrichment_alerts(
        warehouse_config, unacknowledged_only=True
    )
    assert {item["alert_type"] for item in alerts} >= {
        "source_unavailable",
        "stale_symbol",
    }
    stale = next(item for item in alerts if item["alert_type"] == "stale_symbol")
    assert stale["affected_symbols"] == ["NVDA"]
    assert enrichment_scheduler.acknowledge_enrichment_alert(
        alerts[0]["alert_id"], warehouse_config
    )


@pytest.mark.unit
def test_failed_cycle_remains_retryable(warehouse_config):
    schedule = enrichment_scheduler.save_enrichment_schedule(
        "Failing refresh",
        ["AAPL"],
        warehouse_config,
        min_interval_seconds=0,
        max_attempts=2,
    )

    def collector(*_args):
        raise RuntimeError("provider unavailable")

    report = enrichment_scheduler.execute_enrichment_schedule(
        schedule,
        "2026-10-23",
        warehouse_config,
        collector=collector,
        sleep_fn=lambda _seconds: None,
    )
    saved = enrichment_scheduler.get_enrichment_schedule(
        schedule["schedule_id"], warehouse_config
    )

    assert report["status"] == "failed"
    assert report["failed_count"] == 1
    assert saved["last_run_date"] is None
    assert saved["last_status"] == "failed"
    assert any(
        item["alert_type"] == "refresh_failed"
        for item in enrichment_scheduler.list_enrichment_alerts(warehouse_config)
    )


@pytest.mark.unit
def test_operations_status_summarizes_scheduler_state(warehouse_config):
    enrichment_scheduler.save_enrichment_schedule(
        "Core", ["AAPL"], warehouse_config, min_interval_seconds=0
    )
    status = enrichment_scheduler.get_enrichment_operations_status(warehouse_config)

    assert status["schedule_count"] == 1
    assert status["active_schedule_count"] == 1
    assert status["source_health"] == []
    assert status["recent_runs"] == []


@pytest.mark.unit
def test_source_circuit_skips_redundant_calls_and_recovers_automatically(
    warehouse_config,
):
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    schedule = enrichment_scheduler.save_enrichment_schedule(
        "Circuit protected",
        symbols,
        warehouse_config,
        timezone="UTC",
        weekdays=list(range(7)),
        local_time="00:00",
        max_symbols_per_run=5,
        min_interval_seconds=0,
        max_attempts=1,
        stale_after_hours=720,
    )
    disabled_seen = []

    def unavailable_collector(symbol, as_of_date, config):
        disabled = config.get("enrichment_disabled_datasets", {})
        disabled_seen.append((symbol, set(disabled)))
        snapshot = _snapshot(symbol, as_of_date, macro_available=False)
        if "macro_calendar" in disabled:
            macro = next(
                item for item in snapshot["sources"] if item["dataset"] == "macro_calendar"
            )
            macro["circuit_open"] = True
            macro["retry_at"] = disabled["macro_calendar"]
        return snapshot

    first = enrichment_scheduler.execute_enrichment_schedule(
        schedule,
        "2026-10-23",
        warehouse_config,
        collector=unavailable_collector,
        sleep_fn=lambda _seconds: None,
    )

    assert first["source_failure_count"] == 3
    assert first["source_circuit_skip_count"] == 2
    assert first["open_circuit_count"] == 1
    assert first["next_retry_at"]
    assert disabled_seen[:3] == [
        ("AAPL", set()),
        ("MSFT", set()),
        ("NVDA", set()),
    ]
    assert all("macro_calendar" in disabled for _symbol, disabled in disabled_seen[3:])
    alerts = enrichment_scheduler.list_enrichment_alerts(
        warehouse_config, unacknowledged_only=True
    )
    source_alerts = [
        item for item in alerts if item["alert_type"] == "source_unavailable"
    ]
    assert len(source_alerts) == 1
    assert source_alerts[0]["affected_symbol_count"] == 5

    connection = enrichment_scheduler._connect(warehouse_config)
    connection.execute(
        """UPDATE enrichment_source_circuits
           SET retry_at = ?, state = 'open' WHERE dataset = 'macro_calendar'""",
        [datetime.now(UTC)],
    )
    connection.close()
    refreshed = enrichment_scheduler.get_enrichment_schedule(
        schedule["schedule_id"], warehouse_config
    )

    def recovered_collector(symbol, as_of_date, config):
        assert "macro_calendar" not in config.get("enrichment_disabled_datasets", {})
        return _snapshot(symbol, as_of_date, macro_available=True)

    second = enrichment_scheduler.execute_enrichment_schedule(
        refreshed,
        "2026-10-23",
        warehouse_config,
        collector=recovered_collector,
        sleep_fn=lambda _seconds: None,
    )
    health = {
        item["dataset"]: item
        for item in enrichment_scheduler.list_enrichment_source_health(warehouse_config)
    }
    assert second["open_circuit_count"] == 0
    assert second["next_retry_at"] is None
    assert health["macro_calendar"]["circuit_state"] == "closed"
    assert not any(
        item["alert_type"] == "source_unavailable"
        for item in enrichment_scheduler.list_enrichment_alerts(
            warehouse_config, unacknowledged_only=True
        )
    )


@pytest.mark.unit
def test_open_circuit_retry_becomes_due_after_cooldown():
    retry_at = datetime(2026, 10, 23, 18, 30, tzinfo=UTC)
    schedule = {
        "active": True,
        "timezone": "UTC",
        "weekdays": [4],
        "local_time": "20:00",
        "last_run_date": "2026-10-23",
        "last_summary": {"next_retry_at": retry_at.isoformat()},
    }

    assert not enrichment_scheduler.enrichment_schedule_is_due(
        schedule, retry_at - timedelta(seconds=1)
    )
    assert enrichment_scheduler.enrichment_schedule_is_due(
        schedule, retry_at + timedelta(seconds=1)
    )


@pytest.mark.unit
def test_legacy_symbol_alerts_are_grouped_and_acknowledged_together(warehouse_config):
    schedule = enrichment_scheduler.save_enrichment_schedule(
        "Grouped alerts", ["AAPL", "MSFT"], warehouse_config
    )
    connection = enrichment_scheduler._connect(warehouse_config)
    for symbol in ("AAPL", "MSFT"):
        enrichment_scheduler._create_alert(
            connection,
            schedule_id=schedule["schedule_id"],
            run_id="run-1",
            as_of_date="2026-10-23",
            alert_type="source_unavailable",
            severity="high",
            title="Macro Calendar unavailable",
            message=f"{symbol} returned no usable macro data.",
            symbol=symbol,
            dataset="macro_calendar",
        )
    connection.close()

    alerts = enrichment_scheduler.list_enrichment_alerts(
        warehouse_config, unacknowledged_only=True
    )

    assert len(alerts) == 1
    assert sorted(alerts[0]["affected_symbols"]) == ["AAPL", "MSFT"]
    assert alerts[0]["grouped_alert_count"] == 2
    assert alerts[0]["symbol"] is None
    assert "consolidated" in alerts[0]["message"]
    assert enrichment_scheduler.acknowledge_enrichment_alert(
        alerts[0]["alert_id"], warehouse_config
    )
    assert (
        enrichment_scheduler.list_enrichment_alerts(
            warehouse_config, unacknowledged_only=True
        )
        == []
    )
