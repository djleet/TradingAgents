from __future__ import annotations

from datetime import UTC, datetime

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
    assert {item["symbol"] for item in alerts if item["alert_type"] == "stale_symbol"} == {
        "NVDA"
    }
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
