from __future__ import annotations

from datetime import date

import pytest

from tradingagents.dataflows import candidate_pipeline, paper_portfolio, warehouse

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path, monkeypatch):
    config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }
    monkeypatch.setattr(warehouse, "get_config", lambda: config)
    return config


@pytest.mark.unit
def test_vendor_results_are_deduplicated_but_revised_payloads_are_retained(warehouse_config):
    args = ("AAPL", "2026-01-01", "2026-01-31")
    assert warehouse.capture_vendor_result("get_stock_data", "openbb", args, {}, "Date,Close\n2026-01-30,1")
    assert warehouse.capture_vendor_result("get_stock_data", "openbb", args, {}, "Date,Close\n2026-01-30,1")
    assert warehouse.capture_vendor_result("get_stock_data", "openbb", args, {}, "Date,Close\n2026-01-30,2")

    status = warehouse.get_warehouse_status(warehouse_config)
    assert status == {
        "enabled": True,
        "dependency_available": True,
        "ready": True,
        "observation_count": 2,
        "market_bar_count": 2,
        "quantitative_snapshot_count": 0,
        "validation_experiment_count": 0,
        "portfolio_validation_experiment_count": 0,
        "opportunity_scan_count": 0,
        "scan_universe_count": 0,
        "active_scan_schedule_count": 0,
        "unacknowledged_alert_count": 0,
        "quality_failures": 0,
    }

    connection = duckdb.connect(warehouse_config["data_warehouse_path"], read_only=True)
    try:
        row = connection.execute(
            "SELECT source, dataset, symbol, query_start, query_end, as_of_date "
            "FROM vendor_observations ORDER BY payload_json LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    assert row[0:3] == ("openbb", "get_stock_data", "AAPL")
    assert tuple(str(value) for value in row[3:]) == (
        "2026-01-01",
        "2026-01-31",
        "2026-01-31",
    )


@pytest.mark.unit
def test_ohlcv_normalization_rejects_future_and_impossible_bars(warehouse_config):
    payload = """# vendor response
Date,Open,High,Low,Close,Volume
2026-01-30,10,12,9,11,100
2026-01-31,10,8,9,11,100
2026-02-01,11,12,10,11,100
"""
    assert warehouse.capture_vendor_result(
        "get_stock_data",
        "test",
        ("AAPL", "2026-01-01", "2026-01-31"),
        {},
        payload,
    )

    connection = duckdb.connect(warehouse_config["data_warehouse_path"], read_only=True)
    try:
        bars = connection.execute(
            "SELECT bar_date, close FROM market_bars ORDER BY bar_date"
        ).fetchall()
        quality = connection.execute(
            "SELECT row_count, stored_row_count, invalid_ohlc, future_rows, passed "
            "FROM ingestion_quality"
        ).fetchone()
    finally:
        connection.close()

    assert [(str(day), close) for day, close in bars] == [("2026-01-30", 11.0)]
    assert quality == (3, 1, 1, 1, False)


@pytest.mark.unit
def test_ohlcv_parser_handles_vendor_prose_and_adjusted_close():
    frame = warehouse.parse_ohlcv_payload(
        "# Stock data\n# Total records: 1\n\n"
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2026-01-02,10,11,9,10.5,10.4,123\n"
    )
    assert frame.to_dict("records") == [
        {
            "Date": date(2026, 1, 2),
            "Open": 10,
            "High": 11,
            "Low": 9,
            "Close": 10.5,
            "Adj Close": 10.4,
            "Volume": 123,
        }
    ]


@pytest.mark.unit
def test_archive_failure_never_breaks_a_market_data_call(warehouse_config, monkeypatch):
    monkeypatch.setattr(warehouse, "_connect", lambda _path: (_ for _ in ()).throw(OSError("disk full")))
    assert warehouse.capture_vendor_result("get_news", "test", ("AAPL",), {}, "news") is False


@pytest.mark.unit
def test_quantitative_snapshots_are_audited_and_deduplicated(warehouse_config):
    snapshot = {
        "model_version": "transparent-trend-v1",
        "symbol": "AAPL",
        "as_of_date": "2026-01-31",
        "latest_bar_date": "2026-01-30",
        "score": 42.0,
        "label": "bullish",
        "risk_regime": "normal",
        "confidence": 0.9,
        "bars_used": 252,
        "metrics": {"momentum_20": 0.08},
    }
    assert warehouse.capture_quantitative_snapshot(snapshot)
    assert warehouse.capture_quantitative_snapshot(snapshot)

    status = warehouse.get_warehouse_status(warehouse_config)
    assert status["quantitative_snapshot_count"] == 1


@pytest.mark.unit
def test_validation_experiments_are_persisted_and_listed(warehouse_config):
    report = {
        "model_version": "transparent-trend-v1",
        "symbol": "AAPL",
        "benchmark_symbol": "SPY",
        "as_of_date": "2026-01-30",
        "robustness": "mixed",
        "configuration": {"train_sessions": 252, "test_sessions": 63},
        "metrics": {
            "oos_return": 0.12,
            "benchmark_return": 0.10,
            "max_drawdown": -0.08,
            "fold_count": 4,
        },
        "points": [{"date": "2026-01-30", "strategy_equity": 1.12}],
    }

    experiment_id = warehouse.capture_validation_experiment(report, warehouse_config)
    assert experiment_id == warehouse.capture_validation_experiment(report, warehouse_config)
    items = warehouse.list_validation_experiments(warehouse_config)

    assert len(items) == 1
    assert items[0]["experiment_id"] == experiment_id
    assert items[0]["symbol"] == "AAPL"
    assert warehouse.get_warehouse_status(warehouse_config)["validation_experiment_count"] == 1


@pytest.mark.unit
def test_portfolio_validation_experiments_are_persisted_and_listed(warehouse_config):
    report = {
        "model_version": "portfolio-walk-forward-v1",
        "as_of_date": "2026-01-30",
        "benchmark_symbol": "SPY",
        "symbols": ["AAPL", "MSFT"],
        "verdict": "improved",
        "configuration": {"train_sessions": 252, "rebalance_sessions": 63},
        "block_win_rate": 0.75,
        "strategies": {
            "current": {"total_return": 0.10, "max_drawdown": -0.12},
            "risk_balanced": {"total_return": 0.11, "max_drawdown": -0.08},
            "benchmark": {"total_return": 0.09, "max_drawdown": -0.15},
        },
        "points": [{"date": "2026-01-30", "balanced_equity": 1.11}],
    }

    experiment_id = warehouse.capture_portfolio_validation_experiment(
        report, warehouse_config
    )
    assert experiment_id == warehouse.capture_portfolio_validation_experiment(
        report, warehouse_config
    )
    items = warehouse.list_portfolio_validation_experiments(warehouse_config)

    assert len(items) == 1
    assert items[0]["experiment_id"] == experiment_id
    assert items[0]["symbols"] == ["AAPL", "MSFT"]
    assert items[0]["verdict"] == "improved"
    assert warehouse.get_warehouse_status(warehouse_config)[
        "portfolio_validation_experiment_count"
    ] == 1


@pytest.mark.unit
def test_opportunity_scans_are_persisted_and_listed(warehouse_config):
    report = {
        "as_of_date": "2026-01-30",
        "benchmark_symbol": "SPY",
        "symbols": ["AAPL", "MSFT", "FAIL"],
        "requested_symbols": 3,
        "scanned_symbols": 2,
        "eligible_symbols": 1,
        "bucket_counts": {
            "top_opportunity": 1,
            "watch": 0,
            "neutral": 0,
            "avoid": 0,
            "excluded": 1,
        },
        "filters": {
            "min_dollar_volume": 5_000_000,
            "max_volatility": 0.8,
            "min_quality_score": 70,
        },
        "validation_configuration": {
            "sessions": 252,
            "train_sessions": 252,
            "test_sessions": 63,
        },
        "items": [{"symbol": "AAPL", "opportunity_score": 72}],
        "errors": [{"symbol": "FAIL", "message": "unavailable"}],
    }

    scan_id = warehouse.capture_opportunity_scan(report, warehouse_config)
    assert scan_id == warehouse.capture_opportunity_scan(report, warehouse_config)
    items = warehouse.list_opportunity_scans(warehouse_config)

    assert len(items) == 1
    assert items[0]["scan_id"] == scan_id
    assert items[0]["benchmark_symbol"] == "SPY"
    assert items[0]["requested_count"] == 3
    assert items[0]["top_opportunity_count"] == 1
    assert warehouse.get_warehouse_status(warehouse_config)["opportunity_scan_count"] == 1


@pytest.mark.unit
def test_candidate_pipeline_tracks_evidence_decisions_and_paper_outcomes(
    warehouse_config,
):
    evidence = {
        "symbol": "AAPL",
        "opportunity_score": 72,
        "rank": 2,
        "bucket": "top_opportunity",
        "quality": {"score": 91},
        "volatility_60": 0.22,
        "validation": {
            "alpha_vs_asset": 0.04,
            "profitable_fold_rate": 0.75,
            "sharpe_ratio": 1.1,
        },
        "decision_confidence": {"score": 76, "label": "supported"},
    }
    created = candidate_pipeline.upsert_candidate(
        "AAPL", evidence, warehouse_config, review_date="2020-01-01"
    )
    assert created["stage"] == "discovered"
    assert created["priority"] == "high"
    assert created["validation_verdict"] == "supported"
    assert created["review_due"] is True

    validating = candidate_pipeline.update_candidate(
        created["candidate_id"],
        {"stage": "validating", "note": "Begin formal review"},
        warehouse_config,
    )
    qualified = candidate_pipeline.update_candidate(
        created["candidate_id"],
        {"stage": "qualified", "note": "Evidence threshold passed"},
        warehouse_config,
    )
    assert validating["stage"] == "validating"
    assert qualified["stage"] == "qualified"

    refreshed = candidate_pipeline.refresh_candidates_from_scan(
        {"items": [{**evidence, "opportunity_score": 81, "rank": 1}]},
        "scan-2",
        warehouse_config,
    )
    assert refreshed == 1
    current = candidate_pipeline.list_candidates(warehouse_config)
    assert current["items"][0]["latest_score"] == 81
    assert current["items"][0]["latest_rank"] == 1

    open_position = {
        "open_positions": [{"symbol": "AAPL", "position_id": "position-open"}],
        "closed_positions": [],
    }
    assert candidate_pipeline.sync_candidates_with_paper(open_position, warehouse_config) == 1
    assert candidate_pipeline.list_candidates(warehouse_config)["items"][0]["stage"] == "paper_position"

    closed_position = {
        "open_positions": [],
        "closed_positions": [{"symbol": "AAPL", "position_id": "position-open"}],
    }
    assert candidate_pipeline.sync_candidates_with_paper(closed_position, warehouse_config) == 1
    final = candidate_pipeline.list_candidates(warehouse_config)
    assert final["items"][0]["stage"] == "closed"
    history = candidate_pipeline.list_candidate_history(
        created["candidate_id"], warehouse_config
    )
    assert {item["event_type"] for item in history} >= {
        "created",
        "stage_changed",
        "evidence_refreshed",
        "paper_synced",
    }


def _opportunity_report(as_of_date="2026-01-30", *, aapl_score=60, aapl_bucket="watch"):
    return {
        "as_of_date": as_of_date,
        "benchmark_symbol": "SPY",
        "symbols": ["AAPL", "MSFT"],
        "requested_symbols": 2,
        "scanned_symbols": 2,
        "eligible_symbols": 2,
        "bucket_counts": {
            "top_opportunity": int(aapl_bucket == "top_opportunity"),
            "watch": int(aapl_bucket == "watch"),
            "neutral": int(aapl_bucket == "neutral") + 1,
            "avoid": 0,
            "excluded": 0,
        },
        "filters": {"min_dollar_volume": 5_000_000},
        "validation_configuration": {"sessions": 252},
        "items": [
            {
                "symbol": "AAPL",
                "rank": 1,
                "opportunity_score": aapl_score,
                "bucket": aapl_bucket,
            },
            {
                "symbol": "MSFT",
                "rank": 2,
                "opportunity_score": 50,
                "bucket": "neutral",
            },
        ],
        "errors": [],
    }


@pytest.mark.unit
def test_opportunity_scans_compare_changes_and_generate_alerts(warehouse_config):
    first = _opportunity_report()
    warehouse.compare_opportunity_scan(first, warehouse_config)
    assert first["changes"]["has_previous"] is False
    first_id = warehouse.capture_opportunity_scan(first, warehouse_config)
    assert first_id

    second = _opportunity_report(
        "2026-02-02", aapl_score=74, aapl_bucket="top_opportunity"
    )
    warehouse.compare_opportunity_scan(second, warehouse_config)

    assert second["changes"]["has_previous"] is True
    assert second["changes"]["entered_top"] == 1
    assert second["items"][0]["change"]["score_delta"] == 14
    second_id = warehouse.capture_opportunity_scan(second, warehouse_config)
    alerts = warehouse.capture_opportunity_alerts(second, second_id, warehouse_config)
    assert alerts[0]["alert_type"] == "entered_top"
    assert warehouse.get_warehouse_status(warehouse_config)["unacknowledged_alert_count"] == 1
    assert warehouse.acknowledge_opportunity_alert(alerts[0]["alert_id"], warehouse_config)
    assert warehouse.list_opportunity_alerts(
        warehouse_config, unacknowledged_only=True
    ) == []


@pytest.mark.unit
def test_saved_universes_and_daily_schedules_are_durable(warehouse_config):
    universe = warehouse.save_scan_universe(
        "Core tech", ["AAPL", "MSFT", "NVDA"], warehouse_config
    )
    assert universe
    saved = warehouse.list_scan_universes(warehouse_config)
    assert saved[0]["symbols"] == ["AAPL", "MSFT", "NVDA"]

    schedule = warehouse.save_scan_schedule(
        "Core tech close",
        saved[0]["symbols"],
        "SPY",
        {"validation_sessions": 252, "min_quality_score": 70},
        warehouse_config,
        universe_id=universe["universe_id"],
        local_time="16:30",
    )
    assert schedule
    schedules = warehouse.list_scan_schedules(warehouse_config, active_only=True)
    assert schedules[0]["name"] == "Core tech close"
    assert warehouse.record_scan_schedule_run(
        schedule["schedule_id"],
        "2026-02-02",
        "completed",
        warehouse_config,
        scan_id="scan-id",
    )
    assert warehouse.list_scan_schedules(warehouse_config)[0]["last_scan_id"] == "scan-id"
    assert warehouse.set_scan_schedule_active(
        schedule["schedule_id"], False, warehouse_config
    )
    assert warehouse.get_warehouse_status(warehouse_config)["active_scan_schedule_count"] == 0
    assert warehouse.delete_scan_schedule(schedule["schedule_id"], warehouse_config)
    assert warehouse.delete_scan_universe(universe["universe_id"], warehouse_config)


@pytest.mark.unit
def test_paper_portfolio_sizes_marks_closes_and_journals(warehouse_config):
    portfolio_id = paper_portfolio.ensure_default_paper_portfolio(warehouse_config)

    opened = paper_portfolio.open_paper_position(
        "AAPL",
        "2026-01-30",
        100.0,
        warehouse_config,
        portfolio_id=portfolio_id,
        risk_pct=1.0,
        stop_pct=5.0,
        target_pct=15.0,
        decision_note="Strong evidence with a defined invalidation level.",
        evidence={"reasons": ["positive out-of-sample alpha"]},
        scan_id="scan-1",
        initial_score=74.0,
        initial_bucket="top_opportunity",
        initial_rank=1,
        benchmark_price=500.0,
    )

    assert opened["sizing"]["shares_selected"] == 200
    assert opened["open_positions"][0]["stop_price"] == 95.0
    assert opened["open_positions"][0]["target_price"] == 115.0
    assert opened["journal"][0]["decision_type"] == "open"
    assert opened["journal"][0]["snapshot"]["score"] == 74.0

    marked = paper_portfolio.mark_paper_portfolio(
        {"AAPL": 105.0},
        "2026-02-02",
        warehouse_config,
        portfolio_id=portfolio_id,
        benchmark_price=505.0,
    )
    assert marked["unrealized_pnl"] == pytest.approx(990.0)
    assert marked["benchmark_return"] == pytest.approx(0.01)
    assert len(marked["snapshots"]) == 2

    closed = paper_portfolio.close_paper_position(
        opened["opened_position_id"],
        "2026-02-03",
        110.0,
        warehouse_config,
        exit_reason="Target review",
    )
    assert closed["open_positions"] == []
    assert closed["realized_pnl"] == pytest.approx(1979.0)
    assert closed["hit_rate"] == 1.0
    assert closed["journal"][0]["decision_type"] == "close"
    assert closed["cohorts"] == [
        {
            "cohort": "top_opportunity",
            "trades": 1,
            "hit_rate": 1.0,
            "average_return": pytest.approx(1979.0 / 20010.0),
        }
    ]


@pytest.mark.unit
def test_paper_portfolio_rejects_duplicate_open_positions(warehouse_config):
    portfolio_id = paper_portfolio.ensure_default_paper_portfolio(warehouse_config)
    paper_portfolio.open_paper_position(
        "AAPL", "2026-01-30", 100.0, warehouse_config, portfolio_id=portfolio_id
    )
    with pytest.raises(ValueError, match="already open"):
        paper_portfolio.open_paper_position(
            "AAPL", "2026-02-02", 101.0, warehouse_config, portfolio_id=portfolio_id
        )


@pytest.mark.unit
def test_paper_monitor_persists_context_alerts_and_schedule(warehouse_config):
    portfolio_id = paper_portfolio.ensure_default_paper_portfolio(warehouse_config)
    opened = paper_portfolio.open_paper_position(
        "AAPL",
        "2026-01-30",
        100.0,
        warehouse_config,
        portfolio_id=portfolio_id,
        stop_pct=5.0,
        evidence={"research_context": {"overlay": {"stance": "supportive"}}},
        benchmark_price=500.0,
    )
    position_id = opened["opened_position_id"]
    assert opened["open_positions"][0]["latest_context"]["overlay"]["stance"] == "supportive"

    context = {
        "overlay": {"stance": "caution"},
        "events": {
            "flags": [
                {
                    "type": "estimated_reporting_window",
                    "severity": "high",
                    "message": "Estimated reporting window is near.",
                }
            ]
        },
    }
    marked = paper_portfolio.mark_paper_portfolio(
        {"AAPL": 94.0},
        "2026-02-02",
        warehouse_config,
        portfolio_id=portfolio_id,
        benchmark_price=505.0,
        contexts={"AAPL": context},
    )

    assert marked["alerts"] == 2
    assert {item["alert_type"] for item in marked["risk_alerts"]} == {
        "stop_breached",
        "event_estimated_reporting_window",
    }
    assert marked["open_positions"][0]["latest_context"]["overlay"]["stance"] == "caution"
    assert paper_portfolio.acknowledge_paper_alert(
        marked["risk_alerts"][0]["alert_id"], warehouse_config
    )
    after_ack = paper_portfolio.get_paper_portfolio(
        warehouse_config, portfolio_id=portfolio_id
    )
    assert after_ack["alerts"] == 1

    configured = paper_portfolio.update_paper_monitoring(
        True,
        "17:05",
        "America/New_York",
        warehouse_config,
        portfolio_id=portfolio_id,
    )
    assert configured["monitoring"]["enabled"] is True
    assert configured["monitoring"]["local_time"] == "17:05"
    assert position_id == configured["open_positions"][0]["position_id"]
