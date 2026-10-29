from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import calibration_engine, warehouse

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _scan_report(as_of_date: str, index: int) -> dict:
    aapl_score = 72 + index
    return {
        "as_of_date": as_of_date,
        "benchmark_symbol": "SPY",
        "symbols": ["AAPL", "MSFT"],
        "requested_symbols": 2,
        "scanned_symbols": 2,
        "eligible_symbols": 2,
        "bucket_counts": {
            "top_opportunity": 1,
            "watch": 1,
            "neutral": 0,
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
                "bucket": "top_opportunity",
                "passes_filters": True,
                "risk_regime": "normal",
                "quality": {"score": 95},
                "validation": {"robustness": "promising"},
                "score_components": {
                    "signal": 80 + index,
                    "risk": 70,
                    "quality": 95,
                    "liquidity": 90,
                    "oos_sharpe": 75 + index,
                    "oos_alpha": 70 + index,
                    "profitable_folds": 75,
                    "threshold_stability": 80,
                },
            },
            {
                "symbol": "MSFT",
                "rank": 2,
                "opportunity_score": 54 - index,
                "bucket": "watch",
                "passes_filters": True,
                "risk_regime": "elevated",
                "quality": {"score": 90},
                "validation": {"robustness": "mixed"},
                "score_components": {
                    "signal": 40 - index,
                    "risk": 55,
                    "quality": 90,
                    "liquidity": 88,
                    "oos_sharpe": 35 - index,
                    "oos_alpha": 40 - index,
                    "profitable_folds": 45,
                    "threshold_stability": 60,
                },
            },
        ],
        "errors": [],
    }


def _price_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2025-12-15", "2026-09-30")
    drift = {"AAPL": 0.0020, "MSFT": -0.0002, "SPY": 0.0005}[symbol]
    prices = [100 * math.exp(drift * index) for index in range(len(dates))]
    return pd.DataFrame({"Date": dates.date, "Close": prices})


@pytest.mark.unit
def test_calibration_engine_measures_only_forward_outcomes_and_persists_runs(
    warehouse_config,
):
    decision_dates = pd.bdate_range("2026-01-05", periods=8, freq="10B")
    for index, decision_date in enumerate(decision_dates):
        warehouse.capture_opportunity_scan(
            _scan_report(decision_date.date().isoformat(), index), warehouse_config
        )

    loader_calls = []

    def fake_loader(symbols, start_date, end_date):
        loader_calls.append((symbols, start_date, end_date))
        return ({symbol: _price_frame(symbol) for symbol in symbols}, [])

    report = calibration_engine.build_calibration_report(
        "2026-09-30",
        warehouse_config,
        horizons=[5, 21, 63],
        transaction_cost_bps=10,
        minimum_observations=5,
        price_loader=fake_loader,
    )

    assert loader_calls[0][2] == "2026-09-30"
    assert report["coverage"]["source_observations"] == 16
    assert report["coverage"]["matured_outcomes"] == 48
    assert report["verdict"] == "positive"
    assert {item["horizon_sessions"] for item in report["horizons"]} == {5, 21, 63}
    assert all(item["entry_date"] > item["decision_date"] for item in report["outcomes"])
    top = next(item for item in report["cohorts"] if item["cohort"] == "top_opportunity")
    assert top["average_excess_return"] > 0
    assert report["shadow_recommendations"]["recommended_qualification_threshold"] is not None
    assert report["shadow_recommendations"]["notice"].startswith("Exploratory")

    run_id = calibration_engine.capture_calibration_run(report, warehouse_config)
    assert calibration_engine.capture_calibration_run(report, warehouse_config) == run_id
    history = calibration_engine.list_calibration_runs(warehouse_config)
    assert history[0]["run_id"] == run_id
    assert history[0]["matured_outcomes"] == 48
    latest = calibration_engine.get_latest_calibration_run(warehouse_config)
    assert latest["registry"]["run_id"] == run_id
    assert latest["outcomes"][0]["entry_date"] > latest["outcomes"][0]["decision_date"]


@pytest.mark.unit
def test_calibration_engine_reports_insufficient_data_without_history(
    warehouse_config,
):
    report = calibration_engine.build_calibration_report(
        "2026-06-30",
        warehouse_config,
        horizons=[21],
        minimum_observations=5,
        price_loader=lambda *_: ({}, []),
    )

    assert report["verdict"] == "insufficient_data"
    assert report["coverage"]["source_observations"] == 0
    assert report["horizons"][0]["observations"] == 0
    assert report["drift"]["status"] == "insufficient_data"
    assert report["shadow_recommendations"]["recommended_qualification_threshold"] is None
