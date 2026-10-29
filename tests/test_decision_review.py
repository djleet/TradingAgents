from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import decision_review

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _observations():
    rule = {
        "metric": "validation_sharpe",
        "label": "Validation Sharpe",
        "operator": "lte",
        "threshold": 0,
        "severity": "critical",
        "triggered": True,
    }
    common = {
        "benchmark_symbol": "SPY",
        "decision_date": "2026-01-05",
        "recorded_at": "2026-01-05T18:00:00+00:00",
        "candidate_id": "candidate-1",
        "from_stage": "watchlist",
        "note": None,
        "score": 70,
        "decision_confidence": 68,
    }
    return [
        {
            **common,
            "observation_id": "qualified-1",
            "observation_type": "candidate_decision",
            "symbol": "AAPL",
            "decision": "qualified",
            "triggered_rules": [],
            "evidence_date": None,
        },
        {
            **common,
            "observation_id": "rejected-1",
            "observation_type": "candidate_decision",
            "symbol": "MSFT",
            "decision": "rejected",
            "triggered_rules": [],
            "evidence_date": None,
        },
        {
            **common,
            "observation_id": "invalidated-1",
            "observation_type": "thesis_alert",
            "symbol": "NVDA",
            "decision": "invalidated",
            "triggered_rules": [rule],
            "evidence_date": "2026-01-05",
        },
        {
            **common,
            "observation_id": "warning-1",
            "observation_type": "thesis_alert",
            "symbol": "TSLA",
            "decision": "warning",
            "triggered_rules": [rule],
            "evidence_date": "2026-01-05",
        },
    ]


def _price_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=100)
    drift = {
        "AAPL": 0.0020,
        "MSFT": -0.0010,
        "NVDA": -0.0020,
        "TSLA": 0.0025,
        "SPY": 0.0005,
    }[symbol]
    prices = [100 * math.exp(drift * index) for index in range(len(dates))]
    return pd.DataFrame({"Date": dates.date, "Close": prices})


@pytest.mark.unit
def test_decision_review_measures_decisions_and_alerts_without_look_ahead(
    warehouse_config,
):
    loader_calls = []

    def fake_loader(symbols, start_date, end_date):
        loader_calls.append((symbols, start_date, end_date))
        return ({symbol: _price_frame(symbol) for symbol in symbols}, [])

    report = decision_review.build_decision_review(
        "2026-06-30",
        warehouse_config,
        horizons=[5, 21],
        transaction_cost_bps=0,
        minimum_observations=1,
        observations=_observations(),
        price_loader=fake_loader,
    )

    assert loader_calls[0][2] == "2026-06-30"
    assert report["coverage"]["source_observations"] == 4
    assert report["coverage"]["matured_outcomes"] == 8
    assert report["summary"]["decisions_reviewed"] == 2
    assert report["summary"]["decision_accuracy"] == 1
    assert report["summary"]["helpful_alert_count"] == 1
    assert report["summary"]["false_alert_count"] == 1
    assert {item["classification"] for item in report["timeline"]} == {
        "helpful_decision",
        "helpful_alert",
        "false_alert",
    }
    assert all(item["entry_date"] > item["decision_date"] for item in report["outcomes"])
    assert all(
        item["counterfactual_delta"] > 0
        for item in report["timeline"]
        if item["classification"] in {"helpful_decision", "helpful_alert"}
    )
    assert report["rule_attribution"][0]["helpful_count"] == 1
    assert report["rule_attribution"][0]["false_alert_count"] == 1
    assert report["policy"]["network_fetches"] is False

    registry = decision_review.capture_decision_review(report, warehouse_config)
    duplicate = decision_review.capture_decision_review(report, warehouse_config)
    assert duplicate["run_id"] == registry["run_id"]
    latest = decision_review.get_latest_decision_review(warehouse_config)
    assert latest["registry"]["run_id"] == registry["run_id"]
    assert decision_review.list_decision_reviews(warehouse_config)[0][
        "matured_outcome_count"
    ] == 8
    reviewed = decision_review.mark_decision_review_reviewed(
        registry["run_id"], warehouse_config
    )
    assert reviewed["reviewed_at"]


@pytest.mark.unit
def test_decision_review_keeps_immature_outcomes_pending(warehouse_config):
    report = decision_review.build_decision_review(
        "2026-01-09",
        warehouse_config,
        horizons=[21],
        minimum_observations=1,
        observations=_observations()[:1],
        price_loader=lambda symbols, *_: (
            {symbol: _price_frame(symbol) for symbol in symbols},
            [],
        ),
    )

    assert report["verdict"] == "insufficient_data"
    assert report["coverage"]["matured_outcomes"] == 0
    assert report["coverage"]["pending_outcomes"] == 1
    assert report["timeline"] == []
