from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.dataflows import candidate_pipeline, shadow_lab

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _experiment(config):
    return shadow_lab.create_shadow_experiment(
        "Raise qualification threshold",
        "A stricter opportunity threshold should improve forward excess returns.",
        "opportunity_score",
        70,
        80,
        config,
        horizon_sessions=21,
        transaction_cost_bps=0,
        minimum_observations=3,
        now=datetime(2026, 1, 1, 12, tzinfo=UTC),
    )


def _observations():
    rows = []
    for index, (symbol, score) in enumerate(
        zip(
            ["LOW1", "LOW2", "LOW3", "HIGH1", "HIGH2", "HIGH3"],
            [72, 74, 76, 82, 86, 90],
            strict=True,
        )
    ):
        rows.append(
            {
                "observation_id": f"observation-{index}",
                "scan_id": "scan-1",
                "symbol": symbol,
                "benchmark_symbol": "SPY",
                "decision_date": "2026-01-05",
                "recorded_at": "2026-01-05T20:00:00+00:00",
                "metric": "opportunity_score",
                "metric_value": score,
                "control_selected": True,
                "challenger_selected": score >= 80,
                "score": score,
                "rank": index + 1,
            }
        )
    return rows


def _price_frame(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=100)
    if symbol.startswith("HIGH"):
        drift = 0.0020
    elif symbol.startswith("LOW"):
        drift = -0.0010
    else:
        drift = 0.0005
    prices = [100 * math.exp(drift * index) for index in range(len(dates))]
    return pd.DataFrame({"Date": dates.date, "Close": prices})


@pytest.mark.unit
def test_shadow_lab_runs_prospectively_and_persists_review(warehouse_config):
    experiment = _experiment(warehouse_config)

    report = shadow_lab.build_shadow_evaluation(
        experiment["experiment_id"],
        "2026-06-30",
        warehouse_config,
        observations=_observations(),
        price_loader=lambda symbols, *_: (
            {symbol: _price_frame(symbol) for symbol in symbols},
            [],
        ),
    )

    assert report["verdict"] == "challenger_promising"
    assert report["control"]["matured_outcomes"] == 6
    assert report["challenger"]["matured_outcomes"] == 3
    assert report["comparison"]["excess_delta"] > 0
    assert report["comparison"]["control_only_outcomes"] == 3
    assert report["comparison"]["overlap_outcomes"] == 3
    assert all(item["entry_date"] > item["decision_date"] for item in report["timeline"])
    assert report["policy"]["production_changes"] is False
    assert report["policy"]["network_fetches"] is False

    registry = shadow_lab.capture_shadow_evaluation(report, warehouse_config)
    duplicate = shadow_lab.capture_shadow_evaluation(report, warehouse_config)
    assert duplicate["evaluation_id"] == registry["evaluation_id"]
    latest = shadow_lab.get_latest_shadow_evaluation(
        experiment["experiment_id"], warehouse_config
    )
    assert latest["registry"]["evaluation_id"] == registry["evaluation_id"]
    assert shadow_lab.list_shadow_evaluations(
        experiment["experiment_id"], warehouse_config
    )[0]["verdict"] == "challenger_promising"

    reviewed = shadow_lab.mark_shadow_evaluation_reviewed(
        registry["evaluation_id"], warehouse_config
    )
    assert reviewed["reviewed_at"]
    closed = shadow_lab.close_shadow_experiment(
        experiment["experiment_id"],
        warehouse_config,
        now=datetime(2026, 7, 1, 12, tzinfo=UTC),
    )
    assert closed["status"] == "closed"
    assert shadow_lab.get_shadow_lab_status(warehouse_config)["closed_count"] == 1


@pytest.mark.unit
def test_shadow_lab_withholds_verdict_until_both_arms_mature(warehouse_config):
    experiment = _experiment(warehouse_config)
    report = shadow_lab.build_shadow_evaluation(
        experiment["experiment_id"],
        "2026-01-09",
        warehouse_config,
        observations=_observations(),
        price_loader=lambda symbols, *_: (
            {symbol: _price_frame(symbol) for symbol in symbols},
            [],
        ),
    )

    assert report["verdict"] == "insufficient_data"
    assert report["control"]["matured_outcomes"] == 0
    assert report["challenger"]["matured_outcomes"] == 0
    assert report["comparison"]["excess_delta"] is None
    assert report["coverage"]["pending_unique_outcomes"] == 6


@pytest.mark.unit
def test_shadow_lab_rejects_identical_arms(warehouse_config):
    with pytest.raises(ValueError, match="must differ"):
        shadow_lab.create_shadow_experiment(
            "Invalid experiment",
            "Both arms accidentally use the same exact threshold.",
            "opportunity_score",
            70,
            70,
            warehouse_config,
        )


@pytest.mark.unit
def test_shadow_lab_excludes_scans_recorded_before_registration(warehouse_config):
    experiment = shadow_lab.create_shadow_experiment(
        "Strict prospective boundary",
        "Only scans recorded after registration should enter this comparison.",
        "opportunity_score",
        70,
        80,
        warehouse_config,
        minimum_observations=3,
        now=datetime(2026, 1, 10, 12, tzinfo=UTC),
    )
    connection = candidate_pipeline._connect(warehouse_config)
    try:
        for scan_id, symbol, created_at in (
            ("old-scan", "OLD", datetime(2026, 1, 5, 20, tzinfo=UTC)),
            ("new-scan", "NEW", datetime(2026, 1, 12, 20, tzinfo=UTC)),
        ):
            result = {
                "items": [
                    {
                        "symbol": symbol,
                        "opportunity_score": 85,
                        "passes_filters": True,
                    }
                ]
            }
            connection.execute(
                """
                INSERT INTO opportunity_scans VALUES (
                    ?, '2026-01-02', 'SPY', NULL, NULL, 1, 1, 1, 1, '{}', ?, ?
                )
                """,
                [scan_id, json.dumps(result), created_at],
            )
    finally:
        connection.close()

    report = shadow_lab.build_shadow_evaluation(
        experiment["experiment_id"],
        "2026-06-30",
        warehouse_config,
        price_loader=lambda symbols, *_: (
            {symbol: _price_frame(symbol) for symbol in symbols},
            [],
        ),
    )

    assert report["coverage"]["source_scans"] == 1
    assert {item["symbol"] for item in report["timeline"]} == {"NEW"}
    assert report["timeline"][0]["decision_date"] == "2026-01-12"
