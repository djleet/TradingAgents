from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from tradingagents.dataflows import model_governance, shadow_lab

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _prices(symbol: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", "2027-12-31")
    drift = 0.002 if symbol.startswith("HIGH") else -0.001
    values = [100 * math.exp(drift * index) for index in range(len(dates))]
    return pd.DataFrame({"Date": dates.date, "Close": values})


def _observations(recorded_at: str, decision_date: str, prefix: str):
    rows = []
    for index, score in enumerate((72, 74, 76, 82, 86, 90)):
        high = score >= 80
        rows.append(
            {
                "observation_id": f"{prefix}-{index}",
                "scan_id": f"scan-{prefix}",
                "symbol": f"{'HIGH' if high else 'LOW'}{prefix}{index}",
                "benchmark_symbol": "SPY",
                "decision_date": decision_date,
                "recorded_at": recorded_at,
                "metric": "opportunity_score",
                "metric_value": score,
                "control_selected": True,
                "challenger_selected": high,
                "score": score,
                "rank": index + 1,
            }
        )
    return rows


def _reviewed_evaluation(config):
    experiment = shadow_lab.create_shadow_experiment(
        "Govern a stronger threshold",
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
    report = shadow_lab.build_shadow_evaluation(
        experiment["experiment_id"],
        "2026-06-30",
        config,
        observations=_observations(
            "2026-01-05T20:00:00+00:00", "2026-01-05", "initial"
        ),
        price_loader=lambda symbols, *_: (
            {symbol: _prices(symbol) for symbol in symbols},
            [],
        ),
    )
    registry = shadow_lab.capture_shadow_evaluation(report, config)
    shadow_lab.mark_shadow_evaluation_reviewed(registry["evaluation_id"], config)
    return registry["evaluation_id"]


@pytest.mark.unit
def test_governance_requires_approvals_probation_and_supports_rollback(
    warehouse_config,
):
    evaluation_id = _reviewed_evaluation(warehouse_config)
    proposal = model_governance.create_promotion_proposal(
        evaluation_id,
        "Advance the stronger threshold through controlled governance.",
        warehouse_config,
        minimum_probation_observations=3,
        now=datetime(2026, 10, 27, 12, tzinfo=UTC),
    )
    assert proposal["status"] == "pending_approval"

    proposal = model_governance.record_governance_approval(
        proposal["proposal_id"],
        "research",
        "approved",
        "Research evidence is prospective and sufficiently supported.",
        warehouse_config,
        now=datetime(2026, 10, 28, 12, tzinfo=UTC),
    )
    assert proposal["status"] == "pending_approval"
    proposal = model_governance.record_governance_approval(
        proposal["proposal_id"],
        "risk",
        "approved",
        "Risk accepts a bounded probation window and explicit rollback.",
        warehouse_config,
        now=datetime(2026, 10, 29, 12, tzinfo=UTC),
    )
    assert proposal["status"] == "approved"

    proposal = model_governance.start_governance_probation(
        proposal["proposal_id"],
        warehouse_config,
        now=datetime(2026, 11, 1, 12, tzinfo=UTC),
    )
    proposal = model_governance.refresh_governance_probation(
        proposal["proposal_id"],
        "2027-06-30",
        warehouse_config,
        observations=_observations(
            "2026-11-05T20:00:00+00:00", "2026-11-05", "probation"
        ),
        price_loader=lambda symbols, *_: (
            {symbol: _prices(symbol) for symbol in symbols},
            [],
        ),
    )
    assert proposal["probation"]["verdict"] == "eligible"

    proposal = model_governance.promote_governance_proposal(
        proposal["proposal_id"],
        "Promote after successful prospective probation and dual approval.",
        warehouse_config,
        now=datetime(2027, 7, 1, 12, tzinfo=UTC),
    )
    assert proposal["status"] == "promoted"
    assert model_governance.get_active_governed_policies(warehouse_config)[0][
        "threshold"
    ] == 80
    assert not model_governance.evaluate_governed_policy(
        {"latest_score": 75}, warehouse_config
    )["allowed"]
    assert model_governance.evaluate_governed_policy({}, warehouse_config)["allowed"]

    proposal = model_governance.rollback_governed_policy(
        proposal["proposal_id"],
        "Restore the previous champion after an explicit operator decision.",
        warehouse_config,
        now=datetime(2027, 7, 2, 12, tzinfo=UTC),
    )
    assert proposal["status"] == "rolled_back"
    assert model_governance.get_active_governed_policies(warehouse_config)[0][
        "threshold"
    ] == 70
    assert model_governance.evaluate_governed_policy(
        {"latest_score": 75}, warehouse_config
    )["allowed"]


@pytest.mark.unit
def test_governance_rejects_unreviewed_shadow_result(warehouse_config):
    experiment = shadow_lab.create_shadow_experiment(
        "Unreviewed threshold",
        "A result must be reviewed before governance can begin.",
        "opportunity_score",
        70,
        80,
        warehouse_config,
        minimum_observations=3,
        now=datetime(2026, 1, 1, 12, tzinfo=UTC),
    )
    report = shadow_lab.build_shadow_evaluation(
        experiment["experiment_id"],
        "2026-06-30",
        warehouse_config,
        observations=_observations(
            "2026-01-05T20:00:00+00:00", "2026-01-05", "unreviewed"
        ),
        price_loader=lambda symbols, *_: (
            {symbol: _prices(symbol) for symbol in symbols},
            [],
        ),
    )
    registry = shadow_lab.capture_shadow_evaluation(report, warehouse_config)

    with pytest.raises(ValueError, match="must be reviewed"):
        model_governance.create_promotion_proposal(
            registry["evaluation_id"],
            "This proposal has not passed the human review checkpoint.",
            warehouse_config,
            minimum_probation_observations=3,
        )
