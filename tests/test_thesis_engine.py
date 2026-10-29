from __future__ import annotations

import pytest

from tradingagents.dataflows import candidate_pipeline, thesis_engine

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _candidate(config):
    return candidate_pipeline.upsert_candidate(
        "AAPL",
        {
            "opportunity_score": 72,
            "rank": 3,
            "decision_confidence": {"score": 68},
            "quality": {"score": 81},
        },
        config,
        thesis="Revenue durability supports the current valuation.",
    )


def _bundle(*, score=42, sharpe=-0.3):
    return {
        "bundle_id": "b" * 64,
        "items": [
            {
                "dataset": "opportunity_scan",
                "facts": {"opportunity_score": score},
            },
            {
                "dataset": "validation",
                "facts": {"sharpe_ratio": sharpe, "oos_return": -0.08},
            },
        ],
    }


@pytest.mark.unit
def test_thesis_engine_versions_and_invalidates_without_changing_stage(
    warehouse_config,
):
    candidate = _candidate(warehouse_config)
    rules = [
        {
            "metric": "opportunity_score",
            "operator": "lt",
            "threshold": 45,
            "severity": "critical",
            "rationale": "The market signal no longer supports the thesis.",
        },
        {
            "metric": "validation_sharpe",
            "operator": "lte",
            "threshold": 0,
            "severity": "warning",
            "rationale": "Out-of-sample risk-adjusted performance weakened.",
        },
    ]
    first = thesis_engine.create_thesis_version(
        candidate["candidate_id"], candidate["thesis"], rules, warehouse_config
    )
    duplicate = thesis_engine.create_thesis_version(
        candidate["candidate_id"], candidate["thesis"], rules, warehouse_config
    )

    assert first["version_number"] == 1
    assert duplicate["version_id"] == first["version_id"]
    assert duplicate["created"] is False

    evaluation = thesis_engine.evaluate_thesis(
        candidate["candidate_id"],
        "2026-10-23",
        warehouse_config,
        bundle=_bundle(),
        candidate=candidate,
    )
    updated = candidate_pipeline.attach_candidate_thesis_evaluation(
        candidate["candidate_id"], evaluation, warehouse_config
    )

    assert evaluation["verdict"] == "invalidated"
    assert evaluation["critical_count"] == 1
    assert evaluation["triggered_count"] == 2
    assert evaluation["policy"]["decision_effect"] == "alert_only"
    assert updated["stage"] == "discovered"
    assert updated["evidence"]["thesis_engine"]["evaluation_id"] == evaluation[
        "evaluation_id"
    ]
    history = candidate_pipeline.list_candidate_history(
        candidate["candidate_id"], warehouse_config
    )
    assert history[0]["event_type"] == "thesis_invalidated"


@pytest.mark.unit
def test_thesis_engine_keeps_immutable_version_history(warehouse_config):
    candidate = _candidate(warehouse_config)
    first = thesis_engine.create_thesis_version(
        candidate["candidate_id"],
        candidate["thesis"],
        [
            {
                "metric": "event_risk",
                "operator": "gte",
                "threshold": 70,
                "severity": "critical",
                "rationale": "A major event would invalidate the setup.",
            }
        ],
        warehouse_config,
    )
    second = thesis_engine.create_thesis_version(
        candidate["candidate_id"],
        "Revenue durability and improving margins support the valuation.",
        [
            {
                "metric": "event_risk",
                "operator": "gte",
                "threshold": 60,
                "severity": "critical",
                "rationale": "A major event would invalidate the setup.",
            }
        ],
        warehouse_config,
        note="Tightened risk tolerance",
    )
    state = thesis_engine.get_thesis_state(candidate["candidate_id"], warehouse_config)

    assert first["version_number"] == 1
    assert second["version_number"] == 2
    assert [item["version_number"] for item in state["versions"]] == [2, 1]
    assert state["active_version"]["note"] == "Tightened risk tolerance"


@pytest.mark.unit
def test_missing_thesis_metrics_are_visible_but_never_triggered(warehouse_config):
    candidate = _candidate(warehouse_config)
    thesis_engine.create_thesis_version(
        candidate["candidate_id"],
        candidate["thesis"],
        [
            {
                "metric": "analyst_target_upside_pct",
                "operator": "lt",
                "threshold": -5,
                "severity": "critical",
                "rationale": "Analyst targets imply downside.",
            }
        ],
        warehouse_config,
    )
    evaluation = thesis_engine.evaluate_thesis(
        candidate["candidate_id"],
        "2026-10-23",
        warehouse_config,
        bundle={"bundle_id": "b" * 64, "items": []},
        candidate=candidate,
    )

    assert evaluation["verdict"] == "insufficient"
    assert evaluation["triggered_count"] == 0
    assert evaluation["missing_rule_count"] == 1
    status = thesis_engine.get_thesis_status(warehouse_config)
    assert status["versioned_candidate_count"] == 1
    assert status["invalidated_count"] == 0
