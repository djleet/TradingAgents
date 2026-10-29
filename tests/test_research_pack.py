from __future__ import annotations

import pytest

from tradingagents.dataflows import research_pack

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _candidate():
    return {
        "candidate_id": "candidate-aapl",
        "symbol": "AAPL",
        "stage": "validating",
        "priority": "high",
        "review_due": True,
        "review_date": "2026-10-23",
        "latest_score": 68,
        "latest_rank": 14,
        "risk_level": "high",
        "decision_confidence": 61,
        "next_action": "Review event risk",
        "evidence": {
            "change": {
                "state": "deteriorated",
                "previous_score": 79,
                "score_delta": -11,
                "previous_rank": 4,
                "rank_delta": -10,
                "previous_bucket": "top_opportunity",
            }
        },
    }


def _snapshots():
    previous = {
        "coverage": {"score": 86},
        "earnings": {"earnings_date": "2026-11-10", "target_upside": 0.18},
        "options": {"expected_move_pct": 0.04, "atm_iv": 0.32},
        "sources": [
            {"dataset": "macro_calendar", "status": "available"},
        ],
        "decision_flags": [],
    }
    current = {
        "coverage": {"score": 62},
        "earnings": {"earnings_date": "2026-11-07", "target_upside": 0.05},
        "options": {"expected_move_pct": 0.11, "atm_iv": 0.47},
        "sources": [
            {"dataset": "macro_calendar", "status": "unavailable"},
        ],
        "decision_flags": [
            {
                "type": "earnings_window",
                "severity": "high",
                "title": "Confirmed earnings window is near",
                "message": "Earnings are inside the configured risk window.",
            }
        ],
    }
    return [current, previous]


def _consensus_report(*_args, **_kwargs):
    return {
        "report_id": "r" * 64,
        "verdict": "aligned",
        "consensus_score": 74,
        "agreement_rate": 80,
        "conflicts": [],
        "qualification_gate": {
            "allowed": True,
            "unresolved_critical_count": 0,
        },
    }


def _thesis_evaluation(*_args, **_kwargs):
    return {
        "evaluation_id": "t" * 64,
        "verdict": "warning",
        "triggered_count": 1,
        "critical_count": 0,
        "message": "Decision confidence warning triggered.",
        "version": {"version_number": 2},
    }


@pytest.mark.unit
def test_research_pack_detects_material_changes_and_routes_evidence(
    warehouse_config,
):
    routed = []
    attached = []
    consensus_attached = []
    thesis_attached = []

    def router(symbol, as_of_date, _config, *, purpose):
        routed.append((symbol, as_of_date, purpose))
        return {
            "bundle_id": "b" * 64,
            "verdict": "ready",
            "coverage_score": 74,
            "eligible_count": 6,
            "blocked_count": 1,
            "missing_count": 1,
        }

    def attach(candidate_id, bundle, _config):
        attached.append((candidate_id, bundle["bundle_id"]))

    pack = research_pack.build_research_pack(
        "2026-10-23",
        warehouse_config,
        symbols=["aapl"],
        trigger="enrichment_schedule",
        candidate_payload={"items": [_candidate()]},
        enrichment_snapshots={"AAPL": _snapshots()},
        router=router,
        attach=attach,
        consensus_builder=_consensus_report,
        consensus_attach=lambda candidate_id, report, _config: consensus_attached.append(
            (candidate_id, report["report_id"])
        ),
        thesis_evaluator=_thesis_evaluation,
        thesis_attach=lambda candidate_id, report, _config: thesis_attached.append(
            (candidate_id, report["evaluation_id"])
        ),
    )

    assert pack["status"] == "attention_required"
    assert pack["summary"]["candidate_count"] == 1
    assert pack["summary"]["high_risk_count"] == 1
    assert pack["summary"]["ready_evidence_count"] == 1
    assert pack["items"][0]["impact_score"] == 100
    assert {item["type"] for item in pack["items"][0]["changes"]} >= {
        "scanner_change",
        "coverage_change",
        "earnings_date_changed",
        "source_status_changed",
        "decision_flag_added",
    }
    assert routed == [("AAPL", "2026-10-23", "candidate")]
    assert attached == [("candidate-aapl", "b" * 64)]
    assert consensus_attached == [("candidate-aapl", "r" * 64)]
    assert pack["items"][0]["consensus"]["verdict"] == "aligned"
    assert thesis_attached == [("candidate-aapl", "t" * 64)]
    assert pack["items"][0]["thesis_engine"]["verdict"] == "warning"
    assert pack["summary"]["consensus_failure_count"] == 0
    assert pack["registry"]["pack_id"] == pack["pack_id"]


@pytest.mark.unit
def test_research_pack_is_content_addressed_reviewable_and_listed(warehouse_config):
    kwargs = {
        "candidate_payload": {"items": [_candidate()]},
        "enrichment_snapshots": {"AAPL": _snapshots()},
        "router": lambda *_args, **_kwargs: {
            "bundle_id": "c" * 64,
            "verdict": "partial",
            "coverage_score": 42,
            "eligible_count": 3,
            "blocked_count": 2,
            "missing_count": 3,
        },
        "attach": lambda *_args, **_kwargs: None,
        "consensus_builder": _consensus_report,
        "consensus_attach": lambda *_args, **_kwargs: None,
        "thesis_evaluator": _thesis_evaluation,
        "thesis_attach": lambda *_args, **_kwargs: None,
    }
    first = research_pack.build_research_pack(
        "2026-10-23", warehouse_config, **kwargs
    )
    second = research_pack.build_research_pack(
        "2026-10-23", warehouse_config, **kwargs
    )

    assert first["pack_id"] == second["pack_id"]
    reviewed = research_pack.mark_research_pack_reviewed(
        first["pack_id"], warehouse_config
    )
    assert reviewed["reviewed_at"]
    latest = research_pack.get_latest_research_pack(warehouse_config)
    assert latest["registry"]["reviewed_at"] == reviewed["reviewed_at"]
    history = research_pack.list_research_packs(warehouse_config)
    assert len(history) == 1
    assert history[0]["material_change_count"] > 0


@pytest.mark.unit
def test_research_pack_keeps_quiet_days_explicit(warehouse_config):
    pack = research_pack.build_research_pack(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
    )

    assert pack["status"] == "quiet"
    assert pack["items"] == []
    assert pack["summary"]["candidate_count"] == 0
    assert "No material" in pack["headline"]
