from __future__ import annotations

import pytest

from tradingagents.dataflows import candidate_pipeline, consensus_engine

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _item(dataset, confidence, facts):
    return {
        "dataset": dataset,
        "confidence_score": confidence,
        "facts": facts,
    }


def _bundle(*items):
    return {
        "bundle_id": "b" * 64,
        "symbol": "AAPL",
        "analysis_date": "2026-10-23",
        "purpose": "candidate",
        "verdict": "ready",
        "coverage_score": 72,
        "eligible_count": len(items),
        "blocked_count": 0,
        "missing_count": 8 - len(items),
        "items": list(items),
        "blocked": [],
        "missing": [],
    }


@pytest.mark.unit
def test_consensus_engine_blocks_only_unresolved_critical_conflicts(warehouse_config):
    bundle = _bundle(
        _item(
            "opportunity_scan",
            82,
            {"opportunity_score": 78, "bucket": "top_opportunity"},
        ),
        _item(
            "validation",
            76,
            {
                "robustness": "weak",
                "sharpe_ratio": -0.4,
                "oos_return": -0.08,
                "profitable_fold_rate": 0.25,
            },
        ),
    )

    report = consensus_engine.build_consensus_report(
        "aapl", "2026-10-23", warehouse_config, bundle=bundle
    )

    assert report["verdict"] == "blocked"
    assert report["qualification_gate"]["allowed"] is False
    assert report["qualification_gate"]["unresolved_critical_count"] == 1
    assert report["conflicts"][0]["type"] == "scanner_vs_validation"
    assert report["confidence"]["contradiction_penalty"] == 18

    resolution = consensus_engine.resolve_consensus_conflict(
        report["conflicts"][0]["conflict_id"],
        "accepted_risk",
        "Validation sample is short; keep the exception visible.",
        warehouse_config,
    )
    assert resolution["resolution"] == "accepted_risk"

    rebuilt = consensus_engine.build_consensus_report(
        "AAPL", "2026-10-23", warehouse_config, bundle=bundle
    )
    assert rebuilt["qualification_gate"]["allowed"] is True
    assert rebuilt["qualification_gate"]["unresolved_critical_count"] == 0
    assert rebuilt["verdict"] == "resolved"
    assert rebuilt["conflicts"][0]["resolution"]["note"].startswith(
        "Validation sample"
    )


@pytest.mark.unit
def test_consensus_engine_explains_multiple_source_disagreements(warehouse_config):
    bundle = _bundle(
        _item("opportunity_scan", 80, {"opportunity_score": 76}),
        _item("earnings_estimates", 68, {"target_upside": 0.32}),
        _item(
            "options_implied_volatility",
            70,
            {"put_call_open_interest": 2.1},
        ),
        _item("research_context", 74, {"fundamental_score": 34}),
    )
    candidate = {
        "symbol": "AAPL",
        "risk_level": "high",
        "event_risk": 78,
        "decision_confidence": 81,
        "validation_verdict": "not_run",
    }

    report = consensus_engine.build_consensus_report(
        "AAPL",
        "2026-10-23",
        warehouse_config,
        bundle=bundle,
        candidate=candidate,
    )

    types = {item["type"] for item in report["conflicts"]}
    assert types >= {
        "scanner_vs_fundamentals",
        "analyst_vs_options",
        "score_vs_event_risk",
    }
    assert report["confidence"]["adjusted_score"] < 81
    assert report["signal_count"] == 4
    assert report["evidence"]["bundle_id"] == "b" * 64


@pytest.mark.unit
def test_consensus_engine_keeps_missing_evidence_non_blocking(warehouse_config):
    bundle = {
        **_bundle(),
        "verdict": "insufficient",
        "coverage_score": 0,
        "eligible_count": 0,
        "missing_count": 8,
        "missing": ["validation", "opportunity_scan"],
    }
    report = consensus_engine.build_consensus_report(
        "AAPL", "2026-10-23", warehouse_config, bundle=bundle
    )

    assert report["verdict"] == "insufficient"
    assert report["qualification_gate"]["allowed"] is True
    assert report["conflicts"] == []
    assert set(report["gaps"]) == {"opportunity_scan", "validation"}
    status = consensus_engine.get_consensus_status(warehouse_config)
    assert status["evaluated_symbol_count"] == 1
    assert status["blocked_symbol_count"] == 0
    assert consensus_engine.list_consensus_reports(warehouse_config)[0][
        "qualification_allowed"
    ] is True


@pytest.mark.unit
def test_consensus_report_attaches_to_candidate_audit_trail(warehouse_config):
    candidate = candidate_pipeline.upsert_candidate(
        "AAPL",
        {"opportunity_score": 72, "rank": 3},
        warehouse_config,
    )
    report = consensus_engine.build_consensus_report(
        "AAPL",
        "2026-10-23",
        warehouse_config,
        bundle=_bundle(_item("opportunity_scan", 80, {"opportunity_score": 72})),
        candidate=candidate,
    )

    updated = candidate_pipeline.attach_candidate_consensus(
        candidate["candidate_id"], report, warehouse_config
    )
    history = candidate_pipeline.list_candidate_history(
        candidate["candidate_id"], warehouse_config
    )

    assert updated["stage"] == "discovered"
    assert updated["evidence"]["consensus_engine"]["report_id"] == report["report_id"]
    assert history[0]["event_type"] == "consensus_evaluated"
