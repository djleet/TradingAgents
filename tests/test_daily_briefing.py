from __future__ import annotations

import pytest

from tradingagents.dataflows import daily_briefing

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _briefing(warehouse_config):
    return daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={
            "items": [
                {
                    "candidate_id": "candidate-aapl",
                    "symbol": "AAPL",
                    "stage": "qualified",
                    "priority": "high",
                    "review_due": True,
                    "review_date": "2026-10-22",
                    "next_action": "Review context evidence",
                    "latest_score": 81,
                    "decision_confidence": 74,
                    "validation_verdict": "promising",
                    "position_id": None,
                }
            ]
        },
        opportunity_alerts=[
            {
                "alert_id": "scanner-alert",
                "severity": "high",
                "alert_type": "entered_top",
                "symbol": "MSFT",
                "title": "MSFT entered the top bucket",
                "message": "Its evidence improved materially.",
                "current_score": 78,
                "previous_score": 63,
            }
        ],
        portfolio={
            "nav": 101_500,
            "total_return": 0.015,
            "alpha_vs_benchmark": 0.006,
            "max_drawdown": -0.02,
            "open_positions": [{"position_id": "position-1", "symbol": "NVDA"}],
            "risk_alerts": [
                {
                    "alert_id": "paper-alert",
                    "severity": "high",
                    "alert_type": "stop_breach",
                    "symbol": "NVDA",
                    "title": "NVDA crossed its paper stop",
                    "message": "Review the simulated exit assumption.",
                    "as_of_date": "2026-10-23",
                }
            ],
        },
        calibration={
            "verdict": "degrading",
            "as_of_date": "2026-10-23",
            "drift": {"status": "degrading", "excess_delta": -0.035},
            "registry": {"run_id": "calibration-1"},
        },
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "universe_name": "Core quality",
            "eligible_symbols": 18,
            "errors": [],
        },
        schedules=[{"schedule_id": "schedule-1", "last_status": "completed"}],
        warehouse_status={
            "ready": True,
            "market_bar_count": 5_000,
            "quality_failures": 0,
        },
    )


@pytest.mark.unit
def test_daily_briefing_prioritizes_risk_and_builds_actionable_checklist(
    warehouse_config,
):
    report = _briefing(warehouse_config)

    assert report["posture"] == "immediate_review"
    assert report["priorities"][0]["category"] == "paper_risk"
    assert report["priorities"][0]["severity"] == "critical"
    assert report["counts"]["urgent"] == 5
    assert report["summary"] == {
        "candidate_count": 1,
        "due_candidate_count": 1,
        "qualified_count": 1,
        "open_position_count": 1,
        "opportunity_alert_count": 1,
        "paper_alert_count": 1,
    }
    assert {item["target_id"] for item in report["priorities"]} >= {
        "paper-portfolio",
        "candidate-pipeline",
        "opportunity-scanner",
        "calibration-engine",
    }
    assert all(item["action_label"] for item in report["priorities"])
    assert report["checklist"][0]["count"] == 5
    assert report["notice"].startswith("Research workflow only")


@pytest.mark.unit
def test_daily_briefing_snapshots_are_content_addressed_and_reviewable(
    warehouse_config,
):
    report = _briefing(warehouse_config)
    first = daily_briefing.capture_daily_briefing(report, warehouse_config)
    second = daily_briefing.capture_daily_briefing(report, warehouse_config)

    assert first["briefing_id"] == second["briefing_id"]
    reviewed = daily_briefing.mark_briefing_reviewed(
        first["briefing_id"], warehouse_config
    )
    assert reviewed["reviewed_at"]

    latest = daily_briefing.get_latest_daily_briefing(warehouse_config)
    assert latest["registry"]["briefing_id"] == first["briefing_id"]
    assert latest["registry"]["reviewed_at"] == reviewed["reviewed_at"]
    history = daily_briefing.list_daily_briefings(warehouse_config)
    assert len(history) == 1
    assert history[0]["priority_count"] == report["counts"]["total"]


@pytest.mark.unit
def test_daily_briefing_exposes_missing_baselines_without_inventing_evidence(
    warehouse_config,
):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={},
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 0, "quality_failures": 0},
    )

    assert report["posture"] == "routine_review"
    assert [item["category"] for item in report["priorities"]] == [
        "workflow",
        "model_health",
    ]
    assert report["data_health"]["latest_scan_date"] is None
    assert report["summary"]["candidate_count"] == 0


@pytest.mark.unit
def test_daily_briefing_surfaces_saved_enrichment_flags(warehouse_config):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={
            "symbol": "NVDA",
            "observed_at": "2026-10-23T20:00:00+00:00",
            "coverage": {"score": 81, "label": "robust"},
            "decision_flags": [
                {
                    "type": "earnings_window",
                    "severity": "high",
                    "title": "Confirmed earnings window is near",
                    "message": "Earnings are due in five days.",
                }
            ],
            "registry": {"snapshot_id": "snapshot-1"},
        },
    )

    item = next(item for item in report["priorities"] if item["category"] == "enrichment")
    assert item["severity"] == "high"
    assert item["target_id"] == "data-enrichment-hub"
    assert report["enrichment"]["coverage_score"] == 81


@pytest.mark.unit
def test_daily_briefing_surfaces_enrichment_scheduler_health(warehouse_config):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={
            "active_schedule_count": 1,
            "healthy_source_count": 4,
            "degraded_source_count": 1,
            "unacknowledged_alert_count": 1,
            "stale_symbol_count": 1,
            "schedules": [],
            "alerts": [
                {
                    "alert_id": "alert-1",
                    "schedule_id": "schedule-1",
                    "symbol": "AAPL",
                    "dataset": "macro_calendar",
                    "alert_type": "source_unavailable",
                    "severity": "high",
                    "title": "Macro calendar unavailable",
                    "message": "AAPL returned no macro calendar data.",
                }
            ],
        },
    )

    priority = next(
        item
        for item in report["priorities"]
        if item.get("evidence", {}).get("alert_id") == "alert-1"
    )
    checklist = next(
        item for item in report["checklist"] if item["check_id"] == "enrichment_health"
    )
    assert priority["severity"] == "high"
    assert priority["target_id"] == "data-enrichment-hub"
    assert checklist["count"] == 1
    assert report["enrichment_operations"]["degraded_source_count"] == 1


@pytest.mark.unit
def test_daily_briefing_surfaces_unreviewed_research_pack(warehouse_config):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={},
        research_pack={
            "pack_id": "p" * 64,
            "status": "attention_required",
            "headline": "One high-priority update requires review",
            "summary": {
                "candidate_count": 1,
                "material_change_count": 4,
                "high_risk_count": 1,
                "ready_evidence_count": 1,
            },
            "registry": {"pack_id": "a" * 64, "reviewed_at": None},
        },
    )

    priority = next(
        item for item in report["priorities"] if item["category"] == "research_pack"
    )
    checklist = next(
        item
        for item in report["checklist"]
        if item["check_id"] == "research_pack_review"
    )
    assert priority["severity"] == "high"
    assert priority["target_id"] == "research-pack"
    assert checklist["count"] == 1
    assert report["research_pack"]["material_change_count"] == 4


@pytest.mark.unit
def test_daily_briefing_surfaces_critical_consensus_conflicts(warehouse_config):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={},
        research_pack={},
        consensus_status={
            "evaluated_symbol_count": 2,
            "blocked_symbol_count": 1,
            "critical_conflict_count": 2,
            "conflict_count": 3,
            "aligned_symbol_count": 1,
            "reports": [
                {"symbol": "AAPL", "qualification_allowed": False},
                {"symbol": "MSFT", "qualification_allowed": True},
            ],
        },
    )

    priority = next(
        item for item in report["priorities"] if item["category"] == "evidence_conflict"
    )
    checklist = next(
        item
        for item in report["checklist"]
        if item["check_id"] == "consensus_conflicts"
    )
    assert priority["severity"] == "high"
    assert priority["target_id"] == "consensus-engine"
    assert priority["evidence"]["symbols"] == ["AAPL"]
    assert checklist["count"] == 2
    assert report["consensus"]["blocked_symbol_count"] == 1


@pytest.mark.unit
def test_daily_briefing_surfaces_thesis_invalidations_without_changing_decisions(
    warehouse_config,
):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={},
        research_pack={},
        consensus_status={},
        thesis_status={
            "active_candidate_count": 2,
            "versioned_candidate_count": 2,
            "invalidated_count": 1,
            "warning_count": 1,
            "intact_count": 0,
            "unevaluated_count": 0,
            "reports": [
                {"symbol": "AAPL", "verdict": "invalidated"},
                {"symbol": "MSFT", "verdict": "warning"},
            ],
        },
    )

    priority = next(
        item
        for item in report["priorities"]
        if item["category"] == "thesis_invalidation"
    )
    checklist = next(
        item
        for item in report["checklist"]
        if item["check_id"] == "thesis_invalidations"
    )
    assert priority["severity"] == "high"
    assert priority["evidence"]["symbols"] == ["AAPL", "MSFT"]
    assert "does not change" in priority["rationale"]
    assert checklist["count"] == 2
    assert report["thesis_engine"]["invalidated_count"] == 1


@pytest.mark.unit
def test_daily_briefing_surfaces_unreviewed_decision_learning_gap(
    warehouse_config,
):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={},
        research_pack={},
        consensus_status={},
        thesis_status={},
        decision_review={
            "verdict": "learning_required",
            "coverage": {"matured_outcomes": 12},
            "summary": {
                "decision_accuracy": 0.5,
                "helpful_alert_count": 1,
                "false_alert_count": 3,
            },
            "registry": {"run_id": "review-1", "reviewed_at": None},
        },
    )

    priority = next(
        item for item in report["priorities"] if item["category"] == "decision_review"
    )
    checklist = next(
        item for item in report["checklist"] if item["check_id"] == "decision_review"
    )
    assert priority["severity"] == "high"
    assert priority["target_id"] == "decision-review-engine"
    assert priority["evidence"]["false_alert_count"] == 3
    assert checklist["count"] == 1
    assert report["decision_review"]["matured_outcomes"] == 12


@pytest.mark.unit
def test_daily_briefing_surfaces_promising_shadow_challenger(warehouse_config):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={},
        research_pack={},
        consensus_status={},
        thesis_status={},
        decision_review={},
        shadow_lab_status={
            "experiment_count": 2,
            "active_count": 1,
            "evaluated_count": 2,
            "actionable_unreviewed_count": 1,
            "challenger_promising_count": 1,
            "control_leads_count": 0,
        },
    )

    priority = next(
        item for item in report["priorities"] if item["category"] == "shadow_experiment"
    )
    checklist = next(
        item for item in report["checklist"] if item["check_id"] == "shadow_experiments"
    )
    assert priority["severity"] == "high"
    assert priority["target_id"] == "shadow-lab"
    assert checklist["count"] == 1
    assert report["shadow_lab"]["challenger_promising_count"] == 1


@pytest.mark.unit
def test_daily_briefing_surfaces_promotion_eligible_governance(warehouse_config):
    report = daily_briefing.build_daily_briefing(
        "2026-10-23",
        warehouse_config,
        candidate_payload={"items": []},
        opportunity_alerts=[],
        portfolio={"open_positions": [], "risk_alerts": []},
        calibration={},
        scan_state={
            "scan_id": "scan-1",
            "as_of_date": "2026-10-23",
            "eligible_symbols": 10,
            "errors": [],
        },
        schedules=[],
        warehouse_status={"ready": True, "market_bar_count": 10, "quality_failures": 0},
        enrichment={},
        enrichment_operations={},
        research_pack={},
        consensus_status={},
        thesis_status={},
        decision_review={},
        shadow_lab_status={},
        governance_status={
            "proposal_count": 1,
            "pending_approval_count": 0,
            "probation_count": 1,
            "promotion_eligible_count": 1,
            "promoted_count": 0,
            "rollback_available_count": 0,
            "active_policies": [],
        },
    )

    priority = next(
        item for item in report["priorities"] if item["category"] == "model_governance"
    )
    checklist = next(
        item for item in report["checklist"] if item["check_id"] == "model_governance"
    )
    assert priority["severity"] == "high"
    assert priority["target_id"] == "model-governance"
    assert checklist["count"] == 1
    assert report["model_governance"]["promotion_eligible_count"] == 1
