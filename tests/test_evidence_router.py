from __future__ import annotations

from datetime import date

import pytest

from tradingagents.dataflows import candidate_pipeline, enrichment_hub, evidence_router
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _source(dataset, data, *, point_in_time, status="available"):
    return {
        "dataset": dataset,
        "provider": "Test provider",
        "authority": "official" if point_in_time else "aggregated_analyst",
        "status": status,
        "completeness": 1.0 if status == "available" else 0.0,
        "point_in_time": point_in_time,
        "caveat": "Test caveat",
        "error": None,
        "data": data,
    }


def _snapshot(config):
    report = enrichment_hub.build_enrichment_snapshot(
        "AAPL",
        "2000-01-01",
        config,
        source_payloads={
            "earnings_estimates": _source(
                "earnings_estimates",
                {"target_upside": 0.424242, "earnings_date": "2000-02-01"},
                point_in_time=False,
            ),
            "options_implied_volatility": _source(
                "options_implied_volatility",
                {"atm_iv": 0.333333, "nearest_expiry": "2000-02-18"},
                point_in_time=False,
            ),
            "institutional_ownership": _source(
                "institutional_ownership",
                {"institutional_holders": [{"Holder": "Fund"}]},
                point_in_time=False,
            ),
            "sec_insider_filings": _source(
                "sec_insider_filings",
                {"filings": [{"filing_date": "1999-12-30"}]},
                point_in_time=True,
            ),
            "macro_calendar": _source(
                "macro_calendar",
                {"releases": [{"name": "CPI", "date": "2000-01-10"}]},
                point_in_time=True,
            ),
        },
    )
    enrichment_hub.capture_enrichment_snapshot(report, config)
    return report


@pytest.mark.unit
def test_router_blocks_live_sources_after_historical_cutoff(warehouse_config):
    _snapshot(warehouse_config)

    bundle = evidence_router.build_evidence_bundle(
        "AAPL", "2000-01-01", warehouse_config, purpose="backtest"
    )

    assert {item["dataset"] for item in bundle["items"]} == {
        "sec_insider_filings",
        "macro_calendar",
    }
    assert {item["dataset"] for item in bundle["blocked"]} == {
        "earnings_estimates",
        "options_implied_volatility",
        "institutional_ownership",
    }
    assert not set(bundle["missing"]).intersection(
        {"earnings_estimates", "options_implied_volatility", "institutional_ownership"}
    )
    rendered = evidence_router.render_evidence_context(bundle)
    assert "0.4242" not in rendered
    assert "Blocked by the temporal gate" in rendered


@pytest.mark.unit
def test_router_allows_live_snapshot_once_observation_date_is_known(warehouse_config):
    _snapshot(warehouse_config)

    bundle = evidence_router.build_evidence_bundle(
        "AAPL", date.today().isoformat(), warehouse_config, purpose="analysis"
    )

    datasets = {item["dataset"] for item in bundle["items"]}
    assert set(enrichment_hub.DATASETS).issubset(datasets)
    assert bundle["blocked_count"] == 0
    assert "target upside=0.4242" in evidence_router.render_evidence_context(bundle)


@pytest.mark.unit
def test_router_persists_deduplicated_audit_history(warehouse_config):
    _snapshot(warehouse_config)
    first = evidence_router.build_evidence_bundle(
        "AAPL", "2000-01-01", warehouse_config, purpose="backtest"
    )
    second = evidence_router.build_evidence_bundle(
        "AAPL", "2000-01-01", warehouse_config, purpose="backtest"
    )

    assert first["bundle_id"] == second["bundle_id"]
    history = evidence_router.list_evidence_bundles(
        warehouse_config, symbol="AAPL", purpose="backtest"
    )
    assert len(history) == 1
    latest = evidence_router.get_latest_evidence_bundle(
        warehouse_config, symbol="AAPL", purpose="backtest"
    )
    assert latest["bundle_id"] == first["bundle_id"]


@pytest.mark.unit
def test_candidate_attachment_is_audited_only_when_bundle_changes(warehouse_config):
    candidate = candidate_pipeline.upsert_candidate(
        "AAPL", {"opportunity_score": 70, "rank": 1}, warehouse_config
    )
    bundle = evidence_router.build_evidence_bundle(
        "AAPL", date.today().isoformat(), warehouse_config, purpose="candidate"
    )

    attached = candidate_pipeline.attach_candidate_evidence(
        candidate["candidate_id"], bundle, warehouse_config
    )
    candidate_pipeline.attach_candidate_evidence(
        candidate["candidate_id"], bundle, warehouse_config
    )
    history = candidate_pipeline.list_candidate_history(
        candidate["candidate_id"], warehouse_config
    )

    assert attached["evidence"]["evidence_router"]["bundle_id"] == bundle["bundle_id"]
    assert [item["event_type"] for item in history].count("evidence_routed") == 1


@pytest.mark.unit
def test_graph_run_state_injects_routed_context(monkeypatch):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"data_warehouse_enabled": False}
    graph.propagator = Propagator()
    graph.resolve_instrument_context = lambda *args: "Instrument: AAPL"
    bundle = {
        "analysis_date": "2000-01-01",
        "verdict": "partial",
        "coverage_score": 20,
        "items": [],
        "blocked": [],
        "missing": [],
    }
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.build_evidence_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.render_evidence_context",
        lambda routed: "ROUTED CONTEXT",
    )

    state = graph.create_run_state("AAPL", "2000-01-01")

    assert state["evidence_bundle"] is bundle
    assert state["instrument_context"] == "Instrument: AAPL\n\nROUTED CONTEXT"
