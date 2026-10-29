from __future__ import annotations

import pytest

from tradingagents.dataflows import enrichment_hub

pytest.importorskip("duckdb")


@pytest.fixture
def warehouse_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market_data.duckdb"),
    }


def _source(
    dataset: str,
    data: dict,
    *,
    provider: str = "Test source",
    authority: str = "official",
    point_in_time: bool = True,
    status: str = "available",
    completeness: float = 1.0,
):
    return {
        "dataset": dataset,
        "provider": provider,
        "authority": authority,
        "status": status,
        "completeness": completeness,
        "point_in_time": point_in_time,
        "caveat": "Test evidence policy.",
        "error": None,
        "data": data,
    }


def _payloads():
    return {
        "earnings_estimates": _source(
            "earnings_estimates",
            {
                "earnings_date": "2026-10-29",
                "price_targets": {"current": 100, "mean": 120},
                "target_upside": 0.2,
            },
            provider="Yahoo Finance",
            authority="aggregated_analyst",
            point_in_time=False,
        ),
        "options_implied_volatility": _source(
            "options_implied_volatility",
            {
                "nearest_expiry": "2026-11-20",
                "atm_iv": 0.52,
                "expected_move_pct": 0.11,
                "put_call_open_interest": 1.8,
            },
            provider="Yahoo Finance",
            authority="exchange_derived",
            point_in_time=False,
        ),
        "institutional_ownership": _source(
            "institutional_ownership",
            {"institutional_holders": [{"Holder": "Fund A"}]},
            authority="aggregated_filings",
            point_in_time=False,
        ),
        "sec_insider_filings": _source(
            "sec_insider_filings",
            {"filing_count": 2, "recent_30d_count": 1, "filings": []},
        ),
        "macro_calendar": _source(
            "macro_calendar",
            {
                "indicators": [{"key": "vix", "value": 19.5}],
                "releases": [{"name": "Consumer Price Index", "date": "2026-11-12"}],
            },
        ),
    }


@pytest.mark.unit
def test_enrichment_hub_scores_sources_and_keeps_live_data_out_of_history(
    warehouse_config,
):
    report = enrichment_hub.build_enrichment_snapshot(
        "aapl",
        "2026-10-23",
        warehouse_config,
        source_payloads=_payloads(),
    )

    assert report["symbol"] == "AAPL"
    assert report["coverage"]["label"] == "robust"
    assert report["coverage"]["available_sources"] == 5
    assert report["coverage"]["point_in_time_sources"] == 2
    assert set(report["policy"]["live_only"]) == {
        "earnings_estimates",
        "options_implied_volatility",
        "institutional_ownership",
    }
    assert set(report["policy"]["historical_eligible"]) == {
        "sec_insider_filings",
        "macro_calendar",
    }
    assert {item["type"] for item in report["decision_flags"]} >= {
        "earnings_window",
        "options_expected_move",
        "put_call_skew",
    }
    assert all(0 <= item["confidence_score"] <= 100 for item in report["sources"])


@pytest.mark.unit
def test_enrichment_hub_persists_content_addressed_snapshots(warehouse_config):
    report = enrichment_hub.build_enrichment_snapshot(
        "MSFT", "2026-10-23", warehouse_config, source_payloads=_payloads()
    )
    first = enrichment_hub.capture_enrichment_snapshot(report, warehouse_config)
    second = enrichment_hub.capture_enrichment_snapshot(report, warehouse_config)

    assert first["snapshot_id"] == second["snapshot_id"]
    latest = enrichment_hub.get_latest_enrichment_snapshot(
        warehouse_config, symbol="msft"
    )
    assert latest["registry"]["snapshot_id"] == first["snapshot_id"]
    assert latest["coverage"]["label"] == "robust"
    history = enrichment_hub.list_enrichment_snapshots(
        warehouse_config, symbol="MSFT"
    )
    assert len(history) == 1
    assert history[0]["available_sources"] == 5


@pytest.mark.unit
def test_enrichment_hub_preserves_explicit_source_gaps(warehouse_config):
    payloads = _payloads()
    payloads["macro_calendar"] = {
        "dataset": "macro_calendar",
        "provider": "FRED/ALFRED",
        "authority": "official",
        "status": "unavailable",
        "completeness": 0,
        "point_in_time": True,
        "caveat": "Add FRED_API_KEY.",
        "error": "FRED is not configured.",
        "data": {},
    }
    report = enrichment_hub.build_enrichment_snapshot(
        "NVDA", "2026-10-23", warehouse_config, source_payloads=payloads
    )

    assert report["coverage"]["unavailable_sources"] == 1
    macro = next(item for item in report["sources"] if item["dataset"] == "macro_calendar")
    assert macro["confidence_score"] == 0
    assert macro["error"] == "FRED is not configured."
    assert any(item["type"] == "coverage_gap" for item in report["decision_flags"])

