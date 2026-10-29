from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows import context_validation, research_context

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def validation_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market.duckdb"),
    }


def _opportunity(score=78.0):
    return {
        "opportunity_score": score,
        "confidence": 0.9,
        "quality": {"score": 94.0},
        "validation": {
            "robustness": "promising",
            "sharpe_ratio": 1.2,
            "profitable_fold_rate": 0.75,
            "threshold_stability": 0.85,
        },
    }


def _context(fundamental=74.0, event_risk=12, coverage=0.9):
    return {
        "fundamentals": {"score": fundamental},
        "events": {"risk_score": event_risk},
        "overlay": {"data_coverage": coverage},
    }


@pytest.mark.unit
def test_decision_confidence_is_separate_explainable_and_coverage_aware():
    result = research_context.build_decision_confidence(
        _opportunity(), _context()
    )

    assert result["score"] >= 70
    assert result["label"] in {"high_conviction", "supported"}
    assert result["evidence_coverage"] == 1.0
    assert set(result["components"]) == {
        "opportunity_evidence",
        "rolling_validation",
        "fundamental_support",
        "event_resilience",
        "market_data_quality",
        "evidence_depth",
    }
    assert "unchanged opportunity score" in result["methodology"]

    caution = research_context.build_decision_confidence(
        _opportunity(82), _context(fundamental=30, event_risk=65)
    )
    assert caution["label"] == "caution"
    assert len(caution["disagreements"]) == 2


@pytest.mark.unit
def test_context_validation_summary_measures_incremental_lift():
    observations = []
    for index in range(8):
        qualified = index < 4
        observations.append(
            {
                "symbol": f"S{index % 4}",
                "passes_filters": True,
                "opportunity_score": 65,
                "fundamental_score": 70 if qualified else 40,
                "event_risk": 10,
                "context_coverage": 0.9,
                "decision_confidence": 72 if qualified else 50,
                "decision_label": "supported" if qualified else "mixed",
                "forward_return": 0.06 if qualified else 0.0,
                "excess_return": 0.06 if qualified else 0.0,
            }
        )

    result = context_validation.summarize_context_validation(observations)
    cohorts = {item["cohort"]: item for item in result["cohorts"]}

    assert cohorts["scanner_selected"]["observations"] == 8
    assert cohorts["context_qualified"]["observations"] == 4
    assert cohorts["decision_supported"]["observations"] == 4
    assert result["lift"]["decision_vs_scanner_excess_return"] == pytest.approx(0.03)
    assert result["verdict"] == "promising"


@pytest.mark.unit
def test_context_validation_replays_and_persists_experiment(
    validation_config, monkeypatch
):
    dates = pd.bdate_range(end="2026-01-30", periods=900)
    frame = pd.DataFrame(
        {
            "Date": dates.date,
            "Open": range(100, 1000),
            "High": range(101, 1001),
            "Low": range(99, 999),
            "Close": range(100, 1000),
            "Adj Close": range(100, 1000),
            "Volume": [2_000_000] * 900,
        }
    )
    monkeypatch.setattr(
        context_validation,
        "route_to_vendor",
        lambda *args: frame.to_csv(index=False),
    )
    monkeypatch.setattr(
        context_validation,
        "rank_opportunity",
        lambda symbol, *args, **kwargs: {
            **_opportunity(70 if symbol == "AAPL" else 60),
            "symbol": symbol,
            "passes_filters": True,
            "bucket": "watch",
        },
    )

    def fake_context(symbol, as_of, config, *, opportunity, closing_price):
        result = {
            "symbol": symbol,
            "as_of_date": as_of,
            **_context(fundamental=68, event_risk=10, coverage=0.9),
        }
        result["decision_confidence"] = research_context.build_decision_confidence(
            opportunity, result
        )
        return result

    monkeypatch.setattr(context_validation, "build_research_context", fake_context)

    report = context_validation.build_context_validation(
        ["AAPL", "MSFT"],
        "SPY",
        "2026-01-30",
        validation_config,
        checkpoints=3,
        forward_sessions=21,
    )

    assert len(report["observations"]) == 6
    assert len(report["checkpoints"]) == 3
    assert report["registry"]["saved"] is True
    history = context_validation.list_context_validations(validation_config)
    assert history[0]["experiment_id"] == report["registry"]["experiment_id"]
    assert history[0]["checkpoint_count"] == 3
