from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import opportunity_scanner


def _frame(
    periods: int = 1100,
    *,
    amplitude: float = 12,
    daily_drift: float = 0.06,
    volume: float = 2_000_000,
) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-01-30", periods=periods)
    index = pd.Series(range(periods), dtype=float)
    close = 75 + index.mul(daily_drift) + index.map(
        lambda value: math.sin(value / 18) * amplitude
    )
    return pd.DataFrame(
        {
            "Date": dates.date,
            "Open": close - 0.2,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": volume,
        }
    )


@pytest.mark.unit
def test_rank_opportunity_combines_signal_validation_and_quality(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.quantitative.capture_quantitative_snapshot",
        lambda _: False,
    )

    item = opportunity_scanner.rank_opportunity(
        "aapl",
        _frame(),
        _frame(amplitude=6, daily_drift=0.04),
        "2026-01-30",
        validation_sessions=252,
        train_sessions=252,
        test_sessions=63,
    )

    assert item["symbol"] == "AAPL"
    assert 0 <= item["opportunity_score"] <= 100
    assert item["passes_filters"] is True
    assert item["quality"]["score"] >= 90
    assert item["validation"]["available"] is True
    assert item["validation"]["fold_count"] == 4
    assert {"signal", "quality", "liquidity", "oos_sharpe"}.issubset(
        item["score_components"]
    )
    assert item["bucket"] in {"top_opportunity", "watch", "neutral", "avoid"}
    assert 1 <= len(item["reasons"]) <= 3


@pytest.mark.unit
def test_rank_opportunity_excludes_assets_below_liquidity_floor(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.quantitative.capture_quantitative_snapshot",
        lambda _: False,
    )

    item = opportunity_scanner.rank_opportunity(
        "thin",
        _frame(volume=100),
        _frame(amplitude=6),
        "2026-01-30",
        min_dollar_volume=5_000_000,
    )

    assert item["bucket"] == "excluded"
    assert item["passes_filters"] is False
    assert item["reasons"] == ["Below the selected liquidity floor"]


@pytest.mark.unit
def test_scan_ranks_partial_results_and_fetches_benchmark_once(monkeypatch):
    calls: list[str] = []

    def fake_vendor(method, symbol, start_date, end_date):
        assert method == "get_stock_data"
        assert end_date == "2026-01-30"
        calls.append(symbol)
        if symbol == "FAIL":
            raise RuntimeError("provider rejected symbol")
        return _frame(amplitude=6 if symbol == "SPY" else 12).to_csv(index=False)

    def fake_rank(symbol, *_args, **_kwargs):
        return {
            "symbol": symbol,
            "opportunity_score": {"AAPL": 72, "SPY": 55}[symbol],
            "confidence": 0.8,
            "passes_filters": True,
            "bucket": "top_opportunity" if symbol == "AAPL" else "watch",
        }

    monkeypatch.setattr(opportunity_scanner, "route_to_vendor", fake_vendor)
    monkeypatch.setattr(opportunity_scanner, "rank_opportunity", fake_rank)

    report = opportunity_scanner.scan_opportunities(
        ["aapl", "spy", "fail", "AAPL"],
        "2026-01-30",
        benchmark_symbol="SPY",
        max_workers=1,
    )

    assert calls.count("SPY") == 1
    assert calls == ["SPY", "AAPL", "FAIL"]
    assert [item["symbol"] for item in report["items"]] == ["AAPL", "SPY"]
    assert [item["rank"] for item in report["items"]] == [1, 2]
    assert report["requested_symbols"] == 3
    assert report["scanned_symbols"] == 2
    assert report["eligible_symbols"] == 2
    assert report["bucket_counts"]["top_opportunity"] == 1
    assert report["errors"] == [
        {"symbol": "FAIL", "message": "Point-in-time evidence unavailable"}
    ]


@pytest.mark.unit
@pytest.mark.parametrize("symbols", [["AAPL"], [f"S{index}" for index in range(501)]])
def test_scan_rejects_unsupported_universe_sizes(symbols):
    with pytest.raises(ValueError, match="between 2 and 500"):
        opportunity_scanner.scan_opportunities(symbols, "2026-01-30")
