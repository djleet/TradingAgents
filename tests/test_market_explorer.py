from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows import market_explorer


def _payload(periods: int = 320) -> str:
    dates = pd.bdate_range(end="2026-01-30", periods=periods)
    close = pd.Series(range(periods), dtype=float) * 0.25 + 100
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.7,
            "Close": close,
            "Volume": 1_000_000 + pd.Series(range(periods)) * 100,
        }
    ).to_csv(index=False)


@pytest.mark.unit
def test_market_chart_contains_point_in_time_bars_and_indicators(monkeypatch):
    monkeypatch.setattr(market_explorer, "route_to_vendor", lambda *args: _payload())
    monkeypatch.setattr(
        "tradingagents.dataflows.quantitative.capture_quantitative_snapshot",
        lambda snapshot: False,
    )

    result = market_explorer.build_market_chart("AAPL", "2026-01-30", sessions=90)

    assert result["symbol"] == "AAPL"
    assert result["visible_sessions"] == 90
    assert result["bars"][-1]["date"] == "2026-01-30"
    assert result["bars"][-1]["sma20"] is not None
    assert result["bars"][-1]["sma50"] is not None
    assert result["snapshot"]["label"] == "bullish"


@pytest.mark.unit
def test_comparison_keeps_successes_when_one_symbol_has_no_data(monkeypatch):
    def fake_snapshot(symbol, *args, **kwargs):
        if symbol == "BAD":
            raise ValueError("missing")
        return {
            "symbol": symbol,
            "latest_bar_date": "2026-01-30",
            "latest_close": 100.0,
            "score": 25.0,
            "label": "bullish",
            "confidence": 0.8,
            "risk_regime": "normal",
            "metrics": {
                "momentum_20": 0.1,
                "momentum_60": 0.2,
                "annualized_volatility_60": 0.25,
                "max_drawdown_252": -0.1,
                "rsi_14": 60.0,
            },
        }

    monkeypatch.setattr(market_explorer, "compute_quantitative_snapshot", fake_snapshot)

    result = market_explorer.compare_market_snapshots(["AAA", "BAD"], "2026-01-30")

    assert [item["symbol"] for item in result["items"]] == ["AAA"]
    assert result["errors"] == [
        {"symbol": "BAD", "message": "Point-in-time market data unavailable"}
    ]
