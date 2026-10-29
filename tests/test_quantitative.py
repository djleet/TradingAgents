from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows import quantitative
from tradingagents.dataflows.errors import NoMarketDataError


def _payload(periods: int = 300, *, rising: bool = True) -> str:
    dates = pd.bdate_range(end="2026-01-30", periods=periods)
    close = pd.Series(range(periods), dtype=float)
    close = 100 + close * (0.4 if rising else -0.15)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    return frame.to_csv(index=False)


@pytest.mark.unit
def test_quantitative_snapshot_is_point_in_time_and_audited(monkeypatch):
    stored = []
    monkeypatch.setattr(quantitative, "route_to_vendor", lambda *args: _payload())
    monkeypatch.setattr(quantitative, "capture_quantitative_snapshot", stored.append)

    snapshot = quantitative.compute_quantitative_snapshot("aapl", "2026-01-30")

    assert snapshot["symbol"] == "AAPL"
    assert snapshot["latest_bar_date"] == "2026-01-30"
    assert snapshot["label"] == "bullish"
    assert snapshot["score"] >= 20
    assert snapshot["bars_used"] == 300
    assert stored == [snapshot]


@pytest.mark.unit
def test_existing_frame_can_be_scored_without_another_vendor_call(monkeypatch):
    stored = []
    monkeypatch.setattr(quantitative, "capture_quantitative_snapshot", stored.append)
    frame = quantitative.parse_ohlcv_payload(_payload())

    snapshot = quantitative.calculate_quantitative_snapshot(frame, "AAPL", "2026-01-30")

    assert snapshot["bars_used"] == 300
    assert stored == [snapshot]


@pytest.mark.unit
def test_historical_score_series_matches_latest_snapshot(monkeypatch):
    monkeypatch.setattr(quantitative, "capture_quantitative_snapshot", lambda _: False)
    frame = quantitative.parse_ohlcv_payload(_payload())

    snapshot = quantitative.calculate_quantitative_snapshot(frame, "AAPL", "2026-01-30")
    scores = quantitative.quantitative_score_series(frame["Close"])

    assert scores.iloc[-1]["score"] == pytest.approx(snapshot["score"], abs=0.01)


@pytest.mark.unit
def test_quantitative_snapshot_rejects_an_insufficient_history(monkeypatch):
    monkeypatch.setattr(quantitative, "route_to_vendor", lambda *args: _payload(20))

    with pytest.raises(NoMarketDataError, match="at least 30 bars"):
        quantitative.compute_quantitative_snapshot("AAPL", "2026-01-30")


@pytest.mark.unit
def test_quantitative_report_explains_scope(monkeypatch):
    monkeypatch.setattr(quantitative, "route_to_vendor", lambda *args: _payload())
    monkeypatch.setattr(quantitative, "capture_quantitative_snapshot", lambda _: False)

    report = quantitative.build_quantitative_report("AAPL", "2026-01-30")

    assert "Independent quantitative snapshot" in report
    assert "deterministic cross-check" in report
    assert "not as a prediction" in report
