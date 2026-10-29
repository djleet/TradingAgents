from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import validation_lab
from tradingagents.dataflows.errors import NoMarketDataError


def _frame(periods: int = 1300, *, benchmark: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-01-30", periods=periods)
    index = pd.Series(range(periods), dtype=float)
    amplitude = 7 if benchmark else 16
    close = 80 + index.mul(0.05) + index.map(lambda value: math.sin(value / 19) * amplitude)
    adjusted = close.mul(0.98)
    return pd.DataFrame(
        {
            "Date": dates.date,
            "Open": close - 0.3,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": adjusted,
            "Volume": 1_000_000,
        }
    )


@pytest.mark.unit
def test_validation_uses_strict_rolling_folds_and_adjusted_prices():
    result = validation_lab.validate_signal_frames(
        _frame(),
        _frame(benchmark=True),
        "AAPL",
        "SPY",
        "2026-01-30",
        sessions=504,
        train_sessions=252,
        test_sessions=63,
        mode="long_short",
    )

    assert result["oos_sessions"] == 504
    assert result["price_basis"] == {"asset": "adjusted_close", "benchmark": "adjusted_close"}
    assert len(result["folds"]) == 8
    assert all(fold["train_end"] < fold["test_start"] for fold in result["folds"])
    assert all(fold["selected_threshold"] in validation_lab.THRESHOLD_CANDIDATES for fold in result["folds"])
    assert sum(item["sessions"] for item in result["regimes"]) == 504
    assert result["points"][0]["strategy_equity"] == 1.0
    assert result["robustness"] in {"promising", "mixed", "weak"}


@pytest.mark.unit
def test_validation_cost_model_charges_execution_and_short_borrow():
    result = validation_lab.validate_signal_frames(
        _frame(),
        _frame(benchmark=True),
        "TEST",
        "SPY",
        "2026-01-30",
        sessions=378,
        train_sessions=252,
        test_sessions=63,
        mode="long_short",
        commission_bps=5,
        slippage_bps=7,
        spread_bps=8,
        short_borrow_bps=500,
    )

    assert result["metrics"]["execution_cost"] > 0
    assert result["metrics"]["borrow_cost"] > 0
    assert result["configuration"]["short_borrow_bps"] == 500
    assert "unseen test block" in result["methodology"]


@pytest.mark.unit
def test_validation_fetches_asset_and_benchmark_to_same_cutoff(monkeypatch):
    calls = []

    def fake_vendor(method, symbol, start_date, end_date):
        calls.append((method, symbol, start_date, end_date))
        return _frame(900, benchmark=symbol == "SPY").to_csv(index=False)

    monkeypatch.setattr(validation_lab, "route_to_vendor", fake_vendor)
    result = validation_lab.build_validation_report(
        "AAPL", "SPY", "2026-01-30", sessions=252, train_sessions=252, test_sessions=63
    )

    assert [call[1] for call in calls] == ["AAPL", "SPY"]
    assert all(call[-1] == "2026-01-30" for call in calls)
    assert result["benchmark_symbol"] == "SPY"


@pytest.mark.unit
def test_validation_rejects_insufficient_aligned_history():
    with pytest.raises(NoMarketDataError, match="aligned bars"):
        validation_lab.validate_signal_frames(
            _frame(200),
            _frame(200, benchmark=True),
            "TEST",
            "SPY",
            "2026-01-30",
            train_sessions=252,
            test_sessions=63,
        )
