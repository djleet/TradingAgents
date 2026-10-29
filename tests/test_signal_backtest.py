from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import signal_backtest
from tradingagents.dataflows.errors import NoMarketDataError


def _frame(periods: int = 700, *, oscillating: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-01-30", periods=periods)
    index = pd.Series(range(periods), dtype=float)
    if oscillating:
        close = 100 + index.mul(0.04) + index.map(lambda value: math.sin(value / 8) * 12)
    else:
        close = 80 + index.mul(0.18)
    return pd.DataFrame(
        {
            "Date": dates.date,
            "Open": close - 0.2,
            "High": close + 0.8,
            "Low": close - 0.7,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


@pytest.mark.unit
def test_walk_forward_positions_lag_scores_and_curves_are_normalized():
    result = signal_backtest.simulate_signal_backtest(
        _frame(), "AAPL", "2026-01-30", sessions=252, fee_bps=5
    )

    points = result["points"]
    assert len(points) == 252
    assert points[0]["strategy_equity"] == 1.0
    assert points[0]["benchmark_equity"] == 1.0
    assert all(
        current["position"] == previous["desired_signal"]
        for previous, current in zip(points, points[1:], strict=False)
    )
    assert result["metrics"]["exposure"] > 0
    assert "next session" in result["methodology"]
    assert "not a forecast" in result["limitations"]


@pytest.mark.unit
def test_fees_reduce_results_when_signal_changes():
    frame = _frame(oscillating=True)
    free = signal_backtest.simulate_signal_backtest(
        frame, "TEST", "2026-01-30", sessions=504, fee_bps=0, mode="long_short"
    )
    charged = signal_backtest.simulate_signal_backtest(
        frame, "TEST", "2026-01-30", sessions=504, fee_bps=25, mode="long_short"
    )

    assert charged["metrics"]["trade_count"] > 0
    assert charged["metrics"]["estimated_fees"] > 0
    assert charged["metrics"]["total_return"] < free["metrics"]["total_return"]


@pytest.mark.unit
def test_adjusted_prices_and_realistic_costs_are_reported():
    frame = _frame(oscillating=True)
    frame["Adj Close"] = frame["Close"] * 0.95

    result = signal_backtest.simulate_signal_backtest(
        frame,
        "TEST",
        "2026-01-30",
        sessions=252,
        fee_bps=2,
        slippage_bps=3,
        spread_bps=4,
        short_borrow_bps=300,
        mode="long_short",
    )

    assert result["price_basis"] == "adjusted_close"
    assert result["metrics"]["slippage_bps"] == 3
    assert result["metrics"]["spread_bps"] == 4
    assert result["metrics"]["estimated_fees"] > 0


@pytest.mark.unit
def test_backtest_fetch_is_point_in_time(monkeypatch):
    calls = []
    monkeypatch.setattr(
        signal_backtest,
        "route_to_vendor",
        lambda *args: calls.append(args) or _frame(500).to_csv(index=False),
    )

    result = signal_backtest.build_signal_backtest(
        "MSFT", "2026-01-30", sessions=252, fee_bps=7, mode="long_only"
    )

    assert calls[0][0] == "get_stock_data"
    assert calls[0][1] == "MSFT"
    assert calls[0][-1] == "2026-01-30"
    assert result["metrics"]["fee_bps"] == 7


@pytest.mark.unit
def test_backtest_rejects_short_history():
    with pytest.raises(NoMarketDataError, match="at least 90 bars"):
        signal_backtest.simulate_signal_backtest(
            _frame(60), "TEST", "2026-01-30", sessions=252
        )
