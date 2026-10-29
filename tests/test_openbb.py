from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import openbb


class _Result:
    def __init__(self, frame):
        self.frame = frame

    def to_df(self):
        return self.frame


def _client(frame, captured):
    def historical(**kwargs):
        captured.update(kwargs)
        return _Result(frame)

    return SimpleNamespace(
        equity=SimpleNamespace(price=SimpleNamespace(historical=historical))
    )


@pytest.mark.unit
def test_openbb_prices_are_normalized_and_bounded_by_the_query(monkeypatch):
    frame = pd.DataFrame(
        {
            "open": [10, 11, 12],
            "high": [11, 12, 13],
            "low": [9, 10, 11],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 200, 300],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    frame.index.name = "date"
    captured = {}
    monkeypatch.setattr(openbb, "_load_obb", lambda: _client(frame, captured))

    output = openbb.get_stock("AAPL", "2026-01-01", "2026-01-02")

    assert captured == {
        "symbol": "AAPL",
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
    }
    assert output.splitlines()[0] == "Date,Open,High,Low,Close,Volume"
    assert "2026-01-02" in output
    assert "2026-01-03" not in output


@pytest.mark.unit
def test_openbb_empty_window_is_typed_as_no_market_data(monkeypatch):
    frame = pd.DataFrame({"date": ["2025-01-01"], "close": [10]})
    monkeypatch.setattr(openbb, "_load_obb", lambda: _client(frame, {}))
    with pytest.raises(NoMarketDataError, match="no rows between"):
        openbb.get_stock("AAPL", "2026-01-01", "2026-01-02")

