from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import portfolio_risk


def _price_matrix() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-23", periods=180).date
    aaa = []
    bbb = []
    ccc = []
    spy = []
    values = [100.0, 80.0, 65.0, 400.0]
    for index in range(180):
        market_move = 0.0007 + math.sin(index / 6) * 0.007
        moves = [
            market_move + math.sin(index / 3) * 0.002,
            market_move * 0.85 + math.cos(index / 5) * 0.003,
            market_move * 0.25 + math.sin(index / 2) * 0.011,
            market_move,
        ]
        values = [value * (1 + move) for value, move in zip(values, moves, strict=True)]
        aaa.append(values[0])
        bbb.append(values[1])
        ccc.append(values[2])
        spy.append(values[3])
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc, "SPY": spy}, index=dates)


@pytest.mark.unit
def test_portfolio_risk_is_complete_point_in_time_and_cash_aware(monkeypatch):
    monkeypatch.setattr(portfolio_risk, "load_price_matrix", lambda *args, **kwargs: _price_matrix())

    result = portfolio_risk.analyze_portfolio_risk(
        {"aaa": 0.40, "bbb": 0.25, "ccc": 0.15},
        "2026-10-23",
        portfolio_value=100_000,
    )

    assert result["last_bar_date"] == "2026-10-23"
    assert result["invested_weight"] == pytest.approx(0.80)
    assert result["cash_weight"] == pytest.approx(0.20)
    assert result["summary"]["annualized_volatility"] > 0
    assert result["summary"]["expected_shortfall"] >= result["summary"]["historical_var"]
    assert sum(item["risk_contribution"] for item in result["positions"]) == pytest.approx(
        1.0, abs=1e-5
    )
    assert sum(
        item["suggested_weight"]
        for item in result["risk_balanced"]["recommendations"]
    ) == pytest.approx(0.80, abs=1e-5)
    assert result["correlation"]["AAA"]["AAA"] == 1.0
    assert len(result["scenarios"]) == 4
    assert all(item["impact"] <= 0 for item in result["scenarios"])
    assert result["scenarios"][0]["impact_value"] < 0


@pytest.mark.unit
def test_portfolio_risk_emits_explainable_budget_alerts(monkeypatch):
    monkeypatch.setattr(portfolio_risk, "load_price_matrix", lambda *args, **kwargs: _price_matrix())

    result = portfolio_risk.analyze_portfolio_risk(
        {"AAA": 0.70, "BBB": 0.20, "CCC": 0.10},
        "2026-10-23",
        max_position_weight=0.25,
        annual_volatility_limit=0.05,
        daily_es_limit=0.005,
    )

    alert_types = {item["type"] for item in result["alerts"]}
    assert "position_weight" in alert_types
    assert result["overall_level"] in {"watch", "high"}
    aaa = next(item for item in result["risk_balanced"]["recommendations"] if item["symbol"] == "AAA")
    assert aaa["change"] < 0
    assert aaa["reason"].startswith("Reduce")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"AAA": 1.0}, "2-30"),
        ({"AAA": 0.8, "BBB": 0.4}, "100%"),
        ({"AAA": -0.1, "BBB": 0.5}, "non-negative"),
    ],
)
def test_portfolio_risk_rejects_invalid_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        portfolio_risk.analyze_portfolio_risk(weights, "2026-10-23")
