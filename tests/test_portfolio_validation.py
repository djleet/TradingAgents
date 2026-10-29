from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows import portfolio_validation


def _prices(periods: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-10-23", periods=periods).date
    values = {"AAA": 100.0, "BBB": 80.0, "CCC": 60.0, "SPY": 400.0}
    rows = []
    for index, date in enumerate(dates):
        market = 0.00045 + math.sin(index / 11) * 0.006
        moves = {
            "AAA": market * 0.75 + math.sin(index / 4) * 0.003,
            "BBB": market * 1.05 + math.cos(index / 7) * 0.005,
            "CCC": market * 0.35 + math.sin(index / 3) * 0.009,
            "SPY": market,
        }
        for symbol, move in moves.items():
            values[symbol] *= 1 + move
        rows.append({"date": date, **values})
    return pd.DataFrame(rows).set_index("date")


@pytest.mark.unit
def test_portfolio_validation_compares_identical_oos_periods():
    result = portfolio_validation.validate_portfolio_prices(
        _prices(),
        {"AAA": 0.4, "BBB": 0.35, "CCC": 0.15},
        "SPY",
        "2026-10-23",
        sessions=504,
        train_sessions=252,
        rebalance_sessions=63,
        transaction_cost_bps=7,
    )

    assert result["model_version"] == "portfolio-walk-forward-v1"
    assert result["oos_sessions"] == 504
    assert result["cash_weight"] == pytest.approx(0.10)
    assert len(result["points"]) == 504
    assert len(result["rebalance_blocks"]) == 8
    assert result["verdict"] in {"improved", "mixed", "rejected"}
    assert set(result["strategies"]) == {"current", "risk_balanced", "benchmark"}
    assert result["strategies"]["current"]["transaction_cost"] > 0
    assert result["strategies"]["risk_balanced"]["turnover"] > 0
    assert result["robustness"]["samples"] == 500
    assert sum(result["rebalance_blocks"][0]["balanced_weights"].values()) == pytest.approx(
        0.90, abs=1e-5
    )


@pytest.mark.unit
def test_first_oos_weights_do_not_see_future_prices():
    original = _prices()
    altered = original.copy()
    first_oos_position = len(altered) - 252
    for offset, index in enumerate(altered.index[first_oos_position + 25 :], start=1):
        altered.loc[index, "CCC"] *= 1 + offset * 0.002

    kwargs = {
        "sessions": 252,
        "train_sessions": 126,
        "rebalance_sessions": 21,
    }
    first = portfolio_validation.validate_portfolio_prices(
        original, {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}, "SPY", "2026-10-23", **kwargs
    )
    second = portfolio_validation.validate_portfolio_prices(
        altered, {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}, "SPY", "2026-10-23", **kwargs
    )

    assert first["rebalance_blocks"][0]["balanced_weights"] == second["rebalance_blocks"][0][
        "balanced_weights"
    ]
    assert first["rebalance_blocks"][0]["train_end"] < first["first_session"]


@pytest.mark.unit
def test_portfolio_validation_exposes_regimes_and_budget_frequency():
    result = portfolio_validation.validate_portfolio_prices(
        _prices(),
        {"AAA": 0.55, "BBB": 0.25, "CCC": 0.20},
        "SPY",
        "2026-10-23",
        annual_volatility_limit=0.05,
        daily_es_limit=0.005,
    )

    assert result["regimes"]
    risk_budget = result["strategies"]["current"]["risk_budget"]
    assert risk_budget["observations"] > 0
    assert risk_budget["breach_frequency"] > 0
    assert result["verdict_reasons"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("weights", "rebalance", "message"),
    [
        ({"AAA": 1.0}, 63, "2-30"),
        ({"AAA": 0.8, "BBB": 0.4}, 63, "100%"),
        ({"AAA": 0.5, "BBB": 0.5}, 42, "21, 63 or 126"),
    ],
)
def test_portfolio_validation_rejects_invalid_configuration(weights, rebalance, message):
    with pytest.raises(ValueError, match=message):
        portfolio_validation.validate_portfolio_prices(
            _prices(), weights, "SPY", "2026-10-23", rebalance_sessions=rebalance
        )
