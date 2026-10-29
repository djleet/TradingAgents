from __future__ import annotations

import pandas as pd
import pytest

from tradingagents import portfolio_optimization


def _prices() -> pd.DataFrame:
    index = pd.bdate_range(end="2026-01-30", periods=120).date
    return pd.DataFrame(
        {
            "AAA": [100 + index * 0.2 + (index % 3) for index in range(120)],
            "BBB": [80 + index * 0.1 + (index % 7) * 0.5 for index in range(120)],
            "CCC": [60 + index * 0.05 + (index % 5) * 0.8 for index in range(120)],
        },
        index=index,
    )


@pytest.mark.unit
def test_inverse_volatility_optimization_is_complete_and_normalized(monkeypatch):
    monkeypatch.setattr(portfolio_optimization, "load_price_matrix", lambda *a, **k: _prices())

    result = portfolio_optimization.optimize_portfolio(
        ["aaa", "bbb", "ccc"], "2026-01-30", method="inverse_volatility"
    )

    assert result["backend"] == "native_inverse_volatility"
    assert result["common_sessions"] == 120
    assert set(result["weights"]) == {"AAA", "BBB", "CCC"}
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert result["metrics"]["historical_annual_volatility"] > 0


@pytest.mark.unit
def test_optional_backend_has_an_explicit_fallback(monkeypatch):
    monkeypatch.setattr(portfolio_optimization, "load_price_matrix", lambda *a, **k: _prices())
    monkeypatch.setattr(portfolio_optimization, "pypfopt_available", lambda: False)

    result = portfolio_optimization.optimize_portfolio(
        ["AAA", "BBB"], "2026-01-30", method="minimum_variance"
    )

    assert result["requested_method"] == "minimum_variance"
    assert result["applied_method"] == "inverse_volatility"
    assert "not installed" in result["warning"]


@pytest.mark.unit
def test_optional_backend_is_used_when_available(monkeypatch):
    monkeypatch.setattr(portfolio_optimization, "load_price_matrix", lambda *a, **k: _prices())
    monkeypatch.setattr(portfolio_optimization, "pypfopt_available", lambda: True)
    monkeypatch.setattr(
        portfolio_optimization,
        "_pypfopt_weights",
        lambda prices, method: {"AAA": 0.7, "BBB": 0.2, "CCC": 0.1},
    )

    result = portfolio_optimization.optimize_portfolio(
        ["AAA", "BBB", "CCC"], "2026-01-30", method="hierarchical_risk_parity"
    )

    assert result["backend"] == "pypfopt"
    assert result["applied_method"] == "hierarchical_risk_parity"
    assert result["weights"] == {"AAA": 0.7, "BBB": 0.2, "CCC": 0.1}


@pytest.mark.unit
def test_optimizer_rejects_bad_universes():
    with pytest.raises(ValueError, match="2-30"):
        portfolio_optimization.optimize_portfolio(["AAPL"], "2026-01-30")
