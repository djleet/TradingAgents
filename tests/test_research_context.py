from __future__ import annotations

import pytest

from tradingagents.dataflows import research_context

duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def context_config(tmp_path):
    return {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market.duckdb"),
    }


def _statement(title, headers, rows):
    return "\n".join(
        [f"# {title}", "", "," + ",".join(headers)]
        + [f"{label}," + ",".join(str(value) for value in values) for label, values in rows]
    )


@pytest.mark.unit
def test_research_context_builds_separate_fundamental_and_event_pillars(
    context_config, monkeypatch
):
    quarterly_income = _statement(
        "Income",
        ["2025-06-30", "2025-09-30"],
        [
            ("Revenue", [100, 120]),
            ("Operating Income", [20, 30]),
            ("Net Income", [10, 18]),
        ],
    )
    balance = _statement(
        "Balance",
        ["2025-06-30", "2025-09-30"],
        [
            ("Total Assets", [500, 550]),
            ("Current Assets", [200, 220]),
            ("Total Liabilities", [200, 210]),
            ("Current Liabilities", [100, 100]),
        ],
    )
    cashflow = _statement(
        "Cashflow",
        ["2025-06-30", "2025-09-30"],
        [("Operating Cash Flow", [22, 28]), ("Capital Expenditure", [4, 5])],
    )
    annual = _statement(
        "Annual income",
        ["2024-12-31", "2025-12-31"],
        [("Diluted EPS", [4, 5])],
    )
    filings = """# SEC filings
| Accepted | Form | Report date | Items / description | Primary document |
|---|---|---|---|---|
| 2025-11-10 12:00:00 | 8-K | 2025-11-10 | 2.02 | [event.htm](https://sec/event) |
| 2025-10-30 16:00:00 | 10-Q | 2025-09-30 | Quarterly report | [q3.htm](https://sec/q3) |
| 2025-07-30 16:00:00 | 10-Q | 2025-06-30 | Quarterly report | [q2.htm](https://sec/q2) |
| 2025-04-30 16:00:00 | 10-Q | 2025-03-31 | Quarterly report | [q1.htm](https://sec/q1) |
"""

    def fake_route(method, *args):
        if method == "get_income_statement":
            return annual if args[1] == "annual" else quarterly_income
        return {
            "get_balance_sheet": balance,
            "get_cashflow": cashflow,
            "get_sec_filings": filings,
        }[method]

    monkeypatch.setattr(research_context, "route_to_vendor", fake_route)
    monkeypatch.setattr(
        research_context, "load_close_prices", lambda symbols, as_of: {symbols[0]: 100.0}
    )

    report = research_context.build_research_context(
        "AAPL", "2025-11-15", context_config
    )

    assert report["fundamentals"]["available"] is True
    assert report["fundamentals"]["metrics"]["sequential_revenue_growth"] == 0.2
    assert report["fundamentals"]["metrics"]["operating_margin"] == 0.25
    assert report["fundamentals"]["metrics"]["trailing_pe_proxy"] == 20.0
    assert report["events"]["recent_8k_count"] == 1
    assert report["events"]["estimated_next_report_date"] == "2026-01-30"
    assert report["overlay"]["stance"] in {"supportive", "balanced"}
    assert report["registry"]["saved"] is True

    connection = duckdb.connect(context_config["data_warehouse_path"], read_only=True)
    try:
        stored = connection.execute(
            "SELECT symbol, fundamental_score, event_risk_score "
            "FROM research_context_snapshots"
        ).fetchone()
    finally:
        connection.close()
    assert stored[0] == "AAPL"
    assert stored[1] == report["fundamentals"]["score"]
    assert stored[2] == report["events"]["risk_score"]


@pytest.mark.unit
def test_research_context_degrades_to_limited_instead_of_fabricating(
    context_config, monkeypatch
):
    monkeypatch.setattr(research_context, "route_to_vendor", lambda *args: "NO_DATA_AVAILABLE:")
    monkeypatch.setattr(
        research_context,
        "load_close_prices",
        lambda *args: (_ for _ in ()).throw(ValueError("missing")),
    )

    report = research_context.build_research_context(
        "UNKNOWN", "2025-11-15", context_config
    )

    assert report["fundamentals"]["score"] is None
    assert report["events"]["risk_level"] == "watch"
    assert report["overlay"]["stance"] == "limited"
    assert len(report["errors"]) == 6
