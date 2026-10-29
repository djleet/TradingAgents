from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.dataflows import research_export, warehouse

duckdb = pytest.importorskip("duckdb")


def _payload(symbol_offset: float) -> str:
    dates = pd.bdate_range(end="2026-01-30", periods=70)
    close = pd.Series(range(70), dtype=float) * 0.1 + 100 + symbol_offset
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Adj Close": close,
            "Volume": 1000,
        }
    ).to_csv(index=False)


@pytest.mark.unit
def test_research_bundle_exports_canonical_qlib_and_lean_files(tmp_path, monkeypatch):
    config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "data_warehouse_enabled": True,
        "data_warehouse_path": str(tmp_path / "market.duckdb"),
    }
    monkeypatch.setattr(warehouse, "get_config", lambda: config)
    for symbol, offset in (("AAA", 0.0), ("BBB", 20.0)):
        assert warehouse.capture_vendor_result(
            "get_stock_data",
            "test",
            (symbol, "2025-01-01", "2026-01-30"),
            {},
            _payload(offset),
        )

    output = tmp_path / "bundle"
    manifest = research_export.export_research_bundle(
        output,
        database_path=config["data_warehouse_path"],
    )

    assert manifest["row_count"] == 140
    assert manifest["symbols"] == ["AAA", "BBB"]
    assert (output / "market_bars.csv").is_file()
    assert (output / "market_bars.parquet").is_file()
    assert (output / "qlib_csv" / "AAA.csv").is_file()
    assert (output / "lean_custom" / "BBB.csv").is_file()
    assert json.loads((output / "manifest.json").read_text())["row_count"] == 140
    assert len(duckdb.read_parquet(str(output / "market_bars.parquet")).fetchall()) == 140


@pytest.mark.unit
def test_research_export_refuses_a_nonempty_destination(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data")

    with pytest.raises(FileExistsError):
        research_export.export_research_bundle(
            destination,
            database_path=tmp_path / "missing.duckdb",
        )
