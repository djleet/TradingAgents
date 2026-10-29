"""Export the audited market bars to reproducible research-tool inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.dataflows.warehouse import dependency_available, warehouse_path


def _latest_bars(connection, source: str | None, symbols: list[str] | None) -> pd.DataFrame:
    filters: list[str] = []
    parameters: list[Any] = []
    if source:
        filters.append("source = ?")
        parameters.append(source)
    if symbols:
        normalized = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
        if not normalized:
            raise ValueError("symbols cannot be empty")
        filters.append(f"symbol IN ({', '.join('?' for _ in normalized)})")
        parameters.extend(normalized)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    return connection.execute(
        f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY symbol, bar_date
                ORDER BY as_of_date DESC NULLS LAST, ingested_at DESC, source
            ) AS vintage_rank
            FROM market_bars
            {where}
        )
        SELECT
            source,
            symbol,
            bar_date,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume,
            as_of_date,
            payload_sha256,
            ingested_at
        FROM ranked
        WHERE vintage_rank = 1
        ORDER BY symbol, bar_date
        """,
        parameters,
    ).fetchdf()


def _write_qlib_csv(frame: pd.DataFrame, directory: Path) -> list[str]:
    directory.mkdir()
    files = []
    for symbol, group in frame.groupby("symbol", sort=True):
        filename = f"{safe_ticker_component(str(symbol))}.csv"
        output = group.rename(columns={"bar_date": "date"}).loc[
            :, ["date", "symbol", "open", "high", "low", "close", "volume"]
        ]
        adjusted = group["adjusted_close"].reset_index(drop=True)
        close = output["close"].reset_index(drop=True)
        factor = (adjusted / close).where(adjusted.notna() & close.gt(0), 1.0)
        output = output.reset_index(drop=True).assign(factor=factor)
        output.to_csv(directory / filename, index=False, date_format="%Y-%m-%d")
        files.append(filename)
    return files


def _write_lean_custom_csv(frame: pd.DataFrame, directory: Path) -> list[str]:
    directory.mkdir()
    files = []
    for symbol, group in frame.groupby("symbol", sort=True):
        filename = f"{safe_ticker_component(str(symbol))}.csv"
        output = group.loc[:, ["bar_date", "open", "high", "low", "close", "volume"]].copy()
        for column in ("open", "high", "low"):
            output[column] = output[column].fillna(output["close"])
        output["volume"] = output["volume"].fillna(0)
        output["bar_date"] = pd.to_datetime(output["bar_date"]).dt.strftime("%Y%m%d 00:00")
        output.to_csv(directory / filename, index=False, header=False)
        files.append(filename)
    return files


def export_research_bundle(
    output_dir: str | Path,
    *,
    database_path: str | Path | None = None,
    source: str | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Create one immutable bundle for DuckDB, Qlib and LEAN research flows.

    The destination must be empty or absent so stale files from an earlier
    universe can never be mistaken for part of the new export.
    """
    if not dependency_available():
        raise RuntimeError("DuckDB is required; install the warehouse extra")
    destination = Path(output_dir).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    database = Path(database_path).expanduser() if database_path else warehouse_path()
    if not database.is_file():
        raise FileNotFoundError(f"Market-data warehouse does not exist: {database}")

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        frame = _latest_bars(connection, source, symbols)
    finally:
        connection.close()
    if frame.empty:
        raise ValueError("No normalized market bars matched the export filters")

    unified_csv = destination / "market_bars.csv"
    unified_parquet = destination / "market_bars.parquet"
    frame.to_csv(unified_csv, index=False, date_format="%Y-%m-%d")
    parquet_connection = duckdb.connect()
    try:
        parquet_connection.register("research_export", frame)
        parquet_connection.execute(
            "COPY research_export TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(unified_parquet)],
        )
    finally:
        parquet_connection.close()

    qlib_files = _write_qlib_csv(frame, destination / "qlib_csv")
    lean_files = _write_lean_custom_csv(frame, destination / "lean_custom")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "point_in_time_policy": (
            "Latest archived vintage per symbol and bar date, selected only from source data "
            "that was already filtered to each observation's as-of date."
        ),
        "filters": {"source": source, "symbols": symbols},
        "row_count": len(frame),
        "symbols": sorted(str(value) for value in frame["symbol"].unique()),
        "first_bar_date": pd.Timestamp(frame["bar_date"].min()).date().isoformat(),
        "last_bar_date": pd.Timestamp(frame["bar_date"].max()).date().isoformat(),
        "outputs": {
            "canonical_csv": unified_csv.name,
            "canonical_parquet": unified_parquet.name,
            "qlib_csv": qlib_files,
            "lean_custom_daily_tradebar": lean_files,
        },
        "qlib_factor": "adjusted_close / close when available, otherwise 1.0",
        "lean_format": "No-header CSV: yyyyMMdd HH:mm,open,high,low,close,volume",
        "lean_missing_field_policy": "Missing OHLC uses close; missing volume uses zero.",
        "disclaimer": "Research data only; corporate-action completeness depends on the source.",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export audited market bars for research tools")
    parser.add_argument("--output", required=True, help="New or empty destination directory")
    parser.add_argument("--warehouse", help="DuckDB path; defaults to configured warehouse")
    parser.add_argument("--source", help="Optional archived vendor source filter")
    parser.add_argument("--symbols", nargs="*", help="Optional ticker filter")
    args = parser.parse_args()
    manifest = export_research_bundle(
        args.output,
        database_path=args.warehouse,
        source=args.source,
        symbols=args.symbols,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
