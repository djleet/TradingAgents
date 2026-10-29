"""Durable, point-in-time audit log for data returned by market vendors.

The agent prompts receive strings, which is convenient for LLM tools but used
to make it impossible to answer a basic research question later: *which exact
vendor response produced this decision?*  This module records every successful
vendor response in DuckDB together with its query window, source and content
hash.  Capturing is deliberately best-effort; an analytics database problem
must never make a paid analysis fail.

DuckDB is imported lazily so the core library remains usable without the
``warehouse`` optional dependency.  The dashboard installs that extra by
default.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_SCHEMA_VERSION = 1


def dependency_available() -> bool:
    """Return whether the optional DuckDB runtime can be imported."""
    return importlib.util.find_spec("duckdb") is not None


def warehouse_path(config: dict[str, Any] | None = None) -> Path:
    """Resolve the warehouse beside the configured data cache by default."""
    cfg = config or get_config()
    configured = cfg.get("data_warehouse_path")
    if configured:
        return Path(configured).expanduser()
    return Path(cfg["data_cache_dir"]).expanduser().parent / "market_data.duckdb"


def _connect(path: Path):
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS vendor_observations (
            observation_id VARCHAR PRIMARY KEY,
            source VARCHAR NOT NULL,
            dataset VARCHAR NOT NULL,
            symbol VARCHAR,
            query_start DATE,
            query_end DATE,
            as_of_date DATE,
            query_json VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL,
            payload_sha256 VARCHAR NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL,
            schema_version INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vendor_observations_lookup
        ON vendor_observations(symbol, dataset, as_of_date, ingested_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_bars (
            observation_id VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            bar_date DATE NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE NOT NULL,
            adjusted_close DOUBLE,
            volume DOUBLE,
            as_of_date DATE,
            payload_sha256 VARCHAR NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (observation_id, bar_date)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_market_bars_lookup
        ON market_bars(symbol, bar_date, as_of_date, ingested_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_quality (
            observation_id VARCHAR PRIMARY KEY,
            source VARCHAR NOT NULL,
            symbol VARCHAR,
            row_count INTEGER NOT NULL,
            stored_row_count INTEGER NOT NULL,
            first_bar DATE,
            last_bar DATE,
            duplicate_dates INTEGER NOT NULL,
            missing_close INTEGER NOT NULL,
            invalid_ohlc INTEGER NOT NULL,
            future_rows INTEGER NOT NULL,
            passed BOOLEAN NOT NULL,
            checked_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quantitative_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            latest_bar_date DATE NOT NULL,
            score DOUBLE NOT NULL,
            label VARCHAR NOT NULL,
            risk_regime VARCHAR NOT NULL,
            confidence DOUBLE NOT NULL,
            bars_used INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_quantitative_snapshots_lookup
        ON quantitative_snapshots(symbol, as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS validation_experiments (
            experiment_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            robustness VARCHAR NOT NULL,
            oos_return DOUBLE,
            benchmark_return DOUBLE,
            max_drawdown DOUBLE,
            fold_count INTEGER NOT NULL,
            config_json VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_validation_experiments_lookup
        ON validation_experiments(symbol, as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_validation_experiments (
            experiment_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            symbols_json VARCHAR NOT NULL,
            verdict VARCHAR NOT NULL,
            current_return DOUBLE,
            balanced_return DOUBLE,
            benchmark_return DOUBLE,
            current_drawdown DOUBLE,
            balanced_drawdown DOUBLE,
            block_win_rate DOUBLE,
            config_json VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_portfolio_validation_lookup
        ON portfolio_validation_experiments(as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunity_scans (
            scan_id VARCHAR PRIMARY KEY,
            as_of_date DATE NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            universe_hash VARCHAR,
            universe_name VARCHAR,
            requested_count INTEGER NOT NULL,
            scanned_count INTEGER NOT NULL,
            eligible_count INTEGER NOT NULL,
            top_opportunity_count INTEGER NOT NULL,
            config_json VARCHAR NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        "ALTER TABLE opportunity_scans ADD COLUMN IF NOT EXISTS universe_hash VARCHAR"
    )
    connection.execute(
        "ALTER TABLE opportunity_scans ADD COLUMN IF NOT EXISTS universe_name VARCHAR"
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_opportunity_scans_lookup
        ON opportunity_scans(as_of_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_opportunity_scans_universe
        ON opportunity_scans(universe_hash, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_universes (
            universe_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            source VARCHAR NOT NULL,
            preset_id VARCHAR,
            symbols_json VARCHAR NOT NULL,
            symbol_count INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scan_universes_updated
        ON scan_universes(updated_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_schedules (
            schedule_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            universe_id VARCHAR,
            symbols_json VARCHAR NOT NULL,
            symbol_count INTEGER NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            local_time VARCHAR NOT NULL,
            timezone VARCHAR NOT NULL,
            weekdays_json VARCHAR NOT NULL,
            configuration_json VARCHAR NOT NULL,
            active BOOLEAN NOT NULL,
            last_run_date DATE,
            last_status VARCHAR,
            last_error VARCHAR,
            last_scan_id VARCHAR,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scan_schedules_active
        ON scan_schedules(active, updated_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunity_alerts (
            alert_id VARCHAR PRIMARY KEY,
            scan_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            alert_type VARCHAR NOT NULL,
            severity VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            message VARCHAR NOT NULL,
            current_score DOUBLE,
            previous_score DOUBLE,
            created_at TIMESTAMPTZ NOT NULL,
            acknowledged_at TIMESTAMPTZ
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_opportunity_alerts_status
        ON opportunity_alerts(acknowledged_at, created_at)
        """
    )
    return connection


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, Path)):
        return str(value)
    return repr(value)


def _safe_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _db_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def _query_metadata(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract common point-in-time fields without knowing each vendor API."""
    query = {"args": list(args), "kwargs": kwargs}
    strings = [value for value in args if isinstance(value, str)]
    dates = [value for value in strings if len(value) == 10 and value[4:5] == "-" and value[7:8] == "-"]

    symbol = None
    if strings and method not in {"get_global_news", "get_macro_indicators", "get_prediction_markets"}:
        symbol = strings[0].strip().upper() or None

    query_start = None
    query_end = None
    as_of_date = None
    if method in {"get_stock_data", "get_news"} and len(dates) >= 2:
        query_start, query_end = dates[0], dates[1]
        as_of_date = query_end
    elif dates:
        as_of_date = dates[-1]

    return {
        "symbol": symbol,
        "query_start": query_start,
        "query_end": query_end,
        "as_of_date": as_of_date,
        "query": query,
    }


def _local_date(value: Any):
    """Keep a vendor timestamp's wall-clock date instead of shifting it to UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.date()


def parse_ohlcv_payload(payload: str) -> pd.DataFrame:
    """Parse the common vendor CSV responses into normalized daily bars.

    Vendor prose before the CSV is ignored. Unknown payloads return an empty
    frame instead of guessing which numbers represent market data.
    """
    columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if not isinstance(payload, str) or not payload.strip():
        return pd.DataFrame(columns=columns)

    lines = payload.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "," in line
            and "close" in line.lower()
            and ("date" in line.lower() or "datetime" in line.lower())
        ),
        None,
    )
    if header_index is None:
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    except (pd.errors.ParserError, UnicodeError, ValueError):
        return pd.DataFrame(columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    normalized = {
        str(column).strip().lower().replace("_", " "): column for column in frame.columns
    }
    rename: dict[Any, str] = {}
    aliases = {
        "date": "Date",
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjusted close": "Adj Close",
        "volume": "Volume",
    }
    for source_name, target_name in aliases.items():
        if source_name in normalized:
            rename[normalized[source_name]] = target_name
    frame = frame.rename(columns=rename)
    if "Date" not in frame or "Close" not in frame:
        return pd.DataFrame(columns=columns)

    frame["Date"] = frame["Date"].map(_local_date)
    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[:, columns].dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def _archive_market_bars(
    connection,
    *,
    observation_id: str,
    source: str,
    symbol: str | None,
    as_of_date: str | None,
    payload_json: str,
    payload_hash: str,
    ingested_at: datetime,
) -> None:
    if not symbol:
        return
    frame = parse_ohlcv_payload(payload_json)
    row_count = len(frame)
    duplicate_dates = int(frame["Date"].duplicated().sum()) if row_count else 0
    missing_close = int(frame["Close"].isna().sum()) if row_count else 0
    invalid = pd.Series(False, index=frame.index)
    if row_count:
        invalid |= frame["Close"].le(0).fillna(False)
        invalid |= frame["High"].lt(frame["Low"]).fillna(False)
        invalid |= frame["Open"].gt(frame["High"]).fillna(False)
        invalid |= frame["Open"].lt(frame["Low"]).fillna(False)
        invalid |= frame["Close"].gt(frame["High"]).fillna(False)
        invalid |= frame["Close"].lt(frame["Low"]).fillna(False)
    invalid_ohlc = int(invalid.sum())

    cutoff = _local_date(as_of_date) if as_of_date else None
    future = frame["Date"].map(lambda value: bool(cutoff and value > cutoff))
    future_rows = int(future.sum())
    usable = frame[~future & ~frame["Close"].isna() & ~invalid].drop_duplicates(
        subset=["Date"], keep="last"
    )
    passed = bool(row_count and not (duplicate_dates or missing_close or invalid_ohlc or future_rows))
    first_bar = min(frame["Date"]) if row_count else None
    last_bar = max(frame["Date"]) if row_count else None

    connection.execute(
        """
        INSERT OR REPLACE INTO ingestion_quality VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            observation_id,
            source,
            symbol,
            row_count,
            len(usable),
            first_bar,
            last_bar,
            duplicate_dates,
            missing_close,
            invalid_ohlc,
            future_rows,
            passed,
            ingested_at,
        ],
    )
    if usable.empty:
        return
    connection.executemany(
        """
        INSERT OR IGNORE INTO market_bars VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            [
                observation_id,
                source,
                symbol,
                row["Date"],
                _db_value(row["Open"]),
                _db_value(row["High"]),
                _db_value(row["Low"]),
                _db_value(row["Close"]),
                _db_value(row["Adj Close"]),
                _db_value(row["Volume"]),
                as_of_date,
                payload_hash,
                ingested_at,
            ]
            for _, row in usable.iterrows()
        ],
    )


def capture_vendor_result(
    method: str,
    source: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> bool:
    """Persist one successful vendor response, returning whether it was stored.

    Identical responses for an identical query are deduplicated. A later vendor
    revision gets a different content hash and is retained as another vintage.
    """
    cfg = get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False
    if result is None or (isinstance(result, str) and not result.strip()):
        return False

    try:
        query = _query_metadata(method, args, kwargs)
        query_json = _safe_json(query["query"])
        payload_json = result if isinstance(result, str) else _safe_json(result)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        identity = _safe_json(
            {
                "source": source,
                "dataset": method,
                "query": query["query"],
                "payload_sha256": payload_hash,
            }
        )
        observation_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        ingested_at = datetime.now(UTC)

        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO vendor_observations VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        observation_id,
                        source,
                        method,
                        query["symbol"],
                        query["query_start"],
                        query["query_end"],
                        query["as_of_date"],
                        query_json,
                        payload_json,
                        payload_hash,
                        ingested_at,
                        _SCHEMA_VERSION,
                    ],
                )
                if method == "get_stock_data":
                    _archive_market_bars(
                        connection,
                        observation_id=observation_id,
                        source=source,
                        symbol=query["symbol"],
                        as_of_date=query["as_of_date"],
                        payload_json=payload_json,
                        payload_hash=payload_hash,
                        ingested_at=ingested_at,
                    )
            finally:
                connection.close()
        return True
    except Exception as exc:  # noqa: BLE001 - capture must never abort an analysis
        logger.warning("Could not archive %s response from %s: %s", method, source, exc)
        return False


def capture_quantitative_snapshot(snapshot: dict[str, Any]) -> bool:
    """Persist a reproducible quantitative result when the warehouse is enabled."""
    cfg = get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False

    try:
        payload_json = _safe_json(snapshot)
        identity = _safe_json(
            {
                "model_version": snapshot["model_version"],
                "symbol": snapshot["symbol"],
                "as_of_date": snapshot["as_of_date"],
                "payload_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            }
        )
        snapshot_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO quantitative_snapshots VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        snapshot_id,
                        snapshot["model_version"],
                        str(snapshot["symbol"]).upper(),
                        snapshot["as_of_date"],
                        snapshot["latest_bar_date"],
                        snapshot["score"],
                        snapshot["label"],
                        snapshot["risk_regime"],
                        snapshot["confidence"],
                        snapshot["bars_used"],
                        payload_json,
                        datetime.now(UTC),
                    ],
                )
            finally:
                connection.close()
        return True
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not archive malformed quantitative snapshot: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - capture must never abort an analysis
        logger.warning("Could not archive quantitative snapshot: %s", exc)
        return False


def capture_validation_experiment(
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str | None:
    """Persist a complete validation report and return its stable id."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    try:
        result_json = _safe_json(report)
        config_json = _safe_json(report["configuration"])
        experiment_id = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        metrics = report["metrics"]
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO validation_experiments VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        experiment_id,
                        report["model_version"],
                        str(report["symbol"]).upper(),
                        str(report["benchmark_symbol"]).upper(),
                        report["as_of_date"],
                        report["robustness"],
                        metrics.get("oos_return"),
                        metrics.get("benchmark_return"),
                        metrics.get("max_drawdown"),
                        metrics["fold_count"],
                        config_json,
                        result_json,
                        datetime.now(UTC),
                    ],
                )
            finally:
                connection.close()
        return experiment_id
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not archive malformed validation experiment: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - persistence must not erase a valid report
        logger.warning("Could not archive validation experiment: %s", exc)
        return None


def list_validation_experiments(
    config: dict[str, Any] | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent experiment summaries without loading large curve payloads."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    bounded_limit = max(1, min(int(limit), 100))
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                rows = connection.execute(
                    """
                    SELECT experiment_id, symbol, benchmark_symbol, as_of_date, robustness,
                           oos_return, benchmark_return, max_drawdown, fold_count, created_at
                    FROM validation_experiments
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [bounded_limit],
                ).fetchall()
                columns = [item[0] for item in connection.description]
            finally:
                connection.close()
        return [
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in zip(columns, row, strict=True)
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001 - history failure is non-critical
        logger.warning("Could not list validation experiments: %s", exc)
        return []


def capture_portfolio_validation_experiment(
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str | None:
    """Persist a complete portfolio walk-forward report with a stable id."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    try:
        result_json = _safe_json(report)
        config_json = _safe_json(report["configuration"])
        experiment_id = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        strategies = report["strategies"]
        current = strategies["current"]
        balanced = strategies["risk_balanced"]
        benchmark = strategies["benchmark"]
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO portfolio_validation_experiments VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        experiment_id,
                        report["model_version"],
                        report["as_of_date"],
                        str(report["benchmark_symbol"]).upper(),
                        _safe_json(report["symbols"]),
                        report["verdict"],
                        current.get("total_return"),
                        balanced.get("total_return"),
                        benchmark.get("total_return"),
                        current.get("max_drawdown"),
                        balanced.get("max_drawdown"),
                        report.get("block_win_rate"),
                        config_json,
                        result_json,
                        datetime.now(UTC),
                    ],
                )
            finally:
                connection.close()
        return experiment_id
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not archive malformed portfolio validation: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - persistence must not erase a valid report
        logger.warning("Could not archive portfolio validation: %s", exc)
        return None


def list_portfolio_validation_experiments(
    config: dict[str, Any] | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent portfolio validation summaries without loading curves."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    bounded_limit = max(1, min(int(limit), 100))
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                rows = connection.execute(
                    """
                    SELECT experiment_id, as_of_date, benchmark_symbol, symbols_json,
                           verdict, current_return, balanced_return, benchmark_return,
                           current_drawdown, balanced_drawdown, block_win_rate, created_at
                    FROM portfolio_validation_experiments
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [bounded_limit],
                ).fetchall()
            finally:
                connection.close()
        return [
            {
                "experiment_id": row[0],
                "as_of_date": row[1].isoformat(),
                "benchmark_symbol": row[2],
                "symbols": json.loads(row[3]),
                "verdict": row[4],
                "current_return": row[5],
                "balanced_return": row[6],
                "benchmark_return": row[7],
                "current_drawdown": row[8],
                "balanced_drawdown": row[9],
                "block_win_rate": row[10],
                "created_at": row[11].isoformat(),
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001 - history failure is non-critical
        logger.warning("Could not list portfolio validation experiments: %s", exc)
        return []


_BUCKET_ORDER = {
    "excluded": 0,
    "avoid": 1,
    "neutral": 2,
    "watch": 3,
    "top_opportunity": 4,
}


def _opportunity_universe_hash(report: dict[str, Any]) -> str:
    symbols = sorted({str(symbol).upper() for symbol in report["symbols"]})
    return hashlib.sha256(_safe_json(symbols).encode("utf-8")).hexdigest()


def compare_opportunity_scan(
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach score, rank and bucket changes from the latest matching universe."""
    cfg = config or get_config()
    empty = {
        "has_previous": False,
        "previous_scan_id": None,
        "previous_as_of_date": None,
        "new_symbols": 0,
        "entered_top": 0,
        "improved": 0,
        "deteriorated": 0,
        "stable": 0,
    }
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        report["changes"] = empty
        return report
    try:
        universe_hash = _opportunity_universe_hash(report)
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                previous_row = connection.execute(
                    """
                    SELECT scan_id, as_of_date, result_json
                    FROM opportunity_scans
                    WHERE universe_hash = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [universe_hash],
                ).fetchone()
            finally:
                connection.close()
        if not previous_row:
            for item in report.get("items", []):
                item["change"] = {
                    "state": "new",
                    "previous_score": None,
                    "score_delta": None,
                    "previous_rank": None,
                    "rank_delta": None,
                    "previous_bucket": None,
                }
            empty["new_symbols"] = len(report.get("items", []))
            report["changes"] = empty
            return report

        previous = json.loads(previous_row[2])
        previous_items = {item["symbol"]: item for item in previous.get("items", [])}
        counts = {"new_symbols": 0, "entered_top": 0, "improved": 0, "deteriorated": 0, "stable": 0}
        for item in report.get("items", []):
            old = previous_items.get(item["symbol"])
            if not old:
                state = "new"
                score_delta = None
                rank_delta = None
                counts["new_symbols"] += 1
            else:
                score_delta = round(
                    float(item["opportunity_score"]) - float(old["opportunity_score"]), 2
                )
                current_rank = item.get("rank")
                previous_rank = old.get("rank")
                rank_delta = (
                    previous_rank - current_rank
                    if isinstance(current_rank, int) and isinstance(previous_rank, int)
                    else None
                )
                entered_top = (
                    item["bucket"] == "top_opportunity"
                    and old.get("bucket") != "top_opportunity"
                )
                bucket_delta = _BUCKET_ORDER.get(item["bucket"], 0) - _BUCKET_ORDER.get(
                    old.get("bucket", "excluded"), 0
                )
                if entered_top:
                    state = "entered_top"
                    counts["entered_top"] += 1
                elif bucket_delta > 0 or score_delta >= 5:
                    state = "improved"
                    counts["improved"] += 1
                elif bucket_delta < 0 or score_delta <= -5:
                    state = "deteriorated"
                    counts["deteriorated"] += 1
                else:
                    state = "stable"
                    counts["stable"] += 1
            item["change"] = {
                "state": state,
                "previous_score": old.get("opportunity_score") if old else None,
                "score_delta": score_delta,
                "previous_rank": old.get("rank") if old else None,
                "rank_delta": rank_delta,
                "previous_bucket": old.get("bucket") if old else None,
            }
        report["changes"] = {
            "has_previous": True,
            "previous_scan_id": previous_row[0],
            "previous_as_of_date": previous_row[1].isoformat(),
            **counts,
        }
        return report
    except Exception as exc:  # noqa: BLE001 - comparison is enrichment, not a scan blocker
        logger.warning("Could not compare opportunity scan: %s", exc)
        report["changes"] = empty
        return report


def capture_opportunity_scan(
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str | None:
    """Persist a complete cross-sectional scan and return its stable id."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    try:
        symbols = [str(symbol).upper() for symbol in report["symbols"]]
        universe_hash = hashlib.sha256(
            _safe_json(sorted(set(symbols))).encode("utf-8")
        ).hexdigest()
        result_json = _safe_json(report)
        identity_payload = {
            "as_of_date": report["as_of_date"],
            "benchmark_symbol": report["benchmark_symbol"],
            "symbols": symbols,
            "filters": report["filters"],
            "validation_configuration": report["validation_configuration"],
            "items": [
                {key: value for key, value in item.items() if key != "change"}
                for item in report.get("items", [])
            ],
            "errors": report.get("errors", []),
        }
        scan_id = hashlib.sha256(_safe_json(identity_payload).encode("utf-8")).hexdigest()
        scan_config = {
            "filters": report["filters"],
            "validation_configuration": report["validation_configuration"],
        }
        bucket_counts = report["bucket_counts"]
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO opportunity_scans (
                        scan_id, as_of_date, benchmark_symbol, universe_hash, universe_name,
                        requested_count, scanned_count, eligible_count, top_opportunity_count,
                        config_json, result_json, created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        scan_id,
                        report["as_of_date"],
                        str(report["benchmark_symbol"]).upper(),
                        universe_hash,
                        report.get("universe_name"),
                        report["requested_symbols"],
                        report["scanned_symbols"],
                        report["eligible_symbols"],
                        bucket_counts.get("top_opportunity", 0),
                        _safe_json(scan_config),
                        result_json,
                        datetime.now(UTC),
                    ],
                )
            finally:
                connection.close()
        return scan_id
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not archive malformed opportunity scan: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - persistence must not erase a valid scan
        logger.warning("Could not archive opportunity scan: %s", exc)
        return None


def list_opportunity_scans(
    config: dict[str, Any] | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return compact scan-history rows without loading every ranked asset."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    bounded_limit = max(1, min(int(limit), 100))
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                rows = connection.execute(
                    """
                    SELECT scan_id, as_of_date, benchmark_symbol, universe_name, requested_count,
                           scanned_count, eligible_count, top_opportunity_count, created_at
                    FROM opportunity_scans
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [bounded_limit],
                ).fetchall()
                columns = [item[0] for item in connection.description]
            finally:
                connection.close()
        return [
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in zip(columns, row, strict=True)
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001 - history failure is non-critical
        logger.warning("Could not list opportunity scans: %s", exc)
        return []


def save_scan_universe(
    name: str,
    symbols: list[str],
    config: dict[str, Any] | None = None,
    *,
    source: str = "custom",
    preset_id: str | None = None,
    universe_id: str | None = None,
) -> dict[str, Any] | None:
    """Create or update a named scan universe."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    cleaned = list(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    now = datetime.now(UTC)
    resolved_id = universe_id or hashlib.sha256(
        f"{name}|{now.isoformat()}".encode()
    ).hexdigest()[:16]
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT INTO scan_universes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (universe_id) DO UPDATE SET
                        name = excluded.name,
                        source = excluded.source,
                        preset_id = excluded.preset_id,
                        symbols_json = excluded.symbols_json,
                        symbol_count = excluded.symbol_count,
                        updated_at = excluded.updated_at
                    """,
                    [
                        resolved_id,
                        name,
                        source,
                        preset_id,
                        _safe_json(cleaned),
                        len(cleaned),
                        now,
                        now,
                    ],
                )
            finally:
                connection.close()
        return {
            "universe_id": resolved_id,
            "name": name,
            "source": source,
            "preset_id": preset_id,
            "symbols": cleaned,
            "symbol_count": len(cleaned),
            "updated_at": now.isoformat(),
        }
    except Exception as exc:  # noqa: BLE001 - UI can continue with an unsaved universe
        logger.warning("Could not save scan universe: %s", exc)
        return None


def list_scan_universes(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """List saved universes with their constituent symbols."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                rows = connection.execute(
                    """
                    SELECT universe_id, name, source, preset_id, symbols_json,
                           symbol_count, created_at, updated_at
                    FROM scan_universes
                    ORDER BY updated_at DESC
                    """
                ).fetchall()
            finally:
                connection.close()
        return [
            {
                "universe_id": row[0],
                "name": row[1],
                "source": row[2],
                "preset_id": row[3],
                "symbols": json.loads(row[4]),
                "symbol_count": row[5],
                "created_at": row[6].isoformat(),
                "updated_at": row[7].isoformat(),
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001 - saved universes are non-critical
        logger.warning("Could not list scan universes: %s", exc)
        return []


def delete_scan_universe(
    universe_id: str,
    config: dict[str, Any] | None = None,
) -> bool:
    """Delete one saved universe and deactivate schedules that reference it."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    "UPDATE scan_schedules SET active = false, updated_at = ? WHERE universe_id = ?",
                    [datetime.now(UTC), universe_id],
                )
                deleted = connection.execute(
                    "DELETE FROM scan_universes WHERE universe_id = ? RETURNING universe_id",
                    [universe_id],
                ).fetchone()
            finally:
                connection.close()
        return deleted is not None
    except Exception as exc:  # noqa: BLE001 - deletion failure is returned to the API
        logger.warning("Could not delete scan universe: %s", exc)
        return False


def save_scan_schedule(
    name: str,
    symbols: list[str],
    benchmark_symbol: str,
    scan_configuration: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    local_time: str = "16:30",
    timezone: str = "America/New_York",
    weekdays: list[int] | None = None,
    universe_id: str | None = None,
    schedule_id: str | None = None,
    active: bool = True,
) -> dict[str, Any] | None:
    """Create or update a daily research scan schedule."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return None
    cleaned = list(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    now = datetime.now(UTC)
    resolved_id = schedule_id or hashlib.sha256(
        f"{name}|{now.isoformat()}".encode()
    ).hexdigest()[:16]
    run_days = weekdays if weekdays is not None else [0, 1, 2, 3, 4]
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                connection.execute(
                    """
                    INSERT INTO scan_schedules (
                        schedule_id, name, universe_id, symbols_json, symbol_count,
                        benchmark_symbol, local_time, timezone, weekdays_json,
                        configuration_json, active, last_run_date, last_status,
                        last_error, last_scan_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)
                    ON CONFLICT (schedule_id) DO UPDATE SET
                        name = excluded.name,
                        universe_id = excluded.universe_id,
                        symbols_json = excluded.symbols_json,
                        symbol_count = excluded.symbol_count,
                        benchmark_symbol = excluded.benchmark_symbol,
                        local_time = excluded.local_time,
                        timezone = excluded.timezone,
                        weekdays_json = excluded.weekdays_json,
                        configuration_json = excluded.configuration_json,
                        active = excluded.active,
                        updated_at = excluded.updated_at
                    """,
                    [
                        resolved_id,
                        name,
                        universe_id,
                        _safe_json(cleaned),
                        len(cleaned),
                        benchmark_symbol.upper(),
                        local_time,
                        timezone,
                        _safe_json(run_days),
                        _safe_json(scan_configuration),
                        active,
                        now,
                        now,
                    ],
                )
            finally:
                connection.close()
        return {
            "schedule_id": resolved_id,
            "name": name,
            "universe_id": universe_id,
            "symbols": cleaned,
            "symbol_count": len(cleaned),
            "benchmark_symbol": benchmark_symbol.upper(),
            "local_time": local_time,
            "timezone": timezone,
            "weekdays": run_days,
            "configuration": scan_configuration,
            "active": active,
            "updated_at": now.isoformat(),
        }
    except Exception as exc:  # noqa: BLE001 - scheduling failure is returned to the API
        logger.warning("Could not save scan schedule: %s", exc)
        return None


def list_scan_schedules(
    config: dict[str, Any] | None = None,
    *,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    """List daily scan schedules and their latest run state."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    try:
        where = "WHERE active" if active_only else ""
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                rows = connection.execute(
                    f"""
                    SELECT schedule_id, name, universe_id, symbols_json, symbol_count,
                           benchmark_symbol, local_time, timezone, weekdays_json,
                           configuration_json, active, last_run_date, last_status,
                           last_error, last_scan_id, created_at, updated_at
                    FROM scan_schedules
                    {where}
                    ORDER BY updated_at DESC
                    """
                ).fetchall()
            finally:
                connection.close()
        return [
            {
                "schedule_id": row[0],
                "name": row[1],
                "universe_id": row[2],
                "symbols": json.loads(row[3]),
                "symbol_count": row[4],
                "benchmark_symbol": row[5],
                "local_time": row[6],
                "timezone": row[7],
                "weekdays": json.loads(row[8]),
                "configuration": json.loads(row[9]),
                "active": row[10],
                "last_run_date": row[11].isoformat() if row[11] else None,
                "last_status": row[12],
                "last_error": row[13],
                "last_scan_id": row[14],
                "created_at": row[15].isoformat(),
                "updated_at": row[16].isoformat(),
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001 - scheduler health remains visible as empty
        logger.warning("Could not list scan schedules: %s", exc)
        return []


def set_scan_schedule_active(
    schedule_id: str,
    active: bool,
    config: dict[str, Any] | None = None,
) -> bool:
    """Pause or resume a scan schedule."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                updated = connection.execute(
                    """
                    UPDATE scan_schedules SET active = ?, updated_at = ?
                    WHERE schedule_id = ? RETURNING schedule_id
                    """,
                    [active, datetime.now(UTC), schedule_id],
                ).fetchone()
            finally:
                connection.close()
        return updated is not None
    except Exception as exc:  # noqa: BLE001 - state failure is returned to the API
        logger.warning("Could not update scan schedule: %s", exc)
        return False


def delete_scan_schedule(
    schedule_id: str,
    config: dict[str, Any] | None = None,
) -> bool:
    """Delete a daily scan schedule without deleting its historical scans."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                deleted = connection.execute(
                    "DELETE FROM scan_schedules WHERE schedule_id = ? RETURNING schedule_id",
                    [schedule_id],
                ).fetchone()
            finally:
                connection.close()
        return deleted is not None
    except Exception as exc:  # noqa: BLE001 - deletion failure is returned to the API
        logger.warning("Could not delete scan schedule: %s", exc)
        return False


def record_scan_schedule_run(
    schedule_id: str,
    run_date: str | None,
    run_status: str,
    config: dict[str, Any] | None = None,
    *,
    error: str | None = None,
    scan_id: str | None = None,
) -> bool:
    """Record one scheduled run without exposing internal exceptions to clients."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                updated = connection.execute(
                    """
                    UPDATE scan_schedules
                    SET last_run_date = ?, last_status = ?, last_error = ?,
                        last_scan_id = ?, updated_at = ?
                    WHERE schedule_id = ? RETURNING schedule_id
                    """,
                    [run_date, run_status, error, scan_id, datetime.now(UTC), schedule_id],
                ).fetchone()
            finally:
                connection.close()
        return updated is not None
    except Exception as exc:  # noqa: BLE001 - scheduler keeps running after registry failure
        logger.warning("Could not record scheduled scan state: %s", exc)
        return False


def capture_opportunity_alerts(
    report: dict[str, Any],
    scan_id: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create actionable, deduplicated alerts from scan-to-scan changes."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    candidates: list[dict[str, Any]] = []
    has_previous = bool(report.get("changes", {}).get("has_previous"))
    for item in report.get("items", []):
        change = item.get("change", {})
        alert_type = None
        severity = "info"
        if change.get("state") == "entered_top":
            alert_type, severity = "entered_top", "high"
            title = f"{item['symbol']} entered Top Opportunities"
        elif not has_previous and item.get("bucket") == "top_opportunity":
            alert_type, severity = "top_opportunity", "high"
            title = f"{item['symbol']} is a Top Opportunity"
        elif change.get("state") == "deteriorated":
            alert_type, severity = "deteriorated", "warning"
            title = f"{item['symbol']} evidence deteriorated"
        elif change.get("score_delta") is not None and change["score_delta"] >= 10:
            alert_type = "score_jump"
            title = f"{item['symbol']} score increased sharply"
        if not alert_type:
            continue
        previous_score = change.get("previous_score")
        delta_text = (
            f" ({change['score_delta']:+.1f} vs prior scan)"
            if change.get("score_delta") is not None
            else ""
        )
        candidates.append(
            {
                "symbol": item["symbol"],
                "alert_type": alert_type,
                "severity": severity,
                "title": title,
                "message": (
                    f"Opportunity score {item['opportunity_score']:.1f}; "
                    f"bucket {item['bucket'].replace('_', ' ')}{delta_text}."
                ),
                "current_score": item.get("opportunity_score"),
                "previous_score": previous_score,
            }
        )
    now = datetime.now(UTC)
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                for alert in candidates:
                    alert_id = hashlib.sha256(
                        f"{scan_id}|{alert['symbol']}|{alert['alert_type']}".encode()
                    ).hexdigest()
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO opportunity_alerts VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL
                        )
                        """,
                        [
                            alert_id,
                            scan_id,
                            alert["symbol"],
                            alert["alert_type"],
                            alert["severity"],
                            alert["title"],
                            alert["message"],
                            alert["current_score"],
                            alert["previous_score"],
                            now,
                        ],
                    )
                    alert["alert_id"] = alert_id
                    alert["created_at"] = now.isoformat()
            finally:
                connection.close()
        return candidates
    except Exception as exc:  # noqa: BLE001 - alert failure must not erase a scan
        logger.warning("Could not capture opportunity alerts: %s", exc)
        return []


def list_opportunity_alerts(
    config: dict[str, Any] | None = None,
    *,
    limit: int = 50,
    unacknowledged_only: bool = False,
) -> list[dict[str, Any]]:
    """List recent opportunity alerts."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return []
    where = "WHERE acknowledged_at IS NULL" if unacknowledged_only else ""
    bounded_limit = max(1, min(int(limit), 200))
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                rows = connection.execute(
                    f"""
                    SELECT alert_id, scan_id, symbol, alert_type, severity, title,
                           message, current_score, previous_score, created_at, acknowledged_at
                    FROM opportunity_alerts
                    {where}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [bounded_limit],
                ).fetchall()
                columns = [item[0] for item in connection.description]
            finally:
                connection.close()
        return [
            {
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in zip(columns, row, strict=True)
            }
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001 - alert center remains usable after read failure
        logger.warning("Could not list opportunity alerts: %s", exc)
        return []


def acknowledge_opportunity_alert(
    alert_id: str,
    config: dict[str, Any] | None = None,
) -> bool:
    """Mark an alert as acknowledged."""
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        return False
    try:
        with _LOCK:
            connection = _connect(warehouse_path(cfg))
            try:
                updated = connection.execute(
                    """
                    UPDATE opportunity_alerts SET acknowledged_at = ?
                    WHERE alert_id = ? RETURNING alert_id
                    """,
                    [datetime.now(UTC), alert_id],
                ).fetchone()
            finally:
                connection.close()
        return updated is not None
    except Exception as exc:  # noqa: BLE001 - acknowledgement failure is returned to the API
        logger.warning("Could not acknowledge opportunity alert: %s", exc)
        return False


def get_warehouse_status(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a non-sensitive health summary for readiness and result metadata."""
    cfg = config or get_config()
    enabled = bool(cfg.get("data_warehouse_enabled", False))
    path = warehouse_path(cfg)
    status: dict[str, Any] = {
        "enabled": enabled,
        "dependency_available": dependency_available(),
        "ready": False,
        "observation_count": 0,
        "market_bar_count": 0,
        "quantitative_snapshot_count": 0,
        "validation_experiment_count": 0,
        "portfolio_validation_experiment_count": 0,
        "opportunity_scan_count": 0,
        "scan_universe_count": 0,
        "active_scan_schedule_count": 0,
        "unacknowledged_alert_count": 0,
        "quality_failures": 0,
    }
    if not enabled or not status["dependency_available"]:
        return status

    try:
        with _LOCK:
            connection = _connect(path)
            try:
                status["observation_count"] = int(
                    connection.execute("SELECT count(*) FROM vendor_observations").fetchone()[0]
                )
                status["market_bar_count"] = int(
                    connection.execute("SELECT count(*) FROM market_bars").fetchone()[0]
                )
                status["quantitative_snapshot_count"] = int(
                    connection.execute("SELECT count(*) FROM quantitative_snapshots").fetchone()[0]
                )
                status["validation_experiment_count"] = int(
                    connection.execute("SELECT count(*) FROM validation_experiments").fetchone()[0]
                )
                status["portfolio_validation_experiment_count"] = int(
                    connection.execute(
                        "SELECT count(*) FROM portfolio_validation_experiments"
                    ).fetchone()[0]
                )
                status["opportunity_scan_count"] = int(
                    connection.execute("SELECT count(*) FROM opportunity_scans").fetchone()[0]
                )
                status["scan_universe_count"] = int(
                    connection.execute("SELECT count(*) FROM scan_universes").fetchone()[0]
                )
                status["active_scan_schedule_count"] = int(
                    connection.execute(
                        "SELECT count(*) FROM scan_schedules WHERE active"
                    ).fetchone()[0]
                )
                status["unacknowledged_alert_count"] = int(
                    connection.execute(
                        "SELECT count(*) FROM opportunity_alerts WHERE acknowledged_at IS NULL"
                    ).fetchone()[0]
                )
                status["quality_failures"] = int(
                    connection.execute(
                        "SELECT count(*) FROM ingestion_quality WHERE NOT passed"
                    ).fetchone()[0]
                )
            finally:
                connection.close()
        status["ready"] = True
    except Exception as exc:  # noqa: BLE001 - health reports degradation
        logger.warning("Market-data warehouse is unavailable: %s", exc)
    return status
