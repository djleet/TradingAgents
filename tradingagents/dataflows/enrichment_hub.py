"""Multi-source enrichment snapshots for the local research dashboard.

Live vendor snapshots are deliberately labelled separately from historical,
point-in-time evidence.  They can inform a decision made after ``observed_at``
but must not be silently injected into a backtest dated earlier than that.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.fred import FredNotConfiguredError, _request
from tradingagents.dataflows.vendors.yahoo.common import yf_retry
from tradingagents.dataflows.warehouse import _connect as _warehouse_connect, warehouse_path

ENRICHMENT_VERSION = "data-enrichment-hub-v1"
DATASETS = (
    "earnings_estimates",
    "options_implied_volatility",
    "institutional_ownership",
    "sec_insider_filings",
    "macro_calendar",
)

_MACRO_SERIES = {
    "fed_funds": {"series_id": "FEDFUNDS", "label": "Fed funds rate", "unit": "%"},
    "treasury_10y": {"series_id": "DGS10", "label": "US 10Y Treasury", "unit": "%"},
    "yield_curve": {"series_id": "T10Y2Y", "label": "10Y–2Y spread", "unit": "pp"},
    "vix": {"series_id": "VIXCLS", "label": "VIX", "unit": "index"},
    "unemployment": {"series_id": "UNRATE", "label": "Unemployment", "unit": "%"},
    "cpi_yoy": {
        "series_id": "CPIAUCSL",
        "label": "CPI year over year",
        "unit": "%",
        "units": "pc1",
    },
}

_IMPORTANT_RELEASES = (
    "consumer price index",
    "employment situation",
    "gross domestic product",
    "personal income and outlays",
    "producer price index",
    "advance monthly sales",
    "fomc",
    "industrial production",
    "job openings",
)


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if math.isfinite(result) else None


def _frame_rows(value: Any, limit: int = 12) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.Series):
        value = value.to_frame().T
    if not isinstance(value, pd.DataFrame) or value.empty:
        return []
    frame = value.reset_index()
    return [_jsonable(row) for row in frame.head(limit).to_dict(orient="records")]


def _dataset(
    name: str,
    provider: str,
    authority: str,
    *,
    data: dict[str, Any] | None = None,
    status: str = "available",
    completeness: float = 1.0,
    point_in_time: bool,
    caveat: str,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": name,
        "provider": provider,
        "authority": authority,
        "status": status,
        "completeness": round(max(0.0, min(float(completeness), 1.0)), 3),
        "point_in_time": point_in_time,
        "caveat": caveat,
        "error": error,
        "data": data or {},
    }


def _live_caveat(name: str) -> str:
    return (
        f"{name} is a current vendor snapshot observed now. It is suitable for new "
        "research decisions, but not as historical evidence for an earlier backtest."
    )


def _collect_earnings_estimates(symbol: str, _as_of_date: str) -> dict[str, Any]:
    def load():
        ticker = yf.Ticker(symbol)
        return {
            "calendar": ticker.calendar,
            "earnings": ticker.earnings_estimate,
            "revenue": ticker.revenue_estimate,
            "growth": ticker.growth_estimates,
            "targets": ticker.analyst_price_targets,
            "recommendations": ticker.recommendations_summary,
        }

    raw = yf_retry(load)
    calendar = _jsonable(raw.get("calendar") or {})
    if isinstance(calendar, list):
        calendar = calendar[0] if calendar else {}
    targets = _jsonable(raw.get("targets") or {})
    earnings_rows = _frame_rows(raw.get("earnings"), 8)
    revenue_rows = _frame_rows(raw.get("revenue"), 8)
    growth_rows = _frame_rows(raw.get("growth"), 12)
    recommendations = _frame_rows(raw.get("recommendations"), 8)

    earnings_date = calendar.get("Earnings Date") or calendar.get("earningsDate")
    if isinstance(earnings_date, list):
        earnings_date = earnings_date[0] if earnings_date else None
    current = _number(targets.get("current"))
    mean = _number(targets.get("mean"))
    upside = (mean / current - 1) if mean is not None and current else None
    populated = sum(
        bool(value)
        for value in (calendar, earnings_rows, revenue_rows, targets, recommendations)
    )
    return _dataset(
        "earnings_estimates",
        "Yahoo Finance",
        "aggregated_analyst",
        data={
            "earnings_date": _jsonable(earnings_date),
            "calendar": calendar,
            "earnings_estimates": earnings_rows,
            "revenue_estimates": revenue_rows,
            "growth_estimates": growth_rows,
            "price_targets": targets,
            "target_upside": _number(upside),
            "recommendations": recommendations,
        },
        status="available" if populated >= 3 else "partial" if populated else "unavailable",
        completeness=populated / 5,
        point_in_time=False,
        caveat=_live_caveat("Analyst and earnings data"),
    )


def _sum_column(frame: pd.DataFrame, column: str) -> float | None:
    if frame is None or frame.empty or column not in frame:
        return None
    value = pd.to_numeric(frame[column], errors="coerce").sum(min_count=1)
    return _number(value, 2)


def _collect_options(symbol: str, _as_of_date: str) -> dict[str, Any]:
    def load():
        ticker = yf.Ticker(symbol)
        expirations = list(ticker.options or [])
        valid = [item for item in expirations if item >= date.today().isoformat()]
        if not valid:
            return {"expirations": expirations, "chain": None, "spot": None}
        expiry = valid[0]
        chain = ticker.option_chain(expiry)
        fast = ticker.fast_info
        try:
            spot = fast["last_price"]
        except (KeyError, TypeError):
            spot = getattr(fast, "last_price", None)
        return {"expirations": expirations, "expiry": expiry, "chain": chain, "spot": spot}

    raw = yf_retry(load)
    chain = raw.get("chain")
    if chain is None:
        return _dataset(
            "options_implied_volatility",
            "Yahoo Finance",
            "exchange_derived",
            status="unavailable",
            completeness=0,
            point_in_time=False,
            caveat=_live_caveat("Options data"),
            error="No listed option chain was available for this symbol.",
        )
    calls = chain.calls
    puts = chain.puts
    spot = _number(raw.get("spot"), 4)
    expiry = raw.get("expiry")
    dte = max(0, (date.fromisoformat(expiry) - date.today()).days)

    def atm_row(frame: pd.DataFrame) -> dict[str, Any]:
        if frame is None or frame.empty or spot is None or "strike" not in frame:
            return {}
        index = (pd.to_numeric(frame["strike"], errors="coerce") - spot).abs().idxmin()
        return _jsonable(frame.loc[index].to_dict())

    call_atm = atm_row(calls)
    put_atm = atm_row(puts)
    call_iv = _number(call_atm.get("impliedVolatility"))
    put_iv = _number(put_atm.get("impliedVolatility"))
    iv_values = [value for value in (call_iv, put_iv) if value is not None]
    atm_iv = sum(iv_values) / len(iv_values) if iv_values else None
    expected_move = atm_iv * math.sqrt(max(dte, 1) / 365) if atm_iv is not None else None
    call_volume = _sum_column(calls, "volume")
    put_volume = _sum_column(puts, "volume")
    call_oi = _sum_column(calls, "openInterest")
    put_oi = _sum_column(puts, "openInterest")
    volume_ratio = put_volume / call_volume if put_volume is not None and call_volume else None
    oi_ratio = put_oi / call_oi if put_oi is not None and call_oi else None
    populated = sum(value is not None for value in (spot, atm_iv, volume_ratio, oi_ratio))
    return _dataset(
        "options_implied_volatility",
        "Yahoo Finance",
        "exchange_derived",
        data={
            "spot": spot,
            "nearest_expiry": expiry,
            "days_to_expiry": dte,
            "expirations": raw.get("expirations", [])[:8],
            "atm_call_iv": call_iv,
            "atm_put_iv": put_iv,
            "atm_iv": _number(atm_iv),
            "iv_skew": _number(put_iv - call_iv) if put_iv is not None and call_iv is not None else None,
            "expected_move_pct": _number(expected_move),
            "put_call_volume": _number(volume_ratio),
            "put_call_open_interest": _number(oi_ratio),
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_open_interest": call_oi,
            "put_open_interest": put_oi,
            "atm_call": call_atm,
            "atm_put": put_atm,
        },
        status="available" if populated >= 3 else "partial",
        completeness=populated / 4,
        point_in_time=False,
        caveat=_live_caveat("Options data"),
    )


def _collect_ownership(symbol: str, _as_of_date: str) -> dict[str, Any]:
    def load():
        ticker = yf.Ticker(symbol)
        return {
            "institutional": ticker.institutional_holders,
            "major": ticker.major_holders,
            "insiders": ticker.insider_transactions,
        }

    raw = yf_retry(load)
    institutional = _frame_rows(raw.get("institutional"), 12)
    major = _frame_rows(raw.get("major"), 10)
    insiders = _frame_rows(raw.get("insiders"), 16)
    populated = sum(bool(value) for value in (institutional, major, insiders))
    return _dataset(
        "institutional_ownership",
        "Yahoo Finance",
        "aggregated_filings",
        data={
            "institutional_holders": institutional,
            "major_holders": major,
            "insider_transactions": insiders,
            "institutional_holder_count": len(institutional),
            "insider_transaction_count": len(insiders),
        },
        status="available" if populated >= 2 else "partial" if populated else "unavailable",
        completeness=populated / 3,
        point_in_time=False,
        caveat=_live_caveat("Ownership tables"),
    )


def _parse_sec_filings(payload: str) -> list[dict[str, Any]]:
    rows = []
    for line in payload.splitlines():
        if not line.startswith("|") or line.startswith(("|---", "| Accepted")):
            continue
        cells = next(csv.reader(io.StringIO(line.strip("|")), delimiter="|"), [])
        cells = [cell.strip() for cell in cells]
        if len(cells) < 5 or len(cells[0]) < 10:
            continue
        accepted_date = cells[0][:10]
        try:
            date.fromisoformat(accepted_date)
        except ValueError:
            continue
        url = cells[4].split("](", 1)[1][:-1] if "](" in cells[4] else None
        rows.append(
            {
                "accepted_at": cells[0].replace(" ", "T", 1),
                "accepted_date": accepted_date,
                "form": cells[1],
                "report_date": None if cells[2] == "—" else cells[2],
                "description": cells[3],
                "url": url,
            }
        )
    return rows


def _collect_sec_insiders(symbol: str, as_of_date: str) -> dict[str, Any]:
    payload = route_to_vendor("get_sec_filings", symbol, as_of_date, "3,4,5", 30)
    filings = _parse_sec_filings(payload)
    return _dataset(
        "sec_insider_filings",
        "SEC EDGAR",
        "official",
        data={
            "filings": filings,
            "filing_count": len(filings),
            "recent_30d_count": sum(
                0 <= (date.fromisoformat(as_of_date) - date.fromisoformat(item["accepted_date"])).days <= 30
                for item in filings
            ),
        },
        status="available",
        completeness=1.0 if filings else 0.65,
        point_in_time=True,
        caveat=(
            "SEC acceptance timestamps are bounded by the selected date. Filing counts "
            "do not by themselves identify purchase or sale direction."
        ),
    )


def _collect_macro(_symbol: str, as_of_date: str) -> dict[str, Any]:
    if not os.getenv("FRED_API_KEY"):
        raise FredNotConfiguredError("FRED_API_KEY is not configured")
    start = (date.fromisoformat(as_of_date) - timedelta(days=430)).isoformat()
    indicators = []
    for key, definition in _MACRO_SERIES.items():
        params = {
            "series_id": definition["series_id"],
            "observation_start": start,
            "observation_end": as_of_date,
            "sort_order": "desc",
            "limit": 14,
            "realtime_start": as_of_date,
            "realtime_end": as_of_date,
        }
        if definition.get("units"):
            params["units"] = definition["units"]
        rows = _request("series/observations", params).get("observations", [])
        usable = [row for row in rows if row.get("value") not in {None, "."}]
        if usable:
            latest = usable[0]
            previous = usable[1] if len(usable) > 1 else None
            latest_value = _number(latest.get("value"))
            previous_value = _number(previous.get("value")) if previous else None
            indicators.append(
                {
                    "key": key,
                    **definition,
                    "value": latest_value,
                    "previous_value": previous_value,
                    "change": _number(latest_value - previous_value)
                    if latest_value is not None and previous_value is not None
                    else None,
                    "observation_date": latest.get("date"),
                }
            )

    release_rows = _request(
        "releases/dates",
        {
            "include_release_dates_with_no_data": "true",
            "sort_order": "desc",
            "order_by": "release_date",
            "limit": 1000,
        },
    ).get("release_dates", [])
    cutoff = date.fromisoformat(as_of_date)
    horizon = cutoff + timedelta(days=45)
    releases = []
    seen = set()
    for item in release_rows:
        name = str(item.get("release_name") or "")
        try:
            release_date = date.fromisoformat(item.get("date"))
        except (TypeError, ValueError):
            continue
        if not cutoff <= release_date <= horizon:
            continue
        if not any(token in name.lower() for token in _IMPORTANT_RELEASES):
            continue
        identity = (name, release_date)
        if identity in seen:
            continue
        seen.add(identity)
        releases.append(
            {
                "release_id": item.get("release_id"),
                "name": name,
                "date": release_date.isoformat(),
                "days_until": (release_date - cutoff).days,
            }
        )
    releases.sort(key=lambda item: (item["date"], item["name"]))
    completeness = (len(indicators) / len(_MACRO_SERIES) + (1 if releases else 0)) / 2
    return _dataset(
        "macro_calendar",
        "Federal Reserve Bank of St. Louis (FRED/ALFRED)",
        "official",
        data={"indicators": indicators, "releases": releases[:20]},
        status="available" if completeness >= 0.75 else "partial",
        completeness=completeness,
        point_in_time=True,
        caveat=(
            "FRED observations are vintage-pinned to the selected date. Calendar dates "
            "come from source-published schedules and may precede FRED availability."
        ),
    )


_COLLECTORS: dict[str, Callable[[str, str], dict[str, Any]]] = {
    "earnings_estimates": _collect_earnings_estimates,
    "options_implied_volatility": _collect_options,
    "institutional_ownership": _collect_ownership,
    "sec_insider_filings": _collect_sec_insiders,
    "macro_calendar": _collect_macro,
}


def collect_enrichment_sources(symbol: str, as_of_date: str) -> dict[str, dict[str, Any]]:
    """Collect independent sources concurrently and preserve partial results."""
    canonical = normalize_symbol(symbol)
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(_COLLECTORS)) as executor:
        futures = {
            executor.submit(collector, canonical, as_of_date): name
            for name, collector in _COLLECTORS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except FredNotConfiguredError:
                results[name] = _dataset(
                    name,
                    "FRED/ALFRED",
                    "official",
                    status="unavailable",
                    completeness=0,
                    point_in_time=True,
                    caveat="Add FRED_API_KEY to enable vintage macro data and release dates.",
                    error="FRED is not configured.",
                )
            except Exception:  # noqa: BLE001 - one failed source must not discard the rest
                provider = "SEC EDGAR" if name == "sec_insider_filings" else "Yahoo Finance"
                if name == "macro_calendar":
                    provider = "FRED/ALFRED"
                results[name] = _dataset(
                    name,
                    provider,
                    "official" if name in {"macro_calendar", "sec_insider_filings"} else "aggregator",
                    status="unavailable",
                    completeness=0,
                    point_in_time=name in {"macro_calendar", "sec_insider_filings"},
                    caveat="The source did not return usable evidence for this refresh.",
                    error="Source temporarily unavailable.",
                )
    return results


def _score_source(source: dict[str, Any], observed_at: str) -> dict[str, Any]:
    authority = {
        "official": 94,
        "exchange_derived": 80,
        "aggregated_filings": 70,
        "aggregated_analyst": 64,
        "aggregator": 60,
    }.get(source.get("authority"), 55)
    completeness = float(source.get("completeness") or 0)
    if source.get("status") == "unavailable":
        confidence = 0.0
    else:
        confidence = authority * 0.62 + completeness * 100 * 0.28 + 10
        if source.get("status") == "partial":
            confidence *= 0.82
    result = {key: value for key, value in source.items() if key != "data"}
    result.update(
        {
            "confidence_score": round(min(confidence, 100), 1),
            "observed_at": observed_at,
            "item_count": _source_item_count(source),
        }
    )
    return result


def _source_item_count(source: dict[str, Any]) -> int:
    data = source.get("data") or {}
    counts = []
    for value in data.values():
        if isinstance(value, list):
            counts.append(len(value))
    return max(counts, default=1 if data else 0)


def _decision_flags(datasets: dict[str, dict[str, Any]], as_of_date: str) -> list[dict[str, Any]]:
    flags = []
    earnings = (datasets.get("earnings_estimates") or {}).get("data") or {}
    earnings_date = earnings.get("earnings_date")
    if earnings_date:
        try:
            days = (date.fromisoformat(str(earnings_date)[:10]) - date.fromisoformat(as_of_date)).days
        except ValueError:
            days = None
        if days is not None and 0 <= days <= 14:
            flags.append(
                {
                    "type": "earnings_window",
                    "severity": "high" if days <= 7 else "medium",
                    "title": "Confirmed earnings window is near",
                    "message": f"The vendor calendar places earnings in {days} day{'s' if days != 1 else ''}.",
                }
            )
    options = (datasets.get("options_implied_volatility") or {}).get("data") or {}
    expected_move = _number(options.get("expected_move_pct"))
    if expected_move is not None and expected_move >= 0.08:
        flags.append(
            {
                "type": "options_expected_move",
                "severity": "medium",
                "title": "Options imply an elevated near-term move",
                "message": f"Nearest-expiry ATM volatility implies approximately {expected_move:.1%}.",
            }
        )
    put_call = _number(options.get("put_call_open_interest"))
    if put_call is not None and put_call >= 1.5:
        flags.append(
            {
                "type": "put_call_skew",
                "severity": "medium",
                "title": "Put open interest outweighs calls",
                "message": f"Nearest-expiry put/call open-interest ratio is {put_call:.2f}.",
            }
        )
    unavailable = [name for name, item in datasets.items() if item.get("status") == "unavailable"]
    if unavailable:
        flags.append(
            {
                "type": "coverage_gap",
                "severity": "info",
                "title": "Enrichment coverage is incomplete",
                "message": f"{len(unavailable)} of {len(DATASETS)} source groups are unavailable.",
            }
        )
    return flags


def build_enrichment_snapshot(
    symbol: str,
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    source_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one explainable enrichment snapshot without an LLM call."""
    del config  # reserved for provider configuration and kept for API symmetry
    canonical = normalize_symbol(symbol)
    date.fromisoformat(as_of_date)
    observed_at = datetime.now(UTC).isoformat()
    supplied = source_payloads if source_payloads is not None else collect_enrichment_sources(canonical, as_of_date)
    datasets = {
        name: supplied.get(name)
        or _dataset(
            name,
            "Not collected",
            "aggregator",
            status="unavailable",
            completeness=0,
            point_in_time=False,
            caveat="This source group was not collected.",
            error="Missing source payload.",
        )
        for name in DATASETS
    }
    sources = [_score_source(datasets[name], observed_at) for name in DATASETS]
    available = [source for source in sources if source["status"] != "unavailable"]
    full = [source for source in sources if source["status"] == "available"]
    partial = [source for source in sources if source["status"] == "partial"]
    avg_confidence = (
        sum(source["confidence_score"] for source in available) / len(available)
        if available
        else 0
    )
    coverage_score = 100 * (
        0.62 * (len(available) / len(DATASETS)) + 0.38 * (avg_confidence / 100)
    )
    if coverage_score >= 78:
        coverage_label = "robust"
    elif coverage_score >= 58:
        coverage_label = "usable"
    elif coverage_score >= 35:
        coverage_label = "partial"
    else:
        coverage_label = "limited"
    point_in_time = [source for source in sources if source["point_in_time"] and source["status"] != "unavailable"]
    return {
        "model_version": ENRICHMENT_VERSION,
        "symbol": canonical,
        "as_of_date": as_of_date,
        "observed_at": observed_at,
        "coverage": {
            "score": round(coverage_score, 1),
            "label": coverage_label,
            "available_sources": len(full),
            "partial_sources": len(partial),
            "unavailable_sources": len(DATASETS) - len(available),
            "point_in_time_sources": len(point_in_time),
            "average_confidence": round(avg_confidence, 1),
        },
        "sources": sources,
        "earnings": datasets["earnings_estimates"]["data"],
        "options": datasets["options_implied_volatility"]["data"],
        "ownership": datasets["institutional_ownership"]["data"],
        "sec_insiders": datasets["sec_insider_filings"]["data"],
        "macro": datasets["macro_calendar"]["data"],
        "decision_flags": _decision_flags(datasets, as_of_date),
        "policy": {
            "historical_eligible": [source["dataset"] for source in point_in_time],
            "live_only": [
                source["dataset"]
                for source in sources
                if not source["point_in_time"] and source["status"] != "unavailable"
            ],
            "notice": (
                "Live-only sources are never treated as if they were known on an earlier "
                "analysis date. Every saved snapshot retains its observation time."
            ),
        },
    }


def _ensure_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            model_version VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            observed_at TIMESTAMPTZ NOT NULL,
            coverage_score DOUBLE NOT NULL,
            coverage_label VARCHAR NOT NULL,
            available_sources INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_enrichment_snapshots_lookup
        ON enrichment_snapshots(symbol, as_of_date, created_at)
        """
    )


def _connect(config: dict[str, Any] | None = None):
    cfg = config or get_config()
    connection = _warehouse_connect(warehouse_path(cfg))
    _ensure_schema(connection)
    return connection


def capture_enrichment_snapshot(
    snapshot: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    semantic = {
        key: value
        for key, value in snapshot.items()
        if key not in {"observed_at", "registry"}
    }
    snapshot_id = hashlib.sha256(_safe_json(semantic).encode()).hexdigest()
    connection = _connect(config)
    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO enrichment_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_id,
                snapshot["model_version"],
                snapshot["symbol"],
                snapshot["as_of_date"],
                snapshot["observed_at"],
                snapshot["coverage"]["score"],
                snapshot["coverage"]["label"],
                snapshot["coverage"]["available_sources"],
                _safe_json(snapshot),
                datetime.now(UTC),
            ],
        )
        row = connection.execute(
            "SELECT created_at FROM enrichment_snapshots WHERE snapshot_id = ?",
            [snapshot_id],
        ).fetchone()
    finally:
        connection.close()
    return {
        "saved": True,
        "snapshot_id": snapshot_id,
        "created_at": row[0].astimezone(UTC).isoformat(),
    }


def get_latest_enrichment_snapshot(
    config: dict[str, Any] | None = None, *, symbol: str | None = None
) -> dict[str, Any] | None:
    connection = _connect(config)
    try:
        if symbol:
            row = connection.execute(
                """
                SELECT snapshot_id, payload_json, created_at FROM enrichment_snapshots
                WHERE symbol = ? ORDER BY created_at DESC LIMIT 1
                """,
                [normalize_symbol(symbol)],
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT snapshot_id, payload_json, created_at FROM enrichment_snapshots
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    snapshot = json.loads(row[1])
    snapshot["registry"] = {
        "saved": True,
        "snapshot_id": row[0],
        "created_at": row[2].astimezone(UTC).isoformat(),
    }
    return snapshot


def list_enrichment_snapshots(
    config: dict[str, Any] | None = None,
    *,
    symbol: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    connection = _connect(config)
    try:
        if symbol:
            rows = connection.execute(
                """
                SELECT snapshot_id, symbol, as_of_date, coverage_score, coverage_label,
                       available_sources, created_at
                FROM enrichment_snapshots WHERE symbol = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                [normalize_symbol(symbol), max(1, min(int(limit), 100))],
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT snapshot_id, symbol, as_of_date, coverage_score, coverage_label,
                       available_sources, created_at
                FROM enrichment_snapshots ORDER BY created_at DESC LIMIT ?
                """,
                [max(1, min(int(limit), 100))],
            ).fetchall()
    finally:
        connection.close()
    return [
        {
            "snapshot_id": row[0],
            "symbol": row[1],
            "as_of_date": row[2].isoformat(),
            "coverage_score": row[3],
            "coverage_label": row[4],
            "available_sources": row[5],
            "created_at": row[6].astimezone(UTC).isoformat(),
        }
        for row in rows
    ]
