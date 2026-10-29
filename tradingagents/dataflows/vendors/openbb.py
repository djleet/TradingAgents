"""Optional OpenBB adapter for historical OHLCV data.

OpenBB is kept behind a lazy import because its provider bundle is large and
not every TradingAgents installation needs it.  Select it explicitly with
``TRADINGAGENTS_CORE_STOCK_VENDORS=openbb,yfinance``.  The trailing Yahoo
fallback is intentional: an OpenBB installation may not include a provider for
every exchange.
"""

from __future__ import annotations

import os

import pandas as pd

from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)


def _load_obb():
    try:
        from openbb import obb
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "OpenBB is not installed; install tradingagents[openbb]"
        ) from exc
    return obb


def _as_frame(result) -> pd.DataFrame:
    if hasattr(result, "to_df"):
        frame = result.to_df()
    elif isinstance(result, pd.DataFrame):
        frame = result.copy()
    else:
        raise VendorUnavailableError("OpenBB returned an unsupported result type")
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return frame


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    """Return an OpenBB historical-price result using the core CSV contract."""
    provider = os.getenv("OPENBB_PROVIDER", "").strip() or None
    kwargs = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
    if provider:
        kwargs["provider"] = provider

    try:
        result = _load_obb().equity.price.historical(**kwargs)
        frame = _as_frame(result)
    except (VendorNotConfiguredError, NoMarketDataError, VendorUnavailableError):
        raise
    except Exception as exc:
        raise VendorUnavailableError(
            f"OpenBB historical-price request failed ({type(exc).__name__})"
        ) from exc

    if frame.empty:
        raise NoMarketDataError(symbol, symbol, "OpenBB returned no historical rows")

    # OpenBB providers do not all shape their date column identically. Normalize
    # the index/column names into the CSV contract consumed by the market agent.
    if "date" not in {str(column).lower() for column in frame.columns}:
        frame = frame.reset_index()
    lower_names = {str(column).lower().replace("_", " "): column for column in frame.columns}
    date_column = lower_names.get("date") or lower_names.get("datetime") or frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce", utc=True)
    frame = frame.dropna(subset=[date_column])
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    frame = frame[(frame[date_column] >= start) & (frame[date_column] <= end)]
    if frame.empty:
        raise NoMarketDataError(
            symbol,
            symbol,
            f"OpenBB returned no rows between {start_date} and {end_date}",
        )

    rename = {date_column: "Date"}
    desired = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjusted close": "Adj Close",
        "volume": "Volume",
    }
    for normalized, canonical in desired.items():
        original = lower_names.get(normalized)
        if original is not None:
            rename[original] = canonical
    frame = frame.rename(columns=rename)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    columns = [name for name in ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume") if name in frame]
    if "Close" not in columns:
        raise VendorUnavailableError("OpenBB response did not contain a close-price column")
    return frame.loc[:, columns].sort_values("Date").to_csv(index=False)

