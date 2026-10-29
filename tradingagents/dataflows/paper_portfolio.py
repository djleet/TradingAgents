"""Durable paper portfolio and decision journal for research validation.

The tracker is deliberately execution-free: it records hypothetical long-only
positions, sizes them from an explicit risk budget, and marks them with the same
point-in-time market data used elsewhere in the research dashboard.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import (
    dependency_available,
    parse_ohlcv_payload,
    warehouse_path,
)

_LOCK = threading.RLock()


class PaperPortfolioUnavailableError(RuntimeError):
    """Raised when durable paper-trading storage is disabled or unavailable."""


def _safe_json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False) or not dependency_available():
        raise PaperPortfolioUnavailableError(
            "The local research warehouse is required for the paper portfolio."
        )
    return cfg


def _connect(config: dict[str, Any] | None = None):
    import duckdb

    cfg = _config(config)
    path = warehouse_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_portfolios (
            portfolio_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            benchmark_symbol VARCHAR NOT NULL,
            initial_cash DOUBLE NOT NULL,
            cash DOUBLE NOT NULL,
            default_risk_pct DOUBLE NOT NULL,
            max_position_pct DOUBLE NOT NULL,
            transaction_cost_bps DOUBLE NOT NULL,
            benchmark_start_price DOUBLE,
            benchmark_current_price DOUBLE,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        "ALTER TABLE paper_portfolios ADD COLUMN IF NOT EXISTS monitoring_enabled BOOLEAN DEFAULT true"
    )
    connection.execute(
        "ALTER TABLE paper_portfolios ADD COLUMN IF NOT EXISTS monitor_local_time VARCHAR DEFAULT '16:45'"
    )
    connection.execute(
        "ALTER TABLE paper_portfolios ADD COLUMN IF NOT EXISTS monitor_timezone VARCHAR DEFAULT 'America/New_York'"
    )
    connection.execute(
        "ALTER TABLE paper_portfolios ADD COLUMN IF NOT EXISTS last_monitor_date DATE"
    )
    connection.execute(
        "ALTER TABLE paper_portfolios ADD COLUMN IF NOT EXISTS last_monitor_status VARCHAR"
    )
    connection.execute(
        "ALTER TABLE paper_portfolios ADD COLUMN IF NOT EXISTS last_monitor_error VARCHAR"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_positions (
            position_id VARCHAR PRIMARY KEY,
            portfolio_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            shares BIGINT NOT NULL,
            entry_date DATE NOT NULL,
            entry_price DOUBLE NOT NULL,
            stop_price DOUBLE NOT NULL,
            target_price DOUBLE NOT NULL,
            entry_fee DOUBLE NOT NULL,
            initial_score DOUBLE,
            initial_bucket VARCHAR,
            initial_rank INTEGER,
            scan_id VARCHAR,
            decision_note VARCHAR NOT NULL,
            evidence_json VARCHAR NOT NULL,
            latest_price DOUBLE NOT NULL,
            latest_price_date DATE NOT NULL,
            exit_date DATE,
            exit_price DOUBLE,
            exit_fee DOUBLE,
            exit_reason VARCHAR,
            status VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_positions_portfolio
        ON paper_positions(portfolio_id, status, updated_at)
        """
    )
    connection.execute(
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS latest_context_json VARCHAR"
    )
    connection.execute(
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS latest_context_date DATE"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_decisions (
            decision_id VARCHAR PRIMARY KEY,
            portfolio_id VARCHAR NOT NULL,
            position_id VARCHAR NOT NULL,
            decision_type VARCHAR NOT NULL,
            decision_date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            price DOUBLE NOT NULL,
            note VARCHAR NOT NULL,
            snapshot_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_decisions_portfolio
        ON paper_decisions(portfolio_id, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_snapshots (
            snapshot_id VARCHAR PRIMARY KEY,
            portfolio_id VARCHAR NOT NULL,
            snapshot_date DATE NOT NULL,
            nav DOUBLE NOT NULL,
            cash DOUBLE NOT NULL,
            market_value DOUBLE NOT NULL,
            realized_pnl DOUBLE NOT NULL,
            unrealized_pnl DOUBLE NOT NULL,
            benchmark_value DOUBLE NOT NULL,
            daily_pnl DOUBLE NOT NULL,
            drawdown DOUBLE NOT NULL,
            prices_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_snapshots_portfolio
        ON paper_snapshots(portfolio_id, snapshot_date, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_alerts (
            alert_id VARCHAR PRIMARY KEY,
            portfolio_id VARCHAR NOT NULL,
            position_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            alert_type VARCHAR NOT NULL,
            severity VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            message VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            detail_json VARCHAR NOT NULL,
            acknowledged_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_alerts_status
        ON paper_alerts(portfolio_id, acknowledged_at, created_at)
        """
    )
    return connection


def load_close_prices(symbols: list[str], as_of_date: str) -> dict[str, float]:
    """Load the last valid close known on or before ``as_of_date``."""
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    start_date = (as_of - timedelta(days=14)).isoformat()
    prices: dict[str, float] = {}
    for symbol in dict.fromkeys(symbols):
        payload = route_to_vendor("get_stock_data", symbol, start_date, as_of_date)
        frame = parse_ohlcv_payload(payload)
        frame = frame[(frame["Date"] <= as_of) & frame["Close"].notna()].copy()
        frame = frame[frame["Close"] > 0].sort_values("Date")
        if frame.empty:
            raise ValueError(f"No usable closing price for {symbol} on {as_of_date}")
        prices[symbol] = float(frame.iloc[-1]["Close"])
    return prices


def ensure_default_paper_portfolio(
    config: dict[str, Any] | None = None,
    *,
    name: str = "Research Portfolio",
    benchmark_symbol: str = "SPY",
    initial_cash: float = 100_000.0,
) -> str:
    """Return the default portfolio id, creating it on first use."""
    now = datetime.now(UTC)
    with _LOCK:
        connection = _connect(config)
        try:
            existing = connection.execute(
                "SELECT portfolio_id FROM paper_portfolios ORDER BY created_at LIMIT 1"
            ).fetchone()
            if existing:
                return str(existing[0])
            portfolio_id = hashlib.sha256(
                f"{name}|{benchmark_symbol}|{now.isoformat()}".encode()
            ).hexdigest()[:24]
            connection.execute(
                """
                INSERT INTO paper_portfolios (
                    portfolio_id, name, benchmark_symbol, initial_cash, cash,
                    default_risk_pct, max_position_pct, transaction_cost_bps,
                    benchmark_start_price, benchmark_current_price, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    portfolio_id,
                    name,
                    benchmark_symbol,
                    initial_cash,
                    initial_cash,
                    1.0,
                    20.0,
                    5.0,
                    None,
                    None,
                    now,
                    now,
                ],
            )
            return portfolio_id
        finally:
            connection.close()


def _portfolio_row(connection, portfolio_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM paper_portfolios WHERE portfolio_id = ?", [portfolio_id]
    ).fetchone()
    if not row:
        raise ValueError("Paper portfolio not found")
    columns = [column[0] for column in connection.description]
    return dict(zip(columns, row, strict=True))


def _position_rows(connection, portfolio_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT * FROM paper_positions
        WHERE portfolio_id = ?
        ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, updated_at DESC
        """,
        [portfolio_id],
    ).fetchall()
    columns = [column[0] for column in connection.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _alert_rows(connection, portfolio_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT alert_id, position_id, symbol, alert_type, severity, title, message,
               as_of_date, detail_json, created_at
        FROM paper_alerts
        WHERE portfolio_id = ? AND acknowledged_at IS NULL
        ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
                 created_at DESC
        """,
        [portfolio_id],
    ).fetchall()
    columns = [column[0] for column in connection.description]
    result = []
    for row in rows:
        item = dict(zip(columns, row, strict=True))
        item["detail"] = json.loads(item.pop("detail_json"))
        result.append(item)
    return result


def _snapshot_rows(connection, portfolio_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT snapshot_date, nav, cash, market_value, realized_pnl, unrealized_pnl,
               benchmark_value, daily_pnl, drawdown
        FROM paper_snapshots
        WHERE portfolio_id = ?
        ORDER BY snapshot_date, created_at
        """,
        [portfolio_id],
    ).fetchall()
    columns = [column[0] for column in connection.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _journal_rows(connection, portfolio_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT decision_id, position_id, decision_type, decision_date, symbol, price,
               note, snapshot_json, created_at
        FROM paper_decisions
        WHERE portfolio_id = ?
        ORDER BY created_at DESC
        LIMIT 100
        """,
        [portfolio_id],
    ).fetchall()
    columns = [column[0] for column in connection.description]
    result = []
    for row in rows:
        item = dict(zip(columns, row, strict=True))
        item["snapshot"] = json.loads(item.pop("snapshot_json"))
        result.append(item)
    return result


def _cohort_summary(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohorts: dict[str, list[float]] = {}
    for position in positions:
        if position["status"] != "closed" or position["exit_price"] is None:
            continue
        label = position["initial_bucket"] or "unclassified"
        invested = position["shares"] * position["entry_price"] + position["entry_fee"]
        pnl = (
            position["shares"] * (position["exit_price"] - position["entry_price"])
            - position["entry_fee"]
            - (position["exit_fee"] or 0.0)
        )
        cohorts.setdefault(label, []).append(pnl / invested if invested else 0.0)
    return [
        {
            "cohort": label,
            "trades": len(returns),
            "hit_rate": sum(value > 0 for value in returns) / len(returns),
            "average_return": sum(returns) / len(returns),
        }
        for label, returns in sorted(cohorts.items())
    ]


def _summary(connection, portfolio_id: str) -> dict[str, Any]:
    portfolio = _portfolio_row(connection, portfolio_id)
    raw_positions = _position_rows(connection, portfolio_id)
    market_value = 0.0
    unrealized_pnl = 0.0
    realized_pnl = 0.0
    closed_returns: list[float] = []
    open_positions: list[dict[str, Any]] = []
    closed_positions: list[dict[str, Any]] = []

    for position in raw_positions:
        shares = int(position["shares"])
        entry_value = shares * float(position["entry_price"])
        if position["status"] == "open":
            current_value = shares * float(position["latest_price"])
            pnl = current_value - entry_value - float(position["entry_fee"])
            market_value += current_value
            unrealized_pnl += pnl
            item = {
                **position,
                "market_value": current_value,
                "unrealized_pnl": pnl,
                "return_pct": pnl / (entry_value + float(position["entry_fee"])),
            }
            context_json = item.pop("latest_context_json", None)
            item["latest_context"] = json.loads(context_json) if context_json else None
            open_positions.append(item)
        else:
            exit_fee = float(position["exit_fee"] or 0.0)
            pnl = shares * (float(position["exit_price"]) - float(position["entry_price"]))
            pnl -= float(position["entry_fee"]) + exit_fee
            invested = entry_value + float(position["entry_fee"])
            realized_pnl += pnl
            closed_returns.append(pnl / invested if invested else 0.0)
            closed_positions.append(
                {
                    **position,
                    "realized_pnl": pnl,
                    "return_pct": pnl / invested if invested else 0.0,
                }
            )

    nav = float(portfolio["cash"]) + market_value
    initial_cash = float(portfolio["initial_cash"])
    for item in open_positions:
        item["weight"] = item["market_value"] / nav if nav else 0.0
        current_price = float(item["latest_price"])
        item["distance_to_stop"] = (current_price - float(item["stop_price"])) / current_price
        states = []
        if current_price <= float(item["stop_price"]):
            states.append("stop_breached")
        if current_price >= float(item["target_price"]):
            states.append("target_reached")
        if item["weight"] > float(portfolio["max_position_pct"]) / 100:
            states.append("concentration")
        item["risk_states"] = states
        item["evidence"] = json.loads(item.pop("evidence_json"))

    for item in closed_positions:
        item["evidence"] = json.loads(item.pop("evidence_json"))

    start_benchmark = portfolio["benchmark_start_price"]
    current_benchmark = portfolio["benchmark_current_price"]
    if start_benchmark and current_benchmark:
        benchmark_return = float(current_benchmark) / float(start_benchmark) - 1
    else:
        benchmark_return = 0.0
    benchmark_value = initial_cash * (1 + benchmark_return)
    total_return = nav / initial_cash - 1 if initial_cash else 0.0
    snapshots = _snapshot_rows(connection, portfolio_id)
    peak = initial_cash
    max_drawdown = 0.0
    for snapshot in snapshots:
        snapshot["snapshot_date"] = snapshot["snapshot_date"].isoformat()
        peak = max(peak, float(snapshot["nav"]))
        max_drawdown = min(max_drawdown, float(snapshot["nav"]) / peak - 1)

    alerts = _alert_rows(connection, portfolio_id)
    return {
        "portfolio_id": portfolio_id,
        "name": portfolio["name"],
        "benchmark_symbol": portfolio["benchmark_symbol"],
        "initial_cash": initial_cash,
        "cash": float(portfolio["cash"]),
        "nav": nav,
        "market_value": market_value,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "benchmark_value": benchmark_value,
        "alpha_vs_benchmark": total_return - benchmark_return,
        "max_drawdown": max_drawdown,
        "hit_rate": (
            sum(value > 0 for value in closed_returns) / len(closed_returns)
            if closed_returns
            else None
        ),
        "default_risk_pct": float(portfolio["default_risk_pct"]),
        "max_position_pct": float(portfolio["max_position_pct"]),
        "transaction_cost_bps": float(portfolio["transaction_cost_bps"]),
        "monitoring": {
            "enabled": bool(portfolio.get("monitoring_enabled", True)),
            "local_time": portfolio.get("monitor_local_time") or "16:45",
            "timezone": portfolio.get("monitor_timezone") or "America/New_York",
            "last_run_date": portfolio.get("last_monitor_date"),
            "last_status": portfolio.get("last_monitor_status"),
            "last_error": portfolio.get("last_monitor_error"),
        },
        "open_positions": open_positions,
        "closed_positions": closed_positions,
        "journal": _journal_rows(connection, portfolio_id),
        "snapshots": snapshots,
        "cohorts": _cohort_summary(raw_positions),
        "alerts": len(alerts),
        "risk_alerts": alerts,
        "updated_at": portfolio["updated_at"],
    }


def get_paper_portfolio(
    config: dict[str, Any] | None = None,
    *,
    portfolio_id: str | None = None,
) -> dict[str, Any]:
    """Return the complete dashboard view for a paper portfolio."""
    resolved_id = portfolio_id or ensure_default_paper_portfolio(config)
    with _LOCK:
        connection = _connect(config)
        try:
            return _summary(connection, resolved_id)
        finally:
            connection.close()


def _record_decision(
    connection,
    *,
    portfolio_id: str,
    position_id: str,
    decision_type: str,
    decision_date: str,
    symbol: str,
    price: float,
    note: str,
    snapshot: dict[str, Any],
) -> None:
    created_at = datetime.now(UTC)
    decision_id = hashlib.sha256(
        f"{position_id}|{decision_type}|{decision_date}|{created_at.isoformat()}".encode()
    ).hexdigest()
    connection.execute(
        "INSERT INTO paper_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            decision_id,
            portfolio_id,
            position_id,
            decision_type,
            decision_date,
            symbol,
            price,
            note,
            _safe_json(snapshot),
            created_at,
        ],
    )


def _upsert_snapshot(connection, portfolio_id: str, snapshot_date: str) -> None:
    summary = _summary(connection, portfolio_id)
    prior = connection.execute(
        """
        SELECT nav FROM paper_snapshots
        WHERE portfolio_id = ? AND snapshot_date < ?
        ORDER BY snapshot_date DESC, created_at DESC LIMIT 1
        """,
        [portfolio_id, snapshot_date],
    ).fetchone()
    previous_nav = float(prior[0]) if prior else float(summary["initial_cash"])
    daily_pnl = float(summary["nav"]) - previous_nav
    historical_peak = connection.execute(
        "SELECT max(nav) FROM paper_snapshots WHERE portfolio_id = ?",
        [portfolio_id],
    ).fetchone()[0]
    peak = max(
        float(summary["initial_cash"]),
        float(summary["nav"]),
        float(historical_peak or summary["initial_cash"]),
    )
    drawdown = float(summary["nav"]) / peak - 1 if peak else 0.0
    prices = {
        item["symbol"]: item["latest_price"] for item in summary["open_positions"]
    }
    snapshot_id = hashlib.sha256(f"{portfolio_id}|{snapshot_date}".encode()).hexdigest()
    connection.execute(
        """
        INSERT OR REPLACE INTO paper_snapshots VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            snapshot_id,
            portfolio_id,
            snapshot_date,
            summary["nav"],
            summary["cash"],
            summary["market_value"],
            summary["realized_pnl"],
            summary["unrealized_pnl"],
            summary["benchmark_value"],
            daily_pnl,
            drawdown,
            _safe_json(prices),
            datetime.now(UTC),
        ],
    )


def _latest_scan_items(
    connection, symbols: set[str], as_of_date: str
) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    try:
        rows = connection.execute(
            """
            SELECT result_json FROM opportunity_scans
            ORDER BY created_at DESC LIMIT 200
            """
        ).fetchall()
    except Exception:  # noqa: BLE001 - the paper portfolio can exist without scans
        return {}
    found: dict[str, dict[str, Any]] = {}
    for (payload,) in rows:
        try:
            report = json.loads(payload)
        except (TypeError, ValueError):
            continue
        report_date = report.get("as_of_date") or report.get("trade_date")
        if report_date and str(report_date) > as_of_date:
            continue
        for item in report.get("items", []):
            symbol = item.get("symbol")
            if symbol in symbols and symbol not in found:
                found[symbol] = item
        if len(found) == len(symbols):
            break
    return found


def _insert_alert(
    connection,
    *,
    portfolio_id: str,
    position_id: str,
    symbol: str,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    as_of_date: str,
    detail: dict[str, Any],
) -> None:
    alert_id = hashlib.sha256(
        f"{portfolio_id}|{position_id}|{alert_type}|{as_of_date}".encode()
    ).hexdigest()
    connection.execute(
        """
        INSERT OR IGNORE INTO paper_alerts VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?
        )
        """,
        [
            alert_id,
            portfolio_id,
            position_id,
            symbol,
            alert_type,
            severity,
            title,
            message,
            as_of_date,
            _safe_json(detail),
            datetime.now(UTC),
        ],
    )


def _capture_monitor_alerts(
    connection,
    portfolio_id: str,
    as_of_date: str,
    contexts: dict[str, dict[str, Any]],
) -> None:
    connection.execute(
        """
        UPDATE paper_alerts SET acknowledged_at = ?
        WHERE portfolio_id = ? AND acknowledged_at IS NULL AND as_of_date < ?
        """,
        [datetime.now(UTC), portfolio_id, as_of_date],
    )
    summary = _summary(connection, portfolio_id)
    latest_scans = _latest_scan_items(
        connection,
        {item["symbol"] for item in summary["open_positions"]},
        as_of_date,
    )
    labels = {
        "stop_breached": ("high", "Stop level breached"),
        "target_reached": ("high", "Target level reached"),
        "concentration": ("warning", "Position concentration above plan"),
    }
    for position in summary["open_positions"]:
        symbol = position["symbol"]
        for state in position["risk_states"]:
            severity, title = labels[state]
            _insert_alert(
                connection,
                portfolio_id=portfolio_id,
                position_id=position["position_id"],
                symbol=symbol,
                alert_type=state,
                severity=severity,
                title=f"{symbol}: {title}",
                message=(
                    f"Last close {position['latest_price']:.2f}; stop "
                    f"{position['stop_price']:.2f}; target {position['target_price']:.2f}; "
                    f"portfolio weight {position['weight']:.1%}."
                ),
                as_of_date=as_of_date,
                detail={
                    "last_price": position["latest_price"],
                    "stop_price": position["stop_price"],
                    "target_price": position["target_price"],
                    "weight": position["weight"],
                },
            )
        latest = latest_scans.get(symbol)
        initial_score = position.get("initial_score")
        current_score = latest.get("opportunity_score") if latest else None
        if initial_score is not None and current_score is not None:
            score_delta = float(current_score) - float(initial_score)
            if score_delta <= -10:
                _insert_alert(
                    connection,
                    portfolio_id=portfolio_id,
                    position_id=position["position_id"],
                    symbol=symbol,
                    alert_type="score_deterioration",
                    severity="high" if score_delta <= -20 else "warning",
                    title=f"{symbol}: opportunity evidence deteriorated",
                    message=(
                        f"Opportunity score changed from {float(initial_score):.1f} "
                        f"to {float(current_score):.1f} ({score_delta:+.1f})."
                    ),
                    as_of_date=as_of_date,
                    detail={
                        "initial_score": initial_score,
                        "current_score": current_score,
                        "score_delta": score_delta,
                        "current_bucket": latest.get("bucket"),
                    },
                )
        stored_context_date = position.get("latest_context_date")
        stored_context_is_known = (
            stored_context_date is None
            or str(stored_context_date) <= as_of_date
        )
        context = contexts.get(symbol) or (
            position.get("latest_context") if stored_context_is_known else None
        ) or {}
        for flag in context.get("events", {}).get("flags", []):
            if flag.get("severity") not in {"warning", "high"}:
                continue
            _insert_alert(
                connection,
                portfolio_id=portfolio_id,
                position_id=position["position_id"],
                symbol=symbol,
                alert_type=f"event_{flag.get('type', 'review')}",
                severity=flag["severity"],
                title=f"{symbol}: event review required",
                message=str(flag.get("message") or "Regulatory event risk changed."),
                as_of_date=as_of_date,
                detail=flag,
            )


def update_paper_monitoring(
    enabled: bool,
    local_time: str,
    timezone: str,
    config: dict[str, Any] | None = None,
    *,
    portfolio_id: str | None = None,
) -> dict[str, Any]:
    """Configure the local weekday mark-to-market schedule."""
    try:
        datetime.strptime(local_time, "%H:%M")
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("Invalid monitor time or timezone") from exc
    resolved_id = portfolio_id or ensure_default_paper_portfolio(config)
    with _LOCK:
        connection = _connect(config)
        try:
            updated = connection.execute(
                """
                UPDATE paper_portfolios
                SET monitoring_enabled = ?, monitor_local_time = ?, monitor_timezone = ?,
                    updated_at = ?
                WHERE portfolio_id = ? RETURNING portfolio_id
                """,
                [enabled, local_time, timezone, datetime.now(UTC), resolved_id],
            ).fetchone()
            if not updated:
                raise ValueError("Paper portfolio not found")
            return _summary(connection, resolved_id)
        finally:
            connection.close()


def record_paper_monitor_run(
    status: str,
    config: dict[str, Any] | None = None,
    *,
    portfolio_id: str | None = None,
    run_date: str | None = None,
    error: str | None = None,
) -> bool:
    """Persist automatic-monitor lifecycle state for restart-safe scheduling."""
    resolved_id = portfolio_id or ensure_default_paper_portfolio(config)
    with _LOCK:
        connection = _connect(config)
        try:
            row = connection.execute(
                """
                UPDATE paper_portfolios
                SET last_monitor_date = coalesce(?, last_monitor_date),
                    last_monitor_status = ?, last_monitor_error = ?, updated_at = ?
                WHERE portfolio_id = ? RETURNING portfolio_id
                """,
                [run_date, status, error, datetime.now(UTC), resolved_id],
            ).fetchone()
            return bool(row)
        finally:
            connection.close()


def acknowledge_paper_alert(
    alert_id: str, config: dict[str, Any] | None = None
) -> bool:
    """Acknowledge one portfolio-monitor alert without deleting its audit trail."""
    with _LOCK:
        connection = _connect(config)
        try:
            row = connection.execute(
                """
                UPDATE paper_alerts SET acknowledged_at = ?
                WHERE alert_id = ? AND acknowledged_at IS NULL RETURNING alert_id
                """,
                [datetime.now(UTC), alert_id],
            ).fetchone()
            return bool(row)
        finally:
            connection.close()


def open_paper_position(
    symbol: str,
    entry_date: str,
    entry_price: float,
    config: dict[str, Any] | None = None,
    *,
    portfolio_id: str | None = None,
    risk_pct: float | None = None,
    stop_pct: float = 8.0,
    target_pct: float = 16.0,
    max_position_pct: float | None = None,
    decision_note: str = "",
    evidence: dict[str, Any] | None = None,
    scan_id: str | None = None,
    initial_score: float | None = None,
    initial_bucket: str | None = None,
    initial_rank: int | None = None,
    benchmark_price: float | None = None,
) -> dict[str, Any]:
    """Open a hypothetical long position sized by stop-loss risk."""
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Entry price must be positive")
    if stop_pct <= 0 or stop_pct >= 100 or target_pct <= 0:
        raise ValueError("Stop and target percentages must be positive")
    resolved_id = portfolio_id or ensure_default_paper_portfolio(config)
    with _LOCK:
        connection = _connect(config)
        try:
            connection.execute("BEGIN TRANSACTION")
            portfolio = _portfolio_row(connection, resolved_id)
            duplicate = connection.execute(
                """
                SELECT position_id FROM paper_positions
                WHERE portfolio_id = ? AND symbol = ? AND status = 'open'
                """,
                [resolved_id, symbol],
            ).fetchone()
            if duplicate:
                raise ValueError(f"{symbol} is already open in the paper portfolio")

            before = _summary(connection, resolved_id)
            effective_risk = float(risk_pct or portfolio["default_risk_pct"])
            effective_cap = float(max_position_pct or portfolio["max_position_pct"])
            if effective_risk <= 0 or effective_cap <= 0 or effective_cap > 100:
                raise ValueError("Risk and position caps must be within valid ranges")
            stop_price = round(entry_price * (1 - stop_pct / 100), 6)
            target_price = round(entry_price * (1 + target_pct / 100), 6)
            risk_capital = before["nav"] * effective_risk / 100
            shares_by_risk = math.floor(risk_capital / (entry_price - stop_price))
            shares_by_cap = math.floor(before["nav"] * effective_cap / 100 / entry_price)
            fee_rate = float(portfolio["transaction_cost_bps"]) / 10_000
            shares_by_cash = math.floor(float(portfolio["cash"]) / (entry_price * (1 + fee_rate)))
            shares = min(shares_by_risk, shares_by_cap, shares_by_cash)
            if shares < 1:
                raise ValueError("Risk budget or available cash is too small for one share")

            entry_value = shares * entry_price
            entry_fee = entry_value * fee_rate
            now = datetime.now(UTC)
            position_id = hashlib.sha256(
                f"{resolved_id}|{symbol}|{entry_date}|{now.isoformat()}".encode()
            ).hexdigest()[:32]
            evidence_payload = evidence or {}
            connection.execute(
                """
                INSERT INTO paper_positions (
                    position_id, portfolio_id, symbol, side, shares, entry_date,
                    entry_price, stop_price, target_price, entry_fee, initial_score,
                    initial_bucket, initial_rank, scan_id, decision_note, evidence_json,
                    latest_price, latest_price_date, exit_date, exit_price, exit_fee,
                    exit_reason, status, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, 'long', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    NULL, NULL, NULL, NULL, 'open', ?, ?
                )
                """,
                [
                    position_id,
                    resolved_id,
                    symbol,
                    shares,
                    entry_date,
                    entry_price,
                    stop_price,
                    target_price,
                    entry_fee,
                    initial_score,
                    initial_bucket,
                    initial_rank,
                    scan_id,
                    decision_note.strip(),
                    _safe_json(evidence_payload),
                    entry_price,
                    entry_date,
                    now,
                    now,
                ],
            )
            entry_context = evidence_payload.get("research_context")
            if isinstance(entry_context, dict):
                connection.execute(
                    """
                    UPDATE paper_positions
                    SET latest_context_json = ?, latest_context_date = ?
                    WHERE position_id = ?
                    """,
                    [_safe_json(entry_context), entry_date, position_id],
                )
            connection.execute(
                """
                UPDATE paper_portfolios
                SET cash = cash - ?,
                    benchmark_start_price = coalesce(benchmark_start_price, ?),
                    benchmark_current_price = coalesce(?, benchmark_current_price),
                    updated_at = ?
                WHERE portfolio_id = ?
                """,
                [entry_value + entry_fee, benchmark_price, benchmark_price, now, resolved_id],
            )
            sizing = {
                "risk_pct": effective_risk,
                "risk_capital": risk_capital,
                "risk_per_share": entry_price - stop_price,
                "max_position_pct": effective_cap,
                "shares_by_risk": shares_by_risk,
                "shares_by_cap": shares_by_cap,
                "shares_by_cash": shares_by_cash,
                "shares_selected": shares,
                "entry_fee": entry_fee,
            }
            _record_decision(
                connection,
                portfolio_id=resolved_id,
                position_id=position_id,
                decision_type="open",
                decision_date=entry_date,
                symbol=symbol,
                price=entry_price,
                note=decision_note.strip(),
                snapshot={
                    "score": initial_score,
                    "bucket": initial_bucket,
                    "rank": initial_rank,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "sizing": sizing,
                    "evidence": evidence_payload,
                },
            )
            _upsert_snapshot(connection, resolved_id, entry_date)
            connection.execute("COMMIT")
            result = _summary(connection, resolved_id)
            result["opened_position_id"] = position_id
            result["sizing"] = sizing
            return result
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def mark_paper_portfolio(
    prices: dict[str, float],
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    portfolio_id: str | None = None,
    benchmark_price: float | None = None,
    contexts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Mark all supplied open positions and store a daily equity snapshot."""
    resolved_id = portfolio_id or ensure_default_paper_portfolio(config)
    with _LOCK:
        connection = _connect(config)
        try:
            connection.execute("BEGIN TRANSACTION")
            now = datetime.now(UTC)
            for symbol, price in prices.items():
                if not math.isfinite(price) or price <= 0:
                    raise ValueError(f"Invalid price for {symbol}")
                connection.execute(
                    """
                    UPDATE paper_positions SET latest_price = ?, latest_price_date = ?,
                        updated_at = ?
                    WHERE portfolio_id = ? AND symbol = ? AND status = 'open'
                    """,
                    [price, as_of_date, now, resolved_id, symbol],
                )
            for symbol, context in (contexts or {}).items():
                connection.execute(
                    """
                    UPDATE paper_positions
                    SET latest_context_json = ?, latest_context_date = ?, updated_at = ?
                    WHERE portfolio_id = ? AND symbol = ? AND status = 'open'
                    """,
                    [_safe_json(context), as_of_date, now, resolved_id, symbol],
                )
            connection.execute(
                """
                UPDATE paper_portfolios
                SET benchmark_current_price = coalesce(?, benchmark_current_price), updated_at = ?
                WHERE portfolio_id = ?
                """,
                [benchmark_price, now, resolved_id],
            )
            _upsert_snapshot(connection, resolved_id, as_of_date)
            _capture_monitor_alerts(connection, resolved_id, as_of_date, contexts or {})
            connection.execute("COMMIT")
            return _summary(connection, resolved_id)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()


def close_paper_position(
    position_id: str,
    exit_date: str,
    exit_price: float,
    config: dict[str, Any] | None = None,
    *,
    exit_reason: str = "Manual exit",
) -> dict[str, Any]:
    """Close a hypothetical position and journal the realized outcome."""
    if not math.isfinite(exit_price) or exit_price <= 0:
        raise ValueError("Exit price must be positive")
    with _LOCK:
        connection = _connect(config)
        try:
            connection.execute("BEGIN TRANSACTION")
            row = connection.execute(
                """
                SELECT position_id, portfolio_id, symbol, shares, entry_price, entry_fee,
                       status, initial_score, initial_bucket
                FROM paper_positions WHERE position_id = ?
                """,
                [position_id],
            ).fetchone()
            if not row:
                raise ValueError("Paper position not found")
            if row[6] != "open":
                raise ValueError("Paper position is already closed")
            portfolio = _portfolio_row(connection, row[1])
            fee_rate = float(portfolio["transaction_cost_bps"]) / 10_000
            proceeds = int(row[3]) * exit_price
            exit_fee = proceeds * fee_rate
            now = datetime.now(UTC)
            connection.execute(
                """
                UPDATE paper_positions
                SET latest_price = ?, latest_price_date = ?, exit_date = ?, exit_price = ?,
                    exit_fee = ?, exit_reason = ?, status = 'closed', updated_at = ?
                WHERE position_id = ?
                """,
                [
                    exit_price,
                    exit_date,
                    exit_date,
                    exit_price,
                    exit_fee,
                    exit_reason.strip(),
                    now,
                    position_id,
                ],
            )
            connection.execute(
                """
                UPDATE paper_portfolios SET cash = cash + ?, updated_at = ?
                WHERE portfolio_id = ?
                """,
                [proceeds - exit_fee, now, row[1]],
            )
            connection.execute(
                """
                UPDATE paper_alerts SET acknowledged_at = ?
                WHERE position_id = ? AND acknowledged_at IS NULL
                """,
                [now, position_id],
            )
            realized_pnl = (
                int(row[3]) * (exit_price - float(row[4]))
                - float(row[5])
                - exit_fee
            )
            _record_decision(
                connection,
                portfolio_id=row[1],
                position_id=position_id,
                decision_type="close",
                decision_date=exit_date,
                symbol=row[2],
                price=exit_price,
                note=exit_reason.strip(),
                snapshot={
                    "realized_pnl": realized_pnl,
                    "entry_price": row[4],
                    "shares": row[3],
                    "initial_score": row[7],
                    "initial_bucket": row[8],
                },
            )
            _upsert_snapshot(connection, row[1], exit_date)
            connection.execute("COMMIT")
            return _summary(connection, row[1])
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
