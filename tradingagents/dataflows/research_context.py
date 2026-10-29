"""Point-in-time fundamental and regulatory-event overlay.

The overlay stays separate from the opportunity score.  It uses SEC facts and
acceptance timestamps as the knowledge boundary, then turns the available
evidence into transparent pillars for review and paper-portfolio journaling.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import threading
from datetime import UTC, date, datetime, timedelta
from typing import Any

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.paper_portfolio import load_close_prices
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.warehouse import dependency_available, warehouse_path

_LOCK = threading.RLock()


def _number(value: Any, digits: int = 6) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, digits) if math.isfinite(result) else None


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _scaled(value: float | None, low: float, high: float, *, inverse: bool = False):
    if value is None or high <= low:
        return None
    score = 100 * (value - low) / (high - low)
    return _clip(100 - score if inverse else score)


def build_decision_confidence(
    opportunity: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Combine independent evidence without changing the opportunity score."""
    validation = opportunity.get("validation") or {}
    validation_parts = [
        _scaled(_number(validation.get("sharpe_ratio")), -1.0, 2.0),
        _scaled(_number(validation.get("profitable_fold_rate")), 0.25, 0.75),
        _scaled(_number(validation.get("threshold_stability")), 0.25, 1.0),
    ]
    robustness_score = {
        "promising": 85.0,
        "mixed": 58.0,
        "weak": 25.0,
    }.get(validation.get("robustness"))
    validation_parts.append(robustness_score)
    usable_validation = [value for value in validation_parts if value is not None]
    validation_score = (
        sum(usable_validation) / len(usable_validation)
        if usable_validation
        else None
    )

    fundamentals = context.get("fundamentals") or {}
    events = context.get("events") or {}
    overlay = context.get("overlay") or {}
    opportunity_confidence = _number(opportunity.get("confidence"))
    context_coverage = _number(overlay.get("data_coverage"))
    evidence_depth_parts = [
        opportunity_confidence * 100 if opportunity_confidence is not None else None,
        context_coverage * 100 if context_coverage is not None else None,
    ]
    evidence_depth_values = [
        value for value in evidence_depth_parts if value is not None
    ]
    evidence_depth = (
        sum(evidence_depth_values) / len(evidence_depth_values)
        if evidence_depth_values
        else None
    )
    event_risk = _number(events.get("risk_score"))
    fundamental_score = _number(fundamentals.get("score"))
    components: dict[str, tuple[float | None, float]] = {
        "opportunity_evidence": (_number(opportunity.get("opportunity_score")), 0.28),
        "rolling_validation": (validation_score, 0.18),
        "fundamental_support": (fundamental_score, 0.20),
        "event_resilience": (100 - event_risk if event_risk is not None else None, 0.14),
        "market_data_quality": (
            _number((opportunity.get("quality") or {}).get("score")),
            0.10,
        ),
        "evidence_depth": (evidence_depth, 0.10),
    }
    available = {
        name: (float(value), weight)
        for name, (value, weight) in components.items()
        if value is not None
    }
    available_weight = sum(weight for _, weight in available.values())
    raw_score = (
        sum(value * weight for value, weight in available.values()) / available_weight
        if available_weight
        else None
    )
    # Missing pillars do not receive neutral values. The visible coverage factor
    # modestly discounts a score supported by a narrow evidence set.
    score = (
        raw_score * (0.75 + 0.25 * available_weight)
        if raw_score is not None
        else None
    )
    if score is None:
        label = "unavailable"
    elif event_risk is not None and event_risk >= 50:
        label = "caution"
    elif context_coverage is None or context_coverage < 0.45:
        label = "limited"
    elif (
        score >= 75
        and available_weight >= 0.75
        and fundamental_score is not None
        and validation_score is not None
    ):
        label = "high_conviction"
    elif score >= 60:
        label = "supported"
    elif score >= 45:
        label = "mixed"
    else:
        label = "weak"

    opportunity_score = _number(opportunity.get("opportunity_score"))
    disagreements = []
    if (
        opportunity_score is not None
        and opportunity_score >= 70
        and fundamental_score is not None
        and fundamental_score < 40
    ):
        disagreements.append("Strong market evidence conflicts with weak filed fundamentals.")
    if event_risk is not None and event_risk >= 50:
        disagreements.append("Elevated filing-event risk requires review before sizing.")
    if available_weight < 0.65:
        disagreements.append("Evidence coverage is too narrow for high conviction.")
    elif context_coverage is None or context_coverage < 0.45:
        disagreements.append("SEC context coverage is too limited for high conviction.")

    return {
        "score": round(score, 2) if score is not None else None,
        "raw_score": round(raw_score, 2) if raw_score is not None else None,
        "label": label,
        "evidence_coverage": round(available_weight, 3),
        "components": {
            name: round(value, 2) for name, (value, _) in available.items()
        },
        "weights": {name: weight for name, (_, weight) in components.items()},
        "disagreements": disagreements,
        "methodology": (
            "Independent weighted synthesis of the unchanged opportunity score, rolling "
            "validation, filed fundamentals, event resilience, market-data quality and "
            "evidence depth. Missing pillars are not assigned neutral values."
        ),
    }


def _valid_payload(payload: str) -> bool:
    text = (payload or "").strip()
    return bool(text) and not text.startswith(("DATA_UNAVAILABLE:", "NO_DATA_AVAILABLE:"))


def _statement_rows(payload: str) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Parse the compact CSV emitted by the point-in-time statement vendors."""
    lines = [line for line in payload.splitlines() if line and not line.startswith("#")]
    if not lines:
        return [], {}
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if not rows or len(rows[0]) < 2:
        return [], {}
    headers = [value.strip() for value in rows[0][1:]]
    result: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        if not row:
            continue
        # SEC appends the reported unit to non-USD rows (for example
        # ``Diluted EPS (USD/shares)``); pillar labels stay unit-agnostic.
        label = row[0].strip().split(" (", 1)[0]
        values: dict[str, float] = {}
        for header, raw in zip(headers, row[1:], strict=False):
            value = _number(raw)
            if value is not None:
                values[header] = value
        result[label] = values
    return headers, result


def _row_values(
    headers: list[str],
    rows: dict[str, dict[str, float]],
    label: str,
    *,
    shortest_span_only: bool = False,
) -> list[tuple[str, float]]:
    values = rows.get(label, {})
    ordered = []
    for header in headers:
        if shortest_span_only and "(" in header:
            continue
        if header in values:
            ordered.append((header, values[header]))
    return ordered


def _latest_common(
    headers: list[str],
    rows: dict[str, dict[str, float]],
    labels: list[str],
    *,
    shortest_span_only: bool = False,
) -> tuple[str | None, dict[str, float]]:
    for header in reversed(headers):
        if shortest_span_only and "(" in header:
            continue
        collected = {label: rows.get(label, {}).get(header) for label in labels}
        if all(value is not None for value in collected.values()):
            return header, {label: float(value) for label, value in collected.items()}
    return None, {}


def _parse_filings(payload: str) -> list[dict[str, Any]]:
    filings = []
    for line in payload.splitlines():
        if not line.startswith("|") or line.startswith(("|---", "| Accepted")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        accepted = cells[0].replace(" ", "T", 1)
        accepted_date = accepted[:10]
        try:
            date.fromisoformat(accepted_date)
        except ValueError:
            continue
        link = None
        if "](" in cells[4] and cells[4].endswith(")"):
            link = cells[4].split("](", 1)[1][:-1]
        filings.append(
            {
                "accepted_at": accepted,
                "accepted_date": accepted_date,
                "form": cells[1],
                "report_date": None if cells[2] == "—" else cells[2],
                "description": cells[3],
                "url": link,
            }
        )
    return filings


def _fundamental_pillar(
    quarterly_income: str,
    quarterly_balance: str,
    quarterly_cashflow: str,
    annual_income: str,
    price: float | None,
) -> dict[str, Any]:
    income_headers, income_rows = _statement_rows(quarterly_income)
    balance_headers, balance_rows = _statement_rows(quarterly_balance)
    cash_headers, cash_rows = _statement_rows(quarterly_cashflow)
    annual_headers, annual_rows = _statement_rows(annual_income)

    revenue_series = _row_values(
        income_headers, income_rows, "Revenue", shortest_span_only=True
    )
    sequential_growth = None
    if len(revenue_series) >= 2 and revenue_series[-2][1]:
        sequential_growth = revenue_series[-1][1] / revenue_series[-2][1] - 1

    income_period, income = _latest_common(
        income_headers,
        income_rows,
        ["Revenue", "Operating Income", "Net Income"],
        shortest_span_only=True,
    )
    operating_margin = None
    net_margin = None
    if income and income["Revenue"]:
        operating_margin = income["Operating Income"] / income["Revenue"]
        net_margin = income["Net Income"] / income["Revenue"]

    balance_period, balance = _latest_common(
        balance_headers,
        balance_rows,
        ["Total Assets", "Current Assets", "Total Liabilities", "Current Liabilities"],
    )
    current_ratio = None
    liabilities_to_assets = None
    if balance:
        if balance["Current Liabilities"]:
            current_ratio = balance["Current Assets"] / balance["Current Liabilities"]
        if balance["Total Assets"]:
            liabilities_to_assets = balance["Total Liabilities"] / balance["Total Assets"]

    cash_period, cash = _latest_common(
        cash_headers,
        cash_rows,
        ["Operating Cash Flow", "Capital Expenditure"],
    )
    free_cashflow = None
    if cash:
        free_cashflow = cash["Operating Cash Flow"] - abs(cash["Capital Expenditure"])
    latest_revenue = income.get("Revenue") if income else None
    free_cashflow_margin = (
        free_cashflow / latest_revenue
        if free_cashflow is not None and latest_revenue
        else None
    )

    annual_eps = _row_values(annual_headers, annual_rows, "Diluted EPS")
    latest_eps = annual_eps[-1][1] if annual_eps else None
    trailing_pe = price / latest_eps if price and latest_eps and latest_eps > 0 else None

    components = {
        "sequential_revenue_growth": _scaled(sequential_growth, -0.20, 0.20),
        "operating_margin": _scaled(operating_margin, -0.05, 0.30),
        "net_margin": _scaled(net_margin, -0.10, 0.25),
        "current_ratio": _scaled(current_ratio, 0.50, 2.00),
        "balance_sheet_leverage": _scaled(
            liabilities_to_assets, 0.30, 0.90, inverse=True
        ),
        "free_cashflow_margin": _scaled(free_cashflow_margin, -0.15, 0.25),
        "valuation": (
            _clip(100 - abs(trailing_pe - 22) * 2.5)
            if trailing_pe is not None
            else None
        ),
    }
    weights = {
        "sequential_revenue_growth": 0.16,
        "operating_margin": 0.18,
        "net_margin": 0.15,
        "current_ratio": 0.12,
        "balance_sheet_leverage": 0.14,
        "free_cashflow_margin": 0.15,
        "valuation": 0.10,
    }
    available = {
        name: float(score)
        for name, score in components.items()
        if score is not None
    }
    total_weight = sum(weights[name] for name in available)
    score = (
        sum(value * weights[name] for name, value in available.items()) / total_weight
        if total_weight
        else None
    )
    coverage = len(available) / len(components)
    if score is None:
        label = "unavailable"
    elif score >= 70:
        label = "strong"
    elif score >= 55:
        label = "supportive"
    elif score >= 40:
        label = "mixed"
    else:
        label = "weak"

    observations = []
    metric_labels = [
        (sequential_growth, "Sequential revenue growth", True),
        (operating_margin, "Operating margin", True),
        (net_margin, "Net margin", True),
        (current_ratio, "Current ratio", False),
        (liabilities_to_assets, "Liabilities / assets", True),
        (free_cashflow_margin, "Free-cash-flow margin", True),
        (trailing_pe, "Price / latest annual diluted EPS", False),
    ]
    for value, name, percent in metric_labels:
        if value is not None:
            observations.append(
                {
                    "label": name,
                    "value": round(value, 4),
                    "format": "percent" if percent else "number",
                }
            )
    return {
        "available": score is not None,
        "score": round(score, 2) if score is not None else None,
        "label": label,
        "coverage": round(coverage, 3),
        "components": {name: round(value, 2) for name, value in available.items()},
        "metrics": {
            "sequential_revenue_growth": _number(sequential_growth),
            "operating_margin": _number(operating_margin),
            "net_margin": _number(net_margin),
            "current_ratio": _number(current_ratio),
            "liabilities_to_assets": _number(liabilities_to_assets),
            "free_cashflow": _number(free_cashflow, 2),
            "free_cashflow_margin": _number(free_cashflow_margin),
            "annual_diluted_eps": _number(latest_eps),
            "trailing_pe_proxy": _number(trailing_pe, 2),
        },
        "periods": {
            "income": income_period,
            "balance_sheet": balance_period,
            "cashflow": cash_period,
            "annual_eps": annual_eps[-1][0] if annual_eps else None,
        },
        "observations": observations,
        "source": "SEC EDGAR company facts; accepted filings only",
        "caveat": (
            "Growth is sequential, not seasonally adjusted. The valuation proxy uses the "
            "selected close divided by the latest annual diluted EPS filed by that date."
        ),
    }


def _event_pillar(filings: list[dict[str, Any]], as_of_date: str) -> dict[str, Any]:
    as_of = date.fromisoformat(as_of_date)
    recent_30 = [
        filing
        for filing in filings
        if 0 <= (as_of - date.fromisoformat(filing["accepted_date"])).days <= 30
    ]
    recent_8k = [filing for filing in recent_30 if filing["form"] == "8-K"]
    insider = [filing for filing in recent_30 if filing["form"] in {"3", "4", "5"}]
    reports = [filing for filing in filings if filing["form"] in {"10-Q", "10-K"}]
    report_dates = sorted(
        {date.fromisoformat(filing["accepted_date"]) for filing in reports}, reverse=True
    )
    latest_report = report_dates[0] if report_dates else None
    days_since_report = (as_of - latest_report).days if latest_report else None
    estimated_next = None
    days_to_estimated = None
    if len(report_dates) >= 2:
        ascending = sorted(report_dates[:5])
        intervals = [
            (right - left).days
            for left, right in zip(ascending, ascending[1:], strict=False)
            if 45 <= (right - left).days <= 150
        ]
        if intervals:
            estimated_next = latest_report + timedelta(days=round(statistics.median(intervals)))
            days_to_estimated = (estimated_next - as_of).days

    risk_points = 0
    flags = []
    if recent_8k:
        points = min(30, 10 + len(recent_8k) * 5)
        risk_points += points
        flags.append(
            {
                "type": "recent_8k",
                "severity": "warning" if len(recent_8k) < 3 else "high",
                "message": f"{len(recent_8k)} material-event filing(s) accepted in 30 days.",
            }
        )
    if len(insider) >= 3:
        risk_points += min(20, 8 + len(insider) * 2)
        flags.append(
            {
                "type": "insider_activity",
                "severity": "info",
                "message": (
                    f"{len(insider)} insider ownership filing(s) in 30 days; "
                    "filing count does not imply buying or selling direction."
                ),
            }
        )
    if days_to_estimated is not None and -7 <= days_to_estimated <= 14:
        risk_points += 30
        flags.append(
            {
                "type": "estimated_reporting_window",
                "severity": "high",
                "message": (
                    "Estimated reporting window is near; this is inferred from filing "
                    "cadence, not a confirmed earnings date."
                ),
            }
        )
    if days_since_report is not None and days_since_report > 180:
        risk_points += 15
        flags.append(
            {
                "type": "stale_reporting",
                "severity": "warning",
                "message": f"Latest 10-Q/K was accepted {days_since_report} days ago.",
            }
        )
    if not filings:
        risk_points += 25
        flags.append(
            {
                "type": "limited_event_data",
                "severity": "warning",
                "message": "No supported SEC filing history was available for this instrument.",
            }
        )
    risk_score = int(_clip(risk_points))
    risk_level = "elevated" if risk_score >= 50 else "watch" if risk_score >= 25 else "quiet"
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recent_8k_count": len(recent_8k),
        "insider_filing_count_30d": len(insider),
        "latest_report_date": latest_report.isoformat() if latest_report else None,
        "days_since_report": days_since_report,
        "estimated_next_report_date": estimated_next.isoformat() if estimated_next else None,
        "days_to_estimated_report": days_to_estimated,
        "estimated_date_notice": (
            "Cadence estimate only; not a confirmed company earnings date."
            if estimated_next
            else None
        ),
        "flags": flags,
        "recent_filings": filings[:8],
        "source": "SEC EDGAR acceptance timestamps",
    }


def _connect(config: dict[str, Any] | None = None):
    if not dependency_available():
        return None
    import duckdb

    cfg = config or get_config()
    if not cfg.get("data_warehouse_enabled", False):
        return None
    path = warehouse_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS research_context_snapshots (
            context_id VARCHAR PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            as_of_date DATE NOT NULL,
            fundamental_score DOUBLE,
            event_risk_score DOUBLE NOT NULL,
            result_json VARCHAR NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_research_context_lookup
        ON research_context_snapshots(symbol, as_of_date, created_at)
        """
    )
    return connection


def capture_research_context(
    report: dict[str, Any], config: dict[str, Any] | None = None
) -> str | None:
    connection = _connect(config)
    if connection is None:
        return None
    identity = {key: value for key, value in report.items() if key not in {"registry"}}
    context_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    try:
        with _LOCK:
            connection.execute(
                """
                INSERT OR IGNORE INTO research_context_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    context_id,
                    report["symbol"],
                    report["as_of_date"],
                    report["fundamentals"]["score"],
                    report["events"]["risk_score"],
                    json.dumps(identity, sort_keys=True, default=str),
                    datetime.now(UTC),
                ],
            )
        return context_id
    finally:
        connection.close()


def build_research_context(
    symbol: str,
    as_of_date: str,
    config: dict[str, Any] | None = None,
    *,
    opportunity: dict[str, Any] | None = None,
    closing_price: float | None = None,
) -> dict[str, Any]:
    """Build and persist separate fundamental and event-risk pillars."""
    payloads: dict[str, str] = {}
    errors = []
    calls = {
        "quarterly_income": ("get_income_statement", symbol, "quarterly", as_of_date),
        "quarterly_balance": ("get_balance_sheet", symbol, "quarterly", as_of_date),
        "quarterly_cashflow": ("get_cashflow", symbol, "quarterly", as_of_date),
        "annual_income": ("get_income_statement", symbol, "annual", as_of_date),
        "filings": (
            "get_sec_filings",
            symbol,
            as_of_date,
            "8-K,10-K,10-Q,3,4,5",
            40,
        ),
    }
    for name, call in calls.items():
        try:
            payload = route_to_vendor(*call)
            if _valid_payload(payload):
                payloads[name] = payload
            else:
                errors.append({"dataset": name, "message": "Point-in-time data unavailable"})
        except Exception:  # noqa: BLE001 - a partial overlay remains useful
            errors.append({"dataset": name, "message": "Point-in-time data unavailable"})

    if closing_price is not None and math.isfinite(closing_price) and closing_price > 0:
        price = float(closing_price)
    else:
        try:
            price = load_close_prices([symbol], as_of_date)[symbol]
        except Exception:  # noqa: BLE001 - statements and filings still remain useful
            price = None
            errors.append(
                {"dataset": "closing_price", "message": "Closing price unavailable"}
            )

    fundamentals = _fundamental_pillar(
        payloads.get("quarterly_income", ""),
        payloads.get("quarterly_balance", ""),
        payloads.get("quarterly_cashflow", ""),
        payloads.get("annual_income", ""),
        price,
    )
    filings = _parse_filings(payloads.get("filings", ""))
    events = _event_pillar(filings, as_of_date)
    coverage = fundamentals["coverage"] * 0.75 + (0.25 if filings else 0.0)
    fundamental_score = fundamentals["score"]
    event_risk = events["risk_score"]
    if coverage < 0.45:
        stance = "limited"
    elif event_risk >= 50:
        stance = "caution"
    elif fundamental_score is not None and fundamental_score >= 65 and event_risk < 25:
        stance = "supportive"
    else:
        stance = "balanced"
    modifier = _clip(
        0.70
        + (0.004 * (fundamental_score - 50) if fundamental_score is not None else 0)
        - 0.003 * event_risk
        + 0.20 * coverage,
        0.50,
        1.10,
    )
    report = {
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "closing_price": price,
        "fundamentals": fundamentals,
        "events": events,
        "overlay": {
            "stance": stance,
            "confidence_modifier": round(modifier, 3),
            "data_coverage": round(coverage, 3),
            "notice": (
                "The overlay does not change the opportunity score. It qualifies confidence "
                "and records risks that require human review."
            ),
        },
        "errors": errors,
    }
    if opportunity is not None:
        report["decision_confidence"] = build_decision_confidence(
            opportunity, report
        )
    context_id = capture_research_context(report, config)
    report["registry"] = {"saved": context_id is not None, "context_id": context_id}
    return report
