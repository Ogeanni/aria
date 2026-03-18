"""
monitoring/logger.py
Structured logging for ARIA.

Every agent run, decision, and execution is logged in structured JSON
so logs can be queried, aggregated, and alerted on programmatically.

Two outputs:
  1. Console  — human-readable (already set up via basicConfig)
  2. File     — structured JSON, one event per line (JSON Lines format)

JSON Lines format means every log entry is a valid JSON object on its own
line. Easy to stream into any log aggregator (Datadog, Grafana Loki,
CloudWatch, etc.) without a parser.

Usage:
    from monitoring.logger import log_run, log_decision, log_execution
    log_run(run_summary)
    log_decision(decision)
    log_execution(result)
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_DIR  = ROOT / "logs"
LOG_FILE = LOG_DIR / "aria.jsonl"

LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("aria.logger")


def _write(event: dict):
    """Append one JSON event to the log file."""
    event.setdefault("logged_at", datetime.utcnow().isoformat())
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        log.warning(f"Failed to write structured log: {e}")


def log_run(summary: dict):
    """Log an agent run summary."""
    _write({
        "event":            "agent_run",
        "run_id":           summary.get("run_id"),
        "status":           summary.get("status"),
        "products_reviewed":summary.get("products_reviewed", 0),
        "executed":         summary.get("executed", 0),
        "held":             summary.get("held", 0),
        "pending_approval": summary.get("pending_approval", 0),
        "errors":           summary.get("errors", 0),
        "elapsed_seconds":  summary.get("elapsed_seconds", 0),
        "dry_run":          summary.get("dry_run", False),
    })


def log_decision(decision, demand: dict = None):
    """Log a routing decision."""
    _write({
        "event":             "decision",
        "product_id":        decision.product_id,
        "product_name":      decision.product_name,
        "category":          decision.category,
        "action":            decision.action,
        "layer":             decision.layer,
        "rule_triggered":    decision.rule_triggered,
        "confidence":        decision.confidence,
        "current_price":     decision.current_price,
        "recommended_price": decision.recommended_price,
        "change_pct":        decision.change_pct,
        "comp_price_med":    decision.comp_price_med,
        "trend_index":       decision.trend_index,
        "trend_direction":   decision.trend_direction,
        "inventory_pressure":decision.inventory_pressure,
        "requires_approval": decision.requires_approval,
        "rationale":         decision.rationale,
    })


def log_execution(result: dict):
    """Log an execution result."""
    _write({
        "event":       "execution",
        "status":      result.get("status"),
        "decision_id": result.get("decision_id"),
        "product_id":  result.get("product_id"),
        "action":      result.get("action"),
        "old_price":   result.get("old_price"),
        "new_price":   result.get("new_price"),
        "change_pct":  result.get("change_pct"),
        "error":       result.get("error"),
    })


def log_model_metrics(category: str, mae: float, rmse: float, model_type: str = "prophet"):
    """Log ML model evaluation metrics."""
    _write({
        "event":      "model_metrics",
        "model_type": model_type,
        "category":   category,
        "mae":        mae,
        "rmse":       rmse,
    })


def log_drift_alert(product_id: int, product_name: str, metric: str,
                    current_value: float, baseline_value: float, pct_change: float):
    """Log when a metric drifts beyond the alert threshold."""
    _write({
        "event":          "drift_alert",
        "product_id":     product_id,
        "product_name":   product_name,
        "metric":         metric,
        "current_value":  current_value,
        "baseline_value": baseline_value,
        "pct_change":     pct_change,
    })


def read_recent_logs(n: int = 50, event_type: str = None) -> list:
    """
    Read the last N log entries from the JSON Lines file.
    Optionally filter by event_type.
    """
    if not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text().strip().split("\n")
        events = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event_type is None or event.get("event") == event_type:
                    events.append(event)
            except json.JSONDecodeError:
                continue
        return events[-n:]
    except Exception as e:
        log.warning(f"Failed to read logs: {e}")
        return []