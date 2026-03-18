"""
monitoring/alerts.py
Alerting rules for ARIA.

Checks metrics against thresholds and fires alerts when breached.
Alerts are designed to be actionable — every alert has a clear cause
and a recommended action.

Alert levels:
  WARNING  — worth knowing, monitor closely
  CRITICAL — requires immediate action

Current alert channels:
  - Console log (always)
  - Structured JSON log (always, via monitoring/logger.py)
  - Webhook (optional — set ALERT_WEBHOOK_URL in .env)

To add Slack, email, or PagerDuty:
  Replace _send_webhook() with the appropriate integration.

Usage:
    python monitoring/alerts.py           # Run all checks
    python monitoring/alerts.py --watch   # Run every 5 minutes
"""
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("alerts")

from config.settings import get_settings
settings = get_settings()


# ── Alert thresholds ──────────────────────────────────────────────────
THRESHOLDS = {
    # Business
    "pending_approval_warning":    5,    # >5 items pending approval
    "pending_approval_critical":   15,   # >15 items pending approval
    "llm_escalation_warning":      15,   # >15% of decisions reaching LLM
    "llm_escalation_critical":     30,   # >30% — rules/ML not working
    "error_rate_warning":          10,   # >10% of runs erroring
    "price_position_warning":      20,   # avg price >20% above market

    # Data freshness
    "stale_data_warning":          3,    # >3 products with stale competitor data
    "stale_data_critical":         8,    # >8 products stale

    # Model health
    "prophet_stale_days":          7,    # Prophet model older than 7 days
    "xgb_stale_days":              30,   # XGBoost model older than 30 days
}


def _fire_alert(level: str, alert_type: str, message: str,
                details: dict = None, recommended_action: str = None):
    """
    Fire an alert. Logs to console and structured log.
    Sends webhook if configured.
    """
    from monitoring.logger import _write

    alert = {
        "event":              "alert",
        "level":              level,
        "alert_type":         alert_type,
        "message":            message,
        "details":            details or {},
        "recommended_action": recommended_action,
        "fired_at":           datetime.utcnow().isoformat(),
    }

    # Console
    if level == "CRITICAL":
        log.critical(f"[ALERT/{level}] {alert_type}: {message}")
    else:
        log.warning(f"[ALERT/{level}] {alert_type}: {message}")

    if recommended_action:
        log.info(f"  Action: {recommended_action}")

    # Structured log
    _write(alert)

    # Webhook (optional)
    _send_webhook(alert)


def _send_webhook(alert: dict):
    """
    Send alert to configured webhook URL.
    Compatible with Slack incoming webhooks and generic JSON webhooks.
    Set ALERT_WEBHOOK_URL in .env to activate.
    """
    webhook_url = getattr(settings, "alert_webhook_url", None)
    if not webhook_url:
        return

    try:
        import requests
        payload = {
            "text": (
                f"*[ARIA {alert['level']}]* {alert['alert_type']}\n"
                f"{alert['message']}\n"
                f"_Action: {alert.get('recommended_action', 'N/A')}_"
            )
        }
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        log.warning(f"Webhook delivery failed: {e}")


# ── Alert checks ──────────────────────────────────────────────────────

def check_approval_queue() -> list:
    """Alert if too many decisions are sitting in the approval queue."""
    alerts = []
    from db.models import get_db, ApprovalQueue

    with get_db() as db:
        pending = db.query(ApprovalQueue).filter(ApprovalQueue.status == "pending").count()

    if pending >= THRESHOLDS["pending_approval_critical"]:
        _fire_alert(
            "CRITICAL", "approval_queue_overloaded",
            f"{pending} decisions pending human approval",
            {"pending_count": pending},
            "Review and action the approval queue immediately: python agent/aria.py --status",
        )
        alerts.append({"type": "approval_queue_overloaded", "level": "CRITICAL", "value": pending})

    elif pending >= THRESHOLDS["pending_approval_warning"]:
        _fire_alert(
            "WARNING", "approval_queue_growing",
            f"{pending} decisions pending human approval",
            {"pending_count": pending},
            "Review approval queue: python agent/aria.py --status",
        )
        alerts.append({"type": "approval_queue_growing", "level": "WARNING", "value": pending})

    return alerts


def check_llm_escalation_rate() -> list:
    """Alert if too many decisions are reaching the LLM layer."""
    alerts = []
    from monitoring.metrics import get_business_metrics
    metrics = get_business_metrics(days=1)

    rate = metrics.get("llm_escalation_pct", 0)

    if rate >= THRESHOLDS["llm_escalation_critical"]:
        _fire_alert(
            "CRITICAL", "high_llm_escalation",
            f"LLM escalation rate is {rate:.1f}% (threshold: {THRESHOLDS['llm_escalation_critical']}%)",
            {"rate_pct": rate, "total_decisions": metrics.get("total_decisions")},
            "Review rules engine — rules may be too narrow or ML confidence too low. "
            "Run: python src/model_router.py --all",
        )
        alerts.append({"type": "high_llm_escalation", "level": "CRITICAL", "value": rate})

    elif rate >= THRESHOLDS["llm_escalation_warning"]:
        _fire_alert(
            "WARNING", "elevated_llm_escalation",
            f"LLM escalation rate is {rate:.1f}%",
            {"rate_pct": rate},
            "Monitor — may indicate rules need tuning or ML model needs retraining.",
        )
        alerts.append({"type": "elevated_llm_escalation", "level": "WARNING", "value": rate})

    return alerts


def check_data_freshness() -> list:
    """Alert if competitor price data is stale."""
    alerts = []
    from monitoring.metrics import get_data_freshness
    freshness = get_data_freshness()

    stale = freshness["stale_count"]

    if stale >= THRESHOLDS["stale_data_critical"]:
        _fire_alert(
            "CRITICAL", "stale_competitor_data",
            f"{stale}/{freshness['n_products']} products have stale competitor price data (>24h)",
            {"stale_count": stale, "avg_age_hours": freshness.get("avg_age_hours")},
            "Refresh competitor data: python scripts/fetch_competitors.py",
        )
        alerts.append({"type": "stale_competitor_data", "level": "CRITICAL", "value": stale})

    elif stale >= THRESHOLDS["stale_data_warning"]:
        _fire_alert(
            "WARNING", "aging_competitor_data",
            f"{stale} products have stale competitor data",
            {"stale_count": stale},
            "Consider refreshing: python scripts/fetch_competitors.py",
        )
        alerts.append({"type": "aging_competitor_data", "level": "WARNING", "value": stale})

    return alerts


def check_model_health() -> list:
    """Alert if ML models are stale or have degraded metrics."""
    alerts = []
    from monitoring.metrics import get_model_metrics
    m = get_model_metrics()

    # XGBoost staleness
    xgb = m.get("xgboost", {})
    if isinstance(xgb, dict) and xgb.get("stale"):
        _fire_alert(
            "WARNING", "xgboost_model_stale",
            f"XGBoost pricing model is {xgb.get('age_days')} days old",
            {"age_days": xgb.get("age_days"), "trained_at": xgb.get("trained_at")},
            "Retrain: python src/features.py && python src/pricing_model.py",
        )
        alerts.append({"type": "xgboost_stale", "level": "WARNING"})

    # Prophet staleness
    for category, status in m.get("prophet", {}).items():
        if isinstance(status, dict) and status.get("stale"):
            _fire_alert(
                "WARNING", "prophet_model_stale",
                f"Prophet model for '{category}' is {status.get('age_days')} days old",
                {"category": category, "age_days": status.get("age_days")},
                f"Retrain: python src/demand_forecast.py -c {category}",
            )
            alerts.append({"type": "prophet_stale", "level": "WARNING", "category": category})

    return alerts


def check_price_position() -> list:
    """Alert if our average prices drift too far above market."""
    alerts = []
    from monitoring.metrics import get_business_metrics
    metrics = get_business_metrics(days=3)

    position = metrics.get("avg_price_position", 0)

    if position > THRESHOLDS["price_position_warning"]:
        _fire_alert(
            "WARNING", "prices_above_market",
            f"Average price position is {position:+.1f}% above market median",
            {"avg_position_pct": position},
            "Review pricing — run competitor fetch and re-route: "
            "python scripts/fetch_competitors.py && python src/model_router.py --all",
        )
        alerts.append({"type": "prices_above_market", "level": "WARNING", "value": position})

    return alerts


def run_all_checks() -> list:
    """Run all alert checks. Returns list of fired alerts."""
    log.info(f"Running alert checks at {datetime.utcnow().strftime('%H:%M UTC')}")
    all_alerts = []

    checks = [
        ("Approval queue",    check_approval_queue),
        ("LLM escalation",    check_llm_escalation_rate),
        ("Data freshness",    check_data_freshness),
        ("Model health",      check_model_health),
        ("Price position",    check_price_position),
    ]

    for name, check_fn in checks:
        try:
            fired = check_fn()
            all_alerts.extend(fired)
            status = f"{len(fired)} alert(s)" if fired else "OK"
            log.info(f"  {name:<20} {status}")
        except Exception as e:
            log.error(f"  {name:<20} check failed: {e}")

    if all_alerts:
        log.warning(f"\n{len(all_alerts)} alert(s) fired")
    else:
        log.info("\nAll checks passed — no alerts")

    return all_alerts


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ARIA alert checks")
    parser.add_argument("--watch", action="store_true", help="Run every 5 minutes")
    args = parser.parse_args()

    if args.watch:
        log.info("Alert monitor started — checking every 5 minutes. Ctrl+C to stop.")
        while True:
            try:
                run_all_checks()
                time.sleep(300)
            except KeyboardInterrupt:
                log.info("Alert monitor stopped")
                break
    else:
        alerts = run_all_checks()
        sys.exit(0 if not alerts else 1)