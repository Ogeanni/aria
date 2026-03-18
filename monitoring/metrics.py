"""
monitoring/metrics.py
Business and system metrics for ARIA.

Three categories of metrics:

1. Business metrics — did repricing improve outcomes?
   - reprice_success_rate: % of executed repricings that improved revenue
   - avg_price_vs_market:  how our prices sit vs competitor median
   - approval_queue_depth: backlog of decisions awaiting human review

2. Model quality metrics — is the ML pipeline healthy?
   - prediction_drift:  are model recommendations shifting over time?
   - layer_distribution: how often does each routing layer fire?
   - llm_escalation_rate: % of decisions reaching the LLM

3. System metrics — is the pipeline running correctly?
   - run_success_rate: % of agent runs completing without error
   - avg_run_duration: how long each run takes
   - data_freshness:   how old is the competitor price data?

Usage:
    python monitoring/metrics.py          # Print full metrics report
    python monitoring/metrics.py --json   # Output as JSON
"""
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("metrics")

from db.models import (
    get_db, Product, AgentDecision, PriceHistory,
    CompetitorPrice, RepricingOutcome, ApprovalQueue
)


def get_business_metrics(days: int = 7) -> dict:
    """
    Business metrics over the last N days.
    These are the metrics that matter to the merchant.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    with get_db() as db:
        # Total decisions made
        total_decisions = (
            db.query(AgentDecision)
            .filter(AgentDecision.created_at >= cutoff)
            .count()
        )

        # Executed (actual price changes)
        executed = (
            db.query(AgentDecision)
            .filter(
                AgentDecision.created_at >= cutoff,
                AgentDecision.was_executed == True,
            )
            .count()
        )

        # Pending approval backlog
        pending_approval = (
            db.query(ApprovalQueue)
            .filter(ApprovalQueue.status == "pending")
            .count()
        )

        # Decision breakdown by action
        from sqlalchemy import func
        action_counts = dict(
            db.query(AgentDecision.decision_type, func.count(AgentDecision.id))
            .filter(AgentDecision.created_at >= cutoff)
            .group_by(AgentDecision.decision_type)
            .all()
        )

        # Layer distribution
        layer_counts = dict(
            db.query(AgentDecision.decision_source, func.count(AgentDecision.id))
            .filter(AgentDecision.created_at >= cutoff)
            .group_by(AgentDecision.decision_source)
            .all()
        )

        # Average price position vs market (how competitive are we?)
        comp_rows = (
            db.query(CompetitorPrice)
            .filter(CompetitorPrice.scraped_at >= cutoff)
            .all()
        )
        comp_data = [{"product_id": r.product_id,
                      "competitor_price": float(r.competitor_price)} for r in comp_rows]

        products = db.query(Product).filter(Product.is_active == True).all()
        product_data = [{"id": p.id, "current_price": float(p.current_price)} for p in products]

    # Compute avg price position
    price_positions = []
    for p in product_data:
        comps = [c["competitor_price"] for c in comp_data if c["product_id"] == p["id"]]
        if comps:
            import numpy as np
            median = float(np.median(comps))
            if median > 0:
                position = (p["current_price"] - median) / median * 100
                price_positions.append(position)

    avg_position = round(sum(price_positions) / len(price_positions), 2) if price_positions else 0.0

    # LLM escalation rate
    llm_count    = layer_counts.get("llm", 0)
    llm_rate     = round(llm_count / total_decisions * 100, 1) if total_decisions > 0 else 0.0

    # Execution rate
    exec_rate    = round(executed / total_decisions * 100, 1) if total_decisions > 0 else 0.0

    return {
        "window_days":         days,
        "total_decisions":     total_decisions,
        "executed":            executed,
        "execution_rate_pct":  exec_rate,
        "pending_approval":    pending_approval,
        "action_counts":       action_counts,
        "layer_counts":        layer_counts,
        "llm_escalation_pct":  llm_rate,
        "avg_price_position":  avg_position,
        "n_products_tracked":  len(product_data),
    }


def get_model_metrics() -> dict:
    """
    ML model health metrics.
    Checks model ages, recent prediction distribution, and drift signals.
    """
    from config.settings import get_settings
    settings = get_settings()

    # Prophet model freshness
    prophet_status = {}
    for category in ("electronics", "fashion", "home_goods", "sports"):
        meta_path = settings.prophet_dir / f"prophet_{category}_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                trained_at = datetime.fromisoformat(meta["trained_at"])
                age_days   = (datetime.utcnow() - trained_at).days
                prophet_status[category] = {
                    "trained_at": meta["trained_at"],
                    "age_days":   age_days,
                    "mae":        meta.get("mae"),
                    "n_weeks":    meta.get("n_weeks"),
                    "stale":      age_days >= 7,
                }
            except Exception:
                prophet_status[category] = {"status": "error_reading_meta"}
        else:
            prophet_status[category] = {"status": "not_trained"}

    # XGBoost model freshness
    xgb_status = {"status": "not_trained"}
    if settings.xgb_meta_path.exists():
        try:
            meta       = json.loads(settings.xgb_meta_path.read_text())
            trained_at = datetime.fromisoformat(meta["trained_at"])
            age_days   = (datetime.utcnow() - trained_at).days
            xgb_status = {
                "trained_at":  meta["trained_at"],
                "age_days":    age_days,
                "n_products":  meta.get("n_products"),
                "n_features":  meta.get("n_features"),
                "test_mae":    meta.get("metrics", {}).get("test_mae"),
                "test_r2":     meta.get("metrics", {}).get("test_r2"),
                "note":        meta.get("metrics", {}).get("note"),
                "stale":       age_days >= 30,
            }
        except Exception:
            xgb_status = {"status": "error_reading_meta"}

    # Recent prediction drift — compare last 24h avg recommended vs current prices
    cutoff = datetime.utcnow() - timedelta(hours=24)
    with get_db() as db:
        recent_decisions = (
            db.query(AgentDecision)
            .filter(
                AgentDecision.created_at >= cutoff,
                AgentDecision.decision_source == "ml_model",
            )
            .all()
        )
        drift_data = [{
            "change_pct": row.change_pct,
            "confidence": row.confidence,
        } for row in recent_decisions]

    avg_ml_change = 0.0
    if drift_data:
        avg_ml_change = round(
            sum(d["change_pct"] for d in drift_data) / len(drift_data), 2
        )

    return {
        "prophet":       prophet_status,
        "xgboost":       xgb_status,
        "ml_decisions_24h": len(drift_data),
        "avg_ml_change_pct": avg_ml_change,
    }


def get_data_freshness() -> dict:
    """
    How fresh is our data? Flags stale competitor prices and trends.
    Data freshness directly impacts decision quality.
    """
    with get_db() as db:
        # Latest competitor price scrape per product
        from sqlalchemy import func
        latest_scrapes = dict(
            db.query(
                CompetitorPrice.product_id,
                func.max(CompetitorPrice.scraped_at)
            )
            .group_by(CompetitorPrice.product_id)
            .all()
        )

        products = db.query(Product).filter(Product.is_active == True).all()
        product_ids = [p.id for p in products]

    now = datetime.utcnow()
    freshness = []
    stale_count = 0

    for pid in product_ids:
        last = latest_scrapes.get(pid)
        if last:
            age_hours = (now - last).total_seconds() / 3600
            stale     = age_hours > 24
            if stale:
                stale_count += 1
            freshness.append({"product_id": pid, "age_hours": round(age_hours, 1), "stale": stale})
        else:
            stale_count += 1
            freshness.append({"product_id": pid, "age_hours": None, "stale": True})

    avg_age = None
    ages = [f["age_hours"] for f in freshness if f["age_hours"] is not None]
    if ages:
        avg_age = round(sum(ages) / len(ages), 1)

    return {
        "n_products":        len(product_ids),
        "stale_count":       stale_count,
        "fresh_count":       len(product_ids) - stale_count,
        "avg_age_hours":     avg_age,
        "all_fresh":         stale_count == 0,
        "freshness_detail":  freshness,
    }


def get_full_report(days: int = 7) -> dict:
    """Combine all metrics into one report dict."""
    return {
        "generated_at":  datetime.utcnow().isoformat(),
        "business":      get_business_metrics(days=days),
        "models":        get_model_metrics(),
        "data_freshness":get_data_freshness(),
    }


def print_report(report: dict):
    """Human-readable metrics report."""
    b = report["business"]
    m = report["models"]
    f = report["data_freshness"]

    print(f"\n{'='*60}")
    print(f"ARIA METRICS REPORT  —  {report['generated_at'][:16]}")
    print(f"{'='*60}")

    print(f"\nBUSINESS METRICS  (last {b['window_days']} days)")
    print(f"  Total decisions    : {b['total_decisions']}")
    print(f"  Executed           : {b['executed']}  ({b['execution_rate_pct']}%)")
    print(f"  Pending approval   : {b['pending_approval']}")
    print(f"  LLM escalation     : {b['llm_escalation_pct']}%  "
          f"({'healthy' if b['llm_escalation_pct'] < 15 else 'HIGH — review rules'})")
    print(f"  Avg price position : {b['avg_price_position']:+.1f}% vs market")

    if b["action_counts"]:
        print(f"\n  Action breakdown:")
        for action, count in sorted(b["action_counts"].items()):
            pct = count / b["total_decisions"] * 100 if b["total_decisions"] > 0 else 0
            print(f"    {action:<12} {count:>4}  ({pct:.1f}%)")

    if b["layer_counts"]:
        print(f"\n  Layer distribution:")
        for layer, count in sorted(b["layer_counts"].items()):
            pct = count / b["total_decisions"] * 100 if b["total_decisions"] > 0 else 0
            bar = "|" * int(pct / 5)
            print(f"    {layer:<12} {count:>4}  ({pct:>5.1f}%)  {bar}")

    print(f"\nMODEL HEALTH")
    xgb = m["xgboost"]
    if isinstance(xgb, dict) and "trained_at" in xgb:
        stale_flag = "  [STALE]" if xgb.get("stale") else ""
        print(f"  XGBoost  : trained {xgb['age_days']}d ago  "
              f"MAE=${xgb.get('test_mae', '?')}  "
              f"R2={xgb.get('test_r2', '?')}{stale_flag}")
        if xgb.get("note"):
            print(f"             Note: {xgb['note']}")
    else:
        print(f"  XGBoost  : {xgb.get('status', 'unknown')}")

    print(f"  Prophet models:")
    for cat, status in m["prophet"].items():
        if isinstance(status, dict) and "age_days" in status:
            stale_flag = "  [STALE]" if status.get("stale") else ""
            print(f"    {cat:<14} {status['age_days']}d old  "
                  f"MAE={status.get('mae', '?'):.2f}  "
                  f"{status.get('n_weeks', '?')} weeks{stale_flag}")
        else:
            print(f"    {cat:<14} {status.get('status', 'unknown')}")

    print(f"\n  ML decisions (24h): {m['ml_decisions_24h']}  "
          f"avg change: {m['avg_ml_change_pct']:+.2f}%")

    print(f"\nDATA FRESHNESS")
    print(f"  Products tracked   : {f['n_products']}")
    print(f"  Fresh (<24h)       : {f['fresh_count']}")
    print(f"  Stale (>24h)       : {f['stale_count']}"
          f"{'  [ACTION NEEDED]' if f['stale_count'] > 0 else ''}")
    if f["avg_age_hours"] is not None:
        print(f"  Avg data age       : {f['avg_age_hours']:.1f}h")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="ARIA metrics report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    report = get_full_report(days=args.days)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)