"""
monitoring/feedback.py
Feedback loop for ARIA.

Closes the sense -> decide -> act -> LEARN cycle.

What this does:
  1. Finds price changes executed by the agent in the last 7 days
  2. Compares sell-through rate before vs after each change
  3. Labels each decision: positive / neutral / negative outcome
  4. Stores outcome labels in repricing_outcomes table
  5. Computes agent decision quality metrics over time
  6. Triggers model retraining when enough new labeled data accumulates

Why this matters (playbook: feedback loops):
  Without a feedback loop, the agent makes decisions but never learns
  whether they worked. With it, every price change becomes a training
  signal. After 30+ labeled outcomes, you can fine-tune the routing
  thresholds and retrain XGBoost on real outcome data instead of
  the synthetic target formula.

Sell-through proxy:
  We don't have a real sales feed in this build — that would require
  connecting to Shopify Orders API or a POS system.
  Instead we use a realistic simulation:
    - Price decreases that moved us closer to market → positive signal
    - Price increases where demand was rising → positive signal
    - Changes that moved us further from market → negative signal
  This is explicitly documented as a proxy. In production, replace
  simulate_sell_through() with a real sales query.

Usage:
    python monitoring/feedback.py               # Score recent decisions
    python monitoring/feedback.py --days 14     # Score last 14 days
    python monitoring/feedback.py --report      # Print outcome report
    python monitoring/feedback.py --retrain     # Score + trigger retraining if ready
"""
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feedback")

from db.models import (
    get_db, AgentDecision, RepricingOutcome,
    CompetitorPrice, Product
)
from config.settings import get_settings

settings = get_settings()

# Minimum labeled outcomes before triggering XGBoost retraining
RETRAIN_THRESHOLD = 20


# ══════════════════════════════════════════════════════════════════════
# SELL-THROUGH SIMULATION
# ══════════════════════════════════════════════════════════════════════

def simulate_sell_through(
    decision: dict,
    product: dict,
    comp_prices: list,
) -> dict:
    """
    Simulate sell-through rate before/after a price change.

    IN PRODUCTION: Replace this with a real Shopify Orders API query:
        orders_before = shopify.get_orders(product_id, date_range=before_window)
        orders_after  = shopify.get_orders(product_id, date_range=after_window)
        units_before  = sum(o.quantity for o in orders_before) / days_before
        units_after   = sum(o.quantity for o in orders_after)  / days_after

    CURRENT PROXY:
    We simulate based on pricing theory:
      - Moving closer to market median → better sell-through
      - Moving further from market median → worse sell-through
      - Demand signal modifies the base rate
      - Random noise added for realism

    This proxy produces directionally correct labels for the feedback
    loop even without real sales data. Labels will be noisy but not
    random — they reflect real pricing dynamics.
    """
    import numpy as np

    action        = decision.get("decision_type", "hold")
    old_price     = float(decision.get("current_price", 0))
    new_price     = float(decision.get("recommended_price", old_price))
    trend_index   = float(decision.get("trend_index", 50) or 50)
    inv_pressure  = int(decision.get("inventory_pressure", 0) or 0)

    if not comp_prices:
        comp_median = old_price
    else:
        import statistics
        comp_median = statistics.median(comp_prices)

    # Base sell-through rate: units/day at a neutral price
    # Adjusted by inventory pressure and demand
    demand_factor = 0.8 + (trend_index / 100) * 0.4   # 0.8 - 1.2
    base_rate     = 2.0 * demand_factor                 # ~1.6 - 2.4 units/day

    # Position vs market before and after change
    pos_before = (old_price - comp_median) / comp_median if comp_median > 0 else 0
    pos_after  = (new_price - comp_median) / comp_median if comp_median > 0 else 0

    # Price elasticity: 1% price increase -> ~1.5% volume decrease
    ELASTICITY = -1.5
    rate_before = base_rate * (1 + ELASTICITY * pos_before)
    rate_after  = base_rate * (1 + ELASTICITY * pos_after)

    # Add noise (±20%)
    rng = np.random.default_rng(
        abs(hash(f"{decision.get('id', 0)}:{action}")) % (2**31)
    )
    noise_before = rng.uniform(0.80, 1.20)
    noise_after  = rng.uniform(0.80, 1.20)

    rate_before = max(0.1, rate_before * noise_before)
    rate_after  = max(0.1, rate_after  * noise_after)

    # Revenue per day
    rev_before = rate_before * old_price
    rev_after  = rate_after  * new_price

    return {
        "units_before": round(rate_before, 2),
        "units_after":  round(rate_after,  2),
        "rev_before":   round(rev_before,  2),
        "rev_after":    round(rev_after,   2),
    }


# ══════════════════════════════════════════════════════════════════════
# OUTCOME LABELING
# ══════════════════════════════════════════════════════════════════════

def label_outcome(sim: dict, decision: dict) -> int:
    """
    Label a repricing outcome: 1 (positive), 0 (neutral), -1 (negative).

    Labeling logic:
      Positive (1):  Revenue improved by >3%
      Neutral  (0):  Revenue changed by less than 3% (within noise)
      Negative (-1): Revenue declined by >3%

    Revenue is the primary metric because it combines both price
    and volume effects. A price increase that tanks volume is negative
    even if the price itself was "correct". A price decrease that
    drives enough volume to increase revenue is positive.

    Why 3% threshold:
      Our sell-through simulation has ~20% noise. We use 3% as the
      threshold to avoid labeling noise as signal. In production with
      real sales data, you could tighten this to 1-2%.
    """
    rev_before = sim["rev_before"]
    rev_after  = sim["rev_after"]

    if rev_before <= 0:
        return 0

    pct_change = (rev_after - rev_before) / rev_before * 100

    if pct_change > 3:
        return 1
    elif pct_change < -3:
        return -1
    else:
        return 0


# ══════════════════════════════════════════════════════════════════════
# MAIN SCORING PIPELINE
# ══════════════════════════════════════════════════════════════════════

def score_recent_decisions(days: int = 7, dry_run: bool = False) -> list:
    """
    Find executed decisions from the last N days that don't yet have
    outcome labels, simulate sell-through, label outcomes, and save.

    Returns list of outcome dicts.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Load decisions that were executed and not yet scored
    with get_db() as db:
        decisions = (
            db.query(AgentDecision)
            .filter(
                AgentDecision.was_executed == True,
                AgentDecision.decision_type != "hold",
                AgentDecision.created_at >= cutoff,
            )
            .all()
        )
        decision_data = [{
            "id":                 d.id,
            "product_id":         d.product_id,
            "decision_type":      d.decision_type,
            "decision_source":    d.decision_source,
            "current_price":      float(d.current_price),
            "recommended_price":  float(d.recommended_price),
            "change_pct":         d.change_pct,
            "trend_index":        d.trend_index,
            "confidence":         d.confidence,
            "created_at":         d.created_at,
        } for d in decisions]

        # Find which are already scored
        scored_ids = set(
            row[0] for row in
            db.query(RepricingOutcome.decision_id).all()
        )

    unscored = [d for d in decision_data if d["id"] not in scored_ids]
    log.info(f"Found {len(decision_data)} executed decisions, "
             f"{len(unscored)} unscored")

    if not unscored:
        log.info("All decisions already scored")
        return []

    outcomes = []

    for decision in unscored:
        pid = decision["product_id"]

        # Load product and competitor prices
        with get_db() as db:
            product = db.query(Product).filter(Product.id == pid).first()
            if not product:
                continue
            product_data = {
                "id":    product.id,
                "name":  product.name,
                "category": product.category,
            }

            comp_rows = (
                db.query(CompetitorPrice)
                .filter(CompetitorPrice.product_id == pid)
                .order_by(CompetitorPrice.scraped_at.desc())
                .limit(50)
                .all()
            )
            comp_prices = [float(r.competitor_price) for r in comp_rows]

        # Simulate outcome
        sim     = simulate_sell_through(decision, product_data, comp_prices)
        label   = label_outcome(sim, decision)
        rev_chg = round(
            (sim["rev_after"] - sim["rev_before"]) / sim["rev_before"] * 100, 1
        ) if sim["rev_before"] > 0 else 0

        outcome = {
            "decision_id":        decision["id"],
            "product_id":         pid,
            "product_name":       product_data["name"],
            "action":             decision["decision_type"],
            "layer":              decision["decision_source"],
            "old_price":          decision["current_price"],
            "new_price":          decision["recommended_price"],
            "change_pct":         decision["change_pct"],
            "units_before":       sim["units_before"],
            "units_after":        sim["units_after"],
            "rev_before":         sim["rev_before"],
            "rev_after":          sim["rev_after"],
            "rev_change_pct":     rev_chg,
            "outcome_label":      label,
            "outcome_notes":      f"simulated_proxy | rev_change={rev_chg:+.1f}%",
            "measured_at":        datetime.utcnow().isoformat(),
        }
        outcomes.append(outcome)

        label_str = {1: "POSITIVE", 0: "NEUTRAL", -1: "NEGATIVE"}[label]
        log.info(
            f"  [{label_str}] {product_data['name'][:30]:<30} "
            f"{decision['decision_type']:<10} "
            f"${decision['current_price']:.2f} -> ${decision['recommended_price']:.2f} "
            f"rev: {rev_chg:+.1f}%  [{decision['decision_source']}]"
        )

        if not dry_run:
            _save_outcome(outcome)

    return outcomes


def _save_outcome(outcome: dict):
    """Upsert outcome into repricing_outcomes table."""
    with get_db() as db:
        # Check for existing row
        existing = (
            db.query(RepricingOutcome)
            .filter(RepricingOutcome.decision_id == outcome["decision_id"])
            .first()
        )
        if existing:
            existing.outcome_label        = outcome["outcome_label"]
            existing.units_sold_before_7d = outcome["units_before"]
            existing.units_sold_after_7d  = outcome["units_after"]
            existing.revenue_before_7d    = outcome["rev_before"]
            existing.revenue_after_7d     = outcome["rev_after"]
            existing.outcome_notes        = outcome["outcome_notes"]
            existing.measured_at          = datetime.utcnow()
        else:
            db.add(RepricingOutcome(
                product_id=            outcome["product_id"],
                decision_id=           outcome["decision_id"],
                price_before=          outcome["old_price"],
                price_after=           outcome["new_price"],
                units_sold_before_7d=  outcome["units_before"],
                units_sold_after_7d=   outcome["units_after"],
                revenue_before_7d=     outcome["rev_before"],
                revenue_after_7d=      outcome["rev_after"],
                outcome_label=         outcome["outcome_label"],
                outcome_notes=         outcome["outcome_notes"],
                measured_at=           datetime.utcnow(),
                created_at=            datetime.utcnow(),
            ))


# ══════════════════════════════════════════════════════════════════════
# OUTCOME REPORT
# ══════════════════════════════════════════════════════════════════════

def get_outcome_report() -> dict:
    """
    Summarise all labeled outcomes — agent decision quality over time.
    This is the ground truth answer to "is ARIA making good decisions?"
    """
    with get_db() as db:
        outcomes = db.query(RepricingOutcome).all()
        data = [{
            "outcome_label":   o.outcome_label,
            "price_before":    float(o.price_before),
            "price_after":     float(o.price_after),
            "revenue_before":  float(o.revenue_before_7d or 0),
            "revenue_after":   float(o.revenue_after_7d  or 0),
            "decision_id":     o.decision_id,
        } for o in outcomes]

    if not data:
        return {"total": 0, "message": "No outcomes scored yet"}

    total     = len(data)
    positive  = sum(1 for d in data if d["outcome_label"] == 1)
    neutral   = sum(1 for d in data if d["outcome_label"] == 0)
    negative  = sum(1 for d in data if d["outcome_label"] == -1)
    success_rate = round(positive / total * 100, 1)

    # Avg revenue change across all decisions
    rev_changes = []
    for d in data:
        if d["revenue_before"] > 0:
            chg = (d["revenue_after"] - d["revenue_before"]) / d["revenue_before"] * 100
            rev_changes.append(chg)

    avg_rev_change = round(sum(rev_changes) / len(rev_changes), 2) if rev_changes else 0.0

    return {
        "total":            total,
        "positive":         positive,
        "neutral":          neutral,
        "negative":         negative,
        "success_rate_pct": success_rate,
        "avg_rev_change_pct": avg_rev_change,
        "ready_to_retrain": total >= RETRAIN_THRESHOLD,
        "retrain_threshold": RETRAIN_THRESHOLD,
    }


def print_outcome_report(report: dict):
    print(f"\n{'='*55}")
    print("ARIA FEEDBACK LOOP — OUTCOME REPORT")
    print(f"{'='*55}")

    if report.get("total", 0) == 0:
        print("  No outcomes scored yet.")
        print("  Run: python monitoring/feedback.py")
        print(f"{'='*55}\n")
        return

    total = report["total"]
    print(f"  Total scored decisions : {total}")
    print(f"  Positive outcomes      : {report['positive']}  "
          f"({report['positive']/total*100:.1f}%)")
    print(f"  Neutral outcomes       : {report['neutral']}  "
          f"({report['neutral']/total*100:.1f}%)")
    print(f"  Negative outcomes      : {report['negative']}  "
          f"({report['negative']/total*100:.1f}%)")
    print(f"\n  Success rate           : {report['success_rate_pct']}%")
    print(f"  Avg revenue change     : {report['avg_rev_change_pct']:+.2f}%")

    if report.get("ready_to_retrain"):
        print(f"\n  READY TO RETRAIN: {total} outcomes >= threshold {report['retrain_threshold']}")
        print("  Run: python src/features.py && python src/pricing_model.py")
    else:
        remaining = report["retrain_threshold"] - total
        print(f"\n  Need {remaining} more outcomes to trigger retraining "
              f"({total}/{report['retrain_threshold']})")

    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════
# RETRAINING TRIGGER
# ══════════════════════════════════════════════════════════════════════

def maybe_trigger_retraining(report: dict, auto: bool = False) -> bool:
    """
    Trigger XGBoost retraining if enough labeled outcomes have accumulated.

    auto=False: log that retraining is recommended, don't actually run it
    auto=True:  run retraining pipeline automatically

    Why we don't auto-retrain by default:
      Automatic retraining in production needs a quality gate — you don't
      want a bad data batch to silently degrade your production model.
      The recommended pattern is:
        1. Accumulate outcomes (this function checks readiness)
        2. Human reviews the outcome report
        3. Human triggers retraining manually or via CI
    """
    if not report.get("ready_to_retrain"):
        return False

    log.info(
        f"RETRAINING RECOMMENDED: {report['total']} labeled outcomes available "
        f"(threshold: {report['retrain_threshold']})"
    )
    log.info(
        f"  Success rate: {report['success_rate_pct']}%  "
        f"Avg rev change: {report['avg_rev_change_pct']:+.2f}%"
    )

    if not auto:
        log.info("  To retrain: python src/features.py && python src/pricing_model.py")
        return False

    log.info("  Auto-retraining enabled — starting pipeline...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "features.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if result.returncode != 0:
            log.error(f"  features.py failed: {result.stderr[-500:]}")
            return False

        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "pricing_model.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if result.returncode != 0:
            log.error(f"  pricing_model.py failed: {result.stderr[-500:]}")
            return False

        log.info("  Retraining complete")
        return True
    except Exception as e:
        log.error(f"  Retraining failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ARIA feedback loop")
    parser.add_argument("--days",     type=int, default=7,
                        help="Score decisions from last N days (default: 7)")
    parser.add_argument("--report",   action="store_true",
                        help="Print outcome report only, skip scoring")
    parser.add_argument("--retrain",  action="store_true",
                        help="Score + auto-trigger retraining if ready")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Score decisions but do not save outcomes")
    args = parser.parse_args()

    if args.report:
        report = get_outcome_report()
        print_outcome_report(report)
        return 0

    log.info(f"ARIA Feedback Loop — scoring decisions from last {args.days} days")
    if args.dry_run:
        log.info("DRY RUN — outcomes will not be saved")

    outcomes = score_recent_decisions(days=args.days, dry_run=args.dry_run)

    log.info(f"\nScored {len(outcomes)} decisions:")
    if outcomes:
        pos = sum(1 for o in outcomes if o["outcome_label"] == 1)
        neu = sum(1 for o in outcomes if o["outcome_label"] == 0)
        neg = sum(1 for o in outcomes if o["outcome_label"] == -1)
        log.info(f"  Positive: {pos}  Neutral: {neu}  Negative: {neg}")

    report = get_outcome_report()
    print_outcome_report(report)

    if args.retrain:
        maybe_trigger_retraining(report, auto=True)
    else:
        maybe_trigger_retraining(report, auto=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())