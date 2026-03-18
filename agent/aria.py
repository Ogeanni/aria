"""
agent/aria.py
ARIA — Autonomous Repricing & Inventory Agent

The main agent loop. Runs on a schedule and autonomously:
  1. Fetches fresh competitor prices and demand signals
  2. Rebuilds the feature matrix
  3. Gets demand forecasts from Prophet models
  4. Routes every product through the decision layers (rules -> ML -> LLM)
  5. Executes approved price changes
  6. Queues large changes for human review
  7. Records everything to the audit log
  8. Reports a run summary

This is a sense -> decide -> act loop, not a chatbot.
It runs without human prompting on a configurable schedule.

Usage:
    python agent/aria.py              # Run on schedule (every N minutes per .env)
    python agent/aria.py --once       # Single run then exit
    python agent/aria.py --demo       # Single run, demo mode, verbose output
    python agent/aria.py --dry-run    # Route decisions but do not execute
    python agent/aria.py --status     # Show DB status and pending approvals
"""
import sys
import time
import logging
import argparse
import uuid
from datetime import datetime
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
log = logging.getLogger("aria")

from config.settings import get_settings
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════
# SINGLE AGENT RUN
# ══════════════════════════════════════════════════════════════════════

def run_once(dry_run: bool = False, verbose: bool = False) -> dict:
    """
    Execute one full agent cycle:
      Sense -> Decide -> Act -> Log

    Returns a run summary dict.
    dry_run=True: routes decisions but does not execute or write to DB.
    """
    from agent.memory import RunState, was_recently_repriced
    from agent.executor import execute_decision
    from src.model_router import route_all_products
    from src.demand_forecast import get_all_forecasts
    from src.features import build_feature_matrix
    from db.models import get_db, Product

    run_id = str(uuid.uuid4())[:8]
    state  = RunState(run_id=run_id)

    log.info(f"\n{'='*60}")
    log.info(f"ARIA Agent Run  [{run_id}]  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}  |  "
             f"Auto-approve threshold: ±{settings.agent_auto_approve_max_pct}%")
    log.info(f"{'='*60}")

    # ── 1. SENSE — load demand forecasts ──────────────────────────────
    log.info("\n[1/4] Loading demand forecasts...")
    try:
        forecasts = get_all_forecasts()
        for cat, fc in forecasts.items():
            stale = " [STALE — retrain recommended]" if fc.get("is_stale") else ""
            log.info(
                f"  {cat:<12} {fc['trend_direction']:<8} "
                f"index={fc['current_index']:.0f}  "
                f"forecast={fc['forecast_avg']:.0f}{stale}"
            )
    except Exception as e:
        log.warning(f"  Demand forecasts unavailable: {e} — using neutral signals")
        forecasts = {}

    # ── 2. DECIDE — route all products ────────────────────────────────
    log.info("\n[2/4] Routing decisions...")
    try:
        decisions = route_all_products(forecasts)
    except Exception as e:
        log.error(f"  Routing failed: {e}")
        return {"status": "error", "error": str(e), "run_id": run_id}

    state.products_reviewed = len(decisions)
    log.info(f"  {len(decisions)} products routed")

    # ── 3. ACT — execute decisions ────────────────────────────────────
    log.info(f"\n[3/4] Executing decisions{' [DRY RUN — no changes]' if dry_run else ''}...")

    for decision in decisions:
        # Skip products repriced within cooldown window
        if not dry_run and was_recently_repriced(decision.product_id, cooldown_hours=6):
            log.debug(f"  {decision.product_name}: skipped (cooldown)")
            continue

        if dry_run:
            # In dry run mode — log what would happen but don't execute
            if decision.action != "hold":
                log.info(
                    f"  [DRY RUN] {decision.product_name[:32]:<32} "
                    f"{decision.action:<10} "
                    f"${decision.current_price:.2f} -> ${decision.recommended_price:.2f} "
                    f"({decision.change_pct:+.1f}%)  [{decision.layer}/{decision.confidence}]"
                )
            state.record({"status": "dry_run", "action": decision.action,
                          "product_id": decision.product_id})
        else:
            result = execute_decision(decision)
            state.record(result)

            if verbose and decision.action != "hold":
                log.info(f"    Rationale: {decision.rationale}")

    # ── 4. LOG — run summary ──────────────────────────────────────────
    state.finish()
    log.info(f"\n[4/4] Run complete")
    log.info(f"{'='*60}")
    log.info(f"  {state.summary()}")

    # Action breakdown
    if not dry_run:
        action_counts = {}
        for d in decisions:
            action_counts[d.action] = action_counts.get(d.action, 0) + 1
        log.info(f"  Actions: " + "  ".join(
            f"{a}={c}" for a, c in sorted(action_counts.items())
        ))

        # List pending approvals
        from agent.memory import get_pending_approvals
        pending = get_pending_approvals()
        if pending:
            log.info(f"\n  {len(pending)} pending human approval(s):")
            for p in pending:
                log.info(
                    f"    [{p['product_id']}] "
                    f"${p['current_price']:.2f} -> ${p['proposed_price']:.2f} "
                    f"({p['change_pct']:+.1f}%)"
                )

    log.info(f"{'='*60}\n")

    return {
        "status":           "ok",
        "run_id":           run_id,
        "products_reviewed":state.products_reviewed,
        "executed":         state.executed,
        "held":             state.held,
        "pending_approval": state.pending_approval,
        "errors":           state.errors,
        "elapsed_seconds":  state.elapsed_seconds,
        "dry_run":          dry_run,
    }


# ══════════════════════════════════════════════════════════════════════
# SCHEDULED LOOP
# ══════════════════════════════════════════════════════════════════════

def run_scheduled(interval_minutes: int = None):
    """
    Run the agent on a schedule indefinitely.
    Ctrl+C to stop.
    """
    interval = interval_minutes or settings.agent_schedule_minutes
    log.info(f"ARIA Agent starting — running every {interval} minutes")
    log.info("Press Ctrl+C to stop\n")

    run_count = 0
    while True:
        try:
            run_count += 1
            log.info(f"Starting run #{run_count}")
            result = run_once()
            if result.get("status") == "error":
                log.error(f"Run #{run_count} failed: {result.get('error')}")
        except KeyboardInterrupt:
            log.info("\nARIA Agent stopped by user")
            break
        except Exception as e:
            log.error(f"Unexpected error in run #{run_count}: {e}")

        log.info(f"Next run in {interval} minutes...")
        try:
            time.sleep(interval * 60)
        except KeyboardInterrupt:
            log.info("\nARIA Agent stopped by user")
            break


# ══════════════════════════════════════════════════════════════════════
# STATUS REPORT
# ══════════════════════════════════════════════════════════════════════

def print_status():
    """Print current DB state, recent decisions, and pending approvals."""
    from db.models import get_table_counts, get_db, AgentDecision, Product
    from agent.memory import get_pending_approvals

    print(f"\n{'='*60}")
    print("ARIA STATUS REPORT")
    print(f"{'='*60}")
    print(f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Demo mode       : {settings.demo_mode}")
    print(f"  Auto-approve    : changes <= {settings.agent_auto_approve_max_pct}%")
    print(f"  Schedule        : every {settings.agent_schedule_minutes} min")

    # DB counts
    print(f"\n  Database:")
    counts = get_table_counts()
    for table, count in counts.items():
        if count > 0:
            print(f"    {table:<30} {count:>6,} rows")

    # Recent decisions
    print(f"\n  Recent decisions (last 10):")
    with get_db() as db:
        recent = (
            db.query(AgentDecision)
            .order_by(AgentDecision.created_at.desc())
            .limit(10)
            .all()
        )
        if not recent:
            print("    None yet")
        for r in recent:
            exec_flag = "EXEC" if r.was_executed else "----"
            print(
                f"    [{exec_flag}] product={r.product_id}  "
                f"{r.decision_type:<10} "
                f"${float(r.current_price):.2f} -> ${float(r.recommended_price):.2f} "
                f"({r.change_pct:+.1f}%)  [{r.decision_source}/{r.confidence}]  "
                f"{r.created_at.strftime('%H:%M')}"
            )

    # Pending approvals
    pending = get_pending_approvals()
    print(f"\n  Pending approvals: {len(pending)}")
    for p in pending:
        print(
            f"    [{p['id']}] product={p['product_id']}  "
            f"${p['current_price']:.2f} -> ${p['proposed_price']:.2f} "
            f"({p['change_pct']:+.1f}%)  queued {p['created_at'][:16]}"
        )

    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ARIA Autonomous Repricing Agent")
    parser.add_argument("--once",    action="store_true", help="Single run then exit")
    parser.add_argument("--demo",    action="store_true", help="Single run, verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Route but do not execute")
    parser.add_argument("--status",  action="store_true", help="Show DB status and exit")
    parser.add_argument("--interval",type=int, default=None,
                        help="Override schedule interval (minutes)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    if args.demo:
        result = run_once(dry_run=True, verbose=True)
        print_status()
        return 0 if result.get("status") == "ok" else 1

    if args.once or args.dry_run:
        result = run_once(dry_run=args.dry_run, verbose=True)
        return 0 if result.get("status") == "ok" else 1

    # Default: scheduled loop
    run_scheduled(interval_minutes=args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())