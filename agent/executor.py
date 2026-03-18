"""
agent/executor.py
Price execution layer for ARIA.

Responsibilities:
  1. Validate a routing decision before acting
  2. Write price change to ecommerce platform (Shopify stub)
  3. Record every decision to audit log (agent_decisions table)
  4. Queue decisions requiring human approval
  5. Update price_history and products tables on success

Platform:
  Current: Shopify stub — logs the call, returns success
  Production: Replace _execute_shopify() with real Shopify Admin API
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("executor")

from db.models import get_db, Product, PriceHistory, AgentDecision, ApprovalQueue
from config.settings import get_settings
from src.model_router import RoutingDecision

settings = get_settings()


class ExecutionError(Exception):
    pass


def validate_decision(decision: RoutingDecision) -> tuple:
    """Pre-execution safety checks. Returns (ok, reason)."""
    if decision.action == "hold":
        return True, "hold"

    price = decision.recommended_price
    if price <= 0:
        return False, f"Invalid price ${price:.2f}"
    if price < decision.min_price * 0.99:
        return False, f"${price:.2f} below floor ${decision.min_price:.2f}"
    if price > decision.max_price * 1.01:
        return False, f"${price:.2f} above ceiling ${decision.max_price:.2f}"
    if abs(decision.change_pct) > 50:
        return False, f"Change {decision.change_pct:+.1f}% exceeds 50% safety limit"
    return True, "ok"


def _execute_shopify(product_id: int, new_price: float, product_name: str) -> dict:
    """
    Shopify price update.

    CURRENT: Stub — logs and returns success.
    TO ACTIVATE: Replace with Shopify Admin API PUT request.

    import requests
    url = f"https://{settings.shopify_store_url}/admin/api/2024-01/variants/{variant_id}.json"
    r = requests.put(url, json={"variant": {"price": str(new_price)}},
                     headers={"X-Shopify-Access-Token": settings.shopify_access_token},
                     timeout=10)
    r.raise_for_status()
    """
    log.info(f"  [SHOPIFY STUB] product_id={product_id} ({product_name}) -> ${new_price:.2f}")
    return {"success": True, "platform": "shopify_stub", "new_price": new_price}


def _execute_platform(product: dict, new_price: float) -> dict:
    platform = product.get("platform", "shopify").lower()
    if platform in ("shopify", "shopify_stub"):
        return _execute_shopify(product["id"], new_price, product["name"])
    raise ExecutionError(f"Unsupported platform: {platform}")


def _record_decision(decision, was_executed, requires_approval,
                     approval_status, execution_error=None) -> int:
    with get_db() as db:
        rec = AgentDecision(
            product_id=decision.product_id,
            decision_type=decision.action,
            decision_source=decision.layer,
            current_price=decision.current_price,
            recommended_price=decision.recommended_price,
            change_pct=decision.change_pct,
            competitor_median=decision.comp_price_med,
            trend_index=int(decision.trend_index),
            confidence=decision.confidence,
            reasoning=decision.rationale,
            was_executed=was_executed,
            executed_at=datetime.utcnow() if was_executed else None,
            execution_error=execution_error,
            requires_approval=requires_approval,
            approval_status=approval_status,
            created_at=datetime.utcnow(),
        )
        db.add(rec)
        db.flush()
        return rec.id


def _record_price_change(product_id, old_price, new_price, decision_id):
    change_pct = round((new_price - old_price) / old_price * 100, 2) if old_price > 0 else 0.0
    with get_db() as db:
        db.add(PriceHistory(
            product_id=product_id, old_price=old_price, new_price=new_price,
            change_pct=change_pct, source="agent",
            decision_id=decision_id, recorded_at=datetime.utcnow(),
        ))


def _update_product_price(product_id, new_price):
    with get_db() as db:
        p = db.query(Product).filter(Product.id == product_id).first()
        if p:
            p.current_price = new_price
            p.updated_at    = datetime.utcnow()


def _queue_for_approval(decision, decision_id):
    expires_at = datetime.utcnow() + timedelta(hours=24)
    with get_db() as db:
        db.add(ApprovalQueue(
            decision_id=decision_id, product_id=decision.product_id,
            current_price=decision.current_price,
            proposed_price=decision.recommended_price,
            change_pct=decision.change_pct,
            reasoning=decision.rationale, status="pending",
            expires_at=expires_at, created_at=datetime.utcnow(),
        ))
    log.info(
        f"  APPROVAL QUEUE: {decision.product_name} "
        f"{decision.change_pct:+.1f}% -> ${decision.recommended_price:.2f} "
        f"(expires {expires_at.strftime('%H:%M %d-%b')})"
    )


def execute_decision(decision: RoutingDecision) -> dict:
    """
    Execute a single routing decision end-to-end.

    hold/review  -> record to audit log only
    change       -> validate -> approval check -> platform -> record -> update DB
    """
    # No-ops
    if decision.action in ("hold", "review"):
        did = _record_decision(decision, False, False, None)
        return {
            "status":      "held" if decision.action == "hold" else "review_queued",
            "decision_id": did,
            "product_id":  decision.product_id,
            "action":      decision.action,
            "price":       decision.current_price,
        }

    # Validate
    valid, reason = validate_decision(decision)
    if not valid:
        log.warning(f"  {decision.product_name}: validation failed — {reason}")
        did = _record_decision(decision, False, False, None,
                               execution_error=f"validation_failed: {reason}")
        return {"status": "validation_failed", "decision_id": did,
                "reason": reason, "product_id": decision.product_id}

    # Human approval required
    if decision.requires_approval:
        did = _record_decision(decision, False, True, "pending")
        _queue_for_approval(decision, did)
        return {"status": "pending_approval", "decision_id": did,
                "product_id": decision.product_id,
                "change_pct": decision.change_pct,
                "proposed":   decision.recommended_price}

    # Execute
    try:
        with get_db() as db:
            p = db.query(Product).filter(Product.id == decision.product_id).first()
            if not p:
                raise ExecutionError(f"Product {decision.product_id} not found")
            product_dict = {"id": p.id, "name": p.name, "platform": p.platform or "shopify"}

        result = _execute_platform(product_dict, decision.recommended_price)
        if not result.get("success"):
            raise ExecutionError(f"Platform returned failure: {result}")

        did = _record_decision(decision, True, False, None)
        _record_price_change(decision.product_id, decision.current_price,
                             decision.recommended_price, did)
        _update_product_price(decision.product_id, decision.recommended_price)

        log.info(
            f"  EXECUTED: {decision.product_name} "
            f"${decision.current_price:.2f} -> ${decision.recommended_price:.2f} "
            f"({decision.change_pct:+.1f}%)  [id={did}]"
        )
        return {"status": "executed", "decision_id": did,
                "product_id": decision.product_id,
                "old_price":  decision.current_price,
                "new_price":  decision.recommended_price,
                "change_pct": decision.change_pct}

    except Exception as e:
        log.error(f"  Execution failed {decision.product_name}: {e}")
        did = _record_decision(decision, False, False, None, execution_error=str(e))
        return {"status": "error", "decision_id": did,
                "error": str(e), "product_id": decision.product_id}