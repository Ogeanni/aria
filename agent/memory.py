"""
agent/memory.py
Run state and decision memory for ARIA.

Tracks what happened in the current and previous agent runs.
Used to:
  - Avoid repricing the same product twice in one run
  - Detect when a product was recently repriced (cooldown)
  - Build the run summary for logging and monitoring
  - Feed the feedback loop (did recent reprice work?)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger("memory")

from db.models import get_db, AgentDecision


@dataclass
class RunState:
    """
    Tracks the state of one agent run (one scheduling cycle).
    Created fresh each run, does not persist across runs.
    """
    run_id:       str = ""
    started_at:   datetime = field(default_factory=datetime.utcnow)
    finished_at:  Optional[datetime] = None

    # Counters
    products_reviewed: int = 0
    decisions_made:    int = 0
    executed:          int = 0
    held:              int = 0
    pending_approval:  int = 0
    errors:            int = 0

    # Decision log for this run
    decisions: list = field(default_factory=list)

    def record(self, result: dict):
        """Record one execution result into run state."""
        self.decisions_made += 1
        status = result.get("status", "unknown")
        if status == "executed":
            self.executed += 1
        elif status in ("held", "hold"):
            self.held += 1
        elif status == "pending_approval":
            self.pending_approval += 1
        elif status == "error":
            self.errors += 1
        self.decisions.append(result)

    def finish(self):
        self.finished_at = datetime.utcnow()

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    def summary(self) -> str:
        return (
            f"Run {self.run_id} | "
            f"{self.products_reviewed} reviewed | "
            f"{self.executed} executed | "
            f"{self.held} held | "
            f"{self.pending_approval} pending approval | "
            f"{self.errors} errors | "
            f"{self.elapsed_seconds:.1f}s"
        )


def was_recently_repriced(product_id: int, cooldown_hours: int = 6) -> bool:
    """
    Returns True if the product was repriced within the cooldown window.

    Prevents the agent from flip-flopping on a price that was just changed.
    Default cooldown: 6 hours — agent runs hourly but won't re-execute
    a price change within 6 hours of the last one.
    """
    cutoff = datetime.utcnow() - timedelta(hours=cooldown_hours)
    with get_db() as db:
        recent = (
            db.query(AgentDecision)
            .filter(
                AgentDecision.product_id == product_id,
                AgentDecision.was_executed == True,
                AgentDecision.created_at >= cutoff,
            )
            .first()
        )
        return recent is not None


def get_recent_decisions(product_id: int, limit: int = 5) -> list:
    """
    Returns the last N agent decisions for a product.
    Used by the feedback loop to check if recent repricings improved outcomes.
    """
    with get_db() as db:
        rows = (
            db.query(AgentDecision)
            .filter(AgentDecision.product_id == product_id)
            .order_by(AgentDecision.created_at.desc())
            .limit(limit)
            .all()
        )
        return [{
            "id":                row.id,
            "decision_type":     row.decision_type,
            "decision_source":   row.decision_source,
            "current_price":     float(row.current_price),
            "recommended_price": float(row.recommended_price),
            "change_pct":        row.change_pct,
            "was_executed":      row.was_executed,
            "confidence":        row.confidence,
            "created_at":        row.created_at.isoformat(),
        } for row in rows]


def get_pending_approvals() -> list:
    """Returns all decisions currently pending human approval."""
    from db.models import ApprovalQueue, Product
    with get_db() as db:
        rows = (
            db.query(ApprovalQueue)
            .filter(ApprovalQueue.status == "pending")
            .order_by(ApprovalQueue.created_at.desc())
            .all()
        )
        return [{
            "id":             row.id,
            "decision_id":    row.decision_id,
            "product_id":     row.product_id,
            "current_price":  float(row.current_price),
            "proposed_price": float(row.proposed_price),
            "change_pct":     row.change_pct,
            "reasoning":      row.reasoning,
            "status":         row.status,
            "created_at":     row.created_at.isoformat(),
            "expires_at":     row.expires_at.isoformat() if row.expires_at else None,
        } for row in rows]