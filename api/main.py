"""
api/main.py — ARIA FastAPI REST API

Swagger UI at /docs  |  ReDoc at /redoc

Endpoints:
  GET  /                        Health check + system info
  GET  /products                List all products
  GET  /products/{id}           Single product
  POST /agent/run               Trigger agent cycle (dry-run or live)
  GET  /agent/decisions         Audit log
  GET  /agent/approvals         Pending human approvals
  POST /agent/approve/{id}      Approve a price change
  POST /agent/reject/{id}       Reject a price change
  GET  /metrics                 Business + model metrics
  GET  /metrics/alerts          Run alert checks
  GET  /recommendations         ML price recommendations (all products)
  GET  /recommendations/{id}    ML recommendation (one product)
  POST /simulate                Simulate competitor prices
  GET  /forecasts               Demand forecasts (all categories)
  GET  /forecasts/{category}    Demand forecast (one category)
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_settings
settings = get_settings()

app = FastAPI(
    title="ARIA — Autonomous Repricing & Inventory Agent",
    description="""
## ARIA API

A production AI agentic system for autonomous ecommerce repricing.

**Three-layer decision architecture:**
1. **Rules engine** — ~60% of decisions (free, instant)
2. **XGBoost ML model** — ~30% (fast, no API cost)
3. **LLM reasoning** — ~10% edge cases only

**Key demo endpoints:**
- `POST /agent/run` — trigger a full repricing cycle
- `GET /recommendations` — current ML price recommendations
- `GET /metrics` — business and model health
- `POST /simulate` — see simulated competitor prices for any product
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────

class ProductOut(BaseModel):
    id: int
    name: str
    sku: Optional[str] = None
    category: str
    current_price: float
    base_price: float
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    inventory_qty: int
    is_active: bool
    platform: str


class DecisionOut(BaseModel):
    id: int
    product_id: int
    product_name: Optional[str] = None
    decision_type: str
    decision_source: str
    current_price: float
    recommended_price: float
    change_pct: float
    confidence: Optional[str] = None
    reasoning: Optional[str] = None
    was_executed: bool
    requires_approval: bool
    approval_status: Optional[str] = None
    created_at: str


class ApprovalOut(BaseModel):
    id: int
    decision_id: int
    product_id: int
    product_name: Optional[str] = None
    current_price: float
    proposed_price: float
    change_pct: float
    reasoning: Optional[str] = None
    status: str
    created_at: str
    expires_at: Optional[str] = None


class RunRequest(BaseModel):
    dry_run: bool = True
    model_config = {"json_schema_extra": {"example": {"dry_run": True}}}


class RunResponse(BaseModel):
    status: str
    run_id: str
    products_reviewed: int
    executed: int
    held: int
    pending_approval: int
    errors: int
    elapsed_seconds: float
    dry_run: bool


class SimulateRequest(BaseModel):
    product_name: str
    category: str
    our_price: float
    model_config = {"json_schema_extra": {
        "example": {"product_name": "Wireless Headphones",
                    "category": "electronics", "our_price": 89.99}
    }}


class ReviewRequest(BaseModel):
    reviewed_by: str = "api_user"
    note: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────

@app.get("/", tags=["System"], summary="Health check")
def root():
    """Health check. Returns system status and DB table counts."""
    try:
        from db.models import get_table_counts
        counts = get_table_counts()
        db_ok  = True
    except Exception as e:
        counts = {}
        db_ok  = False
    return {
        "status":       "ok",
        "service":      "ARIA Autonomous Repricing Agent",
        "version":      "1.0.0",
        "demo_mode":    settings.demo_mode,
        "llm_provider": settings.llm_provider,
        "db_connected": db_ok,
        "table_counts": counts,
        "timestamp":    datetime.utcnow().isoformat(),
    }


# ── Products ──────────────────────────────────────────────────────────

@app.get("/products", tags=["Products"], response_model=list[ProductOut], summary="List all products")
def list_products(
    category: Optional[str] = Query(None, description="electronics | fashion | home_goods | sports"),
    active_only: bool = Query(True),
):
    """Returns all products with current prices and inventory levels."""
    from db.models import get_db, Product
    with get_db() as db:
        q = db.query(Product)
        if active_only:
            q = q.filter(Product.is_active == True)
        if category:
            q = q.filter(Product.category == category)
        rows = q.order_by(Product.category, Product.name).all()
        return [_product_out(p) for p in rows]


@app.get("/products/{product_id}", tags=["Products"], response_model=ProductOut)
def get_product(product_id: int):
    """Returns full detail for a single product."""
    from db.models import get_db, Product
    with get_db() as db:
        p = db.query(Product).filter(Product.id == product_id).first()
        if not p:
            raise HTTPException(404, f"Product {product_id} not found")
        return _product_out(p)


def _product_out(p) -> ProductOut:
    return ProductOut(
        id=p.id, name=p.name, sku=p.sku, category=p.category,
        current_price=float(p.current_price), base_price=float(p.base_price),
        min_price=float(p.min_price) if p.min_price else None,
        max_price=float(p.max_price) if p.max_price else None,
        inventory_qty=p.inventory_qty or 0,
        is_active=bool(p.is_active), platform=p.platform or "shopify",
    )


# ── Agent ─────────────────────────────────────────────────────────────

@app.post("/agent/run", tags=["Agent"], response_model=RunResponse, summary="Trigger one agent cycle")
def run_agent(request: RunRequest):
    """
    Triggers one full ARIA repricing cycle.

    **dry_run=true** (default): routes decisions and shows what would happen — no changes made.
    **dry_run=false**: executes real price changes and queues approvals for large changes.
    """
    try:
        from agent.aria import run_once
        result = run_once(dry_run=request.dry_run, verbose=False)
        if result.get("status") == "error":
            raise HTTPException(500, result.get("error"))
        return RunResponse(
            status=result.get("status", "ok"),
            run_id=result.get("run_id", ""),
            products_reviewed=result.get("products_reviewed", 0),
            executed=result.get("executed", 0),
            held=result.get("held", 0),
            pending_approval=result.get("pending_approval", 0),
            errors=result.get("errors", 0),
            elapsed_seconds=result.get("elapsed_seconds", 0.0),
            dry_run=result.get("dry_run", True),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/agent/decisions", tags=["Agent"], response_model=list[DecisionOut], summary="Agent audit log")
def get_decisions(
    limit: int = Query(20, ge=1, le=100),
    executed_only: bool = Query(False),
):
    """Returns recent agent decisions. Every decision is logged — holds, executions, and errors."""
    from db.models import get_db, AgentDecision, Product
    with get_db() as db:
        q = db.query(AgentDecision)
        if executed_only:
            q = q.filter(AgentDecision.was_executed == True)
        rows = q.order_by(AgentDecision.created_at.desc()).limit(limit).all()
        names = {p.id: p.name for p in db.query(Product).all()}
        return [DecisionOut(
            id=d.id, product_id=d.product_id,
            product_name=names.get(d.product_id),
            decision_type=d.decision_type,
            decision_source=d.decision_source,
            current_price=float(d.current_price),
            recommended_price=float(d.recommended_price),
            change_pct=d.change_pct, confidence=d.confidence,
            reasoning=d.reasoning,
            was_executed=bool(d.was_executed),
            requires_approval=bool(d.requires_approval),
            approval_status=d.approval_status,
            created_at=d.created_at.isoformat(),
        ) for d in rows]


@app.get("/agent/approvals", tags=["Agent"], response_model=list[ApprovalOut], summary="Pending human approvals")
def get_approvals():
    """Price changes exceeding ±{pct}% auto-approve threshold. Approve or reject below.""".format(
        pct=settings.agent_auto_approve_max_pct)
    from db.models import get_db, ApprovalQueue, Product
    with get_db() as db:
        rows = db.query(ApprovalQueue)\
            .filter(ApprovalQueue.status == "pending")\
            .order_by(ApprovalQueue.created_at.desc()).all()
        names = {p.id: p.name for p in db.query(Product).all()}
        return [ApprovalOut(
            id=r.id, decision_id=r.decision_id, product_id=r.product_id,
            product_name=names.get(r.product_id),
            current_price=float(r.current_price),
            proposed_price=float(r.proposed_price),
            change_pct=r.change_pct, reasoning=r.reasoning,
            status=r.status, created_at=r.created_at.isoformat(),
            expires_at=r.expires_at.isoformat() if r.expires_at else None,
        ) for r in rows]


@app.post("/agent/approve/{approval_id}", tags=["Agent"], summary="Approve a price change")
def approve(approval_id: int, req: ReviewRequest):
    """Approves a queued price change and executes it immediately."""
    from db.models import get_db, ApprovalQueue, AgentDecision
    with get_db() as db:
        row = db.query(ApprovalQueue).filter(ApprovalQueue.id == approval_id).first()
        if not row:
            raise HTTPException(404, "Approval not found")
        if row.status != "pending":
            raise HTTPException(400, f"Already {row.status}")
        row.status = "approved"
        row.reviewed_by = req.reviewed_by
        row.reviewed_at = datetime.utcnow()
        row.review_note = req.note
        pid, new_p, old_p, did = (
            row.product_id, float(row.proposed_price),
            float(row.current_price), row.decision_id
        )
    try:
        from agent.executor import _update_product_price, _record_price_change
        _update_product_price(pid, new_p)
        _record_price_change(pid, old_p, new_p, did)
    except Exception as e:
        raise HTTPException(500, f"Execution failed: {e}")
    return {"status": "approved_and_executed", "approval_id": approval_id,
            "new_price": new_p, "reviewed_by": req.reviewed_by}


@app.post("/agent/reject/{approval_id}", tags=["Agent"], summary="Reject a price change")
def reject(approval_id: int, req: ReviewRequest):
    """Rejects a queued price change. Current price unchanged."""
    from db.models import get_db, ApprovalQueue
    with get_db() as db:
        row = db.query(ApprovalQueue).filter(ApprovalQueue.id == approval_id).first()
        if not row:
            raise HTTPException(404, "Approval not found")
        if row.status != "pending":
            raise HTTPException(400, f"Already {row.status}")
        row.status = "rejected"
        row.reviewed_by = req.reviewed_by
        row.reviewed_at = datetime.utcnow()
        row.review_note = req.note
    return {"status": "rejected", "approval_id": approval_id,
            "reviewed_by": req.reviewed_by}


# ── Metrics ───────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"], summary="Business + model metrics")
def get_metrics(days: int = Query(7, ge=1, le=30)):
    """Business metrics, model health, and data freshness for the last N days."""
    try:
        from monitoring.metrics import get_full_report
        return get_full_report(days=days)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/metrics/alerts", tags=["Monitoring"], summary="Run alert checks")
def get_alerts():
    """Runs all 5 alert checks. Returns any that fired."""
    try:
        from monitoring.alerts import run_all_checks
        alerts = run_all_checks()
        return {"total_alerts": len(alerts), "alerts": alerts,
                "checked_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── ML ────────────────────────────────────────────────────────────────

@app.get("/recommendations", tags=["ML"], summary="Price recommendations (all products)")
def get_recommendations():
    """XGBoost price recommendations for every product with rationale."""
    try:
        from src.pricing_model import get_all_recommendations
        df = get_all_recommendations()
        return [] if df.empty else df.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not trained. Run: python src/pricing_model.py")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/recommendations/{product_id}", tags=["ML"], summary="Price recommendation (one product)")
def get_recommendation(product_id: int):
    """XGBoost price recommendation for a single product."""
    try:
        from src.pricing_model import get_price_recommendation
        return get_price_recommendation(product_id=product_id)
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained. Run: python src/pricing_model.py")
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/simulate", tags=["ML"], summary="Simulate competitor prices")
def simulate(request: SimulateRequest):
    """
    Generates realistic simulated competitor prices using category-specific
    market distributions. Demonstrates the data layer without spending API credits.
    """
    try:
        from src.price_simulator import simulate_prices
        band = simulate_prices(request.product_name, request.category, request.our_price)
        band.pop("listings", None)
        band["product_name"]    = request.product_name
        band["our_price"]       = request.our_price
        med = band.get("median", 0)
        band["position_pct"]    = round((request.our_price - med) / med * 100, 1) if med > 0 else 0
        band["position_label"]  = (
            "ABOVE market" if band["position_pct"] > 10
            else "BELOW market" if band["position_pct"] < -10
            else "Competitive"
        )
        return band
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/forecasts", tags=["ML"], summary="Demand forecasts (all categories)")
def get_forecasts():
    """30-day Prophet demand forecasts for all 4 categories."""
    try:
        from src.demand_forecast import get_all_forecasts
        forecasts = get_all_forecasts()
        return {cat: {k: v for k, v in fc.items() if k != "forecast_df"}
                for cat, fc in forecasts.items()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/forecasts/{category}", tags=["ML"],
         summary="Demand forecast (one category)")
def get_forecast(category: str):
    """30-day Prophet demand forecast for one category."""
    try:
        from src.demand_forecast import get_demand_forecast
        fc = get_demand_forecast(category)
        return {k: v for k, v in fc.items() if k != "forecast_df"}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))