"""
db/models.py
SQLAlchemy ORM models for ARIA.

Tables:
  products           — product catalog with current pricing
  price_history      — every price change ever made (audit trail)
  competitor_prices  — scraped / simulated competitor prices
  demand_signals     — Google Trends data per keyword
  agent_decisions    — every decision the agent made (audit log)
  repricing_outcomes — did a reprice improve sell-through? (feedback loop)
  approval_queue     — price changes awaiting human review

Usage:
    python db/models.py --init      # Create all tables
    python db/models.py --status    # Show table row counts
"""
import os
import sys
import logging
import argparse
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Boolean, Text, ForeignKey, Index,
    Numeric, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import StaticPool
import enum

log = logging.getLogger("db.models")

# ── Engine setup ──────────────────────────────────────────────────────
def _get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./aria.db")


def _create_engine():
    url = _get_database_url()
    if url.startswith("sqlite"):
        # SQLite — for local dev/testing only
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    # PostgreSQL (production default)
    # pool_pre_ping: test connections before use (handles dropped connections)
    # pool_size: keep 5 connections warm
    # max_overflow: allow up to 10 extra connections under load
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600,  # Recycle connections every hour
    )


engine = _create_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def get_db():
    """
    Context manager for DB sessions.
    Automatically commits on success, rolls back on error.

    Usage:
        with get_db() as db:
            products = db.query(Product).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Enums ─────────────────────────────────────────────────────────────
class DecisionType(str, enum.Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"
    REVIEW = "review"          # Escalated to human


class DecisionSource(str, enum.Enum):
    RULES = "rules"            # Determined by rule engine (no ML/LLM)
    ML_MODEL = "ml_model"      # Determined by XGBoost
    LLM = "llm"                # Determined by LLM reasoning
    HUMAN = "human"            # Manual override


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# ── Models ────────────────────────────────────────────────────────────
class Product(Base):
    """
    Core product catalog.
    Each product has a current price, inventory level, and category.
    """
    __tablename__ = "products"

    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String(200), nullable=False)
    sku            = Column(String(100), unique=True, nullable=True)
    category       = Column(String(100), nullable=False, index=True)
    base_price     = Column(Numeric(10, 2), nullable=False)
    current_price  = Column(Numeric(10, 2), nullable=False)
    min_price      = Column(Numeric(10, 2), nullable=True)   # Floor — never go below
    max_price      = Column(Numeric(10, 2), nullable=True)   # Ceiling — never go above
    inventory_qty  = Column(Integer, default=100)
    is_active      = Column(Boolean, default=True)
    platform       = Column(String(50), default="shopify")   # ecommerce platform
    external_id    = Column(String(200), nullable=True)      # platform product ID
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    price_history      = relationship("PriceHistory",      back_populates="product")
    competitor_prices  = relationship("CompetitorPrice",   back_populates="product")
    agent_decisions    = relationship("AgentDecision",     back_populates="product")
    repricing_outcomes = relationship("RepricingOutcome",  back_populates="product")

    def __repr__(self):
        return f"<Product id={self.id} name={self.name!r} price=${self.current_price}>"


class PriceHistory(Base):
    """
    Every price change ever made — the complete audit trail.
    Both autonomous agent changes and manual changes are recorded here.
    This table is the feedback loop data source.
    """
    __tablename__ = "price_history"

    id             = Column(Integer, primary_key=True, index=True)
    product_id     = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    old_price      = Column(Numeric(10, 2), nullable=False)
    new_price      = Column(Numeric(10, 2), nullable=False)
    change_pct     = Column(Float, nullable=False)           # signed %
    source         = Column(String(50), nullable=False)      # "agent" | "manual" | "import"
    decision_id    = Column(Integer, ForeignKey("agent_decisions.id"), nullable=True)
    recorded_at    = Column(DateTime, default=datetime.utcnow, index=True)
    note           = Column(Text, nullable=True)

    product        = relationship("Product", back_populates="price_history")

    __table_args__ = (
        Index("ix_price_history_product_date", "product_id", "recorded_at"),
    )


class CompetitorPrice(Base):
    """
    Competitor prices scraped from SerpAPI (Google Shopping) or generated
    by the price simulator in DEMO_MODE.
    Each scrape run adds fresh rows — historical snapshots are preserved.
    """
    __tablename__ = "competitor_prices"

    id                = Column(Integer, primary_key=True, index=True)
    product_id        = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    platform          = Column(String(100), nullable=False)  # "google_shopping" | "simulated"
    retailer          = Column(String(200), nullable=True)   # e.g. "Amazon", "BestBuy"
    competitor_price  = Column(Numeric(10, 2), nullable=False)
    listing_title     = Column(String(500), nullable=True)
    listing_url       = Column(Text, nullable=True)
    is_simulated      = Column(Boolean, default=False)
    scraped_at        = Column(DateTime, default=datetime.utcnow, index=True)

    product = relationship("Product", back_populates="competitor_prices")

    __table_args__ = (
        Index("ix_competitor_product_date", "product_id", "scraped_at"),
    )


class DemandSignal(Base):
    """
    Google Trends demand index per keyword per week.
    Range: 0-100 (100 = peak popularity in timeframe)
    """
    __tablename__ = "demand_signals"

    id           = Column(Integer, primary_key=True, index=True)
    keyword      = Column(String(200), nullable=False, index=True)
    trend_index  = Column(Integer, nullable=False)
    week_date    = Column(DateTime, nullable=False)
    region       = Column(String(10), default="US")
    created_at   = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_demand_keyword_date", "keyword", "week_date"),
    )


class AgentDecision(Base):
    """
    Every decision the agent makes — the complete reasoning audit log.

    This is critical for:
    - Debugging why a price changed
    - Building the feedback loop training dataset
    - Human review and override
    - Regulatory compliance
    """
    __tablename__ = "agent_decisions"

    id                 = Column(Integer, primary_key=True, index=True)
    product_id         = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)

    # Decision
    decision_type      = Column(String(20), nullable=False)   # increase/decrease/hold/review
    decision_source    = Column(String(20), nullable=False)   # rules/ml_model/llm/human
    current_price      = Column(Numeric(10, 2), nullable=False)
    recommended_price  = Column(Numeric(10, 2), nullable=False)
    change_pct         = Column(Float, nullable=False)

    # Context at decision time
    competitor_median  = Column(Numeric(10, 2), nullable=True)
    trend_index        = Column(Integer, nullable=True)
    inventory_qty      = Column(Integer, nullable=True)
    confidence         = Column(String(20), nullable=True)    # high/medium/low

    # Reasoning (LLM output or rule description)
    reasoning          = Column(Text, nullable=True)

    # Execution status
    was_executed       = Column(Boolean, default=False)
    executed_at        = Column(DateTime, nullable=True)
    execution_error    = Column(Text, nullable=True)

    # Approval
    requires_approval  = Column(Boolean, default=False)
    approval_status    = Column(String(20), nullable=True)

    created_at         = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    product  = relationship("Product",    back_populates="agent_decisions")
    outcome  = relationship("RepricingOutcome", back_populates="decision", uselist=False)

    __table_args__ = (
        Index("ix_decisions_product_date", "product_id", "created_at"),
    )


class RepricingOutcome(Base):
    """
    Did a repricing decision actually improve business metrics?
    Tracked 24h, 48h, and 7 days after each price change.

    This is the feedback loop data — used to retrain the model
    and evaluate agent decision quality over time.

    Key metrics:
    - sell_through_rate_before / after: units sold per day
    - revenue_before / after: revenue per day
    - margin_before / after: gross margin %
    """
    __tablename__ = "repricing_outcomes"

    id                        = Column(Integer, primary_key=True, index=True)
    product_id                = Column(Integer, ForeignKey("products.id"), nullable=False)
    decision_id               = Column(Integer, ForeignKey("agent_decisions.id"), nullable=False)

    # Prices
    price_before              = Column(Numeric(10, 2), nullable=False)
    price_after               = Column(Numeric(10, 2), nullable=False)

    # Outcome metrics (collected after 7 days)
    units_sold_before_7d      = Column(Float, nullable=True)   # avg units/day before
    units_sold_after_7d       = Column(Float, nullable=True)   # avg units/day after
    revenue_before_7d         = Column(Float, nullable=True)
    revenue_after_7d          = Column(Float, nullable=True)

    # Outcome label for ML training
    # 1 = positive outcome (revenue or sell-through improved)
    # 0 = neutral
    # -1 = negative outcome (both metrics worsened)
    outcome_label             = Column(Integer, nullable=True)
    outcome_notes             = Column(Text, nullable=True)

    measured_at               = Column(DateTime, nullable=True)
    created_at                = Column(DateTime, default=datetime.utcnow)

    product  = relationship("Product",       back_populates="repricing_outcomes")
    decision = relationship("AgentDecision", back_populates="outcome")


class ApprovalQueue(Base):
    """
    Price changes that exceed the auto-approve threshold.
    Sits in a queue until a human approves or rejects.

    ARIA sends an alert (email/Slack/webhook) when items enter this queue.
    """
    __tablename__ = "approval_queue"

    id              = Column(Integer, primary_key=True, index=True)
    decision_id     = Column(Integer, ForeignKey("agent_decisions.id"), nullable=False)
    product_id      = Column(Integer, ForeignKey("products.id"), nullable=False)

    current_price   = Column(Numeric(10, 2), nullable=False)
    proposed_price  = Column(Numeric(10, 2), nullable=False)
    change_pct      = Column(Float, nullable=False)
    reasoning       = Column(Text, nullable=True)

    status          = Column(String(20), default="pending")  # pending/approved/rejected
    reviewed_by     = Column(String(200), nullable=True)
    reviewed_at     = Column(DateTime, nullable=True)
    review_note     = Column(Text, nullable=True)

    expires_at      = Column(DateTime, nullable=True)        # Auto-expire if not reviewed
    created_at      = Column(DateTime, default=datetime.utcnow, index=True)


# ── Upsert helpers ────────────────────────────────────────────────────
def upsert_demand_signal(db, keyword: str, trend_index: int, week_date, region: str = "US"):
    """
    Upsert a demand signal row.
    If a row for (keyword, week_date) already exists, update trend_index.
    Otherwise insert.
    """
    from sqlalchemy import and_
    week_dt = datetime.combine(week_date, datetime.min.time()) if hasattr(week_date, 'year') and not isinstance(week_date, datetime) else week_date

    existing = db.query(DemandSignal).filter(
        and_(
            DemandSignal.keyword == keyword,
            DemandSignal.week_date == week_dt,
            DemandSignal.region == region,
        )
    ).first()

    if existing:
        existing.trend_index = trend_index
    else:
        db.add(DemandSignal(
            keyword=keyword,
            trend_index=trend_index,
            week_date=week_dt,
            region=region,
        ))


def get_latest_competitor_prices(db, product_id: int, hours: int = 24) -> list:
    """
    Returns competitor prices scraped within the last N hours.
    If none found, returns all available (no time filter).
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(CompetitorPrice)
        .filter(
            CompetitorPrice.product_id == product_id,
            CompetitorPrice.scraped_at >= cutoff,
        )
        .all()
    )
    if not rows:
        # Fall back to all available prices
        rows = (
            db.query(CompetitorPrice)
            .filter(CompetitorPrice.product_id == product_id)
            .order_by(CompetitorPrice.scraped_at.desc())
            .limit(50)
            .all()
        )
    return rows


# ── Database initialization ───────────────────────────────────────────
def init_db():
    """Creates all tables. Safe to run multiple times."""
    Base.metadata.create_all(bind=engine)
    log.info("Database tables created successfully")


def get_table_counts() -> dict:
    """Returns row count for each table — useful for health checks."""
    counts = {}
    with get_db() as db:
        for model in [Product, PriceHistory, CompetitorPrice,
                      DemandSignal, AgentDecision, RepricingOutcome, ApprovalQueue]:
            try:
                counts[model.__tablename__] = db.query(model).count()
            except Exception:
                counts[model.__tablename__] = -1
    return counts


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="ARIA database management")
    parser.add_argument("--init",   action="store_true", help="Create all tables")
    parser.add_argument("--status", action="store_true", help="Show table row counts")
    parser.add_argument("--drop",   action="store_true", help="Drop all tables (DESTRUCTIVE)")
    args = parser.parse_args()

    if args.drop:
        confirm = input("⚠️  This will DELETE all data. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            Base.metadata.drop_all(bind=engine)
            log.info("All tables dropped")
        else:
            log.info("Cancelled")

    if args.init or not any([args.drop, args.status]):
        init_db()

    if args.status:
        counts = get_table_counts()
        print(f"\n{'Table':<30} {'Rows':>8}")
        print("-" * 40)
        for table, count in counts.items():
            print(f"  {table:<28} {count:>8,}")
        print()