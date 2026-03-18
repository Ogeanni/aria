"""
tests/conftest.py
Shared fixtures for the ARIA test suite.

Uses SQLite in-memory database for tests — no real Postgres needed.
Every test gets a fresh database — no state leaks between tests.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force test environment before any imports
os.environ.setdefault("DATABASE_URL",  "sqlite:///:memory:")
os.environ.setdefault("DEMO_MODE",     "true")
os.environ.setdefault("LOG_LEVEL",     "WARNING")
os.environ.setdefault("REDIS_URL",     "")
os.environ.setdefault("SERPAPI_KEY",   "")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base


# ── In-memory SQLite engine ───────────────────────────────────────────

@pytest.fixture(scope="session")
def engine():
    """Single engine for the entire test session."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture(scope="function")
def db_session(engine):
    """
    Fresh database session per test.
    Wraps each test in a transaction and rolls back after —
    so tests are fully isolated without recreating tables.
    """
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ── Sample product fixture ────────────────────────────────────────────

@pytest.fixture
def sample_product(db_session):
    """A realistic product row for testing decisions and routing."""
    from db.models import Product
    p = Product(
        name="Test Wireless Headphones",
        sku="TEST-001",
        category="electronics",
        base_price=79.99,
        current_price=94.99,
        min_price=65.00,
        max_price=140.00,
        inventory_qty=85,
        is_active=True,
        platform="shopify",
    )
    db_session.add(p)
    db_session.flush()
    return p


@pytest.fixture
def sample_demand():
    """A realistic demand forecast dict for routing tests."""
    return {
        "category":        "electronics",
        "keyword":         "wireless headphones",
        "current_index":   72.0,
        "forecast_avg":    78.0,
        "forecast_high":   85.0,
        "forecast_low":    65.0,
        "trend_direction": "rising",
        "confidence_low":  60.0,
        "confidence_high": 90.0,
        "model_age_days":  2,
        "is_stale":        False,
        "demand_signal":   "Demand rising for electronics over next 30 days.",
    }


@pytest.fixture
def sample_feature_row():
    """
    A realistic feature row as a dict — simulates what features.py produces.
    Used for routing and pricing model tests without needing a real DB.
    """
    return {
        "product_id":          1,
        "product_name":        "Test Wireless Headphones",
        "category":            "electronics",
        "current_price":       94.99,
        "base_price":          79.99,
        "min_price":           65.00,
        "max_price":           140.00,
        "comp_price_med":      89.50,
        "comp_price_p25":      82.00,
        "comp_price_p75":      98.00,
        "comp_price_range":    16.00,
        "comp_count":          12,
        "price_position":      6.1,       # 6.1% above market
        "is_price_competitive":1,
        "price_vs_floor":      46.1,
        "price_vs_ceiling":    32.1,
        "trend_index_latest":  72,
        "trend_index_4w_avg":  65.0,
        "trend_slope":         1.5,
        "is_trending":         1,
        "demand_percentile":   78.0,
        "inventory_qty":       85,
        "inventory_tier":      2,
        "inventory_pressure":  0,
        "days_of_stock":       42,
        "price_7d_avg":        94.99,
        "price_30d_avg":       92.50,
        "price_90d_avg":       88.00,
        "price_momentum":      2.7,
        "price_volatility":    1.2,
        "price_vs_base":       18.75,
        "target_price":        91.00,
        "demand_multiplier":   1.11,
        "month":               3,
        "quarter":             1,
        "day_of_week":         2,
        "is_weekend":          0,
        "is_holiday_season":   0,
        "is_back_to_school":   0,
        "is_new_year_fitness": 0,
        "days_until_black_friday":   256,
        "days_until_christmas":      282,
        "days_until_prime_day":      119,
        "cat_electronics":     1,
        "cat_fashion":         0,
        "cat_home_goods":      0,
        "cat_sports":          0,
    }