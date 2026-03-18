"""
scripts/seed_db.py
Seeds the database with realistic ecommerce products for ARIA.

Products are designed to cover all four categories with intentional
pricing scenarios: some overpriced, some underpriced, some spot-on.
This creates interesting repricing decisions for the agent to make.

Usage:
    python scripts/seed_db.py           # Seed all products
    python scripts/seed_db.py --reset   # Drop and re-seed
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import random

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("seed_db")

from db.models import (
    get_db, init_db, Product, PriceHistory,
    DemandSignal, Base
)
from db.models import engine

# ── Product catalog ───────────────────────────────────────────────────
# Designed with realistic pricing scenarios:
# - Some are ABOVE market (good repricing opportunity downward)
# - Some are BELOW market (opportunity to raise)
# - Some are COMPETITIVE (hold)
# current_price deliberately set to create interesting agent decisions

PRODUCTS = [
    # ── Electronics ───────────────────────────────────────────────────
    {
        "name": "Wireless Noise-Cancelling Headphones",
        "sku": "ELEC-001",
        "category": "electronics",
        "base_price": 79.99,
        "current_price": 94.99,   # ABOVE market — agent should lower
        "min_price": 65.00,
        "max_price": 140.00,
        "inventory_qty": 85,
    },
    {
        "name": "Bluetooth Portable Speaker",
        "sku": "ELEC-002",
        "category": "electronics",
        "base_price": 44.99,
        "current_price": 42.99,   # BELOW market — agent should raise
        "min_price": 35.00,
        "max_price": 80.00,
        "inventory_qty": 210,     # Overstocked
    },
    {
        "name": "Mechanical Gaming Keyboard",
        "sku": "ELEC-003",
        "category": "electronics",
        "base_price": 69.99,
        "current_price": 74.99,   # Competitive
        "min_price": 55.00,
        "max_price": 120.00,
        "inventory_qty": 45,
    },
    {
        "name": "Smart Watch Fitness Tracker",
        "sku": "ELEC-004",
        "category": "electronics",
        "base_price": 89.99,
        "current_price": 79.99,   # BELOW market — trending product
        "min_price": 70.00,
        "max_price": 150.00,
        "inventory_qty": 12,      # Low stock — scarcity premium
    },
    {
        "name": "USB-C Hub 7-Port Adapter",
        "sku": "ELEC-005",
        "category": "electronics",
        "base_price": 34.99,
        "current_price": 39.99,
        "min_price": 25.00,
        "max_price": 65.00,
        "inventory_qty": 150,
    },

    # ── Fashion ───────────────────────────────────────────────────────
    {
        "name": "Leather Bifold Wallet",
        "sku": "FASH-001",
        "category": "fashion",
        "base_price": 39.99,
        "current_price": 44.99,
        "min_price": 30.00,
        "max_price": 75.00,
        "inventory_qty": 90,
    },
    {
        "name": "Canvas Tote Bag",
        "sku": "FASH-002",
        "category": "fashion",
        "base_price": 19.99,
        "current_price": 16.99,   # BELOW market — agent should raise
        "min_price": 12.00,
        "max_price": 40.00,
        "inventory_qty": 300,     # Overstocked
    },
    {
        "name": "Minimalist Analog Watch",
        "sku": "FASH-003",
        "category": "fashion",
        "base_price": 59.99,
        "current_price": 74.99,   # ABOVE market
        "min_price": 45.00,
        "max_price": 110.00,
        "inventory_qty": 35,
    },
    {
        "name": "Merino Wool Beanie",
        "sku": "FASH-004",
        "category": "fashion",
        "base_price": 24.99,
        "current_price": 24.99,
        "min_price": 18.00,
        "max_price": 45.00,
        "inventory_qty": 7,       # Critically low stock
    },

    # ── Home Goods ────────────────────────────────────────────────────
    {
        "name": "Bamboo Cutting Board Set",
        "sku": "HOME-001",
        "category": "home_goods",
        "base_price": 29.99,
        "current_price": 34.99,
        "min_price": 22.00,
        "max_price": 55.00,
        "inventory_qty": 125,
    },
    {
        "name": "Stainless Steel Water Bottle 1L",
        "sku": "HOME-002",
        "category": "home_goods",
        "base_price": 24.99,
        "current_price": 22.99,   # BELOW market
        "min_price": 18.00,
        "max_price": 45.00,
        "inventory_qty": 180,
    },
    {
        "name": "Soy Wax Candle Set",
        "sku": "HOME-003",
        "category": "home_goods",
        "base_price": 32.99,
        "current_price": 39.99,   # ABOVE market
        "min_price": 25.00,
        "max_price": 60.00,
        "inventory_qty": 60,
    },
    {
        "name": "Essential Oil Diffuser",
        "sku": "HOME-004",
        "category": "home_goods",
        "base_price": 29.99,
        "current_price": 29.99,
        "min_price": 22.00,
        "max_price": 55.00,
        "inventory_qty": 95,
    },

    # ── Sports ───────────────────────────────────────────────────────
    {
        "name": "Non-Slip Yoga Mat 6mm",
        "sku": "SPRT-001",
        "category": "sports",
        "base_price": 24.99,
        "current_price": 29.99,
        "min_price": 18.00,
        "max_price": 50.00,
        "inventory_qty": 75,
    },
    {
        "name": "Resistance Bands Set 5 Levels",
        "sku": "SPRT-002",
        "category": "sports",
        "base_price": 18.99,
        "current_price": 15.99,   # BELOW market — trending fitness product
        "min_price": 12.00,
        "max_price": 35.00,
        "inventory_qty": 250,
    },
    {
        "name": "Running Shoes Lightweight",
        "sku": "SPRT-003",
        "category": "sports",
        "base_price": 89.99,
        "current_price": 104.99,  # ABOVE market
        "min_price": 70.00,
        "max_price": 160.00,
        "inventory_qty": 42,
    },
    {
        "name": "High-Density Foam Roller",
        "sku": "SPRT-004",
        "category": "sports",
        "base_price": 19.99,
        "current_price": 19.99,
        "min_price": 14.00,
        "max_price": 38.00,
        "inventory_qty": 110,
    },
    {
        "name": "Adjustable Dumbbell Set",
        "sku": "SPRT-005",
        "category": "sports",
        "base_price": 129.99,
        "current_price": 119.99,  # BELOW market — high demand
        "min_price": 100.00,
        "max_price": 220.00,
        "inventory_qty": 8,       # Low stock
    },
]


def seed_products(db) -> list:
    """Insert products, skipping existing SKUs."""
    inserted = []
    for p in PRODUCTS:
        existing = db.query(Product).filter(Product.sku == p["sku"]).first()
        if existing:
            log.info(f"  Skipping existing: {p['name']}")
            continue

        product = Product(**p, is_active=True, platform="shopify")
        db.add(product)
        db.flush()  # Get the ID
        inserted.append(product)
        log.info(f"  + [{product.id:>2}] {product.name} (${product.current_price})")

    return inserted


def seed_price_history(db, products: list):
    """
    Seed 30 days of price history per product.
    Creates realistic price movement patterns.
    """
    if not products:
        return

    now = datetime.utcnow()
    rng = random.Random(42)

    for product in products:
        current = float(product.current_price)
        # Work backwards from current price
        price = current
        for days_ago in range(30, 0, -3):  # Every 3 days
            # Small random walk
            change = rng.uniform(-0.03, 0.03)
            price = max(float(product.min_price or current * 0.5),
                        price * (1 + change))
            price = round(price, 2)

            db.add(PriceHistory(
                product_id=product.id,
                old_price=round(price * (1 - change), 2),
                new_price=price,
                change_pct=round(change * 100, 2),
                source="import",
                recorded_at=now - timedelta(days=days_ago),
            ))


def seed_demand_signals(db):
    """
    Seed 52 weeks of demand signals for category keywords.
    Matches the keywords in fetch_trends.py and features.py.
    """
    import numpy as np

    keywords = {
        "wireless headphones":  {"base": 60, "peak_month": 11, "trend": 0.3},
        "bluetooth speaker":    {"base": 50, "peak_month": 7,  "trend": 0.1},
        "smart watch":          {"base": 55, "peak_month": 11, "trend": 0.5},
        "mechanical keyboard":  {"base": 45, "peak_month": 11, "trend": 0.2},
        "leather wallet":       {"base": 40, "peak_month": 11, "trend": 0.0},
        "tote bag":             {"base": 35, "peak_month": 6,  "trend": 0.1},
        "minimalist watch":     {"base": 30, "peak_month": 4,  "trend": -0.1},
        "wool beanie":          {"base": 25, "peak_month": 10, "trend": 0.0},
        "bamboo cutting board": {"base": 40, "peak_month": 11, "trend": 0.2},
        "stainless steel water bottle": {"base": 55, "peak_month": 6,  "trend": 0.3},
        "soy candle":           {"base": 35, "peak_month": 11, "trend": 0.1},
        "essential oil diffuser":{"base": 38, "peak_month": 11,"trend": 0.1},
        "yoga mat":             {"base": 65, "peak_month": 1,  "trend": 0.4},
        "resistance bands":     {"base": 58, "peak_month": 1,  "trend": 0.3},
        "running shoes":        {"base": 52, "peak_month": 3,  "trend": 0.2},
        "foam roller":          {"base": 45, "peak_month": 1,  "trend": 0.1},
    }

    from datetime import date, timedelta
    today = date.today()
    np.random.seed(42)

    count = 0
    for keyword, params in keywords.items():
        existing = db.query(DemandSignal).filter(
            DemandSignal.keyword == keyword
        ).count()
        if existing > 0:
            log.info(f"  Skipping existing signals for: {keyword}")
            continue

        for week_num in range(52):
            week_date = today - timedelta(weeks=52 - week_num)
            t = week_num / 52

            # Base + trend
            base = params["base"] + params["trend"] * week_num

            # Annual seasonality
            peak_month = params["peak_month"]
            seasonal = 20 * np.sin(2 * np.pi * (t - (peak_month - 1) / 12))

            # Noise
            noise = np.random.normal(0, 4)

            index = int(np.clip(base + seasonal + noise, 0, 100))

            db.add(DemandSignal(
                keyword=keyword,
                trend_index=index,
                week_date=datetime.combine(week_date, datetime.min.time()),
                region="US",
            ))
            count += 1

    log.info(f"  Seeded {count} demand signal rows")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ARIA database seeder")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate all tables before seeding")
    args = parser.parse_args()

    log.info("ARIA — Database Seeder")

    if args.reset:
        log.warning("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)

    # Init tables
    init_db()

    with get_db() as db:
        log.info("\nSeeding products...")
        products = seed_products(db)

        log.info("\nSeeding price history...")
        seed_price_history(db, products)

        log.info("\nSeeding demand signals...")
        seed_demand_signals(db)

    # Summary
    from db.models import get_table_counts
    counts = get_table_counts()

    log.info(f"\n{'='*45}")
    log.info("Seed complete")
    log.info(f"{'='*45}")
    for table, count in counts.items():
        if count > 0:
            log.info(f"  {table:<30} {count:>6,} rows")

    log.info(f"\nNext steps:")
    log.info(f"  python scripts/fetch_competitors.py --demo")
    log.info(f"  python src/features.py")
    log.info(f"  python src/pricing_model.py")
    log.info(f"  python src/demand_forecast.py")
    log.info(f"  python agent/aria.py --demo")


if __name__ == "__main__":
    main()