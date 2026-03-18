"""
src/features.py
Feature engineering pipeline for ARIA.

Builds a feature matrix by joining:
  - products          (base info, current price, inventory, price constraints)
  - price_history     (rolling averages, price momentum, volatility)
  - demand_signals    (Google Trends index, slope, seasonality)
  - competitor_prices (market median, price position, bands)
  + time features     (month, retail events, days-until-holiday)

Output:
  data/processed/features.parquet  — feature matrix ready for XGBoost
  data/processed/features_meta.json — column types, feature list, version

Usage:
    python src/features.py                 # Build and save
    python src/features.py --preview       # Print summary, don't save
    python src/features.py --validate      # Data quality checks only
    python src/features.py --date 2024-11-01  # Historical snapshot
"""
import sys
import json
import logging
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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
log = logging.getLogger("features")

from db.models import get_db, Product, PriceHistory, DemandSignal, CompetitorPrice
from config.settings import get_settings

settings = get_settings()

# ── Category → keyword mapping ────────────────────────────────────────
# Maps each product category to its PRIMARY demand keyword.
# Must match keywords fetched in fetch_trends.py.
# One keyword per category — the most representative for that category.
CATEGORY_KEYWORD = {
    "electronics": "wireless headphones",
    "fashion":     "leather wallet",
    "home_goods":  "bamboo cutting board",
    "sports":      "yoga mat",
}

# ── Retail event calendar ─────────────────────────────────────────────
# (month, day) for major retail events — used to compute days_until_* features
RETAIL_EVENTS = {
    "black_friday":   (11, 29),
    "cyber_monday":   (12,  2),
    "christmas":      (12, 25),
    "valentines":     ( 2, 14),
    "mothers_day":    ( 5, 12),
    "prime_day":      ( 7, 16),
    "back_to_school": ( 8, 15),
    "new_year":       ( 1,  1),
}


# ══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# All loaders convert ORM objects to dicts INSIDE the session block
# to avoid SQLAlchemy DetachedInstanceError
# ══════════════════════════════════════════════════════════════════════

def load_products() -> pd.DataFrame:
    """Load all active products from DB."""
    with get_db() as db:
        rows = db.query(Product).filter(Product.is_active == True).all()
        data = [{
            "product_id":    r.id,
            "product_name":  r.name,
            "category":      r.category,
            "base_price":    float(r.base_price),
            "current_price": float(r.current_price),
            "min_price":     float(r.min_price) if r.min_price else None,
            "max_price":     float(r.max_price) if r.max_price else None,
            "inventory_qty": r.inventory_qty or 0,
        } for r in rows]

    if not data:
        raise ValueError("No products in DB. Run: python scripts/seed_db.py")

    df = pd.DataFrame(data)

    # Fill missing price constraints with sensible defaults
    # min_price defaults to 70% of base_price, max_price to 150%
    df["min_price"] = df["min_price"].fillna(df["base_price"] * 0.70)
    df["max_price"] = df["max_price"].fillna(df["base_price"] * 1.50)

    log.info(f"Loaded {len(df)} products across {df['category'].nunique()} categories")
    return df


def load_price_history() -> pd.DataFrame:
    """
    Load price history — uses new_price column (not 'price').
    Ordered by product_id and recorded_at for correct rolling window computation.
    """
    with get_db() as db:
        rows = (
            db.query(PriceHistory)
            .order_by(PriceHistory.product_id, PriceHistory.recorded_at)
            .all()
        )
        data = [{
            "product_id":  r.product_id,
            "price":       float(r.new_price),  
            "recorded_at": r.recorded_at,
        } for r in rows]

    if not data:
        log.warning("No price_history rows — price features will use current_price as fallback")
        return pd.DataFrame(columns=["product_id", "price", "recorded_at"])

    df = pd.DataFrame(data)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    log.info(f"Loaded {len(df)} price_history rows for {df['product_id'].nunique()} products")
    return df


def load_demand_signals() -> pd.DataFrame:
    """Load demand signals ordered by keyword and date."""
    with get_db() as db:
        rows = (
            db.query(DemandSignal)
            .order_by(DemandSignal.keyword, DemandSignal.week_date)
            .all()
        )
        data = [{
            "keyword":     r.keyword,
            "trend_index": r.trend_index,
            "week_date":   r.week_date,
        } for r in rows]

    if not data:
        log.warning("No demand_signals — trend features default to neutral (50)")
        return pd.DataFrame(columns=["keyword", "trend_index", "week_date"])

    df = pd.DataFrame(data)
    df["week_date"] = pd.to_datetime(df["week_date"])
    log.info(f"Loaded {len(df)} demand signal rows for {df['keyword'].nunique()} keywords")
    return df


def load_competitor_prices() -> pd.DataFrame:
    """
    Load competitor prices.
    Uses only the most recent scrape date per product to avoid
    mixing stale and fresh data in the price band calculation.
    """
    with get_db() as db:
        rows = db.query(CompetitorPrice).all()
        data = [{
            "product_id":       r.product_id,
            "competitor_price": float(r.competitor_price),
            "scraped_at":       r.scraped_at,
        } for r in rows]

    if not data:
        log.warning("No competitor_prices — competitor features default to current_price")
        return pd.DataFrame(columns=["product_id", "competitor_price", "scraped_at"])

    df = pd.DataFrame(data)
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])

    # Keep only listings from the most recent scrape per product
    # (avoids mixing stale simulated data with fresh SerpAPI data)
    latest_scrape = df.groupby("product_id")["scraped_at"].max().reset_index()
    latest_scrape.columns = ["product_id", "latest_scrape"]
    df = df.merge(latest_scrape, on="product_id")

    # Allow 1-hour window around latest scrape to catch batch runs
    df = df[df["scraped_at"] >= df["latest_scrape"] - pd.Timedelta(hours=1)]

    log.info(f"Loaded {len(df)} competitor price rows for {df['product_id'].nunique()} products")
    return df


# ══════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_price_features(products_df: pd.DataFrame, price_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product:
      - price_7d_avg      : 7-day rolling average price
      - price_30d_avg     : 30-day rolling average price
      - price_90d_avg     : 90-day rolling average price
      - price_momentum    : % change from 30d avg to current price (signed)
      - price_volatility  : std dev of prices in last 30 days
      - price_vs_base     : % deviation from original base price
    """
    now = datetime.utcnow()
    records = []

    for _, product in products_df.iterrows():
        pid = product["product_id"]
        current = product["current_price"]
        base = product["base_price"]

        if price_history_df.empty:
            ph = pd.DataFrame(columns=["price", "recorded_at"])
        else:
            ph = price_history_df[price_history_df["product_id"] == pid].copy()

        def avg_last_n_days(n: int) -> float:
            if ph.empty:
                return current
            cutoff = now - timedelta(days=n)
            subset = ph[ph["recorded_at"] >= cutoff]["price"]
            return float(subset.mean()) if len(subset) > 0 else current

        avg_7d  = avg_last_n_days(7)
        avg_30d = avg_last_n_days(30)
        avg_90d = avg_last_n_days(90)

        # Momentum: how much has current price drifted from 30d average
        momentum = ((current - avg_30d) / avg_30d * 100) if avg_30d > 0 else 0.0

        # Volatility: std dev of last 30 days of prices
        cutoff_30 = now - timedelta(days=30)
        recent = ph[ph["recorded_at"] >= cutoff_30]["price"] if not ph.empty else pd.Series(dtype=float)
        volatility = float(recent.std()) if len(recent) > 1 else 0.0

        # Price vs base: how far have we drifted from original price
        price_vs_base = ((current - base) / base * 100) if base > 0 else 0.0

        records.append({
            "product_id":      pid,
            "price_7d_avg":    round(avg_7d, 2),
            "price_30d_avg":   round(avg_30d, 2),
            "price_90d_avg":   round(avg_90d, 2),
            "price_momentum":  round(momentum, 3),
            "price_volatility":round(volatility, 3),
            "price_vs_base":   round(price_vs_base, 2),
        })

    return pd.DataFrame(records)


def build_demand_features(products_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product (via category keyword):
      - trend_index_latest  : most recent Google Trends index (0-100)
      - trend_index_4w_avg  : 4-week average trend index
      - trend_slope         : linear slope over last 8 weeks (index pts/week)
      - is_trending         : 1 if latest > 4w avg by >10 points
      - demand_percentile   : where latest index sits in the 52w distribution
    """
    records = []

    for _, product in products_df.iterrows():
        pid = product["product_id"]
        category = product["category"]
        keyword = CATEGORY_KEYWORD.get(category)

        defaults = {
            "product_id":          pid,
            "trend_index_latest":  50,
            "trend_index_4w_avg":  50.0,
            "trend_slope":         0.0,
            "is_trending":         0,
            "demand_percentile":   50.0,
        }

        if demand_df.empty or keyword is None:
            records.append(defaults)
            continue

        kw_data = demand_df[demand_df["keyword"] == keyword].sort_values("week_date")

        if kw_data.empty:
            records.append(defaults)
            continue

        # FIX: was kw_data.loc["trend_index"] — wrong axis
        latest_idx = int(kw_data["trend_index"].iloc[-1])

        last_4w  = kw_data.tail(4)["trend_index"]
        avg_4w   = float(last_4w.mean())

        last_8w  = kw_data.tail(8)
        if len(last_8w) >= 2:
            x = np.arange(len(last_8w))
            y = last_8w["trend_index"].values.astype(float)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0

        is_trending = 1 if (latest_idx - avg_4w) > 10 else 0

        all_indices = kw_data["trend_index"].values
        percentile  = float(np.mean(all_indices <= latest_idx) * 100)

        records.append({
            "product_id":         pid,
            "trend_index_latest": latest_idx,
            "trend_index_4w_avg": round(avg_4w, 1),
            "trend_slope":        round(slope, 3),
            "is_trending":        is_trending,
            "demand_percentile":  round(percentile, 1),
        })

    return pd.DataFrame(records)


def build_competitor_features(products_df: pd.DataFrame, comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product:
      - comp_count            : number of competitor listings
      - comp_price_p25        : 25th percentile competitor price
      - comp_price_med        : median competitor price
      - comp_price_p75        : 75th percentile competitor price
      - comp_price_range      : p75 - p25 (market spread)
      - price_position        : our price vs comp median (positive = above market)
      - is_price_competitive  : 1 if within ±15% of comp median
      - price_vs_floor        : % above our min_price constraint
      - price_vs_ceiling      : % below our max_price constraint
    """
    records = []

    for _, product in products_df.iterrows():
        pid = product["product_id"]
        current = product["current_price"]
        min_p = product.get("min_price") or current * 0.70
        max_p = product.get("max_price") or current * 1.50

        defaults = {
            "product_id":           pid,
            "comp_count":           0,
            "comp_price_p25":       current,
            "comp_price_med":       current,
            "comp_price_p75":       current,
            "comp_price_range":     0.0,
            "price_position":       0.0,
            "is_price_competitive": 1,
            "price_vs_floor":       round((current - min_p) / min_p * 100, 2),
            "price_vs_ceiling":     round((max_p - current) / max_p * 100, 2),
        }

        if comp_df.empty:
            records.append(defaults)
            continue

        comp_prices = comp_df[comp_df["product_id"] == pid]["competitor_price"]

        if len(comp_prices) == 0:
            records.append(defaults)
            continue

        prices = comp_prices.values
        p25    = float(np.percentile(prices, 25))
        median = float(np.percentile(prices, 50))
        p75    = float(np.percentile(prices, 75))
        spread = round(p75 - p25, 2)

        position       = round(((current - median) / median * 100), 2) if median > 0 else 0.0
        is_competitive = 1 if abs(position) <= 15.0 else 0

        # How much room do we have above floor / below ceiling
        price_vs_floor   = round((current - min_p) / min_p * 100, 2)
        price_vs_ceiling = round((max_p - current) / max_p * 100, 2)

        records.append({
            "product_id":           pid,
            "comp_count":           len(prices),
            "comp_price_p25":       round(p25, 2),
            "comp_price_med":       round(median, 2),
            "comp_price_p75":       round(p75, 2),
            "comp_price_range":     spread,
            "price_position":       position,
            "is_price_competitive": is_competitive,
            "price_vs_floor":       price_vs_floor,
            "price_vs_ceiling":     price_vs_ceiling,
        })

    return pd.DataFrame(records)


def build_inventory_features(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per product:
      - inventory_qty      : current stock level
      - inventory_tier     : 0=critical(<10), 1=low(<50), 2=normal, 3=high(>200)
      - inventory_pressure : +1=scarce (price up), 0=normal, -1=overstocked (price down)
      - days_of_stock      : rough estimate at assumed 2 units/day sell rate
    """
    records = []

    for _, product in products_df.iterrows():
        qty = int(product["inventory_qty"])

        if qty < 10:
            tier     = 0
            pressure = 1       # Scarce → premium pricing justified
        elif qty < 50:
            tier     = 1
            pressure = 0
        elif qty <= 200:
            tier     = 2
            pressure = 0
        else:
            tier     = 3
            pressure = -1      # Overstocked → incentivize clearance

        # Rough days of stock remaining at 2 units/day assumed rate
        days_of_stock = min(qty // 2, 365)

        records.append({
            "product_id":        product["product_id"],
            "inventory_qty":     qty,
            "inventory_tier":    tier,
            "inventory_pressure":pressure,
            "days_of_stock":     days_of_stock,
        })

    return pd.DataFrame(records)


def build_time_features(today: date = None) -> dict:
    """
    Global time features — same value for all products on a given day.
    Returned as a dict; broadcast to all rows in the final join.

      - month, day_of_week, is_weekend, quarter
      - is_holiday_season  : Nov or Dec
      - is_back_to_school  : July or August
      - is_new_year_fitness: January (yoga mats, fitness gear peak)
      - days_until_*       : days until each major retail event (capped at 365)
    """
    if today is None:
        today = date.today()

    def days_until(month: int, day: int) -> int:
        target = date(today.year, month, day)
        if target < today:
            target = date(today.year + 1, month, day)
        return min((target - today).days, 365)

    feats = {
        "snapshot_date":        today.isoformat(),
        "month":                today.month,
        "day_of_week":          today.weekday(),
        "is_weekend":           1 if today.weekday() >= 5 else 0,
        "quarter":              (today.month - 1) // 3 + 1,
        "is_holiday_season":    1 if today.month in (11, 12) else 0,
        "is_back_to_school":    1 if today.month in (7, 8) else 0,
        "is_new_year_fitness":  1 if today.month == 1 else 0,
    }

    for event, (month, day) in RETAIL_EVENTS.items():
        feats[f"days_until_{event}"] = days_until(month, day)

    return feats


def build_target(
    products_df: pd.DataFrame,
    comp_feats: pd.DataFrame,
    demand_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the training target price.

    Formula:
        target_price = clamp(comp_price_med * demand_multiplier, min_price, max_price)

    Where:
        demand_multiplier = 1.0 + (trend_index_latest - 50) / 200
            → trend=50  (average): multiplier=1.00 (no adjustment)
            → trend=100 (peak):    multiplier=1.25 (+25%)
            → trend=0   (trough):  multiplier=0.75 (-25%)

    The clamp ensures the target always respects the product's price constraints.
    Without the clamp the model could learn to recommend prices outside business rules.
    """
    merged = (
        products_df[["product_id", "current_price", "min_price", "max_price"]]
        .merge(comp_feats[["product_id", "comp_price_med"]], on="product_id", how="left")
        .merge(demand_feats[["product_id", "trend_index_latest"]], on="product_id", how="left")
    )

    # Base: use competitor median if available, else current price
    merged["price_base"] = merged["comp_price_med"].fillna(merged["current_price"])

    # Demand multiplier
    merged["trend_clamped"]      = merged["trend_index_latest"].fillna(50).clip(0, 100)
    merged["demand_multiplier"]  = 1.0 + (merged["trend_clamped"] - 50) / 200

    raw_target = merged["price_base"] * merged["demand_multiplier"]

    # Clamp to price constraints — target must be within business rules
    merged["target_price"] = np.clip(
        raw_target,
        merged["min_price"],
        merged["max_price"],
    ).round(2)

    return merged[["product_id", "price_base", "demand_multiplier", "target_price"]]


# ══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

def build_feature_matrix(today: date = None) -> pd.DataFrame:
    """
    Orchestrates all feature builders into a single feature matrix.
    One row per product, all features joined on product_id.
    """
    today = today or date.today()
    log.info(f"Building feature matrix — snapshot date: {today}")

    # ── Load raw data ──────────────────────────────────────────────────
    log.info("\n[1/6] Loading products...")
    products_df = load_products()

    log.info("[2/6] Loading price history...")
    price_history_df = load_price_history()

    log.info("[3/6] Loading demand signals...")
    demand_df = load_demand_signals()

    log.info("[4/6] Loading competitor prices...")
    comp_df = load_competitor_prices()

    # ── Build feature groups ───────────────────────────────────────────
    log.info("\n[5/6] Building features...")

    log.info("  price features...")
    price_feats = build_price_features(products_df, price_history_df)

    log.info("  demand features...")
    demand_feats = build_demand_features(products_df, demand_df)

    log.info("  competitor features...")
    comp_feats = build_competitor_features(products_df, comp_df)

    log.info("  inventory features...")
    inv_feats = build_inventory_features(products_df)

    log.info("  time features...")
    time_feats = build_time_features(today)

    log.info("  target prices...")
    target_df = build_target(products_df, comp_feats, demand_feats)

    # ── Join everything on product_id ──────────────────────────────────
    log.info("\n[6/6] Joining feature groups...")

    # Start from products, drop inventory_qty (inv_feats has it + tier + pressure)
    df = products_df.drop(columns=["inventory_qty"], errors="ignore")

    for feats, name in [
        (price_feats,  "price"),
        (demand_feats, "demand"),
        (comp_feats,   "competitor"),
        (inv_feats,    "inventory"),
        (target_df[["product_id", "target_price", "demand_multiplier"]], "target"),
    ]:
        df = df.merge(feats, on="product_id", how="left")
        log.info(f"  after {name} join: {df.shape}")

    # Broadcast time features (scalar → column for all rows)
    for col, val in time_feats.items():
        df[col] = val

    # ── Category encoding ──────────────────────────────────────────────
    # One-hot encode category — makes feature importance plots cleaner
    cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(int)
    df = pd.concat([df, cat_dummies], axis=1)

    # ── Final cleanup ──────────────────────────────────────────────────
    df = df.fillna(0)

    log.info(f"\nFeature matrix complete: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_features(df: pd.DataFrame) -> bool:
    """Data quality checks. Returns True if all pass."""
    log.info("\nRunning validation checks...")
    passed = True

    checks = {
        "No missing product_ids":        df["product_id"].isna().sum() == 0,
        "No missing target prices":      df["target_price"].isna().sum() == 0,
        "All target prices > 0":         (df["target_price"] > 0).all(),
        "No negative inventory":         (df["inventory_qty"] >= 0).all(),
        "Trend index in [0,100]":        df["trend_index_latest"].between(0, 100).all(),
        "Price momentum is finite":      np.isfinite(df["price_momentum"]).all(),
        "No duplicate product_ids":      df["product_id"].nunique() == len(df),
        "Has 4 categories":              df["category"].nunique() >= 4,
        "Target within price bounds":    (
            (df["target_price"] >= df["min_price"]) &
            (df["target_price"] <= df["max_price"])
        ).all(),
    }

    for check, result in checks.items():
        icon = "OK  " if result else "FAIL"
        log.info(f"  [{icon}] {check}")
        if not result:
            passed = False

    log.info(f"\n  {'All checks passed.' if passed else 'Some checks FAILED — review before training.'}")
    return passed


def print_preview(df: pd.DataFrame):
    """Human-readable feature matrix summary."""
    log.info("\n" + "=" * 65)
    log.info("FEATURE MATRIX PREVIEW")
    log.info("=" * 65)
    log.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    display_cols = [
        "product_name", "category", "current_price", "target_price",
        "demand_multiplier", "trend_index_latest", "comp_price_med",
        "price_position", "inventory_qty", "inventory_pressure",
    ]
    available = [c for c in display_cols if c in df.columns]
    pd.set_option("display.max_colwidth", 28)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df[available].to_string(index=False))

    log.info(f"\nAll {len(df.columns)} columns:")
    skip = {"product_id", "product_name", "category", "snapshot_date", "target_price"}
    for i, col in enumerate(df.columns):
        if col in skip:
            continue
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.number):
            log.info(f"  {i+1:>2}. {col:<38} {str(dtype):<10} "
                     f"min={df[col].min():.2f}  max={df[col].max():.2f}")
        else:
            log.info(f"  {i+1:>2}. {col:<38} {str(dtype)}")


def save_features(df: pd.DataFrame):
    """Save feature matrix and metadata."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet
    df.to_parquet(settings.features_path, index=False)
    log.info(f"Features saved → {settings.features_path}")

    # Build metadata
    skip = {"product_id", "product_name", "category", "snapshot_date"}
    feature_cols = [c for c in df.columns if c not in skip and c != "target_price"]

    meta = {
        "version":      "1.0",
        "created_at":   datetime.utcnow().isoformat(),
        "snapshot_date":str(df["snapshot_date"].iloc[0]) if "snapshot_date" in df.columns else None,
        "shape":        list(df.shape),
        "feature_cols": feature_cols,
        "target_col":   "target_price",
        "id_cols":      ["product_id", "product_name", "category"],
        "n_products":   int(df["product_id"].nunique()),
        "categories":   list(df["category"].unique()),
        "feature_groups": {
            "price":      [c for c in feature_cols if "price" in c and "comp" not in c and "target" not in c],
            "demand":     [c for c in feature_cols if "trend" in c or "demand" in c or "is_trending" in c],
            "competitor": [c for c in feature_cols if "comp" in c or "position" in c or "competitive" in c],
            "inventory":  [c for c in feature_cols if "inventory" in c or "stock" in c],
            "time":       [c for c in feature_cols if any(t in c for t in ["month", "day", "quarter", "season", "holiday", "school", "fitness", "days_until"])],
            "category":   [c for c in feature_cols if c.startswith("cat_")],
        },
    }

    meta_path = settings.processed_dir / "features_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Metadata saved  → {meta_path}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ARIA feature engineering pipeline")
    parser.add_argument("--preview",  action="store_true", help="Print summary, skip save")
    parser.add_argument("--validate", action="store_true", help="Validation checks only")
    parser.add_argument("--date",     default=None, help="Snapshot date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    snapshot_date = date.fromisoformat(args.date) if args.date else date.today()

    df = build_feature_matrix(today=snapshot_date)
    passed = validate_features(df)

    if args.preview or args.validate:
        print_preview(df)
        return 0 if passed else 1

    save_features(df)

    # Quick summary regardless
    log.info("\nSample (key columns):")
    key_cols = ["product_name", "current_price", "target_price",
                "trend_index_latest", "comp_price_med",
                "price_position", "inventory_pressure"]
    available = [c for c in key_cols if c in df.columns]
    pd.set_option("display.max_colwidth", 32)
    print(df[available].to_string(index=False))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())