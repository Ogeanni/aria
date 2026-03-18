"""
scripts/fetch_competitors.py
Competitor price ingestion for ARIA.

Data source hierarchy (fallback chain):
  1. Redis cache     — if fresh data exists (< 6h), return immediately
  2. File cache      — if Redis unavailable, check local JSON cache
  3. SerpAPI         — live Google Shopping data (costs credits)
  4. Price simulator — realistic synthetic data (free, always works)

This implements the playbook principle:
  "Every external dependency needs a fallback. Never let one API
   failure take down your serving layer."

DEMO_MODE=true  → skips SerpAPI entirely, uses simulator
DEMO_MODE=false → tries SerpAPI, falls back to simulator on failure

Usage:
    python scripts/fetch_competitors.py              # all products
    python scripts/fetch_competitors.py --category sports
    python scripts/fetch_competitors.py --product-id 1
    python scripts/fetch_competitors.py --dry-run
    python scripts/fetch_competitors.py --demo       # force simulator
"""
import os
import sys
import json
import time
import logging
import argparse
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("fetch_competitors")

# ── Imports ───────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    log.error("requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from db.models import get_db, Product, CompetitorPrice, get_latest_competitor_prices
    from src.price_simulator import PriceSimulator
    from config.settings import get_settings
except ImportError as e:
    log.error(f"Import error: {e}. Run from project root.")
    sys.exit(1)

settings = get_settings()


# ── Redis cache (optional) ────────────────────────────────────────────
def _get_redis():
    """Returns Redis client or None if unavailable."""
    if not settings.has_redis:
        return None
    try:
        import redis
        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


# ── Cache layer ───────────────────────────────────────────────────────
class PriceCache:
    """
    Two-tier cache: Redis (fast) → File (fallback).
    Reduces SerpAPI calls by serving cached results.

    Cache key: sha256(keyword + category)[:16]
    TTL: 6 hours (configurable via settings.cache_ttl_seconds)
    """

    def __init__(self):
        self.redis = _get_redis()
        self.cache_dir = settings.cache_dir
        self.ttl = settings.cache_ttl_seconds

    def _key(self, keyword: str, category: str) -> str:
        raw = f"{keyword.lower().strip()}:{category.lower().strip()}"
        return "aria:prices:" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _file_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, keyword: str, category: str) -> Optional[dict]:
        """Returns cached price data or None if cache miss / expired."""
        key = self._key(keyword, category)

        # Try Redis first
        if self.redis:
            try:
                raw = self.redis.get(key)
                if raw:
                    data = json.loads(raw)
                    log.debug(f"Cache HIT (Redis): {keyword}")
                    return data
            except Exception:
                pass

        # Fall back to file cache
        fpath = self._file_path(key)
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                age_seconds = (datetime.utcnow() - cached_at).total_seconds()
                if age_seconds < self.ttl:
                    log.debug(f"Cache HIT (file, age={age_seconds/3600:.1f}h): {keyword}")
                    return data
                log.debug(f"Cache STALE (file, age={age_seconds/3600:.1f}h): {keyword}")
            except Exception:
                pass

        log.debug(f"Cache MISS: {keyword}")
        return None

    def set(self, keyword: str, category: str, data: dict):
        """Store data in cache with TTL."""
        key = self._key(keyword, category)
        data["cached_at"] = datetime.utcnow().isoformat()
        serialized = json.dumps(data)

        # Write to Redis
        if self.redis:
            try:
                self.redis.setex(key, self.ttl, serialized)
            except Exception:
                pass

        # Always write to file cache as backup
        try:
            self._file_path(key).write_text(serialized)
        except Exception:
            pass

    def invalidate(self, keyword: str, category: str):
        """Remove cached entry (call after fresh scrape)."""
        key = self._key(keyword, category)
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception:
                pass
        fpath = self._file_path(key)
        if fpath.exists():
            fpath.unlink()


# ── SerpAPI fetcher ───────────────────────────────────────────────────
class SerpAPIFetcher:
    """
    Fetches competitor prices from Google Shopping via SerpAPI.

    Free tier: 100 searches/month
    We conserve credits by:
      1. Always checking cache before calling
      2. Grouping products by keyword (shared keywords = 1 call)
      3. Never re-fetching if data is < 6h old
    """

    BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, keyword: str, num_results: int = 10) -> list[dict]:
        """
        Fetch Google Shopping results for a keyword.

        Returns list of dicts:
        [{"price": float, "retailer": str, "title": str, "link": str}, ...]
        """
        params = {
            "engine":   "google_shopping",
            "q":        keyword,
            "api_key":  self.api_key,
            "num":      num_results,
            "gl":       "us",
            "hl":       "en",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            log.error(f"SerpAPI request failed for '{keyword}': {e}")
            return []
        except json.JSONDecodeError as e:
            log.error(f"SerpAPI response parse error for '{keyword}': {e}")
            return []

        # Check for API errors
        if "error" in data:
            log.error(f"SerpAPI error for '{keyword}': {data['error']}")
            return []

        results = []
        for item in data.get("shopping_results", []):
            try:
                # SerpAPI returns price as string like "$29.99" or "29.99"
                price_raw = item.get("price", "")
                price = self._parse_price(price_raw)
                if price is None or price <= 0:
                    continue

                results.append({
                    "price":    price,
                    "retailer": item.get("source", "Unknown"),
                    "title":    item.get("title", keyword),
                    "link":     item.get("link", ""),
                    "platform": "google_shopping",
                })
            except Exception:
                continue

        log.info(f"SerpAPI '{keyword}': {len(results)} results")
        return results

    @staticmethod
    def _parse_price(price_str: str) -> Optional[float]:
        """Parse price string like '$29.99', '29.99', '$1,299.99' → float."""
        if not price_str:
            return None
        cleaned = price_str.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None


# ── Main fetcher orchestrator ─────────────────────────────────────────
class CompetitorFetcher:
    """
    Orchestrates the full fetch pipeline:
    Cache → SerpAPI → Simulator

    This is the single entry point for getting competitor prices.
    All other code calls this, never the underlying fetchers directly.
    """

    def __init__(self, demo_mode: Optional[bool] = None):
        self.demo_mode = demo_mode if demo_mode is not None else settings.demo_mode
        self.cache = PriceCache()
        self.simulator = PriceSimulator()

        if not self.demo_mode and settings.has_serpapi:
            self.serpapi = SerpAPIFetcher(settings.serpapi_key)
            log.info("CompetitorFetcher: SerpAPI enabled")
        else:
            self.serpapi = None
            mode = "DEMO MODE (simulator)" if self.demo_mode else "no SerpAPI key"
            log.info(f"CompetitorFetcher: {mode}")

    def fetch(
        self,
        product_name: str,
        category: str,
        our_price: float,
        product_id: Optional[int] = None,
        force_refresh: bool = False,
    ) -> dict:
        """
        Main fetch method. Returns price band dict.

        Result dict format:
        {
            "keyword":   str,
            "category":  str,
            "count":     int,
            "min":       float,
            "max":       float,
            "median":    float,
            "p25":       float,
            "p75":       float,
            "mean":      float,
            "source":    "serpapi" | "serpapi_cached" | "simulated" | "simulated_cached",
            "listings":  [...],
            "fetched_at": str,
        }
        """
        keyword = self._build_keyword(product_name)

        # ── 1. Check cache ─────────────────────────────────────────────
        if not force_refresh:
            cached = self.cache.get(keyword, category)
            if cached:
                cached["source"] = cached.get("source", "unknown") + "_cached"
                return cached

        # ── 2. Try SerpAPI ─────────────────────────────────────────────
        if not self.demo_mode and self.serpapi:
            listings_raw = self.serpapi.fetch(keyword)
            if listings_raw:
                band = self._compute_band(listings_raw, source="serpapi")
                band["keyword"] = keyword
                band["category"] = category
                band["fetched_at"] = datetime.utcnow().isoformat()
                self.cache.set(keyword, category, band)
                log.info(f"  {product_name}: fetched from SerpAPI "
                         f"(median=${band['median']:.2f}, n={band['count']})")
                return band
            else:
                log.warning(f"  SerpAPI returned no results for '{keyword}' — falling back to simulator")

        # ── 3. Fallback: Price simulator ───────────────────────────────
        band = self.simulator.get_price_band(keyword, category, our_price)
        band["keyword"] = keyword
        band["category"] = category
        band["source"] = "simulated"
        band["fetched_at"] = datetime.utcnow().isoformat()

        # Cache simulated results too (shorter TTL would be ideal, but same for simplicity)
        self.cache.set(keyword, category, band)

        log.info(f"  {product_name}: simulated prices "
                 f"(median=${band['median']:.2f}, n={band['count']})")
        return band

    def _build_keyword(self, product_name: str) -> str:
        """
        Build a clean search keyword from product name.
        Remove noise words that hurt search quality.
        """
        noise = ["Set", "1L", "6mm", "TKL", "7-in-1", "Pro", "Plus",
                 "Edition", "Series", "Model", "Version"]
        keyword = product_name
        for word in noise:
            keyword = keyword.replace(word, "").strip()
        # Collapse whitespace
        keyword = " ".join(keyword.split())
        return keyword

    @staticmethod
    def _compute_band(listings: list[dict], source: str) -> dict:
        """Compute price band statistics from listing list."""
        import numpy as np
        prices = sorted([l["price"] for l in listings])
        n = len(prices)
        if n == 0:
            return {"count": 0, "source": source}
        return {
            "count":    n,
            "min":      round(min(prices), 2),
            "max":      round(max(prices), 2),
            "mean":     round(float(np.mean(prices)), 2),
            "median":   round(float(np.median(prices)), 2),
            "p25":      round(prices[max(0, int(n * 0.25) - 1)], 2),
            "p75":      round(prices[min(n - 1, int(n * 0.75))], 2),
            "source":   source,
            "listings": listings,
        }


def save_to_db(
    product_id: int,
    band: dict,
    dry_run: bool = False,
) -> int:
    """Save fetched listings to competitor_prices table."""
    listings = band.get("listings", [])
    if not listings:
        return 0

    if dry_run:
        log.info(f"  [DRY RUN] Would save {len(listings)} listings for product {product_id}")
        return len(listings)

    is_simulated = band.get("source", "").startswith("simulated")

    with get_db() as db:
        for item in listings:
            # All listings are plain dicts at this point (simulator converts on output)
            price    = item.get("price", 0)
            retailer = item.get("retailer", "Unknown")
            title    = item.get("title", "")
            url      = item.get("link", item.get("listing_url", ""))

            row = CompetitorPrice(
                product_id=product_id,
                platform=band.get("source", "unknown"),
                retailer=retailer,
                competitor_price=price,
                listing_title=title,
                listing_url=url,
                is_simulated=is_simulated,
                scraped_at=datetime.utcnow(),
            )
            db.add(row)

    return len(listings)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ARIA competitor price fetcher")
    parser.add_argument("--category",   "-c", default=None)
    parser.add_argument("--product-id", "-p", type=int, default=None)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--demo",       action="store_true", help="Force simulator mode")
    parser.add_argument("--refresh",    action="store_true", help="Ignore cache")
    args = parser.parse_args()

    demo_mode = args.demo or settings.demo_mode
    fetcher = CompetitorFetcher(demo_mode=demo_mode)

    mode_label = "DEMO (simulator)" if demo_mode else "LIVE (SerpAPI)"
    log.info(f"ARIA Competitor Fetcher — {mode_label}")

    # Load products — convert to plain dicts INSIDE the session
    # to avoid DetachedInstanceError when the session closes
    with get_db() as db:
        query = db.query(Product).filter(Product.is_active == True)
        if args.product_id:
            query = query.filter(Product.id == args.product_id)
        elif args.category:
            query = query.filter(Product.category == args.category)
        products = [
            {
                "id":            p.id,
                "name":          p.name,
                "category":      p.category,
                "current_price": float(p.current_price),
                "min_price":     float(p.min_price) if p.min_price else None,
                "max_price":     float(p.max_price) if p.max_price else None,
                "inventory_qty": p.inventory_qty,
            }
            for p in query.all()
        ]

    if not products:
        log.error("No products found. Run: python scripts/seed_db.py")
        return 1

    log.info(f"Fetching competitor prices for {len(products)} products\n")

    start = datetime.now()
    results = []

    for product in products:
        log.info(f"[{product['id']}] {product['name']} (${product['current_price']})")
        try:
            band = fetcher.fetch(
                product_name=product["name"],
                category=product["category"],
                our_price=product["current_price"],
                product_id=product["id"],
                force_refresh=args.refresh,
            )
            saved = save_to_db(product["id"], band, dry_run=args.dry_run)

            position = (
                (product["current_price"] - band["median"]) / band["median"] * 100
                if band.get("median", 0) > 0 else 0
            )
            status = "ABOVE" if position > 10 else ("BELOW" if position < -10 else "IN LINE")

            log.info(f"  median=${band['median']:.2f}  "
                     f"range=[${band['min']:.2f}, ${band['max']:.2f}]  "
                     f"our=${product['current_price']:.2f} ({position:+.1f}% {status})  "
                     f"source={band['source']}")

            results.append({"product": product["name"], "status": "ok", "saved": saved})

        except Exception as e:
            log.error(f"  Error: {e}")
            results.append({"product": product["name"], "status": "error", "error": str(e)})

        time.sleep(0.5)

    elapsed = int((datetime.now() - start).total_seconds())
    total_saved = sum(r.get("saved", 0) for r in results)
    errors = [r for r in results if r["status"] == "error"]

    log.info(f"\n{'='*55}")
    log.info(f"SUMMARY: {len(results)} products | {total_saved} listings saved | "
             f"{len(errors)} errors | {elapsed}s")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())