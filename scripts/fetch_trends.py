"""
scripts/fetch_trends.py
Google Trends demand signal ingestion for ARIA.

DATA SOURCE HIERARCHY (same fallback pattern as fetch_competitors.py):
  1. Redis/file cache  — if fresh data exists (< 24h), return immediately
  2. SerpAPI           — Google Trends via official API (uses free credits)
  3. Demand simulator  — statistically realistic synthetic trends (free)

DEMO_MODE=true  → skips SerpAPI, uses simulator
DEMO_MODE=false → tries SerpAPI, falls back to simulator on failure

SerpAPI Google Trends docs:
  https://serpapi.com/google-trends-api

Usage:
    python scripts/fetch_trends.py                    # all categories
"""
import os
import sys
import json
import time
import logging
import argparse
import hashlib
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("fetch_trends")

try:
    import requests
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from db.models import get_db, DemandSignal, upsert_demand_signal
    from config.settings import get_settings
except ImportError as e:
    print(f"Import error: {e}. Run from project root.")
    sys.exit(1)

settings = get_settings()

# ── Category → keyword mapping ────────────────────────────────────────
# One primary keyword per category — focused queries give better signal
CATEGORY_KEYWORDS = {
    "electronics": [
        "wireless headphones",
        "bluetooth speaker",
        "smart watch",
        "mechanical keyboard",
    ],
    "fashion": [
        "leather wallet",
        "tote bag",
        "minimalist watch",
        "wool beanie",
    ],
    "home_goods": [
        "bamboo cutting board",
        "stainless steel water bottle",
        "soy candle",
        "essential oil diffuser",
    ],
    "sports": [
        "yoga mat",
        "resistance bands",
        "running shoes",
        "foam roller",
    ],
}

VALID_CATEGORIES = list(CATEGORY_KEYWORDS.keys())

# ── File cache (Redis optional) ───────────────────────────────────────
class TrendsCache:
    """
    Simple file-based cache for trends data.
    TTL: 168 hours / 7 days (trends data is weekly — one fetch per week)
    """
    TTL_HOURS = 168  # 7 days

    def __init__(self):
        self.cache_dir = settings.cache_dir

    def _key(self, keyword: str) -> str:
        h = hashlib.sha256(f"trends:{keyword.lower().strip()}".encode()).hexdigest()[:16]
        return f"trends_{h}"

    def _path(self, keyword: str) -> Path:
        return self.cache_dir / f"{self._key(keyword)}.json"

    def get(self, keyword: str) -> Optional[list]:
        path = self._path(keyword)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
            if age_hours < self.TTL_HOURS:
                log.debug(f"Cache HIT (trends, age={age_hours:.1f}h): {keyword}")
                return data.get("rows", [])
        except Exception:
            pass
        return None

    def set(self, keyword: str, rows: list):
        path = self._path(keyword)
        try:
            path.write_text(json.dumps({
                "cached_at": datetime.utcnow().isoformat(),
                "keyword": keyword,
                "rows": rows,
            }))
        except Exception as e:
            log.warning(f"Cache write failed for '{keyword}': {e}")


# ── SerpAPI trends fetcher ────────────────────────────────────────────
class SerpAPITrendsFetcher:
    """
    Fetches Google Trends interest-over-time data via SerpAPI.

    Returns weekly data points: [{week_date, trend_index, keyword}, ...]
    trend_index range: 0-100 (100 = peak interest in timeframe)

    API docs: https://serpapi.com/google-trends-api
    """
    BASE_URL = "https://serpapi.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, keyword: str, timeframe: str = "today 12-m") -> list[dict]:
        """
        Fetch interest-over-time for a keyword.
        Returns list of weekly data rows.
        """
        params = {
            "engine":      "google_trends",
            "q":           keyword,
            "api_key":     self.api_key,
            "data_type":   "TIMESERIES",
            "tz":          "360",
            "geo":         settings.trends_region,
        }

        # Map pytrends timeframe format to SerpAPI format
        # "today 12-m" → date range for last 12 months
        date_range = self._timeframe_to_date_range(timeframe)
        if date_range:
            params["date"] = date_range

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            log.error(f"SerpAPI trends request failed for '{keyword}': {e}")
            return []
        except json.JSONDecodeError:
            log.error(f"SerpAPI trends parse error for '{keyword}'")
            return []

        if "error" in data:
            log.error(f"SerpAPI trends error for '{keyword}': {data['error']}")
            return []

        return self._parse_response(data, keyword)

    def _parse_response(self, data: dict, keyword: str) -> list[dict]:
        """Parse SerpAPI Google Trends response into row dicts."""
        rows = []

        # SerpAPI returns interest_over_time.timeline_data
        timeline = data.get("interest_over_time", {}).get("timeline_data", [])

        for point in timeline:
            try:
                # Date string: "Nov 27 – Dec 3, 2022" or ISO format
                date_str = point.get("date", "")
                week_date = self._parse_date(date_str)
                if week_date is None:
                    continue

                # Values is a list (one per keyword queried)
                values = point.get("values", [])
                if not values:
                    continue

                # Get value for our keyword (first in list)
                value_data = values[0]
                extracted = value_data.get("extracted_value", 0)
                trend_index = int(extracted) if extracted else 0

                rows.append({
                    "keyword":     keyword,
                    "trend_index": trend_index,
                    "week_date":   week_date,
                    "region":      settings.trends_region,
                })
            except Exception:
                continue

        log.info(f"SerpAPI trends '{keyword}': {len(rows)} weeks")
        return rows

    @staticmethod
    def _parse_date(date_str: str) -> Optional[date]:
        """
        Parse SerpAPI date strings to date objects.
        Handles formats like "Nov 27 - Dec 3, 2022" and "2022-11-27"
        """
        if not date_str:
            return None

        # Try ISO format first
        try:
            return date.fromisoformat(date_str[:10])
        except ValueError:
            pass

        # Try "Nov 27 – Dec 3, 2022" format — take the start date
        try:
            from dateutil import parser as dateutil_parser
            # Take the part before the dash
            start_part = date_str.split("-")[0].strip()
            # Add year if missing (take from end of full string)
            if not any(str(y) in start_part for y in range(2020, 2030)):
                year_match = date_str.split(",")[-1].strip()
                start_part = f"{start_part}, {year_match}"
            return dateutil_parser.parse(start_part).date()
        except Exception:
            pass

        return None

    @staticmethod
    def _timeframe_to_date_range(timeframe: str) -> Optional[str]:
        """
        Convert pytrends-style timeframe to SerpAPI date range.
        "today 12-m" → "2023-03-13 2024-03-13"
        """
        today = date.today()
        if "12-m" in timeframe:
            start = today - timedelta(days=365)
        elif "3-m" in timeframe:
            start = today - timedelta(days=90)
        elif "7-d" in timeframe:
            start = today - timedelta(days=7)
        else:
            return None
        return f"{start.isoformat()} {today.isoformat()}"


# ── Demand simulator for trends ───────────────────────────────────────
class TrendsSimulator:
    """
    Generates realistic weekly Google Trends data when SerpAPI is unavailable.

    Uses the same seasonal patterns as seed_db.py but generates them
    fresh on demand with controllable parameters.

    "Simulate demand signals using category-specific seasonal models
     seeded from historical Google Trends patterns. This generate
     statistically realistic training data without API costs, and
     the system switches to live SerpAPI data in production."
    """

    # Seasonal parameters per keyword — derived from real trend patterns
    KEYWORD_PARAMS = {
        "wireless headphones":         {"base": 60, "peak_month": 11, "amplitude": 25, "trend": 0.3},
        "bluetooth speaker":           {"base": 50, "peak_month": 7,  "amplitude": 20, "trend": 0.1},
        "smart watch":                 {"base": 55, "peak_month": 11, "amplitude": 22, "trend": 0.5},
        "mechanical keyboard":         {"base": 45, "peak_month": 11, "amplitude": 18, "trend": 0.2},
        "leather wallet":              {"base": 40, "peak_month": 11, "amplitude": 15, "trend": 0.0},
        "tote bag":                    {"base": 35, "peak_month": 6,  "amplitude": 18, "trend": 0.1},
        "minimalist watch":            {"base": 30, "peak_month": 4,  "amplitude": 14, "trend":-0.1},
        "wool beanie":                 {"base": 25, "peak_month": 10, "amplitude": 30, "trend": 0.0},
        "bamboo cutting board":        {"base": 40, "peak_month": 11, "amplitude": 20, "trend": 0.2},
        "stainless steel water bottle":{"base": 55, "peak_month": 6,  "amplitude": 22, "trend": 0.3},
        "soy candle":                  {"base": 35, "peak_month": 11, "amplitude": 25, "trend": 0.1},
        "essential oil diffuser":      {"base": 38, "peak_month": 11, "amplitude": 20, "trend": 0.1},
        "yoga mat":                    {"base": 65, "peak_month": 1,  "amplitude": 28, "trend": 0.4},
        "resistance bands":            {"base": 58, "peak_month": 1,  "amplitude": 25, "trend": 0.3},
        "running shoes":               {"base": 52, "peak_month": 3,  "amplitude": 20, "trend": 0.2},
        "foam roller":                 {"base": 45, "peak_month": 1,  "amplitude": 18, "trend": 0.1},
    }

    def generate(self, keyword: str, n_weeks: int = 52) -> list[dict]:
        """Generate n_weeks of realistic trend data for a keyword."""
        params = self.KEYWORD_PARAMS.get(keyword.lower(), {
            "base": 45, "peak_month": 11, "amplitude": 20, "trend": 0.0
        })

        np.random.seed(abs(hash(keyword)) % (2**31))
        today = date.today()
        weeks = [today - timedelta(weeks=(n_weeks - i)) for i in range(n_weeks)]

        rows = []
        for week_num, week_date in enumerate(weeks):
            t = week_num / n_weeks

            # Base + linear trend
            base = params["base"] + params["trend"] * week_num

            # Annual seasonality centered on peak_month
            peak_month = params["peak_month"]
            seasonal = params["amplitude"] * np.sin(
                2 * np.pi * (t - (peak_month - 1) / 12)
            )

            # Noise
            noise = np.random.normal(0, 3.5)

            index = int(np.clip(base + seasonal + noise, 0, 100))

            rows.append({
                "keyword":     keyword,
                "trend_index": index,
                "week_date":   week_date,
                "region":      settings.trends_region,
            })

        log.info(f"Simulated {len(rows)} weeks for '{keyword}'")
        return rows


# ── Orchestrator ──────────────────────────────────────────────────────
class TrendsFetcher:
    """
    Single entry point for demand signal ingestion.
    Handles: cache → SerpAPI → simulator fallback chain.
    """

    def __init__(self, demo_mode: Optional[bool] = None):
        self.demo_mode = demo_mode if demo_mode is not None else settings.demo_mode
        self.cache = TrendsCache()
        self.simulator = TrendsSimulator()

        if not self.demo_mode and settings.has_serpapi:
            self.serpapi = SerpAPITrendsFetcher(settings.serpapi_key)
            log.info("TrendsFetcher: SerpAPI enabled")
        else:
            self.serpapi = None
            mode = "DEMO MODE (simulator)" if self.demo_mode else "no SerpAPI key"
            log.info(f"TrendsFetcher: {mode}")

    def fetch_keyword(self, keyword: str, force_refresh: bool = False) -> list[dict]:
        """
        Fetch demand signal rows for one keyword.
        Returns: [{keyword, trend_index, week_date, region}, ...]
        """
        # 1. Check cache
        if not force_refresh:
            cached = self.cache.get(keyword)
            if cached:
                log.info(f"  '{keyword}': cache hit ({len(cached)} weeks)")
                return cached

        # 2. Try SerpAPI
        if not self.demo_mode and self.serpapi:
            rows = self.serpapi.fetch(keyword)
            if rows:
                self.cache.set(keyword, rows)
                return rows
            log.warning(f"  SerpAPI returned no data for '{keyword}' — falling back to simulator")

        # 3. Simulate
        rows = self.simulator.generate(keyword)
        # Serialize week_date to string for JSON cache compatibility
        serializable = [
            {**r, "week_date": r["week_date"].isoformat()
             if hasattr(r["week_date"], "isoformat") else r["week_date"]}
            for r in rows
        ]
        self.cache.set(keyword, serializable)
        return rows

    def fetch_category(
        self,
        category: str,
        dry_run: bool = False,
        sleep_between: float = 2.0,
        force_refresh: bool = False,
    ) -> dict:
        """Fetch all keywords for a category."""
        keywords = CATEGORY_KEYWORDS.get(category)
        if not keywords:
            log.error(f"Unknown category: '{category}'")
            return {"category": category, "rows": 0, "errors": 1}

        log.info(f"\n{'='*50}")
        log.info(f"Category: {category.upper()} ({len(keywords)} keywords)")
        log.info(f"{'='*50}")

        total_rows = 0
        errors = 0

        for i, keyword in enumerate(keywords):
            try:
                rows = self.fetch_keyword(keyword, force_refresh=force_refresh)
                saved = self._save(rows, dry_run=dry_run)
                total_rows += saved

                # Log latest index
                if rows:
                    latest = rows[-1]
                    idx = latest.get("trend_index", "?") if isinstance(latest, dict) else "?"
                    log.info(f"  OK '{keyword}': {len(rows)} weeks, latest index={idx}")

            except Exception as e:
                errors += 1
                log.error(f"  Error for '{keyword}': {e}")

            if i < len(keywords) - 1 and not self.demo_mode:
                time.sleep(sleep_between)

        return {"category": category, "rows": total_rows, "errors": errors}

    @staticmethod
    def _save(rows: list, dry_run: bool = False) -> int:
        """Upsert rows into demand_signals table."""
        if not rows or dry_run:
            if dry_run and rows:
                log.info(f"  [DRY RUN] Would upsert {len(rows)} rows")
            return 0

        # Normalize trend_index to 0-100 before writing.
        # SerpAPI sometimes returns raw interest values (e.g. 4200) instead of
        # the normalized 0-100 index. Normalize at write time so the DB is
        # always clean regardless of what the API returns.
        raw_indices = [r["trend_index"] for r in rows if isinstance(r.get("trend_index"), (int, float))]
        max_index = max(raw_indices) if raw_indices else 100
        needs_normalize = max_index > 100
        if needs_normalize:
            log.warning(
                f"  trend_index max={max_index:.0f} exceeds 100 — "
                "normalizing to 0-100 before saving"
            )

        with get_db() as db:
            for r in rows:
                week_date = r["week_date"]
                if isinstance(week_date, str):
                    week_date = date.fromisoformat(week_date[:10])

                raw_idx = r["trend_index"]
                if needs_normalize and isinstance(raw_idx, (int, float)) and max_index > 0:
                    trend_index = int(round(raw_idx / max_index * 100))
                else:
                    trend_index = int(raw_idx) if raw_idx is not None else 0

                trend_index = max(0, min(100, trend_index))

                upsert_demand_signal(
                    db,
                    keyword=r["keyword"],
                    trend_index=trend_index,
                    week_date=week_date,
                    region=r.get("region", "US"),
                )

        return len(rows)


# ── Main ──────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="ARIA demand signal ingestion (SerpAPI + simulator)"
    )
    parser.add_argument("--category", "-c", choices=VALID_CATEGORIES, default=None)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--demo",     action="store_true", help="Force simulator mode")
    parser.add_argument("--refresh",  action="store_true", help="Ignore cache")
    parser.add_argument("--sleep",    type=float, default=2.0,
                        help="Seconds between SerpAPI calls (default: 2.0)")
    args = parser.parse_args()

    demo_mode = args.demo or settings.demo_mode
    fetcher = TrendsFetcher(demo_mode=demo_mode)

    mode_label = "DEMO (simulator)" if demo_mode else "LIVE (SerpAPI)"
    log.info(f"ARIA Trends Fetcher — {mode_label}")

    categories = [args.category] if args.category else VALID_CATEGORIES

    start = datetime.now()
    results = []

    for category in categories:
        result = fetcher.fetch_category(
            category=category,
            dry_run=args.dry_run,
            sleep_between=args.sleep,
            force_refresh=args.refresh,
        )
        results.append(result)

    elapsed = int((datetime.now() - start).total_seconds())
    total_rows = sum(r["rows"] for r in results)
    total_errors = sum(r["errors"] for r in results)

    log.info(f"\n{'='*50}")
    log.info("SUMMARY")
    log.info(f"{'='*50}")
    for r in results:
        icon = "OK" if r["errors"] == 0 else "!!"
        log.info(f"  [{icon}] {r['category']:<15} {r['rows']:>4} rows  {r['errors']} errors")
    log.info(f"\n  Total rows : {total_rows}")
    log.info(f"  Errors     : {total_errors}")
    log.info(f"  Elapsed    : {elapsed}s")

    if not args.dry_run and total_rows > 0:
        with get_db() as db:
            from db.models import DemandSignal
            count = db.query(DemandSignal).count()
            log.info(f"\n  DB total demand_signals: {count:,}")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())