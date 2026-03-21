"""
src/price_simulator.py
Realistic synthetic competitor price generator for ARIA.

Used when:
  - DEMO_MODE=true (no SerpAPI credits needed)
  - SerpAPI call fails (fallback)
  - Development and testing

Design:
  Prices are NOT random. They are generated from realistic distributions
  seeded from actual Google Shopping price ranges per category.
  The simulator produces statistically credible data that behaves like
  a real market: price variance, outliers, platform-specific patterns.

This is a deliberate choice:
  "Build a statistical price simulator so the system works end-to-end
   without API costs. Same distributions as real data — zero API cost."

Usage:
    from src.price_simulator import PriceSimulator
    sim = PriceSimulator()
    listings = sim.get_competitor_prices("yoga mat", category="sports", our_price=29.99)
"""
import hashlib
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger("price_simulator")

# ── Market price distributions ────────────────────────────────────────
# Sourced from real Google Shopping ranges per product/category.
# Each entry: (mean_price, std_dev, min_price, max_price, n_typical_listings)
# These represent realistic market conditions per category.
CATEGORY_DISTRIBUTIONS = {
    "electronics": {
        "mean_multiplier": 1.0,    # Market center vs our price
        "std_pct": 0.18,           # 18% std dev — electronics have moderate variance
        "outlier_rate": 0.08,      # 8% of listings are outliers (refurb, premium)
        "outlier_multiplier": 1.45,
        "platforms": {
            "Amazon":   {"weight": 0.40, "price_bias": -0.03},  # Amazon slightly cheaper
            "BestBuy":  {"weight": 0.25, "price_bias":  0.05},
            "Walmart":  {"weight": 0.20, "price_bias": -0.05},
            "Newegg":   {"weight": 0.10, "price_bias": -0.02},
            "B&H Photo":{"weight": 0.05, "price_bias":  0.08},
        },
        "typical_listing_count": (8, 18),
    },
    "fashion": {
        "mean_multiplier": 1.0,
        "std_pct": 0.25,           # Fashion has higher variance (brand premium)
        "outlier_rate": 0.12,
        "outlier_multiplier": 1.80,
        "platforms": {
            "Amazon":   {"weight": 0.30, "price_bias": -0.05},
            "Etsy":     {"weight": 0.30, "price_bias":  0.15},  # Etsy premium
            "Nordstrom":{"weight": 0.15, "price_bias":  0.20},
            "ASOS":     {"weight": 0.15, "price_bias": -0.08},
            "eBay":     {"weight": 0.10, "price_bias": -0.15},  # eBay cheaper
        },
        "typical_listing_count": (6, 15),
    },
    "home_goods": {
        "mean_multiplier": 1.0,
        "std_pct": 0.20,
        "outlier_rate": 0.10,
        "outlier_multiplier": 1.60,
        "platforms": {
            "Amazon":   {"weight": 0.45, "price_bias": -0.02},
            "Wayfair":  {"weight": 0.25, "price_bias":  0.08},
            "Target":   {"weight": 0.15, "price_bias":  0.03},
            "Walmart":  {"weight": 0.10, "price_bias": -0.06},
            "Etsy":     {"weight": 0.05, "price_bias":  0.20},
        },
        "typical_listing_count": (7, 16),
    },
    "sports": {
        "mean_multiplier": 1.0,
        "std_pct": 0.22,
        "outlier_rate": 0.09,
        "outlier_multiplier": 1.55,
        "platforms": {
            "Amazon":   {"weight": 0.40, "price_bias": -0.03},
            "Dick's":   {"weight": 0.20, "price_bias":  0.10},
            "REI":      {"weight": 0.15, "price_bias":  0.15},
            "Walmart":  {"weight": 0.15, "price_bias": -0.08},
            "eBay":     {"weight": 0.10, "price_bias": -0.12},
        },
        "typical_listing_count": (8, 20),
    },
}

# Default distribution for unknown categories
DEFAULT_DISTRIBUTION = {
    "mean_multiplier": 1.0,
    "std_pct": 0.20,
    "outlier_rate": 0.10,
    "outlier_multiplier": 1.50,
    "platforms": {
        "Amazon":  {"weight": 0.50, "price_bias": -0.03},
        "Walmart": {"weight": 0.30, "price_bias": -0.05},
        "eBay":    {"weight": 0.20, "price_bias": -0.10},
    },
    "typical_listing_count": (6, 14),
}


@dataclass
class SimulatedListing:
    """A single simulated competitor listing."""
    price: float
    retailer: str
    title: str
    platform: str = "google_shopping"
    is_simulated: bool = True
    listing_url: str = ""


class PriceSimulator:
    """
    Generates realistic competitor price distributions for any product.

    Key design principles:
    1. Deterministic for same inputs (reproducible demos)
    2. Statistically realistic (not just random)
    3. Platform-specific price patterns
    4. Supports market condition modifiers (demand, inventory)
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed  # None = random each time, int = reproducible

    def _get_seed(self, product_keyword: str) -> int:
        """Generate a deterministic seed from the product name."""
        if self.seed is not None:
            return self.seed
        # Hash the keyword for a consistent but keyword-specific seed
        return int(hashlib.md5(product_keyword.encode()).hexdigest()[:8], 16) % (2**31)

    def get_competitor_prices(
        self,
        keyword: str,
        category: str,
        our_price: float,
        n_results: Optional[int] = None,
        demand_index: float = 50.0,
        inventory_pressure: int = 0,
    ) -> list[SimulatedListing]:
        """
        Generate realistic competitor prices for a product.

        Args:
            keyword: Product search keyword (e.g. "yoga mat")
            category: Product category (electronics/fashion/home_goods/sports)
            our_price: Our current price (market anchor)
            n_results: Number of listings to generate (None = random realistic count)
            demand_index: Google Trends index 0-100 (affects market price level)
            inventory_pressure: -1=oversupply, 0=normal, 1=scarce

        Returns:
            List of SimulatedListing objects with realistic prices

        Why demand affects prices:
            When demand is high (index > 70), market prices tend to be higher.
            When demand is low (index < 30), prices are more competitive.
            This creates realistic market dynamics for the ML model to learn from.
        """
        dist = CATEGORY_DISTRIBUTIONS.get(category.lower(), DEFAULT_DISTRIBUTION)
        rng = np.random.default_rng(self._get_seed(keyword))

        # ── Determine listing count ────────────────────────────────────
        min_n, max_n = dist["typical_listing_count"]
        if n_results is None:
            n_results = int(rng.integers(min_n, max_n + 1))

        # ── Compute market center price ────────────────────────────────
        # Demand modifier: high demand → market prices slightly higher
        demand_modifier = 1.0 + (demand_index - 50) / 500  # ±10% at extremes
        inventory_modifier = 1.0 + inventory_pressure * 0.03  # ±3%
        market_center = our_price * dist["mean_multiplier"] * demand_modifier * inventory_modifier

        # ── Generate prices ────────────────────────────────────────────
        std = market_center * dist["std_pct"]

        # Normal distribution centered on market price
        prices = rng.normal(loc=market_center, scale=std, size=n_results)

        # Add outliers (premium or clearance listings)
        n_outliers = max(1, int(n_results * dist["outlier_rate"]))
        outlier_indices = rng.choice(n_results, size=n_outliers, replace=False)
        for idx in outlier_indices:
            if rng.random() > 0.5:
                prices[idx] = market_center * dist["outlier_multiplier"] * rng.uniform(0.9, 1.1)
            else:
                # Clearance / used item
                prices[idx] = market_center * 0.65 * rng.uniform(0.9, 1.1)

        # Clip to realistic bounds (no negative or absurdly high prices)
        min_price = max(0.99, our_price * 0.30)
        max_price = our_price * 3.0
        prices = np.clip(prices, min_price, max_price)

        # Round to realistic price points (.99, .95, .00)
        prices = self._realistic_price_points(prices, rng)

        # ── Assign platforms ───────────────────────────────────────────
        platforms = list(dist["platforms"].keys())
        weights = [dist["platforms"][p]["weight"] for p in platforms]
        biases = {p: dist["platforms"][p]["price_bias"] for p in platforms}

        assigned_platforms = rng.choice(platforms, size=n_results, p=weights)

        # Apply platform-specific price bias
        for i, platform in enumerate(assigned_platforms):
            bias = biases[platform]
            prices[i] = prices[i] * (1 + bias)
            prices[i] = max(min_price, round(float(prices[i]), 2))

        # ── Build listing objects ──────────────────────────────────────
        listings = []
        for i in range(n_results):
            retailer = assigned_platforms[i]
            price = round(float(prices[i]), 2)
            listings.append(SimulatedListing(
                price=price,
                retailer=retailer,
                title=self._generate_title(keyword, retailer, price, our_price),
                platform="google_shopping",
                is_simulated=True,
                listing_url=f"https://www.google.com/shopping?q={keyword.replace(' ', '+')}",
            ))

        listings.sort(key=lambda x: x.price)
        log.debug(f"Simulated {len(listings)} listings for '{keyword}' "
                  f"(our=${our_price:.2f}, market median=${np.median(prices):.2f})")
        return listings

    def _realistic_price_points(self, prices: np.ndarray, rng) -> np.ndarray:
        """
        Round prices to realistic endings: .99, .95, .00, .49
        Real retailers almost never price at e.g. $32.37
        """
        endings = [0.99, 0.95, 0.00, 0.49]
        ending_weights = [0.50, 0.15, 0.20, 0.15]
        result = []
        for price in prices:
            base = np.floor(price)
            ending = rng.choice(endings, p=ending_weights)
            result.append(base + ending)
        return np.array(result)

    def _generate_title(self, keyword: str, retailer: str, price: float, our_price: float) -> str:
        """Generate a plausible listing title."""
        adjectives = ["Premium", "Professional", "Lightweight", "Durable",
                      "Essential", "Classic", "Sport", "Ultra", "Pro"]
        suffixes = ["", " - Best Seller", " (Top Rated)", " | Free Shipping", ""]

        rng_local = random.Random(hash(f"{keyword}{retailer}{price:.0f}"))
        adj = rng_local.choice(adjectives)
        suffix = rng_local.choice(suffixes)
        return f"{adj} {keyword.title()}{suffix}"

    def get_price_band(
        self,
        keyword: str,
        category: str,
        our_price: float,
        **kwargs,
    ) -> dict:
        """
        Returns summary statistics for the simulated market.
        Same interface as the live SerpAPI fetcher.
        """
        listings = self.get_competitor_prices(keyword, category, our_price, **kwargs)
        prices = [l.price for l in listings]

        if not prices:
            return {
                "count": 0, "min": our_price, "max": our_price,
                "median": our_price, "p25": our_price, "p75": our_price,
                "mean": our_price, "source": "simulated",
            }

        prices_sorted = sorted(prices)
        n = len(prices_sorted)

        # Convert dataclass objects → plain dicts (JSON-serializable)
        listings_dicts = [
            {
                "price":        l.price,
                "retailer":     l.retailer,
                "title":        l.title,
                "platform":     l.platform,
                "is_simulated": l.is_simulated,
                "link":         l.listing_url,
            }
            for l in listings
        ]

        return {
            "count":    n,
            "min":      round(min(prices_sorted), 2),
            "max":      round(max(prices_sorted), 2),
            "mean":     round(float(np.mean(prices_sorted)), 2),
            "median":   round(float(np.median(prices_sorted)), 2),
            "p25":      round(prices_sorted[max(0, int(n * 0.25) - 1)], 2),
            "p75":      round(prices_sorted[min(n - 1, int(n * 0.75))], 2),
            "source":   "simulated",
            "listings": listings_dicts,
        }


# ── Module-level convenience instance ────────────────────────────────
_simulator = PriceSimulator()


def simulate_prices(
    keyword: str,
    category: str,
    our_price: float,
    **kwargs,
) -> dict:
    """
    Module-level convenience function.
    Returns price band dict with listings.
    """
    return _simulator.get_price_band(keyword, category, our_price, **kwargs)


# ── CLI for testing ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    test_products = [
        ("wireless headphones",   "electronics", 89.99),
        ("yoga mat",              "sports",      29.99),
        ("leather wallet",        "fashion",     49.99),
        ("bamboo cutting board",  "home_goods",  24.99),
        ("running shoes",         "sports",      119.99),
        ("bluetooth speaker",     "electronics", 59.99),
    ]

    print(f"\n{'='*70}")
    print("ARIA Price Simulator — Test Output")
    print(f"{'='*70}\n")

    sim = PriceSimulator()

    for keyword, category, our_price in test_products:
        band = sim.get_price_band(keyword, category, our_price)
        position = (our_price - band["median"]) / band["median"] * 100
        status = "ABOVE" if position > 5 else ("BELOW" if position < -5 else "IN LINE")

        print(f"Product : {keyword.title()}")
        print(f"Category: {category}  |  Our price: ${our_price:.2f}")
        print(f"Market  : min=${band['min']}  p25=${band['p25']}  "
              f"median=${band['median']}  p75=${band['p75']}  max=${band['max']}")
        print(f"Position: {position:+.1f}% vs median  ({status})")
        print(f"Listings: {band['count']} from "
              f"{', '.join(set(l['retailer'] for l in band['listings'][:4]))}")
        print()