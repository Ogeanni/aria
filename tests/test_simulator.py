"""
tests/test_simulator.py
Tests for src/price_simulator.py

The simulator is the fallback data layer — if it produces bad data,
every downstream component (features, model, agent) is affected.
These tests verify it produces statistically valid output.
"""
import pytest
from src.price_simulator import PriceSimulator, simulate_prices


class TestPriceSimulator:

    def setup_method(self):
        self.sim = PriceSimulator(seed=42)

    # ── Basic output validation ───────────────────────────────────────

    def test_returns_listings(self):
        band = self.sim.get_price_band("yoga mat", "sports", 29.99)
        assert band["count"] > 0
        assert len(band["listings"]) > 0

    def test_listings_are_dicts(self):
        """Listings must be plain dicts — not dataclass objects."""
        band = self.sim.get_price_band("yoga mat", "sports", 29.99)
        for listing in band["listings"]:
            assert isinstance(listing, dict), "Listings must be dicts, not dataclass objects"
            assert "price" in listing
            assert "retailer" in listing
            assert "title" in listing

    def test_prices_are_positive(self):
        band = self.sim.get_price_band("wireless headphones", "electronics", 89.99)
        for listing in band["listings"]:
            assert listing["price"] > 0

    def test_band_statistics_are_ordered(self):
        band = self.sim.get_price_band("leather wallet", "fashion", 49.99)
        assert band["min"] <= band["p25"] <= band["median"] <= band["p75"] <= band["max"]

    def test_median_near_our_price(self):
        """Market median should be in a reasonable range of our price."""
        our_price = 50.00
        band = self.sim.get_price_band("tote bag", "fashion", our_price)
        # Median should be within 50% of our price — not wildly different
        ratio = band["median"] / our_price
        assert 0.5 <= ratio <= 2.0, f"Median ${band['median']} too far from ${our_price}"

    # ── Reproducibility ───────────────────────────────────────────────

    def test_deterministic_with_seed(self):
        """Same seed + keyword + price should always produce same result."""
        sim1 = PriceSimulator(seed=42)
        sim2 = PriceSimulator(seed=42)
        band1 = sim1.get_price_band("yoga mat", "sports", 29.99)
        band2 = sim2.get_price_band("yoga mat", "sports", 29.99)
        assert band1["median"] == band2["median"]
        assert band1["count"] == band2["count"]

    # ── Category coverage ─────────────────────────────────────────────

    @pytest.mark.parametrize("category,keyword,price", [
        ("electronics", "wireless headphones", 89.99),
        ("fashion",     "leather wallet",      49.99),
        ("home_goods",  "bamboo cutting board", 29.99),
        ("sports",      "yoga mat",             24.99),
        ("unknown",     "random product",       39.99),  # fallback distribution
    ])
    def test_all_categories_produce_output(self, category, keyword, price):
        band = self.sim.get_price_band(keyword, category, price)
        assert band["count"] > 0
        assert band["median"] > 0

    # ── Price constraints ─────────────────────────────────────────────

    def test_prices_not_negative(self):
        """Even with very low input prices, output should not go negative."""
        band = self.sim.get_price_band("cheap item", "home_goods", 0.99)
        for listing in band["listings"]:
            assert listing["price"] >= 0

    def test_demand_modifier_raises_prices(self):
        """High demand index should produce higher median than low demand."""
        low_demand  = self.sim.get_price_band("yoga mat", "sports", 30.00, demand_index=10)
        high_demand = self.sim.get_price_band("yoga mat", "sports", 30.00, demand_index=90)
        assert high_demand["median"] >= low_demand["median"]

    # ── Module-level convenience function ────────────────────────────

    def test_simulate_prices_function(self):
        band = simulate_prices("foam roller", "sports", 19.99)
        assert "median" in band
        assert "listings" in band
        assert band["source"] == "simulated"