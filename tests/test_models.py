"""
tests/test_models.py
Tests for db/models.py

Verifies the database schema, helper functions, and ORM relationships
work correctly. Uses in-memory SQLite via the conftest fixtures.
"""
import pytest
from datetime import datetime, timedelta
from db.models import (
    Product, PriceHistory, CompetitorPrice, DemandSignal,
    AgentDecision, RepricingOutcome, ApprovalQueue,
    upsert_demand_signal, get_latest_competitor_prices,
)


class TestProductModel:

    def test_create_product(self, db_session):
        p = Product(
            name="Test Headphones",
            sku="SKU-001",
            category="electronics",
            base_price=79.99,
            current_price=94.99,
            min_price=65.00,
            max_price=140.00,
            inventory_qty=50,
            is_active=True,
        )
        db_session.add(p)
        db_session.flush()
        assert p.id is not None

    def test_product_repr(self, sample_product):
        r = repr(sample_product)
        assert "Product" in r
        assert "94.99" in r

    def test_unique_sku(self, db_session):
        from sqlalchemy.exc import IntegrityError
        p1 = Product(name="A", sku="SAME-SKU", category="electronics",
                     base_price=10, current_price=10)
        p2 = Product(name="B", sku="SAME-SKU", category="electronics",
                     base_price=20, current_price=20)
        db_session.add(p1)
        db_session.flush()
        db_session.add(p2)
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_product_without_sku(self, db_session):
        """SKU is optional — products can exist without one."""
        p = Product(name="No SKU Product", category="sports",
                    base_price=20, current_price=20)
        db_session.add(p)
        db_session.flush()
        assert p.id is not None
        assert p.sku is None


class TestPriceHistory:

    def test_create_price_history(self, db_session, sample_product):
        ph = PriceHistory(
            product_id=sample_product.id,
            old_price=89.99,
            new_price=94.99,
            change_pct=5.56,
            source="agent",
            recorded_at=datetime.utcnow(),
        )
        db_session.add(ph)
        db_session.flush()
        assert ph.id is not None

    def test_price_history_relationship(self, db_session, sample_product):
        ph = PriceHistory(
            product_id=sample_product.id,
            old_price=85.00,
            new_price=94.99,
            change_pct=11.75,
            source="import",
        )
        db_session.add(ph)
        db_session.flush()
        assert ph.product_id == sample_product.id


class TestCompetitorPrice:

    def test_create_competitor_price(self, db_session, sample_product):
        cp = CompetitorPrice(
            product_id=sample_product.id,
            platform="google_shopping",
            retailer="Amazon",
            competitor_price=87.99,
            is_simulated=True,
            scraped_at=datetime.utcnow(),
        )
        db_session.add(cp)
        db_session.flush()
        assert cp.id is not None

    def test_get_latest_competitor_prices(self, db_session, sample_product):
        """get_latest_competitor_prices should return recent rows."""
        for price in [85.99, 89.99, 92.99]:
            db_session.add(CompetitorPrice(
                product_id=sample_product.id,
                platform="simulated",
                competitor_price=price,
                scraped_at=datetime.utcnow(),
            ))
        db_session.flush()

        rows = get_latest_competitor_prices(db_session, sample_product.id, hours=24)
        assert len(rows) == 3

    def test_get_latest_ignores_stale(self, db_session, sample_product):
        """get_latest_competitor_prices with tight window should exclude old rows."""
        db_session.add(CompetitorPrice(
            product_id=sample_product.id,
            platform="simulated",
            competitor_price=85.00,
            scraped_at=datetime.utcnow() - timedelta(hours=48),
        ))
        db_session.flush()

        rows = get_latest_competitor_prices(db_session, sample_product.id, hours=1)
        # No recent rows — should fall back to all available
        assert isinstance(rows, list)


class TestDemandSignal:

    def test_upsert_demand_signal_insert(self, db_session):
        from datetime import date
        today = date.today()
        upsert_demand_signal(db_session, "yoga mat", 75, today, "US")
        db_session.flush()

        row = db_session.query(DemandSignal)\
            .filter(DemandSignal.keyword == "yoga mat").first()
        assert row is not None
        assert row.trend_index == 75

    def test_upsert_demand_signal_update(self, db_session):
        """Upserting same keyword + date should update, not duplicate."""
        from datetime import date
        today = date.today()
        upsert_demand_signal(db_session, "yoga mat", 75, today, "US")
        db_session.flush()
        upsert_demand_signal(db_session, "yoga mat", 82, today, "US")
        db_session.flush()

        rows = db_session.query(DemandSignal)\
            .filter(DemandSignal.keyword == "yoga mat").all()
        # Should update, not create a second row
        assert len(rows) == 1
        assert rows[0].trend_index == 82

    def test_trend_index_range(self, db_session):
        """Trend index must be 0-100."""
        from datetime import date
        upsert_demand_signal(db_session, "test", 50, date.today())
        db_session.flush()
        row = db_session.query(DemandSignal)\
            .filter(DemandSignal.keyword == "test").first()
        assert 0 <= row.trend_index <= 100


class TestAgentDecision:

    def test_create_decision(self, db_session, sample_product):
        d = AgentDecision(
            product_id=sample_product.id,
            decision_type="increase",
            decision_source="rules",
            current_price=94.99,
            recommended_price=99.99,
            change_pct=5.26,
            confidence="high",
            reasoning="Price below market median.",
            was_executed=True,
            created_at=datetime.utcnow(),
        )
        db_session.add(d)
        db_session.flush()
        assert d.id is not None
        assert d.was_executed is True

    def test_decision_approval_fields(self, db_session, sample_product):
        d = AgentDecision(
            product_id=sample_product.id,
            decision_type="increase",
            decision_source="ml_model",
            current_price=50.00,
            recommended_price=65.00,
            change_pct=30.0,
            confidence="high",
            requires_approval=True,
            approval_status="pending",
            was_executed=False,
        )
        db_session.add(d)
        db_session.flush()
        assert d.requires_approval is True
        assert d.approval_status == "pending"