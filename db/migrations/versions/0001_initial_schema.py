"""Initial schema — all ARIA tables

Revision ID: 0001
Revises:
Create Date: 2026-03-18 00:00:00

This migration creates the complete initial database schema for ARIA.
Tables:
  - products           : product catalog with current pricing
  - price_history      : every price change ever made (audit trail)
  - competitor_prices  : scraped / simulated competitor prices
  - demand_signals     : Google Trends data per keyword
  - agent_decisions    : every decision the agent made (audit log)
  - repricing_outcomes : outcome labels for feedback loop
  - approval_queue     : price changes awaiting human review
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── products ──────────────────────────────────────────────────────
    op.create_table(
        'products',
        sa.Column('id',            sa.Integer(),                    nullable=False),
        sa.Column('name',          sa.String(length=200),           nullable=False),
        sa.Column('sku',           sa.String(length=100),           nullable=True),
        sa.Column('category',      sa.String(length=100),           nullable=False),
        sa.Column('base_price',    sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('current_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('min_price',     sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('max_price',     sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('inventory_qty', sa.Integer(),                    nullable=True),
        sa.Column('is_active',     sa.Boolean(),                    nullable=True),
        sa.Column('platform',      sa.String(length=50),            nullable=True),
        sa.Column('external_id',   sa.String(length=200),           nullable=True),
        sa.Column('created_at',    sa.DateTime(),                   nullable=True),
        sa.Column('updated_at',    sa.DateTime(),                   nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sku'),
    )
    op.create_index('ix_products_category', 'products', ['category'])
    op.create_index('ix_products_id',       'products', ['id'], unique=True)

    # ── price_history ─────────────────────────────────────────────────
    op.create_table(
        'price_history',
        sa.Column('id',          sa.Integer(),                      nullable=False),
        sa.Column('product_id',  sa.Integer(),                      nullable=False),
        sa.Column('old_price',   sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('new_price',   sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('change_pct',  sa.Float(),                        nullable=False),
        sa.Column('source',      sa.String(length=50),              nullable=False),
        sa.Column('decision_id', sa.Integer(),                      nullable=True),
        sa.Column('recorded_at', sa.DateTime(),                     nullable=True),
        sa.Column('note',        sa.Text(),                         nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_price_history_id',           'price_history', ['id'], unique=True)
    op.create_index('ix_price_history_product_id',   'price_history', ['product_id'])
    op.create_index('ix_price_history_recorded_at',  'price_history', ['recorded_at'])
    op.create_index('ix_price_history_product_date', 'price_history', ['product_id', 'recorded_at'])

    # ── competitor_prices ─────────────────────────────────────────────
    op.create_table(
        'competitor_prices',
        sa.Column('id',               sa.Integer(),                      nullable=False),
        sa.Column('product_id',       sa.Integer(),                      nullable=False),
        sa.Column('platform',         sa.String(length=100),             nullable=False),
        sa.Column('retailer',         sa.String(length=200),             nullable=True),
        sa.Column('competitor_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('listing_title',    sa.String(length=500),             nullable=True),
        sa.Column('listing_url',      sa.Text(),                         nullable=True),
        sa.Column('is_simulated',     sa.Boolean(),                      nullable=True),
        sa.Column('scraped_at',       sa.DateTime(),                     nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_competitor_prices_id',           'competitor_prices', ['id'], unique=True)
    op.create_index('ix_competitor_prices_product_id',   'competitor_prices', ['product_id'])
    op.create_index('ix_competitor_prices_scraped_at',   'competitor_prices', ['scraped_at'])
    op.create_index('ix_competitor_product_date',        'competitor_prices', ['product_id', 'scraped_at'])

    # ── demand_signals ────────────────────────────────────────────────
    op.create_table(
        'demand_signals',
        sa.Column('id',          sa.Integer(),      nullable=False),
        sa.Column('keyword',     sa.String(length=200), nullable=False),
        sa.Column('trend_index', sa.Integer(),      nullable=False),
        sa.Column('week_date',   sa.DateTime(),     nullable=False),
        sa.Column('region',      sa.String(length=10), nullable=True),
        sa.Column('created_at',  sa.DateTime(),     nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_demand_signals_id',       'demand_signals', ['id'], unique=True)
    op.create_index('ix_demand_signals_keyword',  'demand_signals', ['keyword'])
    op.create_index('ix_demand_keyword_date',     'demand_signals', ['keyword', 'week_date'])

    # ── agent_decisions ───────────────────────────────────────────────
    op.create_table(
        'agent_decisions',
        sa.Column('id',                sa.Integer(),                      nullable=False),
        sa.Column('product_id',        sa.Integer(),                      nullable=False),
        sa.Column('decision_type',     sa.String(length=20),              nullable=False),
        sa.Column('decision_source',   sa.String(length=20),              nullable=False),
        sa.Column('current_price',     sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('recommended_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('change_pct',        sa.Float(),                        nullable=False),
        sa.Column('competitor_median', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('trend_index',       sa.Integer(),                      nullable=True),
        sa.Column('inventory_qty',     sa.Integer(),                      nullable=True),
        sa.Column('confidence',        sa.String(length=20),              nullable=True),
        sa.Column('reasoning',         sa.Text(),                         nullable=True),
        sa.Column('was_executed',      sa.Boolean(),                      nullable=True),
        sa.Column('executed_at',       sa.DateTime(),                     nullable=True),
        sa.Column('execution_error',   sa.Text(),                         nullable=True),
        sa.Column('requires_approval', sa.Boolean(),                      nullable=True),
        sa.Column('approval_status',   sa.String(length=20),              nullable=True),
        sa.Column('created_at',        sa.DateTime(),                     nullable=True),
        sa.ForeignKeyConstraint(['product_id'], ['products.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_agent_decisions_id',           'agent_decisions', ['id'], unique=True)
    op.create_index('ix_agent_decisions_product_id',   'agent_decisions', ['product_id'])
    op.create_index('ix_agent_decisions_created_at',   'agent_decisions', ['created_at'])
    op.create_index('ix_decisions_product_date',       'agent_decisions', ['product_id', 'created_at'])

    # ── repricing_outcomes ────────────────────────────────────────────
    op.create_table(
        'repricing_outcomes',
        sa.Column('id',                    sa.Integer(),                      nullable=False),
        sa.Column('product_id',            sa.Integer(),                      nullable=False),
        sa.Column('decision_id',           sa.Integer(),                      nullable=False),
        sa.Column('price_before',          sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('price_after',           sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('units_sold_before_7d',  sa.Float(),                        nullable=True),
        sa.Column('units_sold_after_7d',   sa.Float(),                        nullable=True),
        sa.Column('revenue_before_7d',     sa.Float(),                        nullable=True),
        sa.Column('revenue_after_7d',      sa.Float(),                        nullable=True),
        sa.Column('outcome_label',         sa.Integer(),                      nullable=True),
        sa.Column('outcome_notes',         sa.Text(),                         nullable=True),
        sa.Column('measured_at',           sa.DateTime(),                     nullable=True),
        sa.Column('created_at',            sa.DateTime(),                     nullable=True),
        sa.ForeignKeyConstraint(['decision_id'], ['agent_decisions.id']),
        sa.ForeignKeyConstraint(['product_id'],  ['products.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_repricing_outcomes_id', 'repricing_outcomes', ['id'], unique=True)

    # ── approval_queue ────────────────────────────────────────────────
    op.create_table(
        'approval_queue',
        sa.Column('id',            sa.Integer(),                      nullable=False),
        sa.Column('decision_id',   sa.Integer(),                      nullable=False),
        sa.Column('product_id',    sa.Integer(),                      nullable=False),
        sa.Column('current_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('proposed_price',sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('change_pct',    sa.Float(),                        nullable=False),
        sa.Column('reasoning',     sa.Text(),                         nullable=True),
        sa.Column('status',        sa.String(length=20),              nullable=True),
        sa.Column('reviewed_by',   sa.String(length=200),             nullable=True),
        sa.Column('reviewed_at',   sa.DateTime(),                     nullable=True),
        sa.Column('review_note',   sa.Text(),                         nullable=True),
        sa.Column('expires_at',    sa.DateTime(),                     nullable=True),
        sa.Column('created_at',    sa.DateTime(),                     nullable=True),
        sa.ForeignKeyConstraint(['decision_id'], ['agent_decisions.id']),
        sa.ForeignKeyConstraint(['product_id'],  ['products.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_approval_queue_id',         'approval_queue', ['id'], unique=True)
    op.create_index('ix_approval_queue_created_at', 'approval_queue', ['created_at'])


def downgrade() -> None:
    """
    Roll back the initial migration — drops all tables.
    WARNING: This destroys all data. Only use in development.
    In production, never roll back migration 0001.
    """
    op.drop_table('approval_queue')
    op.drop_table('repricing_outcomes')
    op.drop_table('agent_decisions')
    op.drop_table('demand_signals')
    op.drop_table('competitor_prices')
    op.drop_table('price_history')
    op.drop_table('products')