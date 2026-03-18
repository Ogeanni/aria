"""
db/migrations/env.py
Alembic migration environment for ARIA.

Configured to:
  - Read database URL from ARIA's settings (not a hardcoded alembic.ini value)
  - Import all SQLAlchemy models so Alembic can detect schema changes
  - Support both offline (SQL script) and online (live DB) migration modes

Usage:
    alembic revision --autogenerate -m "describe change"  # create migration
    alembic upgrade head                                   # apply all pending
    alembic downgrade -1                                   # roll back one
    alembic history                                        # show all migrations
    alembic current                                        # show current version
"""
import sys
from pathlib import Path
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add project root to path so we can import ARIA modules
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Import all models so Alembic can detect them for autogenerate
from db.models import Base  # noqa: F401 — import registers all models with metadata

# Import settings for database URL
from config.settings import get_settings
settings = get_settings()

# Alembic Config object (from alembic.ini)
config = context.config

# Set up logging from alembic.ini config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url with value from ARIA settings
# This means the DB URL is always sourced from .env, never hardcoded
config.set_main_option("sqlalchemy.url", settings.database_url)

# Target metadata for --autogenerate comparison
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates SQL script without connecting to database.
    Useful for reviewing changes before applying, or for DBAs who
    need to apply migrations manually.

    Usage: alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    Connects to database and applies migrations directly.
    Standard mode for development and production deployments.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Compare column types so type changes are detected
            compare_type=True,
            # Compare server defaults
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()