#!/bin/sh
# ─────────────────────────────────────────────────────────────────────
# docker-entrypoint.sh
# Runs automatically when the Docker container starts.
#
# Steps:
#   1. Wait for PostgreSQL to be ready
#   2. Run any pending database migrations
#   3. Execute the command passed to the container (default: agent)
#
# Why we wait for Postgres here AND have depends_on in docker-compose:
#   depends_on with healthcheck handles Docker networking.
#   This script handles the case where the DB is technically reachable
#   but not yet fully initialised (accepting connections but not queries).
# ─────────────────────────────────────────────────────────────────────
set -e

echo "ARIA entrypoint starting..."

# ── Wait for PostgreSQL ───────────────────────────────────────────────
if [ -n "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    # Extract host and port from DATABASE_URL
    # Format: postgresql://user:pass@host:port/dbname
    DB_HOST=$(echo "$DATABASE_URL" | sed -E 's|.*@([^:/]+).*|\1|')
    DB_PORT=$(echo "$DATABASE_URL" | sed -E 's|.*:([0-9]+)/.*|\1|')
    DB_PORT=${DB_PORT:-5432}

    MAX_RETRIES=30
    RETRY=0
    until python -c "
import psycopg2, os, sys
try:
    psycopg2.connect(os.environ['DATABASE_URL'])
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; do
        RETRY=$((RETRY+1))
        if [ $RETRY -ge $MAX_RETRIES ]; then
            echo "Database not ready after $MAX_RETRIES attempts. Exiting."
            exit 1
        fi
        echo "  Database not ready (attempt $RETRY/$MAX_RETRIES) — retrying in 2s..."
        sleep 2
    done
    echo "Database ready."
fi

# ── Run migrations ────────────────────────────────────────────────────
echo "Running database migrations..."
alembic upgrade head
echo "Migrations complete."

# ── Execute main command ──────────────────────────────────────────────
echo "Starting: $@"
exec "$@"