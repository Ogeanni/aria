#!/bin/sh
# docker-entrypoint.sh
# 1. Wait for Postgres to accept connections
# 2. Run Alembic migrations
# 3. Start the app

set -e

echo "ARIA entrypoint starting..."

# ── Wait for Postgres ─────────────────────────────────────────────────
# Extract host and port from DATABASE_URL
# DATABASE_URL format: postgresql://user:pass@host:port/dbname
DB_HOST=$(echo "$DATABASE_URL" | sed -E 's|postgresql://[^@]+@([^:/]+).*|\1|')
DB_PORT=$(echo "$DATABASE_URL" | sed -E 's|.*:([0-9]+)/.*|\1|')
DB_PORT=${DB_PORT:-5432}

echo "Waiting for postgres at $DB_HOST:$DB_PORT..."

i=0
while ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -q; do
    i=$((i+1))
    if [ $i -ge 30 ]; then
        echo "Postgres not ready after 30 attempts. Exiting."
        exit 1
    fi
    echo "  attempt $i/30 — retrying in 2s..."
    sleep 2
done

echo "Postgres ready."

# ── Run migrations ────────────────────────────────────────────────────
echo "Running migrations..."
alembic upgrade head
echo "Migrations complete."

# ── Start app ────────────────────────────────────────────────────────
echo "Starting: $@"
exec "$@"