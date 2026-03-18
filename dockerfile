# ─────────────────────────────────────────────────────────────────────
# ARIA Dockerfile
# Multi-stage build: keeps the final image lean by separating
# the build environment from the runtime environment.
#
# Stage 1 (builder): installs all dependencies including build tools
# Stage 2 (runtime): copies only what's needed to run, no build tools
#
# Result: ~400MB runtime image vs ~1.2GB single-stage image
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install system build dependencies
# These are needed to compile some Python packages (psycopg2, Prophet)
# but are NOT needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and use a virtual environment
# This makes copying dependencies to the runtime stage straightforward
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies first (Docker layer caching)
# If requirements.txt doesn't change, this layer is cached and
# future builds skip the slow pip install step
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install only runtime system dependencies
# libpq-dev: PostgreSQL client library (psycopg2 needs this at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
# Running as root in a container is a security risk —
# if the container is compromised, the attacker gets root on the host
RUN groupadd -r aria && useradd -r -g aria aria

# Set working directory
WORKDIR /app

# Copy application code
# .dockerignore excludes: .env, __pycache__, *.pyc, models/, data/,
# logs/, results/ — these are runtime artifacts, not source code
COPY --chown=aria:aria . .

# Create runtime directories that are excluded from the image
# These will be populated at runtime, optionally mounted as volumes
RUN mkdir -p \
    data/raw data/processed data/cache \
    models/saved_models models/prophet/saved_models \
    results/prophet \
    logs && \
    chown -R aria:aria /app

# Switch to non-root user
USER aria

# Environment defaults
# These are overridden by docker-compose.yml or --env-file at runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEMO_MODE=true \
    LOG_LEVEL=INFO

# Health check — verifies the application can connect to the database
# Docker marks the container unhealthy if this fails 3 times
# The agent won't receive traffic until this passes
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from db.models import get_table_counts; get_table_counts()" || exit 1

# Default command: run the agent on its configured schedule
# Override with 'docker run aria python agent/aria.py --once' etc.
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "agent/aria.py"]