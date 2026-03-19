# ─────────────────────────────────────────────────────────────────────
# ARIA Dockerfile — multi-stage build
#
# Stage 1 (builder): installs build tools + compiles Python packages
# Stage 2 (runtime): copies compiled packages only — no build tools
#
# WORKDIR /app: this directory is created INSIDE the container.
# It does not need to exist on your local machine.
# Your project files are copied into /app via "COPY . ."
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libpq-dev:          psycopg2 needs this at runtime
# postgresql-client:  provides pg_isready for the entrypoint
# curl:               useful for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r aria && useradd -r -g aria aria

# /app is created here inside the container
# Your local project files are copied into it below
WORKDIR /app

# Copy project files into the container's /app directory
COPY --chown=aria:aria . .

# Create runtime directories
RUN mkdir -p \
    data/raw data/processed data/cache \
    models/saved_models models/prophet/saved_models \
    results/prophet \
    logs && \
    chown -R aria:aria /app

USER aria

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEMO_MODE=true \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from db.models import get_table_counts; get_table_counts()" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "agent/aria.py"]