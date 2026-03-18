# ARIA Deployment Guide

Complete instructions for deploying ARIA to production.
Covers Docker locally, Railway (recommended), Render, and a bare VPS.

---

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database (hosted or Docker)
- SerpAPI key (100-250 free searches/month at serpapi.com)
- Anthropic API key (for LLM layer — claude.ai/api)

---

## Option 1: Docker Compose (Local / VPS)

The simplest full-stack deployment. Runs Postgres, Redis, and the agent
together in Docker on any machine.

### Step 1 — Clone and configure

```bash
git clone https://github.com/yourname/aria.git
cd aria

# Create your .env from the production template
cp .env.production .env

# Edit .env — fill in required values:
#   POSTGRES_PASSWORD=  (strong random password)
#   ANTHROPIC_API_KEY=  (or OPENAI_API_KEY)
#   SERPAPI_KEY=        (your SerpAPI key)
nano .env
```

### Step 2 — Build and start

```bash
# Build the Docker image
docker compose build

# Start the full stack
# This will:
#   1. Start Postgres and Redis
#   2. Run Alembic migrations
#   3. Seed the database with initial products
#   4. Start the ARIA agent on its schedule
docker compose up -d

# Verify everything is running
docker compose ps

# Follow the agent logs
docker compose logs -f agent
```

### Step 3 — Fetch real data and train models

```bash
# Fetch competitor prices and demand signals (uses DEMO_MODE from .env)
docker compose run cli python scripts/fetch_competitors.py
docker compose run cli python scripts/fetch_trends.py

# Build features and train models
docker compose run cli python src/features.py
docker compose run cli python src/pricing_model.py --evaluate
docker compose run cli python src/demand_forecast.py

# Run the agent once to verify
docker compose run cli python agent/aria.py --once
```

### Step 4 — Verify monitoring

```bash
# Check metrics
docker compose run cli python monitoring/metrics.py

# Run alert checks
docker compose run cli python monitoring/alerts.py

# Check agent status and pending approvals
docker compose run cli python agent/aria.py --status
```

### Common Docker commands

```bash
# Restart just the agent
docker compose restart agent

# Stop everything
docker compose down

# Stop and delete all data (DESTRUCTIVE)
docker compose down -v

# Open a shell inside the agent container
docker compose exec agent bash

# View last 100 log lines
docker compose logs --tail=100 agent

# Run a one-off migration
docker compose run cli alembic upgrade head

# Check migration status
docker compose run cli alembic current
```

---

## Option 2: Railway (Recommended for cloud deployment)

Railway is the easiest cloud platform for this stack.
It detects the Dockerfile automatically and provides managed Postgres and Redis.

### Step 1 — Create Railway project

1. Go to railway.app and create an account
2. Click **New Project** → **Deploy from GitHub repo**
3. Connect your GitHub account and select the ARIA repository

### Step 2 — Add services

In the Railway dashboard:

1. **Add PostgreSQL**: Click **+ New** → **Database** → **PostgreSQL**
2. **Add Redis**: Click **+ New** → **Database** → **Redis**

### Step 3 — Set environment variables

In Railway, go to your service → **Variables** tab and add:

```
ANTHROPIC_API_KEY     = sk-ant-...
SERPAPI_KEY           = your-serpapi-key
DEMO_MODE             = false
LOG_LEVEL             = INFO
AGENT_SCHEDULE_MINUTES = 60
```

Railway automatically injects `DATABASE_URL` and `REDIS_URL` from
the linked Postgres and Redis services — no need to set those manually.

### Step 4 — Deploy

Railway automatically deploys when you push to your main branch.

```bash
git add .
git commit -m "deploy: initial ARIA deployment"
git push origin main
```

Watch the deployment in the Railway dashboard. First deploy takes
3-5 minutes to build the Docker image.

### Step 5 — Run initial setup

Use Railway's built-in terminal (or `railway run`):

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway link  # link to your project

# Seed and train
railway run python scripts/seed_db.py
railway run python scripts/fetch_competitors.py
railway run python scripts/fetch_trends.py
railway run python src/features.py
railway run python src/pricing_model.py
railway run python src/demand_forecast.py
```

---

## Option 3: Render

Similar to Railway. Render provides managed Postgres and can run
Docker containers as background workers.

### Setup

1. Create a new **Background Worker** service on Render
2. Connect your GitHub repository
3. Set **Environment** to **Docker**
4. Add a **PostgreSQL** database from Render's dashboard
5. Set environment variables (same as Railway above)
6. Render automatically injects `DATABASE_URL`

For Redis: Render does not include Redis on free tier.
Set `REDIS_URL` to empty — ARIA falls back to file cache automatically.

---

## Option 4: VPS (DigitalOcean, Hetzner, Linode)

For full control and lower cost at scale.

```bash
# On your VPS (Ubuntu 22.04)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git

# Clone repo
git clone https://github.com/yourname/aria.git
cd aria

# Configure
cp .env.production .env
nano .env  # fill in all values

# Start
docker compose up -d

# Enable auto-restart on server reboot
sudo systemctl enable docker
```

---

## Database Migrations in Production

**Never** drop and recreate tables in production. Always use migrations:

```bash
# Apply all pending migrations (safe to run anytime)
alembic upgrade head
# or with Docker:
docker compose run cli alembic upgrade head

# Check current migration version
alembic current

# See migration history
alembic history

# Create a new migration after changing db/models.py
alembic revision --autogenerate -m "add new column"
# Then review the generated file in db/migrations/versions/
# Then apply it:
alembic upgrade head
```

**The migration workflow for schema changes:**

1. Edit `db/models.py` (add column, new table, etc.)
2. Run `alembic revision --autogenerate -m "description"`
3. Review the generated migration file — always check it before applying
4. Run `alembic upgrade head` locally to test
5. Commit the migration file to git
6. Deploy — the entrypoint script runs `alembic upgrade head` automatically

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL connection string |
| `ANTHROPIC_API_KEY` | Yes* | — | Anthropic API key (* or OpenAI) |
| `SERPAPI_KEY` | No | — | SerpAPI key for live data |
| `REDIS_URL` | No | — | Redis URL (falls back to file cache) |
| `DEMO_MODE` | No | `true` | Use simulator instead of live APIs |
| `AGENT_SCHEDULE_MINUTES` | No | `60` | How often agent runs |
| `AGENT_AUTO_APPROVE_MAX_PCT` | No | `10` | Max % change before human approval |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `LANGFUSE_ENABLED` | No | `false` | Enable LLM tracing |
| `ALERT_WEBHOOK_URL` | No | — | Slack/Teams webhook for alerts |

---

## Health Checks

The Docker container includes a health check that verifies database
connectivity. To check manually:

```bash
# Container health status
docker compose ps

# Manual health check
docker compose exec agent python -c "
from db.models import get_table_counts
counts = get_table_counts()
print('DB healthy:', counts)
"

# Full status report
docker compose run cli python agent/aria.py --status
```

---

## Scaling Considerations

**Current architecture** (single container) handles:
- Up to ~500 products
- Hourly pricing reviews
- ~50 LLM calls/day maximum

**When you outgrow this:**
- Separate the data fetcher, agent, and monitoring into separate services
- Add a task queue (Celery + Redis) for parallel product processing
- Move ML model training to a separate scheduled job
- Add a read replica for analytics queries

---

## Security Checklist

Before going live:

- [ ] `.env` is in `.gitignore` and never committed
- [ ] `POSTGRES_PASSWORD` is a strong random string (use `openssl rand -hex 32`)
- [ ] Postgres port 5432 is not publicly exposed (remove ports mapping in docker-compose.yml)
- [ ] Running as non-root user in Docker (already configured in Dockerfile)
- [ ] `DEMO_MODE=false` and `SERPAPI_KEY` set for live competitor data
- [ ] Anthropic or OpenAI API key has spending limits configured