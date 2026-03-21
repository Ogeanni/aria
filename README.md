# ARIA — Autonomous Repricing & Inventory Agent

ARIA is an AI agentic system that autonomously monitors competitor prices, detects repricing opportunities, and executes price changes with little to no human intervention unless. Every decision is audited, large changes require human approval, and the system learns from outcomes over time.

**Live demo:**
- API + Swagger UI: https://aria-production-6e33.up.railway.app/docs
- Dashboard: https://ogeanni-aria-dashboard-y0hiod.streamlit.app

---

## What ARIA do

ARIA runs on a schedule, 24/7. Every hour it:

1. It checks what competitors prices are across 18 products
2. It checks Google Trends demand signals for each category (Electronics, Home Goods, Fashion, and Sports)
3. It routes every product through a three-layer decision architecture
4. It executes price changes within approved limits automatically
5. It queues large changes for human review before acting
6. It records every decision with full reasoning in an audit log
7. It learns from outcomes and improves over time

No human needs to prompt it. It acts on its own.

---

## Three-layer decision architecture

The core design principle: **no LLM first**. The system uses a No-LLM-First architecture, where decisions are handled by business rules first, then ML models, and only escalate to LLMs when necessary, ensuring high efficiency and low operational cost.

| Layer | Method                    | Handles               | Cost                      |
|-------|---------------------------|-----------------------|---------------------------|
| 1     | Rules engine (9 rules)    | ~60% of decisions     | Free · Instant            |
| 2     | XGBoost ML model          | ~30% of decisions     | Near-free · Local         |
| 3     | LLM (OPENAI)              | ~10% edge cases only  | Costs per call · Reserved |

At scale with 1,000 products running hourly, this design keeps AI costs under $5/day. A naive system sending every decision to an LLM would cost more daily for the same workload.

**Current production metrics:**
- LLM escalation rate: 11.1% (target: <15%)
- Test MAE: $3.98 (XGBoost on real market data)
- Test R²: 0.964
- Prophet demand MAE: 8–19 index points across 4 categories

---

## Tech stack

| Component             | Technology                                            |
|-----------------------|-------------------------------------------------------|
| Agent loop            | Python — sense → decide → act                         |
| ML pricing model      | XGBoost — 43 features, trained on live market data    |
| Demand forecasting    | Prophet — per-category, 157 weeks of history          |
| Decision routing      | Custom rules engine + ML + LLM                        |
| Database              | PostgreSQL (Railway managed)                          |
| Cache                 | Redis + file fallback                                 |
| API                   | FastAPI — 16 endpoints, Swagger UI                    |
| Dashboard             | Streamlit — 8-page demo interface                     |
| Model storage         | AWS S3 — downloaded on Railway startup                |
| Deployment            | Railway (API + agent) + Streamlit Cloud (dashboard)   |
| CI/CD                 | GitHub Actions — tests + Docker build + migrations    |
| Containerisation      | Docker + docker-compose                               |
| Migrations            | Alembic — versioned schema management                 |
| Data sources          | SerpAPI (Google Shopping + Google Trends)             |

---

## Project structure

```
aria/
├── agent/
│   ├── aria.py               # Main agent loop — sense → decide → act
│   ├── executor.py           # Price execution + audit log
│   └── memory.py             # Run state + cooldown tracking
│
├── api/
│   └── main.py               # FastAPI — 16 endpoints + Swagger UI
│
├── src/
│   ├── features.py           # 43-column feature matrix
│   ├── pricing_model.py      # XGBoost train + inference
│   ├── demand_forecast.py    # Prophet per-category forecasts
│   ├── model_router.py       # Rules → ML → LLM routing
│   ├── model_store.py        # AWS S3 model upload/download
│   └── price_simulator.py    # Realistic synthetic price data
│
├── scripts/
│   ├── fetch_competitors.py  # SerpAPI + cache + simulator fallback
│   ├── fetch_trends.py       # Google Trends via SerpAPI
│   └── seed_db.py            # Seeds 18 products + history
│
├── db/
│   ├── models.py             # SQLAlchemy ORM — 7 tables
│   └── migrations/           # Alembic versioned migrations
│
├── monitoring/
│   ├── logger.py             # Structured JSON event log
│   ├── metrics.py            # Business + model health metrics
│   ├── alerts.py             # 5 automated threshold checks
│   └── feedback.py           # Outcome scoring + retrain trigger
│
├── config/
│   └── settings.py           # Pydantic settings — env vars > .env
│
├── tests/                    # 65 tests — all passing
├── .github/workflows/ci.yml  # CI — tests + Docker + migrations
├── Dockerfile                # Multi-stage, non-root
├── Dockerfile.dashboard      # Lightweight Streamlit image
├── docker-compose.yml        # 6 services — dedicated port ranges
├── railway.toml              # Railway deployment config
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variable template
```

---

## Local development

### Prerequisites
- Python 3.11+
- PostgreSQL running locally
- Git

### Setup

```bash
git clone https://github.com/Ogeanni/aria.git
cd aria

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set DATABASE_URL, OPENAI_API_KEY

# Create database and run migrations
psql -U postgres -c "CREATE DATABASE aria;"
alembic upgrade head

# Seed and train
python scripts/seed_db.py
python scripts/fetch_competitors.py
python scripts/fetch_trends.py
python src/features.py
python src/pricing_model.py
python src/demand_forecast.py
```

### Run the demo

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000
# Open: http://localhost:8000/docs

# Terminal 2 — Dashboard
streamlit run dashboard.py
# Open: http://localhost:8501

# Terminal 3 — Agent (optional)
python agent/aria.py --once
```

### Docker

```bash
# Start full stack (Postgres, Redis, API, Agent)
docker compose up -d

# Follow logs
docker compose logs -f agent

# Run one-off commands
docker compose run cli python agent/aria.py --status
```

Port strategy — no conflicts with local services:

| Service   | Host port | Container port    |
|-----------|-----------|-------------------|
| Postgres  | 5433      | 5432              |
| Redis     | 6380      | 6379              |
| API       | 8900      | 8000              |

---

## Deployment

### Railway (API + Agent)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login && railway link

# Seed and train against Railway database
export DATABASE_URL="your-railway-public-url"
python scripts/seed_db.py
python scripts/fetch_competitors.py
python src/features.py
python src/pricing_model.py

# Upload models to S3
python src/model_store.py --upload
```

On every deploy, Railway automatically:
1. Runs `alembic upgrade head` — applies pending migrations
2. Downloads ML models from S3 — fast startup, no retraining
3. Starts `uvicorn api.main:app` — FastAPI web service

### Streamlit Cloud (Dashboard)

1. Go to share.streamlit.io → New app
2. Connect GitHub repo → set main file: `dashboard.py`
3. Add secret: `API_BASE = "https://railway-url.up.railway.app"`
4. Deploy

### Environment variables

| Variable                  | Required  | Notes                         |
|---------------------------|-----------|-------------------------------|
| `DATABASE_URL`            | Yes       | Railway injects automatically |
| `ANTHROPIC_API_KEY`       | Yes*      | *or `OPENAI_API_KEY`          |
| `SERPAPI_KEY`             | No        | Falls back to simulator       |
| `AWS_ACCESS_KEY_ID`       | No        | Required for S3 model store   |
| `AWS_SECRET_ACCESS_KEY`   | No        | Required for S3 model store   |
| `S3_BUCKET_NAME`          | No        | Required for S3 model store   |
| `DEMO_MODE`               | No        | `false` = live SerpAPI data   |
| `AGENT_SCHEDULE_MINUTES`  | No        | Default: 60                   |

---

## S3 model store

Model files are binary artifacts — they don't belong in git. ARIA uses AWS S3 to store trained models and downloads them on Railway startup.

```bash
# After training locally
python src/model_store.py --upload

# Check what's in S3
python src/model_store.py --status

# Force re-download (on Railway)
python src/model_store.py --download --force
```

**Startup sequence on Railway:**
```
alembic upgrade head
→ python src/model_store.py --download   (fast — ~5 seconds from S3)
→ python src/features.py                 (regenerate feature matrix from DB)
→ uvicorn api.main:app                   (start API)
```

If S3 download fails, the startup script falls back to retraining from the database automatically.

---

## API endpoints

Full interactive docs at `/docs`. Key endpoints:

| Method    | Path                  | Description                       |
|-----------|-----------------------|-----------------------------------|
| GET       | `/`                   | Health check + table counts       |
| GET       | `/products`           | All products with current prices  |
| POST      | `/agent/run`          | Trigger repricing cycle           |
| GET       | `/agent/decisions`    | Full audit log                    |
| GET       | `/agent/approvals`    | Pending human approval queue      |
| POST      | `/agent/approve/{id}` | Approve a price change            |
| POST      | `/agent/reject/{id}`  | Reject a price change             |
| GET       | `/metrics`            | Business + model health metrics   |
| GET       | `/recommendations`    | ML price recommendations          |
| POST      | `/simulate`           | Simulate competitor prices        |
| GET       | `/forecasts`          | 30-day demand forecasts           |

