"""
dashboard.py
ARIA Streamlit Demo Dashboard

A live visual interface for demonstrating ARIA's capabilities.
Shows the agent running in real time, pricing decisions, monitoring metrics,
and the human approval workflow.

Run alongside the FastAPI server:
    Terminal 1: uvicorn api.main:app --port 8000
    Terminal 2: streamlit run dashboard.py

Or standalone (calls ARIA directly without the API):
    streamlit run dashboard.py -- --standalone
"""
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
import pandas as pd
import requests

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="ARIA — Autonomous Repricing Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

from config.settings import get_settings
settings = get_settings()

# ── Helpers ───────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    """Call the FastAPI backend. Returns (data, error)."""
    try:
        r = getattr(requests, method)(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.ConnectionError:
        return None, "API server not running. Start it with: uvicorn api.main:app --port 8000"
    except Exception as e:
        return None, str(e)


def call_direct(fn, *args, **kwargs):
    """Call ARIA directly without the API (standalone mode)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def fmt_price(p):
    return f"${p:.2f}" if p is not None else "—"


def fmt_pct(p):
    if p is None:
        return "—"
    arrow = "↑" if p > 0 else ("↓" if p < 0 else "→")
    return f"{arrow} {abs(p):.1f}%"


def action_badge(action):
    colors = {
        "increase": "🟢", "decrease": "🔴",
        "hold": "⚪", "review": "🟡"
    }
    return f"{colors.get(action, '⚫')} {action}"


def layer_badge(layer):
    icons = {"rules": "📋", "ml_model": "🤖", "llm": "✨"}
    return f"{icons.get(layer, '?')} {layer}"


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 ARIA")
    st.caption("Autonomous Repricing & Inventory Agent")
    st.divider()

    # Connection status
    health, err = api("get", "/")
    if health:
        st.success("API connected")
        st.caption(f"DB: {'✓' if health.get('db_connected') else '✗'}  "
                   f"Demo mode: {'on' if health.get('demo_mode') else 'off'}")
    else:
        st.error("API offline")
        st.caption(err)

    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🛒 Products",
        "📋 Decisions",
        "✅ Approvals",
        "📊 Metrics",
        "🔮 Recommendations",
        "🧪 Simulator",
    ])
    st.divider()
    st.caption("Built with Python · XGBoost · Prophet · FastAPI · PostgreSQL")


# ── Overview ──────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("ARIA — Autonomous Repricing & Inventory Agent")
    st.markdown("""
    ARIA is a production AI agentic system that **autonomously monitors competitor prices,
    detects repricing opportunities, and executes price changes** without human intervention.

    It uses a three-layer decision architecture:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Layer 1 — Rules engine**\n\n~60% of decisions\n\nFree · Instant · Deterministic")
    with col2:
        st.warning("**Layer 2 — XGBoost ML**\n\n~30% of decisions\n\nFast · Cheap · Local model")
    with col3:
        st.error("**Layer 3 — LLM**\n\n~10% of decisions\n\nEdge cases only · Reserved")

    st.divider()

    # Live stats
    if health:
        counts = health.get("table_counts", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Products", counts.get("products", 0))
        c2.metric("Decisions made", counts.get("agent_decisions", 0))
        c3.metric("Price changes", counts.get("price_history", 0))
        c4.metric("Pending approval", counts.get("approval_queue", 0))

    st.divider()
    st.subheader("System architecture")
    st.code("""
fetch_competitors.py  ─→  features.py  ─→  pricing_model.py
         ↑                                          ↓
  feedback.py         ←─  executor.py  ←─  model_router.py
                               ↓
                        agent_decisions (audit log)
    """)


# ── Products ──────────────────────────────────────────────────────────

elif page == "🛒 Products":
    st.title("Product Catalog")

    col1, col2 = st.columns([3, 1])
    with col1:
        category = st.selectbox("Filter by category",
            ["All", "electronics", "fashion", "home_goods", "sports"])
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    cat_param = None if category == "All" else category
    params = {"active_only": True}
    if cat_param:
        params["category"] = cat_param

    data, err = api("get", "/products", params=params)
    if err:
        st.error(err)
    elif data:
        df = pd.DataFrame(data)
        # Ensure optional columns exist even if API returns None/missing
        df["current_price"] = df["current_price"].apply(fmt_price)
        df["min_price"]     = df.get("min_price", pd.Series(["—"] * len(df))).apply(lambda x: fmt_price(x) if x else "—")
        df["max_price"]     = df.get("max_price", pd.Series(["—"] * len(df))).apply(lambda x: fmt_price(x) if x else "—")

        st.dataframe(
            df[["id", "name", "category", "current_price",
                "min_price", "max_price", "inventory_qty", "platform"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(df)} products")


# ── Run Agent (disabled for public — use API directly) ───────────────

elif page == "⚡ Run Agent":
    st.title("Run Agent")
    st.markdown("""
    Trigger one full ARIA repricing cycle.

    The agent will:
    1. Load demand forecasts from Prophet models
    2. Route every product through rules → ML → LLM
    3. Execute or queue price changes
    """)

    dry_run = st.toggle("Dry run (no real changes)", value=True)

    if not dry_run:
        st.warning(
            "**Live mode** — price changes will be executed and large changes "
            "queued for approval. This makes real database changes."
        )

    if st.button("▶️ Run now", type="primary", use_container_width=True):
        with st.spinner("Agent running..."):
            result, err = api("post", "/agent/run", json={"dry_run": dry_run})

        if err:
            st.error(err)
        elif result:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Products reviewed", result["products_reviewed"])
            col2.metric("Executed",          result["executed"])
            col3.metric("Held",              result["held"])
            col4.metric("Pending approval",  result["pending_approval"])
            col5.metric("Errors",            result["errors"])

            st.success(
                f"Run **{result['run_id']}** complete in "
                f"{result['elapsed_seconds']:.1f}s  |  "
                f"{'DRY RUN' if result['dry_run'] else 'LIVE'}"
            )

            if result["pending_approval"] > 0:
                st.info(
                    f"{result['pending_approval']} decision(s) require human approval. "
                    "Go to ✅ Approvals to review them."
                )


# ── Decisions ─────────────────────────────────────────────────────────

elif page == "📋 Decisions":
    st.title("Agent Decisions — Audit Log")
    st.markdown("Every decision ARIA makes is permanently recorded here.")

    col1, col2 = st.columns([2, 1])
    with col1:
        limit = st.slider("Show last N decisions", 5, 100, 20)
    with col2:
        exec_only = st.checkbox("Executed only")
        if st.button("🔄 Refresh"):
            st.rerun()

    data, err = api("get", "/agent/decisions",
                    params={"limit": limit, "executed_only": exec_only})
    if err:
        st.error(err)
    elif not data:
        st.info("No decisions yet. Run the agent first.")
    else:
        df = pd.DataFrame(data)
        df["action"]   = df["decision_type"].apply(action_badge)
        df["layer"]    = df["decision_source"].apply(layer_badge)
        df["change"]   = df["change_pct"].apply(fmt_pct)
        df["current"]  = df["current_price"].apply(fmt_price)
        df["recommended"] = df["recommended_price"].apply(fmt_price)
        df["executed"] = df["was_executed"].apply(lambda x: "✓" if x else "")
        df["time"]     = pd.to_datetime(df["created_at"]).dt.strftime("%H:%M %d %b")

        st.dataframe(
            df[["time", "product_name", "action", "layer",
                "current", "recommended", "change",
                "confidence", "executed"]],
            use_container_width=True,
            hide_index=True,
        )

        # Decision breakdown
        st.subheader("Distribution")
        col1, col2 = st.columns(2)
        with col1:
            action_counts = df["decision_type"].value_counts()
            st.bar_chart(action_counts)
        with col2:
            layer_counts = df["decision_source"].value_counts()
            st.bar_chart(layer_counts)

        # Rationale expander
        st.subheader("Decision rationale")
        for _, row in df.head(5).iterrows():
            if row.get("reasoning"):
                with st.expander(f"{row['product_name']} — {row['action']}"):
                    st.write(row["reasoning"])


# ── Approvals ─────────────────────────────────────────────────────────

elif page == "✅ Approvals":
    st.title("Human Approval Queue")
    st.markdown(f"""
    Price changes exceeding **±{settings.agent_auto_approve_max_pct}%** require human review.
    ARIA holds them here until approved or rejected.
    """)
    st.info("👀 View-only mode — approve/reject actions are disabled for public access.")

    if st.button("🔄 Refresh"):
        st.rerun()

    data, err = api("get", "/agent/approvals")
    if err:
        st.error(err)
    elif not data:
        st.success("No pending approvals. Queue is clear.")
    else:
        st.warning(f"{len(data)} pending approval(s)")

        for item in data:
            direction = "↑ increase" if item["change_pct"] > 0 else "↓ decrease"
            with st.expander(
                f"**{item['product_name']}** — {fmt_pct(item['change_pct'])} {direction}  "
                f"(#{item['id']})"
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Current price",  fmt_price(item["current_price"]))
                col2.metric("Proposed price", fmt_price(item["proposed_price"]))
                col3.metric("Change",         fmt_pct(item["change_pct"]))

                if item.get("reasoning"):
                    st.info(item["reasoning"])

                col_a, col_r = st.columns(2)
                reviewer = st.text_input("Your name", value="reviewer",
                                         key=f"name_{item['id']}")

                with col_a:
                    st.button("✅ Approve", key=f"approve_{item['id']}",
                             type="primary", disabled=True,
                             help="Disabled in public view")

                with col_r:
                    st.button("❌ Reject", key=f"reject_{item['id']}",
                             disabled=True,
                             help="Disabled in public view")


# ── Metrics ───────────────────────────────────────────────────────────

elif page == "📊 Metrics":
    st.title("Metrics & Monitoring")

    days = st.slider("Lookback window (days)", 1, 30, 7)

    if st.button("🔄 Refresh"):
        st.rerun()

    data, err = api("get", "/metrics", params={"days": days})
    if err:
        st.error(err)
    elif data:
        b = data.get("business", {})
        m = data.get("models", {})
        f = data.get("data_freshness", {})

        # Business metrics
        st.subheader("Business")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total decisions",  b.get("total_decisions", 0))
        col2.metric("Executed",         b.get("executed", 0))
        col3.metric("LLM escalation",   f"{b.get('llm_escalation_pct', 0):.1f}%")
        col4.metric("Avg price vs market", f"{b.get('avg_price_position', 0):+.1f}%")

        if b.get("action_counts"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Actions")
                st.bar_chart(pd.Series(b["action_counts"]))
            with col_b:
                st.subheader("Decision layers")
                st.bar_chart(pd.Series(b.get("layer_counts", {})))

        # Model health
        st.subheader("Model health")
        xgb = m.get("xgboost", {})
        if isinstance(xgb, dict) and "trained_at" in xgb:
            col1, col2, col3 = st.columns(3)
            col1.metric("XGBoost age",    f"{xgb.get('age_days', '?')} days")
            col2.metric("Test MAE",        f"${xgb.get('test_mae', '?')}")
            col3.metric("Test R²",         xgb.get("test_r2", "?"))
            if xgb.get("stale"):
                st.warning("XGBoost model is stale. Run: python src/pricing_model.py")

        prophet = m.get("prophet", {})
        if prophet:
            rows = []
            for cat, status in prophet.items():
                if isinstance(status, dict) and "age_days" in status:
                    rows.append({
                        "Category": cat,
                        "Age (days)": status.get("age_days"),
                        "MAE": status.get("mae"),
                        "Weeks trained": status.get("n_weeks"),
                        "Stale": "⚠️" if status.get("stale") else "✓",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)

        # Data freshness
        st.subheader("Data freshness")
        col1, col2, col3 = st.columns(3)
        col1.metric("Products tracked", f.get("n_products", 0))
        col2.metric("Fresh (<24h)",     f.get("fresh_count", 0))
        col3.metric("Stale (>24h)",     f.get("stale_count", 0))
        if f.get("stale_count", 0) > 0:
            st.warning("Stale data detected. Run: python scripts/fetch_competitors.py")

        # Alerts
        st.subheader("Alert check")
        alerts_data, _ = api("get", "/metrics/alerts")
        if alerts_data:
            if alerts_data["total_alerts"] == 0:
                st.success("All checks passing — no alerts fired.")
            else:
                for alert in alerts_data["alerts"]:
                    level      = alert.get("level", "WARNING")
                    msg        = alert.get("message", alert.get("type", "Unknown alert"))
                    action     = alert.get("recommended_action", "")
                    alert_name = alert.get("alert_type", alert.get("type", "Alert"))
                    if level == "CRITICAL":
                        st.error(f"**{alert_name}**: {msg}\n\n*{action}*")
                    else:
                        st.warning(f"**{alert_name}**: {msg}\n\n*{action}*")


# ── Recommendations ───────────────────────────────────────────────────

elif page == "🔮 Recommendations":
    st.title("ML Price Recommendations")
    st.markdown("XGBoost model recommendations for every product.")

    if st.button("🔄 Refresh"):
        st.rerun()

    data, err = api("get", "/recommendations")
    if err:
        st.error(err)
    elif not data:
        st.warning("No recommendations. Train the model: `python src/pricing_model.py`")
    else:
        df = pd.DataFrame(data)
        df["change"]    = df["pct_vs_current"].apply(fmt_pct)
        df["vs_market"] = df["pct_vs_market"].apply(fmt_pct)
        df["current"]   = df["current_price"].apply(fmt_price)
        df["recommended"] = df["recommended_price"].apply(fmt_price)

        st.dataframe(
            df[["product_name", "category", "current", "recommended",
                "change", "vs_market", "confidence"]],
            use_container_width=True,
            hide_index=True,
        )

        # Rationale expander
        st.subheader("Rationale")
        for _, row in df.iterrows():
            if row.get("rationale"):
                pct = row.get("pct_vs_current", 0)
                icon = "↑" if pct > 0 else ("↓" if pct < 0 else "→")
                with st.expander(f"{icon} {row['product_name']} ({fmt_pct(pct)})"):
                    st.write(row["rationale"])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current",     fmt_price(row["current_price"]))
                    col2.metric("Recommended", fmt_price(row["recommended_price"]))
                    col3.metric("Confidence",  row.get("confidence", "—"))


# ── Simulator ─────────────────────────────────────────────────────────

elif page == "🧪 Simulator":
    st.title("Price Simulator")
    st.markdown("""
    Generate realistic simulated competitor prices for any product.
    Demonstrates the data layer using statistical distributions — no API credits needed.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        product_name = st.text_input("Product name", value="Wireless Headphones")
    with col2:
        category = st.selectbox("Category",
            ["electronics", "fashion", "home_goods", "sports"])
    with col3:
        our_price = st.number_input("Our price ($)", min_value=0.99,
                                    max_value=9999.0, value=89.99)

    if st.button("🎲 Simulate market", type="primary"):
        result, err = api("post", "/simulate", json={
            "product_name": product_name,
            "category": category,
            "our_price": our_price,
        })
        if err:
            st.error(err)
        elif result:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Market min",    fmt_price(result["min"]))
            col2.metric("Market median", fmt_price(result["median"]))
            col3.metric("Market max",    fmt_price(result["max"]))
            col4.metric("Our position",
                        f"{result['position_pct']:+.1f}%",
                        result["position_label"])

            # Price band visualisation
            price_band = {
                "P25 (25th pct)":  result["p25"],
                "Median":          result["median"],
                "Our price":       our_price,
                "P75 (75th pct)":  result["p75"],
            }
            st.bar_chart(pd.Series(price_band))

            # Position assessment
            label = result["position_label"]
            if label == "ABOVE market":
                st.warning(
                    f"Your price is {result['position_pct']:+.1f}% above market median. "
                    "ARIA might recommend a price decrease."
                )
            elif label == "BELOW market":
                st.info(
                    f"Your price is {abs(result['position_pct']):.1f}% below market median. "
                    "ARIA might recommend a price increase."
                )
            else:
                st.success(
                    f"Your price is competitive — within 10% of the market median."
                )