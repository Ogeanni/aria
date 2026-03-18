"""
src/model_router.py
Decision routing layer for ARIA.

Every repricing decision passes through three layers in order.
A decision is resolved at the earliest layer that can handle it confidently.
Only decisions that escape layers 1 and 2 ever reach the LLM.

Layer 1 — Rules engine      (~60% of decisions)
  Deterministic business rules. Zero cost, zero latency.
  Handles clear-cut cases: massive price gap, critical stock, floor/ceiling hits.

Layer 2 — XGBoost model     (~30% of decisions)
  Statistical recommendation. Used when rules don't produce a clear signal.
  Returns decision only if confidence is medium or high.

Layer 3 — LLM reasoning     (~10% of decisions)
  Edge cases: conflicting signals, unusual combinations, ambiguous context.
  Structured output — returns same dict format as layers 1 and 2.

Usage:
    from src.model_router import route_decision
    decision = route_decision(product_row, demand_forecast)
"""
import sys
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("model_router")

from config.settings import get_settings
settings = get_settings()


# ══════════════════════════════════════════════════════════════════════
# DECISION DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RoutingDecision:
    """
    Uniform output from the routing layer — passed to the agent executor.
    Same structure regardless of which layer resolved the decision.
    """
    action:            str
    recommended_price: float
    current_price:     float
    change_pct:        float
    layer:             str
    rule_triggered:    Optional[str]
    confidence:        str
    product_id:        int
    product_name:      str
    category:          str
    comp_price_med:    float
    trend_index:       float
    trend_direction:   str
    inventory_pressure:int
    min_price:         float
    max_price:         float
    rationale:         str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def requires_approval(self) -> bool:
        return abs(self.change_pct) > settings.agent_auto_approve_max_pct

    @property
    def is_no_op(self) -> bool:
        return self.action == "hold"


# ══════════════════════════════════════════════════════════════════════
# LAYER 1 — RULES ENGINE
# ══════════════════════════════════════════════════════════════════════

RULES = [
    {
        "name": "at_price_floor",
        "description": "Price is at or below minimum — cannot decrease further",
        "condition": lambda ctx: ctx["current_price"] <= ctx["min_price"] * 1.01,
        "action": "hold",
        "price_fn": lambda ctx: ctx["current_price"],
        "confidence": "high",
    },
    {
        "name": "at_price_ceiling",
        "description": "Price is at or above maximum — cannot increase further",
        "condition": lambda ctx: ctx["current_price"] >= ctx["max_price"] * 0.99,
        "action": "hold",
        "price_fn": lambda ctx: ctx["current_price"],
        "confidence": "high",
    },
    {
        "name": "critical_stock_hold",
        "description": "Critical low stock — hold price to preserve margin",
        "condition": lambda ctx: ctx["inventory_qty"] < 10 and ctx["price_position"] <= 5,
        "action": "hold",
        "price_fn": lambda ctx: ctx["current_price"],
        "confidence": "high",
    },
    {
        "name": "critical_stock_increase",
        "description": "Critical low stock and priced below market — increase to capture margin",
        "condition": lambda ctx: ctx["inventory_qty"] < 10 and ctx["price_position"] < -10,
        "action": "increase",
        "price_fn": lambda ctx: min(ctx["comp_price_med"] * 1.05, ctx["max_price"]),
        "confidence": "high",
    },
    {
        "name": "severely_overpriced_falling_demand",
        "description": "Price >20% above market AND demand falling",
        "condition": lambda ctx: (
            ctx["price_position"] > 20 and ctx["trend_direction"] == "falling"
        ),
        "action": "decrease",
        "price_fn": lambda ctx: max(ctx["comp_price_med"] * 1.02, ctx["min_price"]),
        "confidence": "high",
    },
    {
        "name": "severely_underpriced_rising_demand",
        "description": "Price >20% below market AND demand rising",
        "condition": lambda ctx: (
            ctx["price_position"] < -20 and ctx["trend_direction"] == "rising"
        ),
        "action": "increase",
        "price_fn": lambda ctx: min(ctx["comp_price_med"] * 0.98, ctx["max_price"]),
        "confidence": "high",
    },
    {
        "name": "overpriced_overstocked",
        "description": "Price >15% above market AND overstocked",
        "condition": lambda ctx: (
            ctx["price_position"] > 15 and ctx["inventory_pressure"] == -1
        ),
        "action": "decrease",
        "price_fn": lambda ctx: max(ctx["comp_price_med"] * 0.97, ctx["min_price"]),
        "confidence": "high",
    },
    {
        "name": "underpriced_scarce_trending",
        "description": "Price below market, low stock, and trending up",
        "condition": lambda ctx: (
            ctx["price_position"] < -10 and
            ctx["inventory_pressure"] >= 0 and
            ctx["trend_direction"] in ("rising", "stable") and
            ctx["trend_index"] >= 55
        ),
        "action": "increase",
        "price_fn": lambda ctx: min(ctx["comp_price_med"] * 1.03, ctx["max_price"]),
        "confidence": "medium",
    },
    {
        "name": "competitive_hold",
        "description": "Price within +-5% of market median — no action needed",
        "condition": lambda ctx: abs(ctx["price_position"]) <= 5,
        "action": "hold",
        "price_fn": lambda ctx: ctx["current_price"],
        "confidence": "high",
    },
    {
    "name": "overstocked_moderate",
    "description": "Overstocked and priced above market — decrease regardless of demand",
    "condition": lambda ctx: (ctx["inventory_pressure"] == -1 and ctx["price_position"] > 10),
    "action": "decrease",
    "price_fn": lambda ctx: max(ctx["comp_price_med"] * 0.98, ctx["min_price"]),
    "confidence": "high",
},

]


def _build_context(product_row, demand: dict) -> dict:
    """Build flat context dict from product feature row and demand forecast."""
    row = product_row if isinstance(product_row, dict) else product_row.to_dict()

    current      = float(row.get("current_price", 0))
    comp_med     = float(row.get("comp_price_med", current))
    min_price    = float(row.get("min_price", current * 0.70))
    max_price    = float(row.get("max_price", current * 1.50))

    return {
        "product_id":        int(row.get("product_id", 0)),
        "product_name":      str(row.get("product_name", "")),
        "category":          str(row.get("category", "")),
        "current_price":     current,
        "comp_price_med":    comp_med,
        "min_price":         min_price,
        "max_price":         max_price,
        "price_position":    float(row.get("price_position", 0)),
        "inventory_qty":     int(row.get("inventory_qty", 100)),
        "inventory_pressure":int(row.get("inventory_pressure", 0)),
        "trend_index":       float(row.get("trend_index_latest", 50)),
        "trend_direction":   demand.get("trend_direction", "stable") if demand else "stable",
        "is_trending":       int(row.get("is_trending", 0)),
        "comp_count":        int(row.get("comp_count", 0)),
    }


def run_rules_engine(ctx: dict) -> Optional[RoutingDecision]:
    """Evaluate rules in priority order. Return first match or None."""
    for rule in RULES:
        try:
            if not rule["condition"](ctx):
                continue

            new_price  = rule["price_fn"](ctx)
            new_price  = round(float(np.clip(new_price, ctx["min_price"], ctx["max_price"])), 2)
            current    = ctx["current_price"]
            change_pct = round((new_price - current) / current * 100, 2) if current > 0 else 0.0

            action = rule["action"]
            if abs(new_price - current) < 0.01:
                action     = "hold"
                new_price  = current
                change_pct = 0.0

            rationale = _build_rule_rationale(rule["name"], ctx, new_price)
            log.debug(f"  Rule matched: {rule['name']} -> {action} ${new_price:.2f}")

            return RoutingDecision(
                action=action, recommended_price=new_price,
                current_price=current, change_pct=change_pct,
                layer="rules", rule_triggered=rule["name"],
                confidence=rule["confidence"],
                product_id=ctx["product_id"], product_name=ctx["product_name"],
                category=ctx["category"], comp_price_med=ctx["comp_price_med"],
                trend_index=ctx["trend_index"], trend_direction=ctx["trend_direction"],
                inventory_pressure=ctx["inventory_pressure"],
                min_price=ctx["min_price"], max_price=ctx["max_price"],
                rationale=rationale,
            )
        except Exception as e:
            log.warning(f"  Rule '{rule['name']}' error: {e}")
            continue
    return None


def _build_rule_rationale(rule_name: str, ctx: dict, new_price: float) -> str:
    templates = {
        "at_price_floor":
            f"Price is at the minimum floor (${ctx['min_price']:.2f}) — holding.",
        "at_price_ceiling":
            f"Price is at the maximum ceiling (${ctx['max_price']:.2f}) — holding.",
        "critical_stock_hold":
            f"Only {ctx['inventory_qty']} units remaining — holding price to preserve margin.",
        "critical_stock_increase":
            f"Only {ctx['inventory_qty']} units remaining and {abs(ctx['price_position']):.1f}% "
            f"below market — increasing to ${new_price:.2f} to capture margin while stock lasts.",
        "severely_overpriced_falling_demand":
            f"Price is {ctx['price_position']:.1f}% above market (${ctx['comp_price_med']:.2f}) "
            f"and demand is falling — reducing to ${new_price:.2f} to stay competitive.",
        "severely_underpriced_rising_demand":
            f"Price is {abs(ctx['price_position']):.1f}% below market and demand is rising — "
            f"increasing to ${new_price:.2f} to capture available value.",
        "overpriced_overstocked":
            f"Price is {ctx['price_position']:.1f}% above market and inventory is high — "
            f"reducing to ${new_price:.2f} to accelerate sell-through.",
        "underpriced_scarce_trending":
            f"Price is {abs(ctx['price_position']):.1f}% below market, demand index "
            f"{ctx['trend_index']:.0f}/100 ({ctx['trend_direction']}), inventory manageable — "
            f"increasing to ${new_price:.2f}.",
        "competitive_hold":
            f"Price (${ctx['current_price']:.2f}) is within 5% of market median "
            f"(${ctx['comp_price_med']:.2f}) — no change needed.",
    }
    return templates.get(rule_name, f"Rule '{rule_name}' triggered. New price: ${new_price:.2f}.")


# ══════════════════════════════════════════════════════════════════════
# LAYER 2 — XGBOOST MODEL
# ══════════════════════════════════════════════════════════════════════

def run_ml_model(product_row, ctx: dict) -> Optional[RoutingDecision]:
    """Get XGBoost recommendation. Returns None if unavailable or low confidence."""
    try:
        from src.pricing_model import get_price_recommendation
        rec = get_price_recommendation(product_row=product_row)
    except FileNotFoundError:
        log.warning("  XGBoost model not trained — skipping ML layer")
        return None
    except Exception as e:
        log.warning(f"  ML model error: {e}")
        return None

    if rec.get("confidence", "low") == "low":
        log.debug("  ML confidence=low — escalating to LLM")
        return None

    new_price  = float(rec["recommended_price"])
    current    = ctx["current_price"]
    change_pct = round((new_price - current) / current * 100, 2) if current > 0 else 0.0
    action     = "hold" if abs(new_price - current) < 0.01 else (
                 "increase" if new_price > current else "decrease")

    return RoutingDecision(
        action=action, recommended_price=new_price,
        current_price=current, change_pct=change_pct,
        layer="ml_model", rule_triggered=None,
        confidence=rec.get("confidence", "medium"),
        product_id=ctx["product_id"], product_name=ctx["product_name"],
        category=ctx["category"], comp_price_med=ctx["comp_price_med"],
        trend_index=ctx["trend_index"], trend_direction=ctx["trend_direction"],
        inventory_pressure=ctx["inventory_pressure"],
        min_price=ctx["min_price"], max_price=ctx["max_price"],
        rationale=rec.get("rationale", "XGBoost model recommendation."),
    )


# ══════════════════════════════════════════════════════════════════════
# LAYER 3 — LLM REASONING
# ══════════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT = """You are a pricing analyst for an ecommerce business.
You will be given a product's pricing context and must recommend a price action.

Respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.

Required format:
{
  "action": "increase" | "decrease" | "hold" | "review",
  "recommended_price": float,
  "confidence": "high" | "medium" | "low",
  "rationale": "one sentence explanation"
}

Rules:
- recommended_price must be between min_price and max_price
- If signals are contradictory and you are uncertain, use action="review"
- Do not recommend changes smaller than $0.50 — use "hold" instead
- Be conservative — only recommend changes with clear evidence"""

LLM_USER_TEMPLATE = """Product: {product_name} ({category})
Current price: ${current_price:.2f}
Competitor median price: ${comp_price_med:.2f}
Price position vs market: {price_position:+.1f}%
Demand trend index: {trend_index:.0f}/100 ({trend_direction})
Inventory: {inventory_qty} units ({inventory_pressure_label})
Price floor: ${min_price:.2f}
Price ceiling: ${max_price:.2f}
Competitor count: {comp_count}

What price action do you recommend?"""


def run_llm(ctx: dict) -> Optional[RoutingDecision]:
    """Ask LLM to reason about an edge case. Validates and clamps output."""
    if not settings.llm_api_key:
        log.warning("  No LLM API key — returning review decision")
        return _make_review_decision(ctx, "no_llm_key")

    inv_labels = {1: "low stock", 0: "normal", -1: "overstocked"}
    prompt = LLM_USER_TEMPLATE.format(
        product_name=ctx["product_name"],
        category=ctx["category"],
        current_price=ctx["current_price"],
        comp_price_med=ctx["comp_price_med"],
        price_position=ctx["price_position"],
        trend_index=ctx["trend_index"],
        trend_direction=ctx["trend_direction"],
        inventory_qty=ctx["inventory_qty"],
        inventory_pressure_label=inv_labels.get(ctx["inventory_pressure"], "normal"),
        min_price=ctx["min_price"],
        max_price=ctx["max_price"],
        comp_count=ctx["comp_count"],
    )

    try:
        raw    = _call_llm(prompt)
        parsed = _parse_llm_response(raw)
    except Exception as e:
        log.error(f"  LLM call failed: {e}")
        return _make_review_decision(ctx, f"llm_error: {e}")

    new_price = round(float(np.clip(
        float(parsed.get("recommended_price", ctx["current_price"])),
        ctx["min_price"], ctx["max_price"]
    )), 2)

    action = parsed.get("action", "review")
    if action not in ("increase", "decrease", "hold", "review"):
        action = "review"
    if abs(new_price - ctx["current_price"]) < 0.50:
        action    = "hold"
        new_price = ctx["current_price"]

    current    = ctx["current_price"]
    change_pct = round((new_price - current) / current * 100, 2) if current > 0 else 0.0

    return RoutingDecision(
        action=action, recommended_price=new_price,
        current_price=current, change_pct=change_pct,
        layer="llm", rule_triggered=None,
        confidence=parsed.get("confidence", "low"),
        product_id=ctx["product_id"], product_name=ctx["product_name"],
        category=ctx["category"], comp_price_med=ctx["comp_price_med"],
        trend_index=ctx["trend_index"], trend_direction=ctx["trend_direction"],
        inventory_pressure=ctx["inventory_pressure"],
        min_price=ctx["min_price"], max_price=ctx["max_price"],
        rationale=parsed.get("rationale", "LLM recommendation."),
    )


def _call_llm(user_prompt: str) -> str:
    if settings.llm_provider == "anthropic":
        import anthropic
        client  = anthropic.Anthropic(api_key=settings.llm_api_key)
        message = client.messages.create(
            model=settings.llm_model, max_tokens=256,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    elif settings.llm_provider == "openai":
        import openai
        client   = openai.OpenAI(api_key=settings.llm_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=256,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def _parse_llm_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {text[:200]}")


def _make_review_decision(ctx: dict, reason: str) -> RoutingDecision:
    return RoutingDecision(
        action="review", recommended_price=ctx["current_price"],
        current_price=ctx["current_price"], change_pct=0.0,
        layer="rules", rule_triggered="fallback_review", confidence="low",
        product_id=ctx["product_id"], product_name=ctx["product_name"],
        category=ctx["category"], comp_price_med=ctx["comp_price_med"],
        trend_index=ctx["trend_index"], trend_direction=ctx["trend_direction"],
        inventory_pressure=ctx["inventory_pressure"],
        min_price=ctx["min_price"], max_price=ctx["max_price"],
        rationale=f"Unable to determine price action ({reason}) — flagged for human review.",
    )


# ══════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════

def route_decision(product_row, demand: Optional[dict] = None) -> RoutingDecision:
    """
    Route a single product through all decision layers.

    Args:
        product_row: pd.Series or dict of product features
        demand:      dict from get_demand_forecast() — can be None

    Returns RoutingDecision with action, price, confidence, layer, rationale.
    """
    ctx = _build_context(product_row, demand)

    log.debug(
        f"Routing {ctx['product_name']} | "
        f"${ctx['current_price']:.2f} | market=${ctx['comp_price_med']:.2f} | "
        f"pos={ctx['price_position']:+.1f}% | demand={ctx['trend_direction']}"
    )

    # Layer 1: Rules
    decision = run_rules_engine(ctx)
    if decision:
        log.info(
            f"  [{ctx['product_name'][:30]}] RULES/{decision.rule_triggered} "
            f"-> {decision.action} ${decision.recommended_price:.2f}"
        )
        return decision

    # Layer 2: XGBoost
    decision = run_ml_model(product_row, ctx)
    if decision:
        log.info(
            f"  [{ctx['product_name'][:30]}] ML/{decision.confidence} "
            f"-> {decision.action} ${decision.recommended_price:.2f}"
        )
        return decision

    # Layer 3: LLM
    log.info(f"  [{ctx['product_name'][:30]}] Escalating to LLM...")
    decision = run_llm(ctx)
    log.info(
        f"  [{ctx['product_name'][:30]}] LLM/{decision.confidence} "
        f"-> {decision.action} ${decision.recommended_price:.2f}"
    )
    return decision


def route_all_products(demand_forecasts: Optional[dict] = None) -> list:
    """Route all products in the feature matrix. Returns list of RoutingDecision."""
    try:
        from src.features import build_feature_matrix
        df = build_feature_matrix()
    except Exception as e:
        log.error(f"Could not build feature matrix: {e}")
        return []

    decisions = []
    for _, row in df.iterrows():
        demand = (demand_forecasts or {}).get(str(row.get("category", "")))
        try:
            decisions.append(route_decision(row, demand))
        except Exception as e:
            log.error(f"  Routing failed for product {row.get('product_id')}: {e}")
    return decisions


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="ARIA model router")
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--product-id", type=int)
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    try:
        from src.demand_forecast import get_all_forecasts
        forecasts = get_all_forecasts()
        log.info(f"Demand forecasts loaded: {list(forecasts.keys())}")
    except Exception as e:
        log.warning(f"Demand forecasts unavailable: {e}")
        forecasts = {}

    if args.product_id and not args.all:
        from src.features import build_feature_matrix
        df      = build_feature_matrix()
        matches = df[df["product_id"] == args.product_id]
        if matches.empty:
            log.error(f"Product {args.product_id} not found"); sys.exit(1)
        decisions = [route_decision(matches.iloc[0], forecasts.get(str(matches.iloc[0].get("category", ""))))]
    else:
        decisions = route_all_products(forecasts)

    layer_counts  = {}
    action_counts = {}

    print(f"\n{'='*92}")
    print(f"  {'Product':<32} {'Layer':<10} {'Action':<10} {'Current':>9} {'Rec':>10} {'Change':>8}  Conf")
    print("-" * 92)

    for d in decisions:
        layer_counts[d.layer]   = layer_counts.get(d.layer, 0) + 1
        action_counts[d.action] = action_counts.get(d.action, 0) + 1
        arrow = "up" if d.action == "increase" else ("dn" if d.action == "decrease" else "ok")
        print(
            f"  {d.product_name[:30]:<30}  {d.layer:<10} {d.action:<10} "
            f"${d.current_price:>8.2f} ${d.recommended_price:>9.2f} "
            f"{d.change_pct:>+7.1f}%  {arrow} [{d.confidence}]"
        )
        if args.verbose:
            print(f"    {d.rationale}\n")

    total = len(decisions)
    print("=" * 92)
    print(f"\nLayer distribution ({total} products):")
    for layer, count in sorted(layer_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {layer:<12} {count:>3}  ({pct:.1f}%)  {'|' * int(pct / 3)}")

    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {action:<12} {count:>3}  ({pct:.1f}%)")

    needs_approval = [d for d in decisions if d.requires_approval]
    if needs_approval:
        print(f"\n  {len(needs_approval)} require human approval (>{settings.agent_auto_approve_max_pct}% change):")
        for d in needs_approval:
            print(f"  {d.product_name}: {d.change_pct:+.1f}% -> ${d.recommended_price:.2f}")