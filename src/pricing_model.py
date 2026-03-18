"""
src/pricing_model.py
XGBoost pricing model for ARIA.

Trains a regression model to predict optimal price for each product
based on competitor prices, demand signals, inventory, and time context.

Design decisions:
  - Small dataset mode (n < 30): skips early stopping, trains on full data
  - Large dataset mode (n >= 30): train/val/test split with early stopping
  - Final model always retrained on all data with best n_estimators
  - Predictions clamped to product min/max price constraints
  - Model version tracked in metadata for drift detection

Usage:
    python src/pricing_model.py               # Train and save
    python src/pricing_model.py --evaluate    # Train + per-product report
    python src/pricing_model.py --predict     # Load saved model, score all products
    python src/pricing_model.py --importance  # Feature importance table
"""
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pricing_model")

try:
    from xgboost import XGBRegressor
except ImportError:
    log.error("xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config.settings import get_settings

settings = get_settings()

# ── Columns excluded from training ───────────────────────────────────
# Identifiers, labels, and leakage columns — must not be features
EXCLUDE_COLS = [
    "product_id",
    "product_name",
    "category",
    "snapshot_date",
    "target_price",       # The label
    "price_base",         # Used to compute target — leakage
    "demand_multiplier",  # Used to compute target — leakage
    "current_price",      # Direct leakage — we're predicting improvement on this
    "min_price",          # Business constraint — not a market signal
    "max_price",          # Business constraint — not a market signal
    "base_price",         # Original list price — leakage
]

SMALL_DATASET_THRESHOLD = 30


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_features() -> pd.DataFrame:
    """Load feature matrix from parquet."""
    path = settings.features_path
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}.\n"
            "Run: python src/features.py"
        )
    df = pd.read_parquet(path)
    log.info(f"Loaded feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns columns to use as features (excludes IDs, labels, leakage)."""
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    log.info(f"Feature columns ({len(cols)}): {cols}")
    return cols


# ══════════════════════════════════════════════════════════════════════
# SPLITTING
# ══════════════════════════════════════════════════════════════════════

def time_based_split(df, val_size=0.15, test_size=0.15):
    """
    Split into train / val / test sorted by product_id.
    In production with daily snapshots, split on snapshot_date instead.
    Only used when n >= SMALL_DATASET_THRESHOLD.
    """
    df_sorted = df.sort_values("product_id").reset_index(drop=True)
    n = len(df_sorted)

    train_end = max(1, int(n * (1 - val_size - test_size)))
    val_end   = max(train_end + 1, int(n * (1 - test_size)))
    val_end   = min(val_end, n - 1)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df   = df_sorted.iloc[train_end:val_end].copy()
    test_df  = df_sorted.iloc[val_end:].copy()

    log.info(f"Split — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

def build_model(n_estimators=300):
    """
    Configure XGBoost pricing model.

    Parameter rationale:
      n_estimators=300      300 trees is sufficient for 20 products.
                            Early stopping trims this to the optimal count.
      max_depth=4           Shallow trees generalise better on small datasets.
      learning_rate=0.05    Conservative step size. Needs more trees but
                            generalises better. Pairs well with early stopping.
      subsample=0.8         Each tree trains on 80% of rows.
                            Variance between trees reduces overfitting.
      colsample_bytree=0.8  Each tree uses 80% of features.
                            Prevents single features dominating all trees.
      min_child_weight=3    Minimum samples in a leaf. Higher = more
                            conservative splits. Critical for small datasets.
      reg_alpha=0.1         L1 regularisation — sparse feature weights.
      reg_lambda=1.5        L2 regularisation — penalises large weights.
      random_state=42       Reproducibility.
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        verbosity=0,
    )


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train(df: pd.DataFrame) -> dict:
    """
    Full training pipeline. Routes to small or large dataset path.

    Small dataset (n < 30):  train on all data, no held-out test set.
    Large dataset (n >= 30): proper split with early stopping and test eval.

    Returns dict with model, feature_cols, metrics, predictions.
    """
    log.info("\n" + "=" * 55)
    log.info("ARIA — XGBoost Pricing Model Training")
    log.info("=" * 55)

    feature_cols = get_feature_columns(df)
    target_col   = "target_price"
    n = len(df)

    log.info(f"\nDataset  : {n} products, {len(feature_cols)} features")
    log.info(f"Target   : ${df[target_col].min():.2f} to ${df[target_col].max():.2f}")

    if n < SMALL_DATASET_THRESHOLD:
        return _train_small(df, feature_cols, target_col)
    return _train_large(df, feature_cols, target_col)


def _train_small(df, feature_cols, target_col):
    """
    Small dataset path (n < 30) — train on full data.

    This is expected for initial ARIA setup with 18 products.
    Metrics will be on training data — optimistic by design.
    Switch to large dataset path once you have 30+ products
    or accumulate daily snapshots.
    """
    log.warning(
        f"Small dataset ({len(df)} rows < {SMALL_DATASET_THRESHOLD}). "
        "Training on all data — metrics are training metrics, not held-out test."
    )

    X = df[feature_cols].values
    y = df[target_col].values

    model  = build_model(n_estimators=100)
    model.fit(X, y, verbose=False)
    y_pred = model.predict(X)

    metrics = _compute_metrics(y, y_pred, y, y_pred)
    metrics["note"] = "small_dataset_mode — metrics on training data"
    metrics["best_iteration"] = 100

    log.info(f"\nMAE: ${metrics['train_mae']:.2f}  "
             f"RMSE: ${metrics['train_rmse']:.2f}  R2: {metrics['train_r2']:.3f}")
    log.warning("Metrics are on training data — expect them to improve on real holdout data.")

    _save_model(model, feature_cols, metrics, df)

    return {
        "model": model, "feature_cols": feature_cols, "metrics": metrics,
        "train_df": df, "test_df": df, "y_pred_test": y_pred,
    }


def _train_large(df, feature_cols, target_col):
    """Large dataset path (n >= 30) — proper split with early stopping."""
    train_df, val_df, test_df = time_based_split(df)

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    log.info(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    log.info("\nFitting with early stopping...")
    model = build_model()
    model.set_params(early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_round = model.best_iteration
    log.info(f"Best iteration: {best_round} / {model.n_estimators}")

    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    metrics      = _compute_metrics(y_test, y_pred_test, y_train, y_pred_train)

    log.info(f"\nTest  — MAE: ${metrics['test_mae']:.2f}  "
             f"RMSE: ${metrics['test_rmse']:.2f}  R2: {metrics['test_r2']:.3f}")
    log.info(f"Train — MAE: ${metrics['train_mae']:.2f}  "
             f"RMSE: ${metrics['train_rmse']:.2f}  R2: {metrics['train_r2']:.3f}")

    # FIX: gap = test_mae - train_mae (train always lower; large gap = overfitting)
    gap = metrics["test_mae"] - metrics["train_mae"]
    if gap > 5:
        log.warning(f"Overfit gap ${gap:.2f} — consider increasing reg_alpha/reg_lambda.")
    else:
        log.info(f"Overfit gap: ${gap:.2f} — healthy")

    # Retrain on train+val with best n_estimators (no early stopping needed)
    log.info("\nRetraining on train+val for production model...")
    tv_df        = pd.concat([train_df, val_df], ignore_index=True)
    final_model  = build_model(n_estimators=max(best_round + 1, 50))
    final_model.fit(tv_df[feature_cols].values, tv_df[target_col].values, verbose=False)

    _save_model(final_model, feature_cols, metrics, df)

    return {
        "model": final_model, "feature_cols": feature_cols, "metrics": metrics,
        "train_df": train_df, "test_df": test_df, "y_pred_test": y_pred_test,
    }


def _compute_metrics(y_test, y_pred_test, y_train, y_pred_train) -> dict:
    return {
        "test_mae":   round(float(mean_absolute_error(y_test,  y_pred_test)),  2),
        "test_rmse":  round(float(np.sqrt(mean_squared_error(y_test,  y_pred_test))),  2),
        "test_r2":    round(float(r2_score(y_test,  y_pred_test)),  3),
        "train_mae":  round(float(mean_absolute_error(y_train, y_pred_train)), 2),
        "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, y_pred_train))), 2),
        "train_r2":   round(float(r2_score(y_train, y_pred_train)), 3),
    }


def _save_model(model, feature_cols, metrics, df):
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(settings.xgb_model_path))
    log.info(f"Model saved    -> {settings.xgb_model_path}")

    meta = {
        "version":      "1.0",
        "trained_at":   datetime.utcnow().isoformat(),
        "n_products":   len(df),
        "n_features":   len(feature_cols),
        "feature_cols": feature_cols,
        "target_col":   "target_price",
        "metrics":      metrics,
        "model_params": model.get_params(),
        "price_range": {
            "target_min":  float(df["target_price"].min()),
            "target_max":  float(df["target_price"].max()),
            "target_mean": float(df["target_price"].mean()),
        },
    }
    settings.xgb_meta_path.write_text(json.dumps(meta, indent=2, default=str))
    log.info(f"Metadata saved -> {settings.xgb_meta_path}")


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def print_evaluation_report(result: dict, df: pd.DataFrame):
    """Per-product table: current → target → predicted → business signal."""
    feature_cols = result["feature_cols"]
    model        = result["model"]
    all_preds    = model.predict(df[feature_cols].values)

    log.info("\n" + "=" * 95)
    log.info("PER-PRODUCT EVALUATION")
    log.info("=" * 95)
    log.info(f"  {'Product':<34} {'Current':>9} {'Target':>9} {'Predicted':>10} {'Error':>8}  Signal")
    log.info("-" * 95)

    for i, (_, row) in enumerate(df.iterrows()):
        current   = float(row["current_price"])
        target    = float(row["target_price"])
        predicted = round(float(all_preds[i]), 2)
        error     = round(predicted - target, 2)
        comp_med  = float(row.get("comp_price_med", current))
        pct       = (predicted - comp_med) / comp_med * 100 if comp_med > 0 else 0
        signal    = "ABOVE" if pct > 15 else ("BELOW" if pct < -15 else "OK   ")

        log.info(
            f"  {str(row['product_name'])[:34]:<34} "
            f"${current:>8.2f} ${target:>8.2f} ${predicted:>9.2f} ${error:>+7.2f}  {signal}"
        )

    log.info("=" * 95)

    # Save CSV
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred     = round(float(all_preds[i]), 2)
        comp_med = float(row.get("comp_price_med", row["current_price"]))
        cur      = float(row["current_price"])
        rows.append({
            "product_id":      int(row["product_id"]),
            "product_name":    row["product_name"],
            "category":        row["category"],
            "current_price":   cur,
            "target_price":    float(row["target_price"]),
            "predicted_price": pred,
            "comp_price_med":  comp_med,
            "pct_vs_market":   round((pred - comp_med) / comp_med * 100, 1) if comp_med > 0 else 0,
            "pct_vs_current":  round((pred - cur) / cur * 100, 1) if cur > 0 else 0,
        })
    path = settings.results_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"\nPredictions saved -> {path}")


def print_feature_importance(model, feature_cols, top_n=15):
    """Terminal bar chart of top N feature importances."""
    pairs = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    log.info(f"\nTop {top_n} Feature Importances:")
    log.info("-" * 55)
    max_imp = pairs[0][1] if pairs else 1.0
    for name, imp in pairs[:top_n]:
        bar = "█" * int((imp / max_imp) * 30)
        log.info(f"  {name:<38} {imp:.4f}  {bar}")
    log.info("-" * 55)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [p[0] for p in pairs[:top_n]]
        imps  = [p[1] for p in pairs[:top_n]]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names[::-1], imps[::-1], color="#1B7F79")
        ax.set_xlabel("Feature Importance (gain)")
        ax.set_title("ARIA XGBoost — Top Feature Importances")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        path = settings.results_dir / "feature_importance.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info(f"Plot saved -> {path}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# INFERENCE API  — called by the ARIA agent
# ══════════════════════════════════════════════════════════════════════

def load_model():
    """Load trained model + metadata. Raises FileNotFoundError if untrained."""
    if not settings.xgb_model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {settings.xgb_model_path}.\n"
            "Run: python src/pricing_model.py"
        )
    model = XGBRegressor()
    model.load_model(str(settings.xgb_model_path))
    meta  = json.loads(settings.xgb_meta_path.read_text())
    return model, meta


def get_price_recommendation(product_id=None, product_row=None) -> dict:
    """
    Public inference API — called by ARIA agent tools.

    Accepts either:
      product_id  — looks up product in features.parquet
      product_row — pre-built feature row (pd.Series or dict)

    Returns:
    {
        product_id, product_name, category,
        current_price, recommended_price,  # clamped to min/max constraints
        lower_bound, upper_bound,
        comp_price_med, pct_vs_market, pct_vs_current,
        trend_index, inventory_pressure,
        confidence,   # "high" | "medium" | "low"
        rationale,    # human-readable explanation for agent
    }
    """
    model, meta  = load_model()
    feature_cols = meta["feature_cols"]

    if product_row is None:
        df      = load_features()
        matches = df[df["product_id"] == product_id]
        if matches.empty:
            raise ValueError(f"Product {product_id} not found in feature matrix.")
        product_row = matches.iloc[0]

    # Predict
    X = product_row[feature_cols].values.reshape(1, -1)
    predicted = round(float(model.predict(X)[0]), 2)

    # Clamp to price constraints — model predicts market signals, not business rules
    min_price = float(product_row.get("min_price", predicted * 0.70))
    max_price = float(product_row.get("max_price", predicted * 1.50))
    predicted = round(float(np.clip(predicted, min_price, max_price)), 2)

    lower = round(max(predicted * 0.92, min_price), 2)
    upper = round(min(predicted * 1.08, max_price), 2)

    # Context
    current      = float(product_row.get("current_price", predicted))
    comp_med     = float(product_row.get("comp_price_med", predicted))
    trend_idx    = float(product_row.get("trend_index_latest", 50))
    inv_pressure = int(product_row.get("inventory_pressure", 0))
    is_trending  = int(product_row.get("is_trending", 0))
    comp_count   = int(product_row.get("comp_count", 0))

    pct_vs_market  = round((predicted - comp_med) / comp_med * 100, 1) if comp_med > 0 else 0.0
    pct_vs_current = round((predicted - current)  / current  * 100, 1) if current  > 0 else 0.0

    # Confidence
    if comp_count >= 5 and trend_idx > 0:
        confidence = "high"
    elif comp_count >= 2 or trend_idx > 0:
        confidence = "medium"
    else:
        confidence = "low"

    # Rationale
    product_name = str(product_row.get("product_name", f"Product {product_id}"))
    parts = [f"Recommended price for {product_name}: ${predicted:.2f}."]

    if abs(pct_vs_market) <= 5:
        parts.append(f"In line with market median (${comp_med:.2f}).")
    elif pct_vs_market > 5:
        parts.append(f"{pct_vs_market:.1f}% above market median — justified by elevated demand.")
    else:
        parts.append(f"{abs(pct_vs_market):.1f}% below market — soft demand or inventory pressure.")

    if is_trending:
        parts.append(f"Demand trending up ({trend_idx:.0f}/100) — supports premium pricing.")
    elif trend_idx >= 60:
        parts.append(f"Demand healthy at {trend_idx:.0f}/100.")
    else:
        parts.append(f"Demand below average ({trend_idx:.0f}/100) — pricing conservatively.")

    if inv_pressure == 1:
        parts.append("Low stock — scarcity supports a price premium.")
    elif inv_pressure == -1:
        parts.append("Overstocked — competitive price helps clear inventory.")

    if pct_vs_current > 0:
        parts.append(f"Increase of ${predicted - current:.2f} ({pct_vs_current:+.1f}%) from current ${current:.2f}.")
    elif pct_vs_current < 0:
        parts.append(f"Reduction of ${current - predicted:.2f} ({pct_vs_current:+.1f}%) from current ${current:.2f}.")
    else:
        parts.append("No change from current price.")

    parts.append(
        f"Confidence: {confidence} "
        f"({comp_count} competitor listings, "
        f"demand signal {'available' if trend_idx > 0 else 'unavailable'})."
    )

    return {
        "product_id":         int(product_row.get("product_id", product_id or 0)),
        "product_name":       product_name,
        "category":           str(product_row.get("category", "unknown")),
        "current_price":      round(current, 2),
        "recommended_price":  predicted,
        "lower_bound":        lower,
        "upper_bound":        upper,
        "comp_price_med":     round(comp_med, 2),
        "pct_vs_market":      pct_vs_market,
        "pct_vs_current":     pct_vs_current,
        "trend_index":        trend_idx,
        "inventory_pressure": inv_pressure,
        "confidence":         confidence,
        "rationale":          " ".join(parts),
    }


def get_all_recommendations() -> pd.DataFrame:
    """Score every product. Returns DataFrame. Used by the agent review cycle."""
    df   = load_features()
    recs = []
    for _, row in df.iterrows():
        try:
            recs.append(get_price_recommendation(product_row=row))
        except Exception as e:
            log.warning(f"  Skipped product {row.get('product_id')}: {e}")
    return pd.DataFrame(recs)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ARIA XGBoost pricing model")
    parser.add_argument("--evaluate",   action="store_true", help="Per-product eval report")
    parser.add_argument("--predict",    action="store_true", help="Score with saved model")
    parser.add_argument("--importance", action="store_true", help="Feature importance table")
    args = parser.parse_args()

    if args.predict:
        try:
            recs = get_all_recommendations()
        except FileNotFoundError as e:
            log.error(str(e)); return 1

        log.info(f"\nRecommendations for {len(recs)} products:")
        log.info("-" * 75)
        for _, r in recs.iterrows():
            d = "up" if r["pct_vs_current"] > 0 else ("down" if r["pct_vs_current"] < 0 else "hold")
            log.info(
                f"  [{r['category']:<12}] {r['product_name'][:30]:<30} "
                f"${r['current_price']:.2f} -> ${r['recommended_price']:.2f} "
                f"{d} ({r['pct_vs_current']:+.1f}%)  [{r['confidence']}]"
            )
        return 0

    try:
        df = load_features()
    except FileNotFoundError as e:
        log.error(str(e)); return 1

    result = train(df)

    if args.importance or args.evaluate:
        print_feature_importance(result["model"], result["feature_cols"])

    if args.evaluate:
        print_evaluation_report(result, df)

    # Quick preview
    log.info("\nSample recommendations:")
    log.info("-" * 75)
    try:
        recs = get_all_recommendations()
        for _, r in recs.head(8).iterrows():
            d = "up" if r["pct_vs_current"] > 0 else ("dn" if r["pct_vs_current"] < 0 else "ok")
            log.info(
                f"  {r['product_name'][:32]:<32} "
                f"${r['current_price']:.2f} -> ${r['recommended_price']:.2f} "
                f"({r['pct_vs_current']:+.1f}%)  [{r['confidence']}]"
            )
    except Exception as e:
        log.warning(f"Preview failed: {e}")

    m = result["metrics"]
    log.info("\n" + "=" * 55)
    log.info("FINAL METRICS")
    log.info("=" * 55)
    log.info(f"  Test  MAE  : ${m['test_mae']:.2f}")
    log.info(f"  Test  RMSE : ${m['test_rmse']:.2f}")
    log.info(f"  Test  R2   : {m['test_r2']:.3f}")
    log.info(f"  Train MAE  : ${m['train_mae']:.2f}")
    log.info(f"  Train R2   : {m['train_r2']:.3f}")
    if "note" in m:
        log.info(f"  Note       : {m['note']}")
    log.info(f"\n  Model    -> {settings.xgb_model_path}")
    log.info(f"  --evaluate for full breakdown  |  --predict to score from saved model")
    return 0


if __name__ == "__main__":
    sys.exit(main())