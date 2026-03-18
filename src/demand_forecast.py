"""
src/demand_forecast.py
Prophet-based demand forecasting for ARIA.

Trains one Prophet model per product category using Google Trends data
as a demand proxy, then generates 30-day forward forecasts.

The forecast output feeds directly into the ARIA agent's repricing decisions:
  - Rising demand  → support pricing at or above market median
  - Falling demand → price competitively to maintain volume
  - Stable demand  → price near market median

One model per category (not per product) because:
  - Demand signals are at keyword/category level, not product level
  - Weekly Google Trends data gives ~52 points/year — enough for one robust model
  - Category models capture seasonal patterns that all products in the category share

Usage:
    python src/demand_forecast.py               # Train all categories
    python src/demand_forecast.py -c sports     # Train single category
    python src/demand_forecast.py --forecast    # Skip training, load saved + forecast
    python src/demand_forecast.py --preview     # Train + save forecast plots
"""
import sys
import json
import pickle
import logging
import argparse
import warnings
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Stan.*")
warnings.filterwarnings("ignore", message=".*cmdstanpy.*")

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
log = logging.getLogger("demand_forecast")

# Suppress Prophet / Stan noise
for noisy in ("prophet", "cmdstanpy", "pystan"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

try:
    from prophet import Prophet
except ImportError:
    log.error("prophet not installed. Run: pip install prophet")
    sys.exit(1)

from db.models import get_db, DemandSignal
from config.settings import get_settings

settings = get_settings()

# ── Category → primary keyword mapping ───────────────────────────────
# Must match CATEGORY_KEYWORD in src/features.py
CATEGORY_KEYWORD = {
    "electronics": "wireless headphones",
    "fashion":     "leather wallet",
    "home_goods":  "bamboo cutting board",
    "sports":      "yoga mat",
}

VALID_CATEGORIES  = list(CATEGORY_KEYWORD.keys())
FORECAST_DAYS     = 30
MIN_WEEKS_NEEDED  = 12
MODEL_STALE_DAYS  = 7    # Flag model as stale if older than this


def _model_path(category: str) -> Path:
    return settings.prophet_dir / f"prophet_{category}.pkl"

def _meta_path(category: str) -> Path:
    return settings.prophet_dir / f"prophet_{category}_meta.json"

def _forecast_path(category: str) -> Path:
    return settings.results_dir / "prophet" / f"forecasts_{category}.csv"


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_keyword_data(keyword: str) -> pd.DataFrame:
    """
    Load demand signals for a keyword from DB.
    Returns DataFrame with [ds, y] columns as required by Prophet.
    Converts ORM objects to dicts inside session to avoid DetachedInstanceError.
    """
    with get_db() as db:
        rows = (
            db.query(DemandSignal)
            .filter(DemandSignal.keyword == keyword)
            .order_by(DemandSignal.week_date)
            .all()
        )
        data = [{"ds": r.week_date, "y": float(r.trend_index)} for r in rows]

    if not data:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"]  = df["y"].astype(float)
    df       = df.drop_duplicates(subset="ds", keep="last").reset_index(drop=True)

    # Normalize y to 0-100 if values are out of range.
    # SerpAPI sometimes returns raw interest values instead of the normalized
    # 0-100 index. Mixed DB state (seeded rows + SerpAPI rows) can also cause
    # scale mismatches. Normalizing here makes Prophet stable regardless.
    y_max = df["y"].max()
    if y_max > 100:
        log.warning(
            f"  trend_index max={y_max:.0f} exceeds 100 — "
            "normalizing to 0-100 (SerpAPI scale mismatch)"
        )
        df["y"] = (df["y"] / y_max * 100).round(1)

    df["y"] = df["y"].clip(0, 100)

    log.info(f"  Loaded {len(df)} weeks for '{keyword}' "
             f"({df['ds'].min().date()} to {df['ds'].max().date()})  "
             f"y range: {df['y'].min():.0f}-{df['y'].max():.0f}")
    return df


def make_synthetic_data(keyword: str, n_weeks: int = 104) -> pd.DataFrame:
    """
    Generates realistic synthetic demand data when real data is unavailable
    or insufficient (< MIN_WEEKS_NEEDED).

    Uses category-appropriate seasonal patterns so the model is still meaningful.
    Runs automatically — a warning is logged when this fallback is used.
    Remove once you have 12+ weeks of real SerpAPI trend data.
    """
    log.warning(f"  No DB data for '{keyword}' — using synthetic fallback for training.")

    np.random.seed(abs(hash(keyword)) % 2**31)
    dates = pd.date_range(end=date.today(), periods=n_weeks, freq="W")

    base = 45 + np.linspace(0, 10, n_weeks)

    # Seasonal peak by keyword — from real Google Trends patterns
    peaks = {
        "wireless headphones":          11,
        "bluetooth speaker":            7,
        "smart watch":                  11,
        "mechanical keyboard":          11,
        "leather wallet":               11,
        "tote bag":                     6,
        "minimalist watch":             4,
        "wool beanie":                  10,
        "bamboo cutting board":         11,
        "stainless steel water bottle": 6,
        "soy candle":                   11,
        "essential oil diffuser":       11,
        "yoga mat":                     1,
        "resistance bands":             1,
        "running shoes":                3,
        "foam roller":                  1,
    }
    peak_month = peaks.get(keyword.lower(), 11)
    # t in years — dividing by 52 means seasonality completes one cycle per year
    t          = np.arange(n_weeks) / 52
    seasonal   = 18 * np.sin(2 * np.pi * (t - (peak_month - 1) / 12))
    noise      = np.random.normal(0, 4, n_weeks)
    y          = np.clip(base + seasonal + noise, 0, 100).round(0)

    return pd.DataFrame({"ds": dates, "y": y})


# ══════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

def build_prophet_model(category: str) -> Prophet:
    """
    Configure Prophet model for weekly Google Trends data.

    Parameter rationale:
      yearly_seasonality=True       Captures annual demand cycles (holiday, summer, etc.)
      weekly_seasonality=False      Trends data is weekly — no intra-week pattern to model
      daily_seasonality=False       Same reason — no daily pattern in weekly data
      changepoint_prior_scale=0.08  Moderate trend flexibility. 0.05=rigid, 0.5=very flexible.
                                    0.08 allows the trend to shift but not wildly overfit.
      seasonality_prior_scale=12.0  Stronger seasonality — appropriate for retail demand.
      interval_width=0.80           80% confidence interval — tighter = more decisive signals.
      seasonality_mode="additive"   Seasonal effect adds to trend rather than multiplying.
                                    Better for stable-amplitude seasonal patterns.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=12.0,
        interval_width=0.80,
        seasonality_mode="additive",
    )

    # US retail holidays — captures Black Friday, Christmas, New Year, etc.
    model.add_country_holidays(country_name="US")

    # Quarterly retail cycle — back-to-school, end-of-quarter promotions
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)

    # Category-specific extra seasonality
    if category in ("electronics", "home_goods"):
        # Strong Nov-Dec holiday spike
        model.add_seasonality(name="holiday_spike", period=365.25 / 6, fourier_order=3)
    elif category == "sports":
        # January fitness peak + summer outdoor peak
        model.add_seasonality(name="fitness_cycle", period=365.25 / 2, fourier_order=3)
    elif category == "fashion":
        # Spring and fall fashion seasons
        model.add_seasonality(name="fashion_season", period=365.25 / 2, fourier_order=4)

    return model


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train_category(category: str) -> dict:
    """
    Train a Prophet model for one category.

    Pipeline:
      1. Load demand signals from DB
      2. Fall back to synthetic data if insufficient
      3. Hold out last 4 weeks for evaluation
      4. Fit on training data, evaluate on holdout
      5. Refit on ALL data (Prophet can only be fit once — new instance needed)
      6. Save model + metadata + forecast CSV

    Returns result dict with model, metrics, forecast.
    """
    keyword = CATEGORY_KEYWORD.get(category)
    if not keyword:
        raise ValueError(f"Unknown category '{category}'. Valid: {VALID_CATEGORIES}")

    log.info(f"\n{'='*55}")
    log.info(f"Category: {category.upper()}  |  Keyword: '{keyword}'")
    log.info(f"{'='*55}")

    # ── Load data ──────────────────────────────────────────────────────
    df = load_keyword_data(keyword)

    if len(df) < MIN_WEEKS_NEEDED:
        log.warning(f"  Only {len(df)} weeks — need {MIN_WEEKS_NEEDED}. Falling back to synthetic.")
        df = make_synthetic_data(keyword)

    log.info(f"  Training on {len(df)} weeks")

    # ── Train/eval split — hold out last 4 weeks ──────────────────────
    split_idx = max(len(df) - 4, int(len(df) * 0.85))
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()
    log.info(f"  Train: {len(train_df)} weeks  |  Holdout: {len(test_df)} weeks")

    # ── Fit on train data ──────────────────────────────────────────────
    log.info("  Fitting Prophet model...")
    eval_model = build_prophet_model(category)
    eval_model.fit(train_df)

    # ── Evaluate on holdout ───────────────────────────────────────────
    if not test_df.empty:
        test_fc  = eval_model.predict(test_df[["ds"]])
        # Clip predictions to valid range before computing metrics
        # Raw Prophet output can exceed [0, 100] — clipping gives realistic MAE
        test_fc["yhat"] = test_fc["yhat"].clip(0, 100)
        merged = test_df.merge(test_fc[["ds", "yhat"]], on="ds", how="inner")
        mae  = float(np.mean(np.abs(merged["y"] - merged["yhat"])))            if not merged.empty else 0.0
        rmse = float(np.sqrt(np.mean((merged["y"] - merged["yhat"]) ** 2)))    if not merged.empty else 0.0
        log.info(f"  Holdout eval ({len(merged)} weeks) — MAE: {mae:.2f}  RMSE: {rmse:.2f}")
    else:
        mae, rmse = 0.0, 0.0
        log.warning("  No holdout data — skipping eval")

    # ── Refit on full data for production model ────────────────────────
    # Prophet cannot be updated — must create a fresh instance and fit from scratch
    log.info("  Refitting on full data for production model...")
    final_model = build_prophet_model(category)
    final_model.fit(df)

    # ── Generate forecast ──────────────────────────────────────────────
    log.info(f"  Generating {FORECAST_DAYS}-day forecast...")
    future   = final_model.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    forecast = final_model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 100).round(2)

    # ── Save model ─────────────────────────────────────────────────────
    settings.prophet_dir.mkdir(parents=True, exist_ok=True)
    with open(_model_path(category), "wb") as f:
        pickle.dump(final_model, f)
    log.info(f"  Model saved  -> {_model_path(category)}")

    # ── Save metadata ──────────────────────────────────────────────────
    upcoming = forecast[forecast["ds"] > df["ds"].max()].head(FORECAST_DAYS)
    meta = {
        "version":       "1.0",
        "trained_at":    datetime.utcnow().isoformat(),
        "category":      category,
        "keyword":       keyword,
        "n_weeks":       len(df),
        "mae":           round(mae, 2),
        "rmse":          round(rmse, 2),
        "forecast_days": FORECAST_DAYS,
        "forecast_avg":  round(float(upcoming["yhat"].mean()), 1) if not upcoming.empty else None,
        "current_index": round(float(df["y"].iloc[-1]), 1),
    }
    _meta_path(category).write_text(json.dumps(meta, indent=2))
    log.info(f"  Meta saved   -> {_meta_path(category)}")

    # ── Save forecast CSV ──────────────────────────────────────────────
    fc_path = _forecast_path(category)
    fc_path.parent.mkdir(parents=True, exist_ok=True)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["category"] = category
    out["keyword"]  = keyword
    out.to_csv(fc_path, index=False)
    log.info(f"  Forecast CSV -> {fc_path}")

    # ── Log 30-day outlook ─────────────────────────────────────────────
    if not upcoming.empty:
        current  = float(df["y"].iloc[-1])
        avg_fc   = float(upcoming["yhat"].mean())
        direction = "rising" if avg_fc > current + 5 else ("falling" if avg_fc < current - 5 else "stable")
        log.info(f"  30-day outlook: avg={avg_fc:.1f}  current={current:.0f}  [{direction}]  "
                 f"CI: {upcoming['yhat_lower'].mean():.1f}–{upcoming['yhat_upper'].mean():.1f}")

    return {
        "category":   category,
        "keyword":    keyword,
        "n_weeks":    len(df),
        "mae":        round(mae, 2),
        "rmse":       round(rmse, 2),
        "model":      final_model,
        "model_path": str(_model_path(category)),
        "forecast":   forecast,
    }


# ══════════════════════════════════════════════════════════════════════
# INFERENCE API — called by the ARIA agent
# ══════════════════════════════════════════════════════════════════════

def load_prophet_model(category: str) -> Prophet:
    """
    Load a trained Prophet model from disk.
    Renamed from load_model to avoid name conflicts when imported.
    """
    path = _model_path(category)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model for '{category}'.\n"
            f"Run: python src/demand_forecast.py -c {category}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def is_model_stale(category: str) -> bool:
    """
    Returns True if the model was trained more than MODEL_STALE_DAYS ago,
    or if no metadata file exists. The agent uses this to decide whether
    to retrain before making pricing decisions.
    """
    meta_path = _meta_path(category)
    if not meta_path.exists():
        return True
    try:
        meta     = json.loads(meta_path.read_text())
        trained  = datetime.fromisoformat(meta["trained_at"])
        age_days = (datetime.utcnow() - trained).days
        return age_days >= MODEL_STALE_DAYS
    except Exception:
        return True


def get_demand_forecast(category: str, days: int = FORECAST_DAYS) -> dict:
    """
    Public inference API — called by the ARIA agent tools.

    Loads the saved Prophet model, generates a fresh forecast,
    and returns a structured signal the agent can reason about.

    Returns:
    {
        category, keyword,
        current_index:   float,   latest known trend index (0-100)
        forecast_avg:    float,   average predicted index over next N days
        forecast_high:   float,   peak in forecast window
        forecast_low:    float,   trough in forecast window
        trend_direction: str,     "rising" | "falling" | "stable"
        confidence_low:  float,   lower bound average
        confidence_high: float,   upper bound average
        model_age_days:  int,     days since model was trained
        is_stale:        bool,    True if model needs retraining
        demand_signal:   str,     human-readable summary for agent context
        forecast_df:     DataFrame
    }
    """
    keyword = CATEGORY_KEYWORD.get(category)
    if not keyword:
        raise ValueError(f"Unknown category: '{category}'")

    model = load_prophet_model(category)

    actuals = load_keyword_data(keyword)
    if actuals.empty:
        actuals = make_synthetic_data(keyword)

    current_index    = float(actuals["y"].iloc[-1]) if not actuals.empty else 50.0
    last_actual_date = actuals["ds"].max() if not actuals.empty else pd.Timestamp.now()

    # Generate forecast
    future   = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 100)

    # Focus on forward window only
    future_fc = forecast[forecast["ds"] > last_actual_date].head(days)
    if future_fc.empty:
        future_fc = forecast.tail(days)

    avg_fc  = float(future_fc["yhat"].mean())
    high_fc = float(future_fc["yhat"].max())
    low_fc  = float(future_fc["yhat"].min())
    conf_lo = float(future_fc["yhat_lower"].mean())
    conf_hi = float(future_fc["yhat_upper"].mean())

    delta     = avg_fc - current_index
    direction = "rising" if delta > 5 else ("falling" if delta < -5 else "stable")

    # Model age
    meta_p = _meta_path(category)
    try:
        meta      = json.loads(meta_p.read_text())
        trained   = datetime.fromisoformat(meta["trained_at"])
        age_days  = (datetime.utcnow() - trained).days
    except Exception:
        age_days = -1

    stale = age_days >= MODEL_STALE_DAYS or age_days == -1

    # Human-readable demand signal for the agent
    if direction == "rising":
        signal = (
            f"Demand for {category} ({keyword}) expected to RISE over the next {days} days. "
            f"Current index: {current_index:.0f}/100, forecast avg: {avg_fc:.0f}/100 "
            f"(peak {high_fc:.0f}). Rising demand supports pricing at or above market median."
        )
    elif direction == "falling":
        signal = (
            f"Demand for {category} ({keyword}) expected to FALL over the next {days} days. "
            f"Current index: {current_index:.0f}/100, forecast avg: {avg_fc:.0f}/100 "
            f"(low {low_fc:.0f}). Falling demand — price competitively to maintain volume."
        )
    else:
        signal = (
            f"Demand for {category} ({keyword}) expected to remain STABLE "
            f"over the next {days} days. "
            f"Current index: {current_index:.0f}/100, forecast avg: {avg_fc:.0f}/100. "
            f"Stable demand — price near market median."
        )

    if stale:
        signal += f" [Note: model is {age_days}d old — consider retraining.]"

    return {
        "category":        category,
        "keyword":         keyword,
        "current_index":   round(current_index, 1),
        "forecast_avg":    round(avg_fc, 1),
        "forecast_high":   round(high_fc, 1),
        "forecast_low":    round(low_fc, 1),
        "trend_direction": direction,
        "confidence_low":  round(conf_lo, 1),
        "confidence_high": round(conf_hi, 1),
        "model_age_days":  age_days,
        "is_stale":        stale,
        "demand_signal":   signal,
        "forecast_df":     future_fc,
    }


def get_all_forecasts(days: int = FORECAST_DAYS) -> dict:
    """
    Returns demand forecasts for all categories.
    Used by the agent's pricing review cycle to get a full market picture
    before making repricing decisions.

    Returns: {category: forecast_dict, ...}
    Skips categories with missing models rather than crashing.
    """
    results = {}
    for category in VALID_CATEGORIES:
        try:
            results[category] = get_demand_forecast(category, days=days)
        except FileNotFoundError:
            log.warning(f"  No model for '{category}' — skipping. Run training first.")
        except Exception as e:
            log.error(f"  Forecast failed for '{category}': {e}")
    return results


# ══════════════════════════════════════════════════════════════════════
# PLOTTING (optional)
# ══════════════════════════════════════════════════════════════════════

def plot_forecast(result: dict):
    """Save a forecast plot to results/prophet/forecast_{category}.png"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    model    = result["model"]
    forecast = result["forecast"]
    category = result["category"]

    fig = model.plot(forecast, figsize=(12, 5))
    plt.title(f"ARIA Demand Forecast — {category.upper()} ({result['keyword']})")
    plt.xlabel("Date")
    plt.ylabel("Trend Index (0–100)")
    plt.tight_layout()

    plot_path = settings.results_dir / "prophet" / f"forecast_{category}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    log.info(f"  Plot saved -> {plot_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ARIA Prophet demand forecasting")
    parser.add_argument("--category", "-c", choices=VALID_CATEGORIES, default=None)
    parser.add_argument("--forecast", action="store_true",
                        help="Skip training — load saved models and forecast")
    parser.add_argument("--preview",  action="store_true",
                        help="Save forecast plots")
    parser.add_argument("--days",     type=int, default=FORECAST_DAYS)
    args = parser.parse_args()

    categories = [args.category] if args.category else VALID_CATEGORIES
    log.info("ARIA — Prophet Demand Forecasting")
    log.info(f"Categories : {categories}")
    log.info(f"Days ahead : {args.days}")

    results = []

    if args.forecast:
        # Inference only — load saved models
        for cat in categories:
            try:
                fc = get_demand_forecast(cat, days=args.days)
                stale_flag = " [STALE]" if fc["is_stale"] else ""
                log.info(
                    f"\n  [{cat}]{stale_flag}  direction={fc['trend_direction']}  "
                    f"current={fc['current_index']}  forecast_avg={fc['forecast_avg']}"
                )
                log.info(f"  Signal: {fc['demand_signal']}")
                results.append({"category": cat, "status": "ok"})
            except FileNotFoundError as e:
                log.error(f"  {e}")
                results.append({"category": cat, "status": "no_model"})
    else:
        # Full training run
        for cat in categories:
            try:
                result = train_category(cat)
                results.append({
                    "category": cat,
                    "status":   "ok",
                    "mae":      result["mae"],
                    "rmse":     result["rmse"],
                    "n_weeks":  result["n_weeks"],
                })
                if args.preview:
                    plot_forecast(result)
            except Exception as e:
                log.error(f"  Failed '{cat}': {e}")
                results.append({"category": cat, "status": "error", "error": str(e)})

    # Summary
    log.info("\n" + "=" * 55)
    log.info("SUMMARY")
    log.info("=" * 55)
    for r in results:
        if r["status"] == "ok" and "mae" in r:
            log.info(f"  [OK] {r['category']:<15}  {r['n_weeks']} weeks  "
                     f"MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}")
        elif r["status"] == "ok":
            log.info(f"  [OK] {r['category']}")
        else:
            log.info(f"  [!!] {r['category']:<15}  {r.get('error', r['status'])}")

    log.info("\nSaved models:")
    for cat in categories:
        path  = _model_path(cat)
        stale = " [STALE]" if is_model_stale(cat) else ""
        size  = f"{path.stat().st_size / 1024:.0f}KB" if path.exists() else "MISSING"
        log.info(f"  {path.name}  ({size}){stale}")

    errors = [r for r in results if r["status"] != "ok"]
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())