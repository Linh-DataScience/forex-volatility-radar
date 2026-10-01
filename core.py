"""
core.py
--------
Shared logic used by every Streamlit page (app.py's Detail + Radar pages,
and later copilot.py): cached data/model loading, the live prediction
pipeline, SHAP driver extraction, and the numeric building blocks for risk
guidance (volatility-based stop sizing, position-size calculation).

Kept deliberately separate from copilot.py: this file only computes NUMBERS
(probabilities, pip distances, lot sizes). Turning those numbers into a
natural-language risk statement is copilot.py's job -- that separation is
what makes copilot.py swappable for a real GPT call later without touching
any of the math here.

Stretch goal included: get_news() -- local VADER sentiment over yfinance
headlines, no API key, no external LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data_loader import PAIRS, load_ohlc_with_flag, to_timeframe
from features import compute_features
from features import feature_columns as base_feature_columns

MODELS_DIR = Path(__file__).parent / "models"
TIMEFRAMES = ["1h", "4h"]

# ---------------------------------------------------------------------------
# 1. Shared constants (dashboard theme, human-readable feature names)
# ---------------------------------------------------------------------------

# Clean light theme (switched from the original dark-navy palette on
# request). Same brand hues throughout (amber accent, red "stormy",
# green "calm") just re-tuned a shade darker where they double as text,
# so everything still passes basic contrast on a white page. These are
# the canonical color values every page/component should reuse from
# here on, so the theme only ever needs to change in one place.
THEME = {
    "bg": "#ffffff",
    "surface": "#f4f5f7",
    "accent": "#b45309",
    "text": "#111827",
    "text_muted": "#6b7280",
    "stormy": "#dc2626",
    "calm": "#16a34a",
}

# English on purpose: this dict feeds directly into dashboard-facing SHAP
# text (deployment content), not just internal debugging output.
FEATURE_LABELS: dict[str, str] = {
    "f_true_range": "the current candle's true range",
    "f_tr_lag1": "the true range one candle ago",
    "f_tr_lag2": "the true range two candles ago",
    "f_atr14": "average true range (14)",
    "f_realized_vol10": "realized volatility (last 10 candles)",
    "f_regime_ratio": "current range vs. its 50-candle median",
    "f_abs_last_return": "the size of the last price move",
    "f_rsi14": "RSI (14)",
    "f_weekday": "the day of the week",
    "f_session_asia": "the Asia session",
    "f_session_london": "the London session",
    "f_session_overlap": "the London/New York overlap",
    "f_session_ny": "the New York session",
}

# One-hot pair/timeframe columns are structural (identical for every
# prediction of that pair/timeframe) -- excluded from the "top drivers"
# shown to the user by default, since they don't explain THIS moment,
# just which series it is.
_STRUCTURAL_PREFIXES = ("f_pair_", "f_tf_")


# ---------------------------------------------------------------------------
# 2. Cached loading (models, feature columns, live candle data)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_calibrated_model():
    """Main model: calibrated XGBoost, used for the probability shown to the user."""
    return joblib.load(MODELS_DIR / "xgb_calibrated.pkl")


@st.cache_resource
def load_raw_model():
    """Raw (uncalibrated) XGBoost, kept separately for TreeSHAP (see shap_drivers)."""
    return joblib.load(MODELS_DIR / "xgb_raw.pkl")


@st.cache_resource
def load_feature_columns() -> list[str]:
    with open(MODELS_DIR / "feature_columns.json") as f:
        return json.load(f)


@st.cache_data(ttl=15 * 60)
def _load_pair_data_cached(pair: str) -> tuple[pd.DataFrame, bool]:
    return load_ohlc_with_flag(pair)


def load_pair_data(pair: str, force_refresh: bool = False) -> tuple[pd.DataFrame, bool]:
    """
    1h candles for one pair. Returns (df, is_synthetic).

    force_refresh=True (the manual "Refresh now" button) skips BOTH the
    Streamlit cache above AND data_loader's own file-cache staleness check
    (use_cache=False) -- otherwise a file cache that's still "fresh enough"
    by the 2h/8h rule could make the button look like it did nothing.
    """
    if force_refresh:
        return load_ohlc_with_flag(pair, use_cache=False)
    return _load_pair_data_cached(pair)


# One local VADER instance -- it's a lexicon lookup, not a model that needs
# retraining, so loading it once at import time is enough.
_VADER = SentimentIntensityAnalyzer()


def _extract_headline(item: dict) -> str | None:
    """
    yfinance's news schema has changed across versions -- sometimes the
    title sits directly on the item, sometimes nested under "content".
    Handle both rather than assuming one shape and breaking silently.
    """
    if isinstance(item.get("title"), str):
        return item["title"]
    content = item.get("content") or {}
    if isinstance(content.get("title"), str):
        return content["title"]
    return None


def get_news(pair: str, max_headlines: int = 5, force_refresh: bool = False) -> dict:
    """
    Recent news headlines for a pair, scored locally with VADER. Same
    force_refresh contract as load_pair_data: bypasses the 15-min cache
    for one manually-triggered call.
    """
    if force_refresh:
        return _fetch_news(pair, max_headlines)
    return _get_news_cached(pair, max_headlines)


@st.cache_data(ttl=15 * 60)
def _get_news_cached(pair: str, max_headlines: int) -> dict:
    return _fetch_news(pair, max_headlines)


def _fetch_news(pair: str, max_headlines: int) -> dict:
    """
    No API key, no LLM call -- just a lexicon-based sentiment score per
    headline. FX news is often sparse or empty; that's a normal quiet day,
    not an error, so this always returns a usable (possibly empty) result
    rather than raising.
    """
    import yfinance as yf

    symbol = PAIRS[pair]
    try:
        raw_items = yf.Ticker(symbol).news or []
    except Exception as exc:
        print(f"[core] news fetch failed for {pair}: {exc}")
        raw_items = []

    headlines = []
    for item in raw_items:
        title = _extract_headline(item)
        if not title:
            continue
        score = _VADER.polarity_scores(title)["compound"]
        label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        headlines.append({"title": title, "score": score, "label": label})
        if len(headlines) >= max_headlines:
            break

    if not headlines:
        return {"headlines": [], "avg_score": 0.0, "overall": "no recent news"}

    avg_score = sum(h["score"] for h in headlines) / len(headlines)
    overall = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"
    return {"headlines": headlines, "avg_score": avg_score, "overall": overall}


# ---------------------------------------------------------------------------
# 3. Live feature row (same logic app.py used in MVP1, now shared so every
#    page builds "now" the exact same way the model was trained)
# ---------------------------------------------------------------------------

def build_live_feature_row(
    raw_1h: pd.DataFrame, pair: str, timeframe: str, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Series]:
    """
    Build exactly one row of model input for "right now". Needs ~50 candles
    of history for the rolling features, not just the last candle alone --
    so this always recomputes features on the full recent series and takes
    the last fully-computed row, never a single isolated candle.
    """
    candles = raw_1h if timeframe == "1h" else to_timeframe(raw_1h, "4h")
    feats = compute_features(candles)
    latest = feats.dropna().iloc[[-1]]
    latest_timestamp = latest.index[0]

    row = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    for col in base_feature_columns(feats):
        if col in row.columns:
            row.at[0, col] = latest.iloc[0][col]

    pair_col, tf_col = f"f_pair_{pair}", f"f_tf_{timeframe}"
    if pair_col in row.columns:
        row.at[0, pair_col] = 1
    if tf_col in row.columns:
        row.at[0, tf_col] = 1

    return row, latest_timestamp, candles.iloc[-1]


# ---------------------------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------------------------

def predict_vol(
    pair: str,
    timeframe: str,
    force_refresh: bool = False,
    raw_1h: pd.DataFrame | None = None,
    is_synthetic: bool | None = None,
) -> dict:
    """
    The main entry point every page calls: builds today's feature row and
    returns everything downstream code needs (dashboard display, SHAP, the
    co-pilot) in one dict, so nothing gets recomputed twice per page.

    force_refresh=True passes through to load_pair_data -- see its
    docstring for why this needs to skip more than just Streamlit's cache.

    raw_1h/is_synthetic: pass these in when the caller already fetched the
    candles itself (the Detail page needs raw_1h for its own chart/warning).
    Without this, force_refresh=True would trigger a SECOND independent
    live-fetch-with-retries here -- doubling the wait on every manual
    refresh, and, since synthetic fallback data is randomly generated,
    potentially returning DIFFERENT candles than the ones shown in the
    chart. Leave both None to fetch normally (e.g. from the Radar page,
    which has no separate chart to keep in sync).
    """
    feature_cols = load_feature_columns()
    model = load_calibrated_model()

    if raw_1h is None or is_synthetic is None:
        raw_1h, is_synthetic = load_pair_data(pair, force_refresh=force_refresh)
    row, latest_timestamp, last_candle = build_live_feature_row(raw_1h, pair, timeframe, feature_cols)

    proba_stormy = float(model.predict_proba(row[feature_cols])[0, 1])

    return {
        "pair": pair,
        "timeframe": timeframe,
        "proba_stormy": proba_stormy,
        "proba_calm": 1.0 - proba_stormy,
        "label": "STORMY" if proba_stormy >= 0.5 else "CALM",
        "confidence": abs(proba_stormy - 0.5) * 2,  # 0 = coin-flip, 1 = maximally sure
        "latest_timestamp": latest_timestamp,
        "last_candle": last_candle,
        "is_synthetic": is_synthetic,
        "feature_row": row,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# 5. SHAP interpretability (TreeSHAP via XGBoost's own predict(), NOT the
#    `shap` library -- see PROJECT HANDOFF: `shap` 0.49 breaks on xgboost
#    3.x's base_score parsing)
# ---------------------------------------------------------------------------

def shap_drivers(prediction: dict, top_n: int = 5, exclude_structural: bool = True) -> dict:
    """
    Explain ONE prediction (the dict returned by predict_vol) using
    XGBoost's built-in TreeSHAP: booster.predict(..., pred_contribs=True).
    Same algorithm as the `shap` library, no extra dependency, robust
    against the xgboost-3.x incompatibility.

    Returns the top_n features by |contribution|, each with a ready-to-
    display English sentence. Contribution > 0 pushed THIS prediction
    toward STORMY, < 0 pushed it toward CALM (additive log-odds
    contributions -- they sum to the model's raw margin output, plus bias).
    """
    raw_model = load_raw_model()
    feature_cols = prediction["feature_cols"]
    row = prediction["feature_row"][feature_cols]

    booster = raw_model.get_booster()
    dmatrix = xgb.DMatrix(row, feature_names=feature_cols)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]

    bias = float(contribs[-1])
    values = contribs[:-1]

    drivers = []
    for col, val in zip(feature_cols, values):
        if exclude_structural and col.startswith(_STRUCTURAL_PREFIXES):
            continue
        drivers.append(
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "contribution": float(val),
                "direction": "stormy" if val > 0 else "calm",
            }
        )

    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    top = drivers[:top_n]

    for d in top:
        verb = "pushed this forecast toward STORMY" if d["direction"] == "stormy" else "pushed this forecast toward CALM"
        label = d["label"][0].upper() + d["label"][1:]
        d["sentence"] = f"{label} {verb} (impact {d['contribution']:+.3f})"

    return {"bias": bias, "top_drivers": top}


# ---------------------------------------------------------------------------
# 6. Risk guidance building blocks (numbers only -- copilot.py turns these
#    into a natural-language statement)
# ---------------------------------------------------------------------------

def _regime_bucket(proba_stormy: float) -> tuple[str, float, float]:
    """(regime label, stop-distance multiplier, position-size factor)."""
    if proba_stormy >= 0.75:
        return "high", 1.5, 0.5
    if proba_stormy >= 0.55:
        return "elevated", 1.2, 0.75
    if proba_stormy <= 0.25:
        return "low", 0.8, 1.1
    return "normal", 1.0, 1.0


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def price_diff_to_pips(pair: str, price_diff: float) -> float:
    return abs(price_diff) / pip_size(pair)


def pip_value_per_lot(pair: str, price: float, lot_size: float = 100_000) -> float:
    """
    USD value of one pip for one standard lot.
    USD-quote pairs (EUR/USD, GBP/USD, AUD/USD): fixed at $10/pip/lot.
    JPY-quote pairs (USD/JPY): pip value is in JPY, converted to USD by
    dividing by the current price.
    """
    if pair.endswith("JPY"):
        return (pip_size(pair) * lot_size) / price
    return pip_size(pair) * lot_size


def vol_recommendation(prediction: dict) -> dict:
    """
    Translate a raw stormy-probability into a volatility-based stop/size
    recommendation. Direction (long/short) is deliberately NOT part of
    this -- that stays the trader's decision, never the model's.
    """
    proba_stormy = prediction["proba_stormy"]
    pair = prediction["pair"]
    atr_price = float(prediction["feature_row"].iloc[0].get("f_atr14", 0.0))
    atr_pips = price_diff_to_pips(pair, atr_price)

    regime, stop_multiplier, size_factor = _regime_bucket(proba_stormy)
    suggested_stop_pips = round(atr_pips * stop_multiplier, 1)

    return {
        "regime": regime,
        "proba_stormy": proba_stormy,
        "confidence": prediction["confidence"],
        "atr_pips": round(atr_pips, 1),
        "stop_multiplier": stop_multiplier,
        "size_factor": size_factor,
        "suggested_stop_pips": suggested_stop_pips,
    }


def position_plan(
    pair: str,
    direction: str,
    entry_price: float,
    account_balance: float,
    risk_pct: float,
    stop_pips: float,
    reward_risk_ratio: float = 1.5,
) -> dict:
    """
    Constant-risk position sizing: lots = risk_amount / (stop_pips * pip_value).
    `direction` ("long" | "short") is a required INPUT the trader provides --
    never derived from the model, which only ever speaks about magnitude.
    """
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")

    risk_amount = account_balance * (risk_pct / 100.0)
    pip_val = pip_value_per_lot(pair, entry_price)
    lots = risk_amount / (stop_pips * pip_val)

    distance = stop_pips * pip_size(pair)
    if direction == "long":
        stop_loss = entry_price - distance
        take_profit = entry_price + distance * reward_risk_ratio
    else:
        stop_loss = entry_price + distance
        take_profit = entry_price - distance * reward_risk_ratio

    return {
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
        "stop_pips": stop_pips,
        "reward_risk_ratio": reward_risk_ratio,
        "risk_amount": round(risk_amount, 2),
        "lots": round(lots, 2),
    }


if __name__ == "__main__":
    # Manual smoke test: python core.py
    # (the @st.cache_* decorators still work outside `streamlit run`; they
    # just skip real caching without an active Streamlit script context)
    pred = predict_vol("EUR/USD", "1h")
    print(f"{pred['pair']} {pred['timeframe']}: {pred['label']} "
          f"({pred['proba_stormy']:.1%} stormy, synthetic={pred['is_synthetic']})")

    shap_result = shap_drivers(pred)
    print("\nTop SHAP drivers:")
    for d in shap_result["top_drivers"]:
        print(" -", d["sentence"])

    vol_rec = vol_recommendation(pred)
    print("\nVolatility recommendation:", vol_rec)

    plan = position_plan(
        pair="EUR/USD", direction="long", entry_price=float(pred["last_candle"]["close"]),
        account_balance=10_000, risk_pct=1.0, stop_pips=vol_rec["suggested_stop_pips"],
    )
    print("\nPosition plan:", plan)
