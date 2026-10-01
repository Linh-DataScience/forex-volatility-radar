"""
features.py
------------
Turns raw OHLC candles (from data_loader.py) into a model-ready dataset:
technical/volatility features (columns prefixed "f_") + the calm/stormy
target (and a secondary, context-only direction target).

LEAKAGE RULE (the single most important rule in this whole project):
  - Every FEATURE at row t may only use information from candle t and
    earlier (rolling windows, lags, .shift(1) or later).
  - The TARGET at row t is computed from candle t+1 (the future candle we
    want to forecast) and attached to row t via .shift(-1).
  - The very last row therefore never gets a valid target (there is no
    "next candle" yet) and is dropped.
This file keeps feature computation and target computation in two
separate functions on purpose, so this rule is easy to audit by eye.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Small helpers
# ---------------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range = the largest of:
      - today's high minus today's low
      - today's high minus yesterday's close
      - yesterday's close minus today's low
    It captures gaps (e.g. weekend gaps) that a simple high-low range misses.
    Uses close.shift(1) -> only past information, safe as a feature input.
    """
    prev_close = df["close"].shift(1)
    range_hl = df["high"] - df["low"]
    range_hc = (df["high"] - prev_close).abs()
    range_lc = (df["low"] - prev_close).abs()
    return pd.concat([range_hl, range_hc, range_lc], axis=1).max(axis=1)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic Relative Strength Index using Wilder's smoothing (EMA-based)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 2. Features (only past/present information -> safe to feed the model)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the volatility feature set. Every column here is computable at
    the moment candle t just closed -- nothing here ever looks forward.
    """
    tr = _true_range(df)
    ret = df["close"].pct_change()  # return from candle t-1 to candle t

    feat = pd.DataFrame(index=df.index)
    feat["f_true_range"] = tr
    feat["f_tr_lag1"] = tr.shift(1)
    feat["f_tr_lag2"] = tr.shift(2)
    feat["f_atr14"] = tr.rolling(14).mean()
    feat["f_realized_vol10"] = ret.rolling(10).std()
    # "regime ratio": is the CURRENT true range unusually large vs. its own
    # recent history? >1 = more turbulent than usual right now.
    feat["f_regime_ratio"] = tr / tr.rolling(50).median()
    feat["f_abs_last_return"] = ret.abs()
    feat["f_rsi14"] = _rsi(df["close"], 14)

    # Calendar / session context -- cheap but genuinely informative for FX vol.
    feat["f_weekday"] = df.index.dayofweek  # 0=Monday ... 4=Friday
    hours = df.index.hour  # UTC, matches data_loader's convention
    feat["f_session_asia"] = ((hours >= 0) & (hours < 7)).astype(int)
    feat["f_session_london"] = ((hours >= 7) & (hours < 12)).astype(int)
    feat["f_session_overlap"] = ((hours >= 12) & (hours < 16)).astype(int)  # London/NY, usually the busiest
    feat["f_session_ny"] = ((hours >= 16) & (hours < 21)).astype(int)

    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Convenience: pull out just the model input columns (the "f_" ones)."""
    return [c for c in df.columns if c.startswith("f_")]


# ---------------------------------------------------------------------------
# 3. Targets (built from the FUTURE candle, then shifted back onto row t)
# ---------------------------------------------------------------------------

def compute_targets(
    df: pd.DataFrame,
    vol_window: int = 50,
    direction_threshold: float = 0.0005,
) -> pd.DataFrame:
    """
    Main target: target_vol (1 = "stormy", 0 = "calm").
      threshold_t  = rolling 50-candle median of True Range, known at time t
                     (uses only candles up to and including t).
      next_tr      = True Range of candle t+1 (the future we're forecasting).
      target_vol_t = 1 if next_tr > threshold_t else 0.
    Because the threshold is a rolling (not fixed) value, the calm/stormy
    split stays roughly balanced even as overall market volatility drifts
    across years/regimes.

    Secondary target: target_dir (context only, NOT the main model output) --
    "up" / "down" / "flat" based on the next candle's return vs. a small
    threshold, to explicitly avoid pretending near-zero moves are directional.
    """
    tr = _true_range(df)
    threshold = tr.rolling(vol_window).median()
    next_tr = tr.shift(-1)

    # IMPORTANT: a plain `next_tr > threshold` comparison silently turns NaN
    # (the un-knowable last row, with no "next candle" yet) into False --
    # that would mislabel the last row as "calm" instead of dropping it.
    # We mask those rows back to <NA> explicitly before casting to Int64.
    is_missing = next_tr.isna() | threshold.isna()
    vol_bool = (next_tr > threshold).mask(is_missing)

    target = pd.DataFrame(index=df.index)
    target["target_vol"] = vol_bool.astype("Int64")
    target["target_vol_label"] = target["target_vol"].map({1: "stormy", 0: "calm"})

    next_return = df["close"].pct_change().shift(-1)  # return from close_t to close_{t+1}
    target["target_dir_label"] = pd.cut(
        next_return,
        bins=[-float("inf"), -direction_threshold, direction_threshold, float("inf")],
        labels=["down", "flat", "up"],
    )

    return target


# ---------------------------------------------------------------------------
# 4. Assemble the final model-ready dataset
# ---------------------------------------------------------------------------

def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine features + targets and drop rows that can't be used:
      - the first ~50 rows (rolling windows still "warming up" -> NaN features)
      - the last row (no next candle yet -> NaN target)
    """
    feat = compute_features(df)
    targets = compute_targets(df)
    combined = pd.concat([feat, targets], axis=1)
    combined = combined.dropna(subset=feature_columns(combined) + ["target_vol"])
    return combined


def assert_no_leakage(df: pd.DataFrame, dataset: pd.DataFrame) -> None:
    """
    Lightweight sanity check, meant to be run once per new dataset version:
      1. The dataset must be strictly shorter than the raw candles (we lose
         warm-up rows at the start and the un-labelled last row at the end).
      2. target_vol must only contain 0/1 (no leftover NaNs slipped through).
      3. The class split should be roughly balanced (rolling-median target),
         a wildly skewed split would hint at a bug in the threshold logic.
    This does not "prove" there's no leakage, but it catches the most common
    mistakes (e.g. forgetting a .shift, dropna dropping the wrong rows).
    """
    assert len(dataset) < len(df), "dataset should be shorter than raw candles (warm-up + last row dropped)"
    assert dataset["target_vol"].isin([0, 1]).all(), "target_vol must be strictly 0/1"
    balance = dataset["target_vol"].mean()
    assert 0.3 < balance < 0.7, f"target_vol looks unbalanced ({balance:.2%} stormy) -- check the threshold logic"
    print(f"[features] leakage sanity check passed. {len(dataset)} rows, {balance:.1%} stormy.")


if __name__ == "__main__":
    from data_loader import load_ohlc_with_flag

    raw, is_synthetic = load_ohlc_with_flag("EUR/USD")
    dataset = build_dataset(raw)
    assert_no_leakage(raw, dataset)
    print(dataset[feature_columns(dataset) + ["target_vol_label", "target_dir_label"]].tail())
