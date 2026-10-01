"""
train_model.py
---------------
Trains and fairly compares three models on the calm/stormy volatility target:
  1. Baseline  -- always predicts the majority class (the floor any real
                  model must beat).
  2. RandomForest
  3. XGBoost   -- the project's main model.

Both RandomForest and XGBoost are evaluated both RAW (straight out of
.fit()) and CALIBRATED (via isotonic regression on a held-out calibration
slice), because a model that is 70% accurate but says "95% stormy" every
time is not trustworthy -- the probability itself has to be honest.

Chronological, leakage-safe split PER (pair, timeframe) series:
  train (60%) -> calib (20%) -> test (20%), always in time order.
Never a random shuffle-split: neighbouring candles are similar, so a random
split would leak information from "the future" into training.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from xgboost import XGBClassifier

from data_loader import CACHE_DIR

DATASET_PATH = CACHE_DIR / "dataset.csv"
MODELS_DIR = Path(__file__).parent / "models"
REPORTS_DIR = Path(__file__).parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load + chronological split (per pair/timeframe series)
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    return df


def build_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add one-hot columns for `pair` and `timeframe` so a single, global model
    can still pick up pair/timeframe-specific behaviour (e.g. JPY pairs are
    just structurally more volatile than EUR/USD). Kept with the same "f_"
    naming convention as features.py for consistency.
    """
    df = df.copy()
    pair_dummies = pd.get_dummies(df["pair"], prefix="f_pair")
    tf_dummies = pd.get_dummies(df["timeframe"], prefix="f_tf")
    df = pd.concat([df, pair_dummies, tf_dummies], axis=1)

    base_features = [c for c in df.columns if c.startswith("f_") and not c.startswith(("f_pair", "f_tf"))]
    all_features = base_features + list(pair_dummies.columns) + list(tf_dummies.columns)
    return df, all_features


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.6, calib_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split EACH (pair, timeframe) series into train/calib/test by position
    along time, then concatenate the pieces across series. This keeps every
    series in correct chronological order while still giving us one
    combined dataset to train a single global model on.
    """
    train_parts, calib_parts, test_parts = [], [], []

    for _, group in df.groupby(["pair", "timeframe"], sort=False):
        group = group.sort_values("timestamp")
        n = len(group)
        n_train = int(n * train_frac)
        n_calib = int(n * calib_frac)

        train_parts.append(group.iloc[:n_train])
        calib_parts.append(group.iloc[n_train:n_train + n_calib])
        test_parts.append(group.iloc[n_train + n_calib:])

    return (
        pd.concat(train_parts).sort_values("timestamp"),
        pd.concat(calib_parts).sort_values("timestamp"),
        pd.concat(test_parts).sort_values("timestamp"),
    )


# ---------------------------------------------------------------------------
# 2. Train + calibrate
# ---------------------------------------------------------------------------

def train_and_calibrate(
    model, X_train: pd.DataFrame, y_train: pd.Series, X_calib: pd.DataFrame, y_calib: pd.Series
):
    """
    Fit `model` on the TRAIN slice only, then wrap the already-fitted model
    in FrozenEstimator and fit CalibratedClassifierCV on the separate CALIB
    slice. Using a held-out calibration set (not the training data itself)
    avoids an overly optimistic calibration -- the model would otherwise be
    calibrated against data it has already memorized.
    """
    model.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_calib, y_calib)
    return model, calibrated


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]  # P(stormy)
    preds = (proba >= 0.5).astype(int)

    return {
        "model": name,
        "brier_score": brier_score_loss(y_test, proba),  # lower = better calibrated
        "auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "stormy_rate_predicted": preds.mean(),
        "stormy_rate_actual": y_test.mean(),
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_dataset()
    df, feature_cols = build_model_features(df)
    train_df, calib_df, test_df = chronological_split(df)

    print(
        f"[train_model] train={len(train_df)} calib={len(calib_df)} "
        f"test={len(test_df)} features={len(feature_cols)}"
    )

    X_train, y_train = train_df[feature_cols], train_df["target_vol"]
    X_calib, y_calib = calib_df[feature_cols], calib_df["target_vol"]
    X_test, y_test = test_df[feature_cols], test_df["target_vol"]

    results = []

    # --- Baseline: always predict the majority class from training data ---
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    results.append(evaluate("baseline", baseline, X_test, y_test))

    # --- RandomForest: raw + calibrated ---
    rf_raw, rf_calibrated = train_and_calibrate(
        RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train, X_calib, y_calib,
    )
    results.append(evaluate("random_forest_raw", rf_raw, X_test, y_test))
    results.append(evaluate("random_forest_calibrated", rf_calibrated, X_test, y_test))

    # --- XGBoost: raw + calibrated (raw is kept separately for SHAP later) ---
    xgb_raw, xgb_calibrated = train_and_calibrate(
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        X_train, y_train, X_calib, y_calib,
    )
    results.append(evaluate("xgboost_raw", xgb_raw, X_test, y_test))
    results.append(evaluate("xgboost_calibrated", xgb_calibrated, X_test, y_test))

    # --- Report ---
    results_df = pd.DataFrame(results).sort_values("brier_score")
    print("\n--- Model comparison (sorted by Brier score, lower = better) ---")
    print(results_df.to_string(index=False))
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # --- Persist artifacts ---
    # Main model = calibrated XGBoost (honest probabilities for the dashboard).
    joblib.dump(xgb_calibrated, MODELS_DIR / "xgb_calibrated.pkl")
    # Raw XGBoost kept separately: SHAP/TreeSHAP needs the raw booster, not
    # the CalibratedClassifierCV wrapper.
    joblib.dump(xgb_raw, MODELS_DIR / "xgb_raw.pkl")
    joblib.dump(rf_calibrated, MODELS_DIR / "rf_calibrated.pkl")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\n[train_model] artifacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
