"""
build_dataset.py
-----------------
Assembles the full training dataset: loops over all FX pairs and both
timeframes (1h, 4h), loads candles, builds leakage-safe features + targets
(via features.py), tags each row with its pair/timeframe/data-source, and
writes everything to data/dataset.csv -- the single input file train_model.py
will read in Woche 2.

Note on 4h data: we do NOT download 4h candles separately from yfinance.
We fetch 1h once and resample to 4h locally via data_loader.to_timeframe().
This halves the number of API calls and guarantees 1h and 4h data are
perfectly consistent with each other (same source candles).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import CACHE_DIR, PAIRS, load_ohlc_with_flag, to_timeframe
from features import assert_no_leakage, build_dataset, feature_columns

TIMEFRAMES = ["1h", "4h"]
OUTPUT_PATH = CACHE_DIR / "dataset.csv"


def build_all() -> pd.DataFrame:
    """
    Loop over every (pair, timeframe) combination, build a leakage-checked
    dataset for each, and stack them into one big DataFrame.
    """
    frames: list[pd.DataFrame] = []

    for pair in PAIRS:
        # Fetch 1h once per pair (live or synthetic fallback).
        raw_1h, is_synthetic = load_ohlc_with_flag(pair)

        for timeframe in TIMEFRAMES:
            raw = raw_1h if timeframe == "1h" else to_timeframe(raw_1h, "4h")

            if len(raw) < 200:
                # Not enough candles to build meaningful rolling features
                # (warm-up windows need ~50 rows) -- skip rather than emit
                # a near-empty, unreliable slice.
                print(f"[build_dataset] skipping {pair} {timeframe}: only {len(raw)} candles")
                continue

            dataset = build_dataset(raw)
            assert_no_leakage(raw, dataset)

            dataset = dataset.copy()
            dataset["pair"] = pair
            dataset["timeframe"] = timeframe
            dataset["is_synthetic"] = is_synthetic
            frames.append(dataset)

            stormy_share = dataset["target_vol"].mean()
            tag = "SYNTHETIC" if is_synthetic else "LIVE"
            print(
                f"[build_dataset] {pair:8s} {timeframe:3s} [{tag:9s}] "
                f"-> {len(dataset):5d} rows, {stormy_share:.1%} stormy"
            )

    full = pd.concat(frames, axis=0)
    # Reset to a plain integer index -- timestamps stay available as a column
    # (from the index) since pair+timeframe alone don't uniquely order rows.
    full = full.reset_index().rename(columns={"index": "timestamp"})
    return full


def save(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[build_dataset] wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    full_dataset = build_all()

    print("\n--- Summary ---")
    print(full_dataset.groupby(["pair", "timeframe"]).agg(
        rows=("target_vol", "size"),
        stormy_share=("target_vol", "mean"),
        any_synthetic=("is_synthetic", "any"),
    ))

    save(full_dataset)

    print(f"\nFeature columns ({len(feature_columns(full_dataset))}):", feature_columns(full_dataset))
