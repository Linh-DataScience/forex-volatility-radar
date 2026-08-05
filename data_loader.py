"""
data_loader.py
---------------
Fetches 1h OHLC (Open/High/Low/Close) candle data for major FX pairs via yfinance,
caches it locally as Parquet, resamples to other timeframes (e.g. 4h), and falls
back to synthetic (but realistic) data if the live download fails.

Design principle: this module ONLY loads and shapes raw price history. It never
looks into the future and never computes model features or targets — that
separation is what makes leakage-checking easy later in features.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# Human-readable pair name -> Yahoo Finance ticker symbol.
# Yahoo represents FX spot pairs as "EURUSD=X" etc.
PAIRS: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
}

# Where cached data lives. Parent module (whoever imports this file) can
# override this by setting data_loader.CACHE_DIR = Path(...) before calling.
CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)

# yfinance only keeps 1h intraday history for ~730 days — this is a Yahoo
# limit, not something we can configure around.
MAX_INTRADAY_PERIOD = "730d"


# ---------------------------------------------------------------------------
# 2. Live data fetch (with retries + local Parquet cache)
# ---------------------------------------------------------------------------

def _cache_path(pair: str, interval: str) -> Path:
    """Build a filesystem-safe cache filename for a given pair/interval."""
    safe_pair = pair.replace("/", "")
    return CACHE_DIR / f"{safe_pair}_{interval}.parquet"


def fetch_ohlc(pair: str, interval: str = "1h", retries: int = 3) -> pd.DataFrame | None:
    """
    Download 1h OHLC candles for one FX pair from Yahoo Finance.

    Returns a DataFrame indexed by UTC timestamp with columns
    [open, high, low, close, volume], or None if the download failed
    after all retries (caller should fall back to synthetic data).
    """
    if pair not in PAIRS:
        raise ValueError(f"Unknown pair '{pair}'. Choose from: {list(PAIRS)}")

    ticker = PAIRS[pair]

    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                ticker,
                interval=interval,
                period=MAX_INTRADAY_PERIOD,
                progress=False,
                auto_adjust=False,
            )
            if raw is None or raw.empty:
                raise ValueError("empty response from yfinance")

            # yfinance sometimes returns MultiIndex columns (ticker, field)
            # when downloading a single symbol depending on version — flatten them.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.rename(columns=str.lower)
            raw = raw[["open", "high", "low", "close", "volume"]].copy()

            # Yahoo timestamps are already tz-aware; normalize everything to UTC
            # so the whole project has one consistent clock.
            if raw.index.tz is None:
                raw.index = raw.index.tz_localize("UTC")
            else:
                raw.index = raw.index.tz_convert("UTC")
            raw.index.name = "timestamp"

            _save_cache(raw, pair, interval)
            return raw

        except Exception as exc:  # noqa: BLE001 - we deliberately catch everything here
            print(f"[data_loader] fetch attempt {attempt}/{retries} for {pair} failed: {exc}")
            time.sleep(1.5 * attempt)  # simple backoff before retrying

    return None


def _save_cache(df: pd.DataFrame, pair: str, interval: str) -> None:
    df.to_parquet(_cache_path(pair, interval))


def _load_cache(pair: str, interval: str) -> pd.DataFrame | None:
    path = _cache_path(pair, interval)
    if path.exists():
        return pd.read_parquet(path)
    return None


# ---------------------------------------------------------------------------
# 3. Synthetic fallback data
# ---------------------------------------------------------------------------

def generate_synthetic_ohlc(
    pair: str,
    interval: str = "1h",
    n_periods: int = 5000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate realistic-looking FX candles when no internet/Yahoo data is
    available. This is NOT random noise — two properties are deliberately
    baked in, because without them the volatility model has nothing to learn
    and looks "broken" during offline testing:

      1. Volatility clustering: log-volatility follows an AR(1) process, so
         calm periods and stormy periods each last for a while (like real FX).
      2. Session seasonality: volatility is higher during the London/New York
         trading overlap and lower during the quiet Asia-only session.
    """
    rng = np.random.default_rng(seed)

    freq = "1h" if interval == "1h" else "4h"
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_periods, freq=freq, tz="UTC")

    # --- volatility clustering via AR(1) on log-volatility ---
    phi = 0.97  # persistence: close to 1 = slow-changing vol regimes
    log_vol = np.zeros(n_periods)
    log_vol[0] = np.log(0.0006)  # baseline vol level, roughly realistic for FX
    noise = rng.normal(0, 0.15, n_periods)
    for t in range(1, n_periods):
        log_vol[t] = phi * log_vol[t - 1] + (1 - phi) * np.log(0.0006) + noise[t]
    vol = np.exp(log_vol)

    # --- session seasonality multiplier (UTC hours) ---
    hours = timestamps.hour
    session_mult = np.select(
        [
            (hours >= 7) & (hours < 16),   # London session -> elevated
            (hours >= 12) & (hours < 16),  # London/NY overlap -> highest
            (hours >= 0) & (hours < 7),    # Asia-only -> quiet
        ],
        [1.3, 1.6, 0.6],
        default=1.0,
    )
    vol = vol * session_mult

    # --- build a price path from the volatility series ---
    start_price = {"EUR/USD": 1.08, "GBP/USD": 1.27, "USD/JPY": 150.0, "AUD/USD": 0.66}.get(
        pair, 1.0
    )
    returns = rng.normal(0, vol)
    log_price = np.log(start_price) + np.cumsum(returns)
    close = np.exp(log_price)

    # Derive open/high/low around close with a bit of intrabar noise.
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    intrabar_range = vol * np.abs(rng.normal(1.0, 0.4, n_periods)) * close
    high = np.maximum(open_, close) + intrabar_range * rng.uniform(0.1, 0.6, n_periods)
    low = np.minimum(open_, close) - intrabar_range * rng.uniform(0.1, 0.6, n_periods)
    volume = rng.integers(100, 1000, n_periods)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# 4. Timeframe resampling (1h -> 4h etc.)
# ---------------------------------------------------------------------------

def to_timeframe(df: pd.DataFrame, target: str = "4h") -> pd.DataFrame:
    """
    Resample 1h OHLC candles into a coarser timeframe (e.g. 4h) using
    standard OHLC aggregation rules: first open, max high, min low, last
    close, summed volume. Only uses information within each completed
    bucket, so no look-ahead is introduced here.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resampled = df.resample(target).agg(agg)
    return resampled.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# 5. Public entry point
# ---------------------------------------------------------------------------

def load_ohlc_with_flag(
    pair: str,
    interval: str = "1h",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """
    Main function the rest of the project should call.

    Order of attempts: local cache -> live yfinance download -> synthetic
    fallback. Returns (dataframe, is_synthetic) so callers (esp. the
    dashboard) can show a visible warning whenever synthetic data is used —
    never fail silently into fake data.
    """
    if use_cache:
        cached = _load_cache(pair, interval)
        if cached is not None and len(cached) > 100:
            return cached, False

    live = fetch_ohlc(pair, interval=interval)
    if live is not None and len(live) > 100:
        return live, False

    print(f"[data_loader] falling back to SYNTHETIC data for {pair} ({interval})")
    synthetic = generate_synthetic_ohlc(pair, interval=interval)
    return synthetic, True


if __name__ == "__main__":
    # Quick manual smoke test: python data_loader.py
    for pair_name in PAIRS:
        data, is_synthetic = load_ohlc_with_flag(pair_name)
        tag = "SYNTHETIC" if is_synthetic else "LIVE"
        print(f"{pair_name}: {len(data)} rows [{tag}] | last close={data['close'].iloc[-1]:.5f}")
