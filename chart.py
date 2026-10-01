"""
chart.py
---------
Candlestick chart for the dashboard's Detail page. Shows recent
price action and marks candles that were, in hindsight, unusually large
(true range above their own trailing 50-candle median -- same idea as the
target definition in features.py, just without the forward shift, since
here we're annotating PAST candles, not forecasting the next one).

No candlestick pattern markers (double-top, engulfing, etc.) -- those
belong to the direction-prediction idea that this project deliberately
dropped (see PROJECT HANDOFF, section 3). This chart is about magnitude
of movement, not shape.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from core import THEME
from features import compute_features

# ---------------------------------------------------------------------------
# 1. Which candles count as "was unusually large" (for the highlight markers)
# ---------------------------------------------------------------------------

def _flag_high_vol_candles(candles: pd.DataFrame) -> pd.Series:
    """
    True where a candle's own true range exceeded its trailing 50-candle
    median at that point in time (f_regime_ratio > 1). Purely descriptive/
    historical -- not the same as a forecast for the NEXT candle.
    """
    feats = compute_features(candles)
    return feats["f_regime_ratio"] > 1.0


# ---------------------------------------------------------------------------
# 2. Chart builder
# ---------------------------------------------------------------------------

def build_candlestick_chart(
    candles: pd.DataFrame,
    pair: str,
    timeframe: str,
    n_candles: int = 100,
) -> go.Figure:
    """
    Build a candlestick chart for the last `n_candles` candles,
    with a small marker above/below any candle that was unusually large.
    """
    recent = candles.tail(n_candles)
    high_vol = _flag_high_vol_candles(candles).reindex(recent.index).fillna(False)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["open"],
            high=recent["high"],
            low=recent["low"],
            close=recent["close"],
            increasing_line_color=THEME["calm"],
            decreasing_line_color=THEME["stormy"],
            increasing_fillcolor=THEME["calm"],
            decreasing_fillcolor=THEME["stormy"],
            name=f"{pair} {timeframe}",
        )
    )

    flagged = recent[high_vol]
    if not flagged.empty:
        # small amber dots above the high of each unusually large candle --
        # a passive annotation, not a trading signal or pattern marker.
        marker_y = flagged["high"] * 1.0015
        fig.add_trace(
            go.Scatter(
                x=flagged.index,
                y=marker_y,
                mode="markers",
                marker=dict(symbol="circle", size=6, color=THEME["accent"]),
                name="Unusually large candle",
                hovertemplate="Unusually large candle<br>%{x}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        title=f"{pair} · {timeframe} — last {len(recent)} candles",
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=THEME["surface"])
    fig.update_yaxes(gridcolor=THEME["surface"])

    return fig


if __name__ == "__main__":
    # Manual smoke test: python chart.py -- builds the chart on synthetic/
    # live data (whichever data_loader falls back to) and writes a PNG so
    # the theme/markers can be checked visually.
    from data_loader import load_ohlc_with_flag, to_timeframe

    raw_1h, is_synthetic = load_ohlc_with_flag("EUR/USD")
    candles = to_timeframe(raw_1h, "4h")

    fig = build_candlestick_chart(candles, "EUR/USD", "4h", n_candles=80)
    fig.write_image("/tmp/chart_smoke_test.png", width=1000, height=550, scale=2)
    print(f"Chart built OK (synthetic={is_synthetic}). Traces: {len(fig.data)}")
    print("Preview written to /tmp/chart_smoke_test.png")
