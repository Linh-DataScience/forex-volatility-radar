"""
app.py
------
FX Volatility Radar -- MVP2 multipage dashboard.

Two pages:
  Detail: pick one pair/timeframe, see the full picture (probability, risk
          co-pilot statement, SHAP explanation, candlestick chart).
  Radar:  all pairs x timeframes at a glance, so you can spot where
          movement is expected right now without drilling into each one.

Everything here is presentation only -- all the actual computation (model
inference, SHAP, risk guidance) lives in core.py / copilot.py / chart.py.
This file's job is layout, not logic.
"""

from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from chart import build_candlestick_chart
from copilot import DISCLAIMER, generate_risk_briefing
from core import PAIRS, THEME, TIMEFRAMES, get_news, load_pair_data, position_plan, predict_vol
from data_loader import to_timeframe

st.set_page_config(
    page_title="FX Volatility Radar",
    page_icon=":zap:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global styling. NOTE: every HTML string below is built as ONE continuous
# line with no embedded newlines / leading indentation -- Streamlit renders
# indented multi-line HTML inside st.markdown as a literal code block
# instead of actual HTML (a bug that bit this project twice already).
# ---------------------------------------------------------------------------

st.markdown(f'<style>.block-container {{padding-top:2rem;max-width:1200px;}} .stMetric {{background:{THEME["surface"]};border-radius:8px;padding:10px 14px;}} div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {{gap:0.6rem;}}</style>', unsafe_allow_html=True)


def _section_header(icon: str, title: str) -> None:
    """
    Consistent card header used above every major Detail-page section: an
    icon, a bold title, and a short accent-colored underline -- the same
    "kicker" language as the project's slide deck, so the dashboard and the
    presentation read as one visual product instead of two unrelated things.
    """
    st.markdown(
        f'<div style="margin:1.6rem 0 0.5rem 0;display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:1.25rem;line-height:1;">{icon}</span>'
        f'<span style="font-size:1.1rem;font-weight:700;color:{THEME["text"]};">{title}</span>'
        f'</div>'
        f'<div style="height:3px;width:40px;background:{THEME["accent"]};border-radius:2px;margin-bottom:0.9rem;"></div>',
        unsafe_allow_html=True,
    )


def _pill(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;background:{color}22;color:{color};'
        f'border:1px solid {color}66;border-radius:999px;padding:2px 11px;'
        f'font-size:0.72rem;font-weight:700;letter-spacing:0.03em;text-transform:uppercase;">{text}</span>'
    )


def _seconds_until_next_refresh(minutes_before_hour: int = 10) -> int:
    """
    Seconds (UTC) until the next "N minutes before the full hour" mark.
    Both 1h and 4h candles always close on the hour, so one hourly trigger
    -- timed shortly before each close -- covers both timeframes; no need
    for a separate 4h-specific schedule.
    """
    now = dt.datetime.now(dt.timezone.utc)
    minute_mark = 60 - minutes_before_hour  # 10 min before -> :50
    target = now.replace(minute=minute_mark, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(hours=1)
    return max(1, int((target - now).total_seconds()))


_next_refresh_secs = _seconds_until_next_refresh()

try:
    from streamlit_autorefresh import st_autorefresh

    # Recomputed fresh on every rerun (including ones the timer itself
    # triggers), so this self-corrects to the next :50 mark each time
    # rather than drifting on a fixed interval.
    st_autorefresh(interval=_next_refresh_secs * 1000, key="autorefresh")
except Exception:
    pass  # optional dependency -- dashboard still works without auto-refresh


def _kpi_card(label: str, value: str, sub: str, color: str) -> str:
    return f'<div style="background:{THEME["surface"]};border-left:4px solid {color};border-radius:8px;padding:14px 18px;"><div style="color:{THEME["text_muted"]};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;">{label}</div><div style="color:{THEME["text"]};font-size:1.9rem;font-weight:700;">{value}</div><div style="color:{THEME["text_muted"]};font-size:0.85rem;">{sub}</div></div>'


def _shap_bar_chart(shap_result: dict) -> go.Figure:
    drivers = list(reversed(shap_result["top_drivers"]))  # largest impact on top
    labels = [d["label"] for d in drivers]
    values = [d["contribution"] for d in drivers]
    colors = [THEME["stormy"] if v > 0 else THEME["calm"] for v in values]

    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Impact on stormy probability (log-odds)",
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Detail page
# ---------------------------------------------------------------------------

def render_detail_page() -> None:
    st.title("FX Volatility Radar")
    st.caption(
        "Forecasts whether the NEXT candle will be calm or stormy -- an unusually "
        "large move, not a direction call. Educational project, not financial advice."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    pair = col1.selectbox("Pair", list(PAIRS.keys()))
    timeframe = col2.selectbox("Timeframe", TIMEFRAMES)
    col3.markdown('<div style="height:1.85rem;"></div>', unsafe_allow_html=True)  # aligns the button with the dropdowns
    force_refresh = col3.button("🔄 Refresh now", help="Force a fresh price + news fetch, bypassing the cache.", use_container_width=True)

    next_refresh_local = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_next_refresh_secs)
    st.caption(
        f"Auto-refreshes ~10 min before every full hour (next: {next_refresh_local.strftime('%H:%M')} UTC) "
        "-- both 1h and 4h candles close on the hour. Use the button above to update immediately."
    )

    raw_1h, is_synthetic = load_pair_data(pair, force_refresh=force_refresh)
    if is_synthetic:
        st.warning(
            "Live data unavailable right now -- showing SYNTHETIC demo data instead. "
            "The forecast below is illustrative, not a real market read."
        )

    briefing = generate_risk_briefing(
        pair, timeframe, force_refresh=force_refresh, raw_1h=raw_1h, is_synthetic=is_synthetic
    )
    prediction = briefing["prediction"]
    vol_rec = briefing["vol_recommendation"]
    shap_result = briefing["shap"]

    st.subheader(f"{pair} · {timeframe}")

    # -----------------------------------------------------------------
    # 1. Price chart
    # -----------------------------------------------------------------
    _section_header("🕯️", "Price chart")
    with st.container(border=True):
        candles = raw_1h if timeframe == "1h" else to_timeframe(raw_1h, "4h")
        st.plotly_chart(build_candlestick_chart(candles, pair, timeframe), use_container_width=True)
        st.caption(f"Based on the candle that closed at {prediction['latest_timestamp']} UTC.")

    # -----------------------------------------------------------------
    # 2. Probability
    # -----------------------------------------------------------------
    _section_header("🎯", "Probability")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.markdown(
            _kpi_card("⚡ Stormy probability", f"{prediction['proba_stormy']:.0%}", "unusually large next candle", THEME["stormy"]),
            unsafe_allow_html=True,
        )
        c2.markdown(
            _kpi_card("☀️ Calm probability", f"{prediction['proba_calm']:.0%}", "typical-sized next candle", THEME["calm"]),
            unsafe_allow_html=True,
        )
        st.caption(
            "**Direction (uncertain):** this model does not predict direction -- our own "
            "cost-aware backtests found no reliable directional edge on 1h/4h FX after "
            "spread costs. Direction below is always YOUR input, never a model output."
        )

    # -----------------------------------------------------------------
    # 3. Risk Co-Pilot
    # -----------------------------------------------------------------
    _section_header("🧭", "Risk Co-Pilot")
    with st.container(border=True):
        st.markdown(
            f'<div style="background:{THEME["bg"]};border-left:4px solid {THEME["accent"]};'
            f'border-radius:8px;padding:16px 20px;color:{THEME["text"]};font-size:1rem;line-height:1.6;">'
            f'{briefing["statement"]}</div>',
            unsafe_allow_html=True,
        )
        source_label = "GPT (Groq)" if briefing["statement_source"] == "llm" else "rule-based"
        st.markdown(f'<div style="margin-top:0.6rem;">{_pill(source_label, THEME["accent"])}</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 4. Position plan
    # -----------------------------------------------------------------
    _section_header("📐", "Position plan")
    with st.container(border=True):
        st.caption("Direction and account details are your inputs -- the model never sets these.")
        p1, p2, p3, p4 = st.columns(4)
        direction = p1.selectbox("Direction", ["long", "short"])
        account_balance = p2.number_input("Account balance (USD)", min_value=100.0, value=10_000.0, step=100.0)
        risk_pct = p3.slider("Risk per trade (%)", 0.25, 5.0, 1.0, step=0.25)
        stop_pips = p4.number_input("Stop distance (pips)", min_value=1.0, value=float(vol_rec["suggested_stop_pips"]), step=0.5)

        entry_price = float(prediction["last_candle"]["close"])
        plan = position_plan(pair, direction, entry_price, account_balance, risk_pct, stop_pips)

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("📊 Position size", f"{plan['lots']} lots")
        r2.metric("🛑 Stop-loss", f"{plan['stop_loss']}")
        r3.metric("🎯 Take-profit", f"{plan['take_profit']}")
        r4.metric("💰 Risk amount", f"${plan['risk_amount']}")

    # -----------------------------------------------------------------
    # 5. Volatility guidance
    # -----------------------------------------------------------------
    _section_header("🌡️", "Volatility guidance")
    with st.container(border=True):
        g1, g2, g3 = st.columns(3)
        g1.metric("🎚️ Regime", vol_rec["regime"].upper())
        g2.metric("🛑 Suggested stop", f"{vol_rec['suggested_stop_pips']} pips")
        g3.metric("📏 Suggested size", f"{vol_rec['size_factor']:.0%}", help="As a share of your normal position size.")

    # -----------------------------------------------------------------
    # Explainability & context -- supporting detail below the core flow,
    # kept visible (not tucked in an expander) since SHAP interpretability
    # is a core project requirement, not an afterthought.
    # -----------------------------------------------------------------
    _section_header("🔍", "Why this forecast & market context")
    with st.container(border=True):
        shap_col, news_col = st.columns(2)

        with shap_col:
            st.markdown("**SHAP drivers**")
            st.plotly_chart(_shap_bar_chart(shap_result), use_container_width=True)
            for d in shap_result["top_drivers"]:
                st.caption(f"• {d['sentence']}")

        with news_col:
            st.markdown("**News sentiment**")
            news = get_news(pair, force_refresh=force_refresh)
            if not news["headlines"]:
                st.caption("No recent news found for this pair -- a quiet news day, not an error.")
            else:
                sentiment_color = {"positive": THEME["calm"], "negative": THEME["stormy"], "neutral": THEME["text_muted"]}
                st.markdown(
                    f'<span style="color:{sentiment_color[news["overall"]]};font-weight:600;">Overall: {news["overall"].upper()}</span> (avg. VADER score {news["avg_score"]:+.2f})',
                    unsafe_allow_html=True,
                )
                for h in news["headlines"]:
                    dot = sentiment_color[h["label"]]
                    st.markdown(f'<span style="color:{dot};">●</span> {h["title"]}', unsafe_allow_html=True)

    with st.expander("How does this work?"):
        st.write(
            "A calibrated XGBoost model looks at recent volatility, momentum and session "
            "features and estimates the probability that the next candle's true range "
            "will exceed its own recent typical level. Calibration means '70% stormy' "
            "should actually happen about 70% of the time -- not just rank moments "
            "relative to each other. SHAP shows which inputs pushed THIS particular "
            "forecast up or down."
        )

    st.divider()
    st.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# Radar page
# ---------------------------------------------------------------------------

def render_radar_page() -> None:
    st.title("Volatility Radar")
    st.caption("All pairs and timeframes at a glance -- where is movement expected right now?")

    force_refresh = st.button("🔄 Refresh now", help="Force a fresh price fetch for all pairs, bypassing the cache.")
    next_refresh_local = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_next_refresh_secs)
    st.caption(f"Auto-refreshes ~10 min before every full hour (next: {next_refresh_local.strftime('%H:%M')} UTC).")

    timeframe_icon = {"1h": "🕐", "4h": "🕓"}
    for timeframe in TIMEFRAMES:
        _section_header(timeframe_icon.get(timeframe, "⏱️"), f"{timeframe} timeframe")
        cols = st.columns(len(PAIRS))
        for pair, col in zip(PAIRS, cols):
            prediction = predict_vol(pair, timeframe, force_refresh=force_refresh)
            color = THEME["stormy"] if prediction["label"] == "STORMY" else THEME["calm"]
            icon = "⚡" if prediction["label"] == "STORMY" else "☀️"
            with col:
                st.markdown(
                    _kpi_card(f"{icon} {pair}", prediction["label"], f"{prediction['proba_stormy']:.0%} stormy", color),
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(DISCLAIMER)


pg = st.navigation(
    [
        st.Page(render_detail_page, title="Detail", icon="🔎", default=True),
        st.Page(render_radar_page, title="Radar", icon="🛰️"),
    ]
)
pg.run()
