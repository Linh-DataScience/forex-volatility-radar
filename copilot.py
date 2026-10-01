"""
copilot.py
-----------
The Risk Co-Pilot: turns the plain numbers from core.py (stormy/calm
probability, volatility regime, stop/size guidance, SHAP drivers) into a
short, natural-language risk statement for the dashboard.

Rule-based for now -- no API key, no cost, nothing that can fail at demo
time. build_prompt() prepares the exact same facts as a GPT-ready prompt
string, so upgrading to a real LLM later (stretch goal) is a one-line swap:
send build_prompt()'s output to the API instead of calling risk_statement().

This file never decides direction (long/short) and never invents numbers --
it only narrates what core.py already computed.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from core import predict_vol, shap_drivers, vol_recommendation

DISCLAIMER = (
    "Educational project. Not investment advice. No performance guaranteed. "
    "Risk/context assistant, not a signal generator. Direction is the "
    "trader's decision, never the model's."
)

_REGIME_INTRO = {
    "low": "looks calmer than usual",
    "normal": "looks like a fairly typical moment",
    "elevated": "looks more turbulent than usual",
    "high": "looks notably turbulent right now",
}


def _confidence_phrase(confidence: float) -> str:
    """confidence = |proba_stormy - 0.5| * 2, so 0 = coin-flip, 1 = maximally sure."""
    if confidence < 0.15:
        return "That's close to a coin flip, so treat it as low-confidence context, not a signal."
    if confidence < 0.4:
        return "The model has moderate confidence in this read."
    return "The model is fairly confident in this read."


# ---------------------------------------------------------------------------
# 1. Rule-based statement (what the dashboard shows today)
# ---------------------------------------------------------------------------

def risk_statement(
    prediction: dict,
    vol_rec: dict,
    shap_result: dict | None = None,
) -> str:
    """
    Build the 3-5 sentence risk statement shown in the "Risk Co-Pilot" panel.
    Pure string templating over already-computed numbers -- no model call,
    no randomness, so it's deterministic and safe to demo live.

    Does NOT append the full legal DISCLAIMER -- that already appears once,
    clearly, in the page header caption and the page footer (see app.py).
    Repeating it inside this box too was redundant; the one line this
    function keeps ("MAGNITUDE, not direction") is the specific reminder
    that matters here, not boilerplate.
    """
    pair, timeframe = prediction["pair"], prediction["timeframe"]
    proba_stormy = prediction["proba_stormy"]
    regime = vol_rec["regime"]

    lines = [
        f"{pair} on the {timeframe} chart {_REGIME_INTRO[regime]} "
        f"({proba_stormy:.0%} probability the next candle is unusually large).",
        _confidence_phrase(prediction["confidence"]),
        f"Suggested stop distance: about {vol_rec['suggested_stop_pips']} pips "
        f"({vol_rec['stop_multiplier']:.1f}x current ATR). "
        f"Suggested position size: {vol_rec['size_factor']:.0%} of your normal size.",
    ]

    if shap_result and shap_result["top_drivers"]:
        top = shap_result["top_drivers"][0]
        lines.append(f"Main driver right now: {top['label']} (a {top['direction']}-leaning signal).")

    lines.append(
        "This describes expected MAGNITUDE of the next candle, not direction -- "
        "going long or short stays your call."
    )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# 2. GPT-ready prompt (not called anywhere yet -- prepared for the stretch
#    goal upgrade to a real LLM, e.g. Groq)
# ---------------------------------------------------------------------------

def build_prompt(prediction: dict, vol_rec: dict, shap_result: dict | None = None) -> str:
    """
    Same facts as risk_statement(), packaged as a prompt an LLM could turn
    into the statement instead. Swapping risk_statement() for a real API
    call later means: send this string, return the response -- nothing
    else in the dashboard needs to change.
    """
    facts = [
        f"Pair: {prediction['pair']}",
        f"Timeframe: {prediction['timeframe']}",
        f"Stormy probability (unusually large next candle): {prediction['proba_stormy']:.1%}",
        f"Calm probability: {prediction['proba_calm']:.1%}",
        f"Model confidence (0=coin-flip, 1=very sure): {prediction['confidence']:.2f}",
        f"Volatility regime: {vol_rec['regime']}",
        f"Suggested stop distance: {vol_rec['suggested_stop_pips']} pips "
        f"({vol_rec['stop_multiplier']}x current ATR of {vol_rec['atr_pips']} pips)",
        f"Suggested position-size factor: {vol_rec['size_factor']} (1.0 = normal size)",
    ]
    if shap_result and shap_result["top_drivers"]:
        drivers_text = "; ".join(d["sentence"] for d in shap_result["top_drivers"][:3])
        facts.append(f"Top model drivers for this forecast: {drivers_text}")

    instructions = (
        "You are a risk assistant for a forex volatility dashboard. Using ONLY the facts "
        "below, write a short (3-4 sentence) plain-English risk statement for a trader. "
        "Describe expected volatility (not direction), give the stop-distance and "
        "position-size guidance, mention the main driver, and end with a one-line reminder "
        "that this is not a trading signal and direction is the trader's own decision. "
        "Do not invent any numbers that are not given below."
    )

    return instructions + "\n\n" + "\n".join(facts)


# ---------------------------------------------------------------------------
# 3. Optional real-LLM upgrade (Groq, free tier) -- inactive by default
# ---------------------------------------------------------------------------

def _groq_api_key() -> str | None:
    """
    Looks for a Groq API key in Streamlit secrets first (the normal place
    for both local dev and Streamlit Community Cloud), then an environment
    variable as a fallback. Returns None if neither is set -- that's the
    default, out-of-the-box state, not an error.
    """
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception as exc:
        # Streamlit raises the SAME exception class (StreamlitSecretNotFoundError,
        # which happens to subclass FileNotFoundError) both when secrets.toml is
        # simply missing AND when it exists but fails to parse -- so we can't
        # tell those apart by exception type. Check the file directly instead:
        # a parse error on an EXISTING file is worth surfacing (e.g. an
        # unquoted value), a missing file is the normal, silent default.
        if Path(".streamlit/secrets.toml").exists():
            print(f"[copilot] secrets.toml exists but couldn't be read ({exc}) -- check its TOML syntax")
    return os.environ.get("GROQ_API_KEY")


def generate_llm_statement(prompt: str, model: str = "openai/gpt-oss-20b") -> str | None:
    """
    Send build_prompt()'s text to Groq's (free) chat completion API and
    return the response -- or None if no key is configured, the `requests`
    call fails, or the response is malformed for any reason. Callers must
    always have the rule-based risk_statement() as a fallback; this
    function is designed to fail quietly, never to crash the dashboard.

    Model note: Groq deprecated llama-3.1-8b-instant for free/developer-tier
    keys on 2026-08-16 and recommends openai/gpt-oss-20b as the replacement
    (see https://console.groq.com/docs/deprecations). If Groq retires this
    one too, a "model_not_found" error in the terminal is the tell -- swap
    the default here for whatever https://console.groq.com/docs/models
    currently lists.
    """
    api_key = _groq_api_key()
    if not api_key:
        return None

    try:
        import requests

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
            timeout=10,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception as exc:
        print(f"[copilot] Groq call failed, falling back to the rule-based statement: {exc}")
        return None


# ---------------------------------------------------------------------------
# 4. Convenience wrapper: everything a dashboard page needs in one call
# ---------------------------------------------------------------------------

def generate_risk_briefing(
    pair: str,
    timeframe: str,
    force_refresh: bool = False,
    raw_1h=None,
    is_synthetic: bool | None = None,
) -> dict:
    """
    One call for app.py: runs the full pipeline (predict -> vol
    recommendation -> SHAP -> statement + prompt) and returns everything
    the Detail page needs to render its Co-Pilot panel.

    Tries the real-LLM statement first (only does anything if a Groq key
    is configured); falls back to the deterministic rule-based statement
    otherwise. "statement_source" tells the caller which one was used, so
    the dashboard can optionally show a small "via GPT" / "rule-based" tag.

    force_refresh=True (the manual "Refresh now" button) passes through to
    predict_vol so the underlying price data is genuinely re-fetched, not
    just re-read from cache. raw_1h/is_synthetic: forwarded to predict_vol
    so it reuses the SAME candles the caller already fetched for its chart,
    instead of triggering a second independent (and, for synthetic
    fallback data, potentially different) fetch -- see predict_vol's
    docstring.
    """
    prediction = predict_vol(
        pair, timeframe, force_refresh=force_refresh, raw_1h=raw_1h, is_synthetic=is_synthetic
    )
    vol_rec = vol_recommendation(prediction)
    shap_result = shap_drivers(prediction, top_n=3)
    prompt = build_prompt(prediction, vol_rec, shap_result)

    llm_statement = generate_llm_statement(prompt)
    if llm_statement:
        statement = llm_statement
    else:
        statement = risk_statement(prediction, vol_rec, shap_result)

    return {
        "prediction": prediction,
        "vol_recommendation": vol_rec,
        "shap": shap_result,
        "statement": statement,
        "statement_source": "llm" if llm_statement else "rule-based",
        "prompt": prompt,
    }


if __name__ == "__main__":
    # Manual smoke test: python copilot.py
    briefing = generate_risk_briefing("EUR/USD", "1h")

    print("--- Risk statement ---")
    print(briefing["statement"])

    print("\n--- GPT-ready prompt (not sent anywhere yet) ---")
    print(briefing["prompt"])
