from __future__ import annotations

import pandas as pd

from .horizon_forecast import build_multi_horizon_forecast

_PATCHED = False


def _rows_from_forecasts(forecasts: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in forecasts:
        rows.append(
            {
                "Horizon": item.get("horizon"),
                "Bias": item.get("bias"),
                "Up %": item.get("up_probability"),
                "Down %": item.get("down_probability"),
                "Flat %": item.get("flat_probability"),
                "Expected return %": item.get("expected_return_pct"),
                "Expected low": item.get("expected_low"),
                "Expected high": item.get("expected_high"),
                "EV buy": item.get("ev_buy_points"),
                "EV sell": item.get("ev_sell_points"),
                "Decision": item.get("decision"),
                "Expiry": item.get("expiry"),
                "Training rows": item.get("training_rows"),
                "Status": item.get("status"),
            }
        )
    return rows


def _patch_streamlit_subheader() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        import streamlit as st
    except Exception:
        return

    original_subheader = st.subheader
    if getattr(original_subheader, "_xauusd_multi_horizon_patch", False):
        _PATCHED = True
        return

    def patched_subheader(body, *args, **kwargs):
        body_text = str(body)
        if body_text in {"Trade levels", "Active trade levels"}:
            forecasts = st.session_state.get("_xauusd_multi_horizon_forecasts")
            scan_id = st.session_state.get("_xauusd_multi_horizon_scan_id")
            rendered_id = st.session_state.get("_xauusd_multi_horizon_rendered_id")
            if forecasts and scan_id != rendered_id:
                original_subheader("5m / 15m / 30m / 1h predictors")
                st.dataframe(pd.DataFrame(_rows_from_forecasts(forecasts)), use_container_width=True, hide_index=True)
                st.caption(
                    "Formula engine: rolling ridge regression on forward log returns plus residual volatility range. "
                    "Use 5m timeframe for exact 5m/15m/30m/1h horizons. Probabilities are model estimates, not guarantees."
                )
                st.session_state["_xauusd_multi_horizon_rendered_id"] = scan_id
        return original_subheader(body, *args, **kwargs)

    patched_subheader._xauusd_multi_horizon_patch = True
    st.subheader = patched_subheader
    _PATCHED = True


def store_multi_horizon_for_streamlit(df, settings: dict) -> None:
    """Compute scan-time predictors and arrange display inside the existing Forecast Manager page.

    This avoids adding a new page. The table is injected immediately before the existing
    Trade Levels section after SCAN NOW completes.
    """
    try:
        import streamlit as st
    except Exception:
        return

    _patch_streamlit_subheader()
    try:
        forecasts = build_multi_horizon_forecast(df, settings)
    except Exception as exc:
        forecasts = [
            {
                "horizon": "model",
                "status": "error",
                "bias": "N/A",
                "up_probability": None,
                "down_probability": None,
                "flat_probability": None,
                "expected_return_pct": None,
                "expected_low": None,
                "expected_high": None,
                "ev_buy_points": None,
                "ev_sell_points": None,
                "decision": f"Forecast error: {exc}",
                "expiry": "N/A",
                "training_rows": 0,
            }
        ]

    st.session_state["_xauusd_multi_horizon_forecasts"] = forecasts
    st.session_state["_xauusd_multi_horizon_scan_id"] = str(pd.Timestamp.utcnow().value)
