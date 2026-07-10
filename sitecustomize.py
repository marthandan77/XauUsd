from __future__ import annotations


def _parse_lookback_days(value, default: int = 30) -> int:
    """Convert stored lookback values such as 30d, 6mo, 1y, or 45 into days."""
    try:
        text = str(value).strip().lower()
        if not text:
            return default
        if text.endswith("d"):
            return int(float(text[:-1]))
        if text.endswith("mo"):
            return int(float(text[:-2]) * 30)
        if text.endswith("y"):
            return int(float(text[:-1]) * 365)
        return int(float(text))
    except Exception:
        return default


def _patch_streamlit_lookback_selectbox() -> None:
    """
    Keep app.py unchanged but replace its sidebar Lookback period selectbox
    with a manual day-entry box at runtime.

    app.py still stores the value as price_period, for example 30d, so the
    existing Twelve Data outputsize logic continues to work without changes.
    """
    try:
        from streamlit.delta_generator import DeltaGenerator
    except Exception:
        return

    original_selectbox = getattr(DeltaGenerator, "selectbox", None)
    if not callable(original_selectbox):
        return
    if getattr(original_selectbox, "_xauusd_manual_lookback_patch", False):
        return

    def patched_selectbox(self, label, options, *args, **kwargs):
        key = kwargs.get("key")
        if str(label) == "Lookback period" and str(key or "").endswith("price_period"):
            try:
                import streamlit as st

                stored_days = _parse_lookback_days(st.session_state.get(key, "30d"), default=30)
                manual_key = f"{key}_manual_days"
                if manual_key not in st.session_state:
                    st.session_state[manual_key] = ""

                raw_days = self.text_input(
                    "Lookback days",
                    value=st.session_state[manual_key],
                    placeholder=str(stored_days),
                    help="Type number of calendar days only. Example: 30",
                    key=manual_key,
                )
                text = str(raw_days).strip()
                days = _parse_lookback_days(text, default=stored_days) if text else stored_days
                days = max(1, min(int(days), 3650))
                st.session_state[key] = f"{days}d"
                return st.session_state[key]
            except Exception:
                return original_selectbox(self, label, options, *args, **kwargs)

        return original_selectbox(self, label, options, *args, **kwargs)

    patched_selectbox._xauusd_manual_lookback_patch = True
    DeltaGenerator.selectbox = patched_selectbox


_patch_streamlit_lookback_selectbox()
