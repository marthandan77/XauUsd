from __future__ import annotations


def _patch_streamlit_lookback_options() -> None:
    """
    Add 14d to the app.py sidebar Lookback period dropdown at runtime.

    This keeps the original app.py control intact and only expands the fixed
    lookback option list from 7d/30d/... to include 14d.
    """
    try:
        from streamlit.delta_generator import DeltaGenerator
    except Exception:
        return

    original_selectbox = getattr(DeltaGenerator, "selectbox", None)
    if not callable(original_selectbox):
        return
    if getattr(original_selectbox, "_xauusd_14d_lookback_patch", False):
        return

    def patched_selectbox(self, label, options, *args, **kwargs):
        if str(label) == "Lookback period":
            try:
                option_list = list(options)
                if "14d" not in option_list:
                    if "7d" in option_list:
                        option_list.insert(option_list.index("7d") + 1, "14d")
                    else:
                        option_list.insert(0, "14d")
                return original_selectbox(self, label, option_list, *args, **kwargs)
            except Exception:
                return original_selectbox(self, label, options, *args, **kwargs)

        return original_selectbox(self, label, options, *args, **kwargs)

    patched_selectbox._xauusd_14d_lookback_patch = True
    DeltaGenerator.selectbox = patched_selectbox


_patch_streamlit_lookback_options()
