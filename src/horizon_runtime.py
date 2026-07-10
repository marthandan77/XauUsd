from __future__ import annotations


def rows_from_forecasts(forecasts: list[dict]) -> list[dict]:
    """Convert horizon forecast dictionaries into UI/log rows.

    This module is intentionally side-effect free. It does not patch Streamlit,
    write session state, or influence trade logic.
    """
    rows: list[dict] = []
    for item in forecasts or []:
        rows.append(
            {
                "Horizon": item.get("horizon"),
                "Role": item.get("role"),
                "Bias": item.get("bias"),
                "Up %": item.get("up_probability"),
                "Down %": item.get("down_probability"),
                "Flat %": item.get("flat_probability"),
                "Expected return %": item.get("expected_return_pct"),
                "Expected low": item.get("expected_low"),
                "Expected high": item.get("expected_high"),
                "EV buy": item.get("ev_buy_points"),
                "EV sell": item.get("ev_sell_points"),
                "Min EV": item.get("min_ev_points"),
                "Decision": item.get("decision"),
                "Training rows": item.get("training_rows"),
                "Status": item.get("status"),
            }
        )
    return rows
