from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.forecast_labels import HORIZONS, add_forward_outcomes


@dataclass
class HorizonForecast:
    horizon: str
    bars_ahead: int
    side: str
    status: str
    entry: float | None
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    tp1_move: float | None
    tp2_move: float | None
    adverse_move: float | None
    historical_direction_probability_pct: float | None
    sample_count: int
    matching_setup: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except Exception:
        return default
    if not np.isfinite(number):
        return default
    return number


def _range_zone(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "lower_range"
    if value > upper:
        return "upper_range"
    return "middle_range"


def _trend_state(row: pd.Series) -> str:
    close = _finite_float(row.get("close"))
    ema_fast = _finite_float(row.get("ema_fast"))
    ema_slow = _finite_float(row.get("ema_slow"))
    if close is None or ema_fast is None or ema_slow is None:
        return "trend_unknown"
    if close >= ema_fast >= ema_slow:
        return "bull_trend"
    if close <= ema_fast <= ema_slow:
        return "bear_trend"
    return "mixed_trend"


def _momentum_state(row: pd.Series) -> str:
    momentum = _finite_float(row.get("kc_momentum"), 0.0) or 0.0
    if momentum > 0:
        return "positive_momentum"
    if momentum < 0:
        return "negative_momentum"
    return "flat_momentum"


def _recommended_side(action: str, final_action: str, candidate_action: str) -> tuple[str, str, str]:
    if final_action == "BUY PLAN":
        return "BUY", "ACTIVE", "Final action is BUY PLAN."
    if final_action == "SELL PLAN":
        return "SELL", "ACTIVE", "Final action is SELL PLAN."
    if candidate_action == "BUY CANDIDATE" or action == "BUY PLAN":
        return "BUY", "CANDIDATE_ONLY", "Current engine only has a BUY candidate/rejected plan; use as preview, not execution."
    if candidate_action == "SELL CANDIDATE" or action == "SELL PLAN":
        return "SELL", "CANDIDATE_ONLY", "Current engine only has a SELL candidate/rejected plan; use as preview, not execution."
    return "HOLD", "NO_TRADE", "No actionable or candidate side from the current engine."


def _add_setup_columns(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = df.copy()
    lookback_window = max(80, int(settings.get("lookback_days", 7)) * 96)
    min_periods = min(lookback_window, 80)
    rolling_high = out["high"].rolling(lookback_window, min_periods=min_periods).max()
    rolling_low = out["low"].rolling(lookback_window, min_periods=min_periods).min()
    range_span = (rolling_high - rolling_low).replace(0, np.nan)
    out["_range_position_pct"] = (((out["close"] - rolling_low) / range_span) * 100.0).clip(0, 100)

    lower = float(settings.get("middle_range_lower_pct", 35))
    upper = float(settings.get("middle_range_upper_pct", 65))
    out["_range_zone"] = out["_range_position_pct"].apply(
        lambda value: _range_zone(float(value), lower, upper) if pd.notna(value) else "range_unknown"
    )
    out["_trend_state"] = out.apply(_trend_state, axis=1)
    out["_momentum_state"] = out.apply(_momentum_state, axis=1)
    out["_squeeze_state"] = out.get("squeeze_on", False).fillna(False).astype(bool).map({True: "squeeze_on", False: "squeeze_off"})
    return out


def _current_setup(latest_row: pd.Series, market, settings: dict) -> tuple[dict, str]:
    lower = float(settings.get("middle_range_lower_pct", 35))
    upper = float(settings.get("middle_range_upper_pct", 65))
    setup = {
        "_range_zone": _range_zone(float(market.range_position_pct), lower, upper),
        "_trend_state": _trend_state(latest_row),
        "_momentum_state": _momentum_state(latest_row),
        "_squeeze_state": "squeeze_on" if bool(latest_row.get("squeeze_on", False)) else "squeeze_off",
    }
    label = ", ".join(f"{key.replace('_', '')}={value}" for key, value in setup.items())
    return setup, label


def _matching_sample(labeled: pd.DataFrame, setup: dict, required_columns: list[str]) -> pd.DataFrame:
    sample = labeled.copy()
    for column, value in setup.items():
        sample = sample[sample[column] == value]
    return sample.dropna(subset=required_columns)


def _level_forecast(
    horizon: str,
    bars_ahead: int,
    sample: pd.DataFrame,
    side: str,
    status: str,
    entry: float,
    setup_label: str,
    settings: dict,
    side_reason: str,
) -> HorizonForecast:
    min_samples = int(settings.get("horizon_min_samples", 30))
    if len(sample) < min_samples:
        return HorizonForecast(
            horizon=horizon,
            bars_ahead=bars_ahead,
            side=side,
            status="INSUFFICIENT_DATA",
            entry=entry,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            tp1_move=None,
            tp2_move=None,
            adverse_move=None,
            historical_direction_probability_pct=None,
            sample_count=int(len(sample)),
            matching_setup=setup_label,
            reason=f"Only {len(sample)} matching historical samples; minimum required is {min_samples}. No forecast levels generated.",
        )

    side_key = side.lower()
    profit_col = f"{horizon}_{side_key}_profit"
    adverse_col = f"{horizon}_{side_key}_adverse"
    close_change_col = f"{horizon}_future_close_change"

    profit = sample[profit_col].astype(float)
    adverse = sample[adverse_col].astype(float)
    close_change = sample[close_change_col].astype(float)

    tp1_quantile = float(settings.get("horizon_tp1_quantile", 0.50))
    tp2_quantile = float(settings.get("horizon_tp2_quantile", 0.75))
    adverse_quantile = float(settings.get("horizon_adverse_quantile", 0.80))

    tp1_move = float(profit.quantile(tp1_quantile))
    tp2_move = float(profit.quantile(tp2_quantile))
    adverse_move = float(adverse.quantile(adverse_quantile))

    if tp1_move <= 0 or tp2_move <= 0 or adverse_move <= 0:
        return HorizonForecast(
            horizon=horizon,
            bars_ahead=bars_ahead,
            side=side,
            status="NO_EDGE",
            entry=entry,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            tp1_move=tp1_move,
            tp2_move=tp2_move,
            adverse_move=adverse_move,
            historical_direction_probability_pct=None,
            sample_count=int(len(sample)),
            matching_setup=setup_label,
            reason="Historical matched outcomes do not show positive favorable/adverse values suitable for TP/SL.",
        )

    if side == "BUY":
        stop_loss = entry - adverse_move
        take_profit_1 = entry + tp1_move
        take_profit_2 = entry + tp2_move
        directional_probability = float((close_change > 0).mean() * 100.0)
    else:
        stop_loss = entry + adverse_move
        take_profit_1 = entry - tp1_move
        take_profit_2 = entry - tp2_move
        directional_probability = float((close_change < 0).mean() * 100.0)

    return HorizonForecast(
        horizon=horizon,
        bars_ahead=bars_ahead,
        side=side,
        status=status,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        tp1_move=tp1_move,
        tp2_move=tp2_move,
        adverse_move=adverse_move,
        historical_direction_probability_pct=directional_probability,
        sample_count=int(len(sample)),
        matching_setup=setup_label,
        reason=(
            f"{side_reason} Levels are empirical quantiles from matching historical outcomes: "
            f"TP1 q={tp1_quantile:.2f}, TP2 q={tp2_quantile:.2f}, SL adverse q={adverse_quantile:.2f}."
        ),
    )


def multi_horizon_forecast(
    bars: pd.DataFrame,
    market,
    action: str,
    veto: dict,
    candidate_action: str,
    settings: dict,
) -> list[HorizonForecast]:
    """Return data-derived forecast levels for 30m, 1h, 4h, and 8h.

    The function uses only actual historical forward outcomes from the loaded OHLC
    data. It does not use ATR projection multipliers, drift assumptions, or synthetic
    forecast paths. If there are not enough matching historical samples, the row is
    marked INSUFFICIENT_DATA.
    """
    if bars.empty:
        return []

    final_action = str(veto.get("final_action", "HOLD"))
    side, status, side_reason = _recommended_side(action, final_action, candidate_action)
    entry = float(market.price)
    if side == "HOLD":
        return [
            HorizonForecast(
                horizon=label,
                bars_ahead=bars_ahead,
                side=side,
                status=status,
                entry=entry,
                stop_loss=None,
                take_profit_1=None,
                take_profit_2=None,
                tp1_move=None,
                tp2_move=None,
                adverse_move=None,
                historical_direction_probability_pct=None,
                sample_count=0,
                matching_setup="none",
                reason=side_reason,
            )
            for label, bars_ahead in HORIZONS.items()
        ]

    labeled = _add_setup_columns(add_forward_outcomes(bars, HORIZONS), settings)
    latest_row = bars.iloc[-1]
    setup, setup_label = _current_setup(latest_row, market, settings)

    forecasts: list[HorizonForecast] = []
    for label, bars_ahead in HORIZONS.items():
        side_key = side.lower()
        required = [
            f"{label}_{side_key}_profit",
            f"{label}_{side_key}_adverse",
            f"{label}_future_close_change",
            "_range_zone",
            "_trend_state",
            "_momentum_state",
            "_squeeze_state",
        ]
        sample = _matching_sample(labeled.iloc[:-bars_ahead].copy(), setup, required)
        forecasts.append(_level_forecast(label, bars_ahead, sample, side, status, entry, setup_label, settings, side_reason))
    return forecasts


def horizon_forecast_table(forecasts: list[HorizonForecast]) -> pd.DataFrame:
    return pd.DataFrame([forecast.to_dict() for forecast in forecasts])
