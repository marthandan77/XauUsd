from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.forecast_labels import add_forward_outcomes, horizons_for_interval, interval_minutes


@dataclass
class BiasValidationRow:
    horizon: str
    side: str
    status: str
    verdict: str
    sample_count: int
    follow_through_pct: float | None
    avg_favorable_move: float | None
    avg_adverse_move: float | None
    edge_score: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _as_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if pd.isna(number):
        return default
    return number


def _bars_per_day(settings: dict) -> int:
    return max(int(1440 / interval_minutes(str(settings.get("price_interval", "15m")))), 1)


def _range_zone(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "lower_range"
    if value > upper:
        return "upper_range"
    return "middle_range"


def _trend_state(row: pd.Series, settings: dict) -> str:
    close = _as_float(row.get("close"))
    ema_fast = _as_float(row.get("ema_fast"), close)
    ema_slow = _as_float(row.get("ema_slow"), close)
    adx = _as_float(row.get("adx"), 0.0)
    adx_min = _as_float(settings.get("adx_minimum"), 20.0)
    if ema_fast > ema_slow and close > ema_fast and adx >= adx_min:
        return "bull_trend"
    if ema_fast < ema_slow and close < ema_fast and adx >= adx_min:
        return "bear_trend"
    return "range_or_transition"


def _momentum_state(row: pd.Series) -> str:
    momentum = _as_float(row.get("kc_momentum"), 0.0)
    previous = _as_float(row.get("kc_momentum_prev"), 0.0)
    if momentum > 0 and momentum >= previous:
        return "positive_rising"
    if momentum > 0 and momentum < previous:
        return "positive_falling"
    if momentum < 0 and momentum <= previous:
        return "negative_falling"
    if momentum < 0 and momentum > previous:
        return "negative_rising"
    return "flat"


def _rsi_zone(row: pd.Series) -> str:
    rsi = _as_float(row.get("rsi"), 50.0)
    if rsi >= 65:
        return "overbought"
    if rsi <= 35:
        return "oversold"
    return "neutral"


def _squeeze_state(row: pd.Series) -> str:
    squeeze_on = bool(row.get("squeeze_on", False))
    squeeze_fired = bool(row.get("squeeze_fired", False))
    momentum = _as_float(row.get("kc_momentum"), 0.0)
    previous = _as_float(row.get("kc_momentum_prev"), 0.0)
    if squeeze_fired and momentum > 0 and momentum > previous:
        return "bullish_release"
    if squeeze_fired and momentum < 0 and momentum < previous:
        return "bearish_release"
    if squeeze_on:
        return "compression"
    return "normal"


def _add_setup_columns(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    out = df.copy()
    bars_per_day = _bars_per_day(settings)
    lookback_window = max(80, int(settings.get("lookback_days", 7)) * bars_per_day)
    min_periods = min(lookback_window, 80)
    rolling_high = out["high"].rolling(lookback_window, min_periods=min_periods).max()
    rolling_low = out["low"].rolling(lookback_window, min_periods=min_periods).min()
    span = (rolling_high - rolling_low).replace(0, pd.NA)
    out["_range_position_pct"] = (((out["close"] - rolling_low) / span) * 100.0).clip(0, 100)

    lower = _as_float(settings.get("middle_range_lower_pct"), 35.0)
    upper = _as_float(settings.get("middle_range_upper_pct"), 65.0)
    out["_range_zone"] = out["_range_position_pct"].apply(
        lambda value: _range_zone(float(value), lower, upper) if pd.notna(value) else "range_unknown"
    )
    out["_trend_state"] = out.apply(lambda row: _trend_state(row, settings), axis=1)
    out["_momentum_state"] = out.apply(_momentum_state, axis=1)
    out["_rsi_zone"] = out.apply(_rsi_zone, axis=1)
    out["_squeeze_state"] = out.apply(_squeeze_state, axis=1)
    return out


def _current_setup(latest_row: pd.Series, market, settings: dict) -> dict:
    lower = _as_float(settings.get("middle_range_lower_pct"), 35.0)
    upper = _as_float(settings.get("middle_range_upper_pct"), 65.0)
    return {
        "_range_zone": _range_zone(float(market.range_position_pct), lower, upper),
        "_trend_state": _trend_state(latest_row, settings),
        "_momentum_state": _momentum_state(latest_row),
        "_rsi_zone": _rsi_zone(latest_row),
        "_squeeze_state": _squeeze_state(latest_row),
    }


def _matching_sample(labeled: pd.DataFrame, setup: dict, required: list[str]) -> pd.DataFrame:
    sample = labeled.copy()
    for column, value in setup.items():
        sample = sample[sample[column] == value]
    return sample.dropna(subset=required)


def _empty_row(horizon: str, side: str, status: str, reason: str) -> BiasValidationRow:
    return BiasValidationRow(
        horizon=horizon,
        side=side,
        status=status,
        verdict="NO EDGE",
        sample_count=0,
        follow_through_pct=None,
        avg_favorable_move=None,
        avg_adverse_move=None,
        edge_score=None,
        reason=reason,
    )


def _validate_horizon(labeled: pd.DataFrame, horizon: str, bars_ahead: int, side: str, setup: dict, settings: dict) -> BiasValidationRow:
    if side not in {"BUY", "SELL"}:
        return _empty_row(horizon, side, "NO_BIAS", "Current engine bias is mixed; no directional validation is possible.")

    side_key = "buy" if side == "BUY" else "sell"
    profit_col = f"{horizon}_{side_key}_profit"
    adverse_col = f"{horizon}_{side_key}_adverse"
    close_change_col = f"{horizon}_future_close_change"
    required = [profit_col, adverse_col, close_change_col, *_SETUP_COLUMNS]
    sample = _matching_sample(labeled.iloc[:-bars_ahead].copy(), setup, required)

    min_samples = int(settings.get("bias_validation_min_samples", 30))
    if len(sample) < min_samples:
        return _empty_row(
            horizon,
            side,
            "LOW_SAMPLE",
            f"Only {len(sample)} similar setups; minimum required is {min_samples}.",
        )

    close_change = sample[close_change_col].astype(float)
    if side == "BUY":
        follow = close_change > 0
    else:
        follow = close_change < 0

    follow_rate = float(follow.mean())
    avg_favorable = float(sample[profit_col].astype(float).mean())
    avg_adverse = float(sample[adverse_col].astype(float).mean())
    edge = follow_rate * avg_favorable - (1.0 - follow_rate) * avg_adverse

    if follow_rate >= 0.58 and edge > 0:
        verdict = "STRONG"
    elif follow_rate >= 0.52 and edge >= 0:
        verdict = "WEAK"
    else:
        verdict = "NO EDGE"

    return BiasValidationRow(
        horizon=horizon,
        side=side,
        status="VALID",
        verdict=verdict,
        sample_count=int(len(sample)),
        follow_through_pct=follow_rate * 100.0,
        avg_favorable_move=avg_favorable,
        avg_adverse_move=avg_adverse,
        edge_score=edge,
        reason="Matched current trend, range zone, KC momentum, RSI zone and squeeze state; measured actual future close follow-through.",
    )


_SETUP_COLUMNS = ["_range_zone", "_trend_state", "_momentum_state", "_rsi_zone", "_squeeze_state"]


def bias_validation_summary(bars: pd.DataFrame, market, scores: dict, settings: dict) -> dict:
    if bars is None or bars.empty:
        return {
            "bias": "mixed",
            "side": "NONE",
            "overall_verdict": "NO EDGE",
            "setup": {},
            "rows": [],
            "warning": "No bars available for bull/bear validation.",
        }

    bias = str(scores.get("bias", "mixed"))
    if bias == "bullish":
        side = "BUY"
    elif bias == "bearish":
        side = "SELL"
    else:
        side = "NONE"

    available_horizons = horizons_for_interval(str(settings.get("price_interval", "15m")))
    wanted = ["5m", "15m", "1h"]
    selected_horizons = {label: bars_ahead for label, bars_ahead in available_horizons.items() if label in wanted}

    latest_row = bars.iloc[-1]
    setup = _current_setup(latest_row, market, settings)
    rows: list[BiasValidationRow] = []
    if selected_horizons:
        labeled = _add_setup_columns(add_forward_outcomes(bars, selected_horizons), settings)
        for horizon, bars_ahead in selected_horizons.items():
            rows.append(_validate_horizon(labeled, horizon, bars_ahead, side, setup, settings))
    else:
        rows.append(_empty_row("N/A", side, "UNAVAILABLE", "Selected timeframe cannot create 5m, 15m or 1h validation exactly."))

    verdicts = [row.verdict for row in rows if row.status == "VALID"]
    if "STRONG" in verdicts:
        overall = "STRONG"
    elif "WEAK" in verdicts:
        overall = "WEAK"
    else:
        overall = "NO EDGE"

    warning = ""
    if "5m" not in selected_horizons:
        warning = "Select Timeframe = 5m to validate 5m, 15m and 1h together."

    return {
        "bias": bias,
        "side": side,
        "overall_verdict": overall,
        "setup": setup,
        "rows": [row.to_dict() for row in rows],
        "warning": warning,
    }


def bias_validation_table(summary: dict) -> pd.DataFrame:
    rows = summary.get("rows", []) if isinstance(summary, dict) else []
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for column in ["follow_through_pct", "avg_favorable_move", "avg_adverse_move", "edge_score"]:
        if column in table:
            table[column] = table[column].apply(lambda value: None if pd.isna(value) else round(float(value), 2))
    return table
