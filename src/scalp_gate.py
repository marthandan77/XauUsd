from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.forecast_labels import add_forward_outcomes, horizons_for_interval, interval_minutes
from src.trade_plan import build_trade_plan, room_ratio


@dataclass
class ScalpHorizon:
    horizon: str
    side: str
    status: str
    entry: float | None
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    tp1_move: float | None
    tp2_move: float | None
    sl_move: float | None
    scalp_edge: float | None
    tp1_hit_rate_pct: float | None
    tp2_hit_rate_pct: float | None
    sl_hit_rate_pct: float | None
    train_samples: int
    validation_samples: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScalpDecision:
    recommendation: str
    side: str
    room_ratio: float
    rsi: float | None
    kc_momentum: float | None
    kc_state: str
    reasons: list[str]
    horizons: list[ScalpHorizon]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["horizons"] = [h.to_dict() for h in self.horizons]
        return data


def _finite_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except Exception:
        return default
    if not np.isfinite(number):
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


def _momentum_state(row: pd.Series) -> str:
    momentum = _finite_float(row.get("kc_momentum"), 0.0) or 0.0
    if momentum > 0:
        return "positive_momentum"
    if momentum < 0:
        return "negative_momentum"
    return "flat_momentum"


def _engine_side(scores: dict) -> str:
    bias = str(scores.get("bias", "mixed"))
    if bias == "bearish":
        return "SELL"
    if bias == "bullish":
        return "BUY"
    return "NONE"


def _add_scalp_setup_columns(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Use a deliberately small setup definition to reduce overfitting.

    KC is kept through momentum_state, but RSI is not used to split historical samples.
    RSI is used as a live guard only. This keeps more validation samples available.
    """
    out = df.copy()
    bars_per_day = _bars_per_day(settings)
    lookback_window = max(80, int(settings.get("lookback_days", 7)) * bars_per_day)
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
    out["_momentum_state"] = out.apply(_momentum_state, axis=1)
    return out


def _current_setup(latest_row: pd.Series, market, settings: dict) -> dict:
    lower = float(settings.get("middle_range_lower_pct", 35))
    upper = float(settings.get("middle_range_upper_pct", 65))
    return {
        "_range_zone": _range_zone(float(market.range_position_pct), lower, upper),
        "_momentum_state": _momentum_state(latest_row),
    }


def _matching_sample(labeled: pd.DataFrame, setup: dict, required_columns: list[str]) -> pd.DataFrame:
    sample = labeled.copy()
    for column, value in setup.items():
        sample = sample[sample[column] == value]
    return sample.dropna(subset=required_columns)


def _split_train_validation(sample: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_fraction = float(settings.get("scalp_validation_fraction", 0.30))
    validation_fraction = min(max(validation_fraction, 0.20), 0.50)
    split_index = int(len(sample) * (1.0 - validation_fraction))
    split_index = max(min(split_index, len(sample) - 1), 1)
    return sample.iloc[:split_index].copy(), sample.iloc[split_index:].copy()


def _empty_horizon(horizon: str, side: str, status: str, entry: float | None, reason: str) -> ScalpHorizon:
    return ScalpHorizon(
        horizon=horizon,
        side=side,
        status=status,
        entry=entry,
        stop_loss=None,
        take_profit_1=None,
        take_profit_2=None,
        tp1_move=None,
        tp2_move=None,
        sl_move=None,
        scalp_edge=None,
        tp1_hit_rate_pct=None,
        tp2_hit_rate_pct=None,
        sl_hit_rate_pct=None,
        train_samples=0,
        validation_samples=0,
        reason=reason,
    )


def _horizon_edge(
    labeled: pd.DataFrame,
    horizon: str,
    bars_ahead: int,
    side: str,
    setup: dict,
    entry: float,
    settings: dict,
) -> ScalpHorizon:
    side_key = side.lower()
    profit_col = f"{horizon}_{side_key}_profit"
    adverse_col = f"{horizon}_{side_key}_adverse"
    required = [profit_col, adverse_col, "_range_zone", "_momentum_state"]
    sample = _matching_sample(labeled.iloc[:-bars_ahead].copy(), setup, required)

    min_samples = int(settings.get("scalp_min_samples", 60))
    min_validation = int(settings.get("scalp_min_validation_samples", 15))
    if len(sample) < min_samples:
        return _empty_horizon(
            horizon,
            side,
            "INSUFFICIENT_DATA",
            entry,
            f"Only {len(sample)} matching samples; minimum required is {min_samples}. No scalp levels generated.",
        )

    train, validation = _split_train_validation(sample, settings)
    if len(validation) < min_validation:
        return _empty_horizon(
            horizon,
            side,
            "INSUFFICIENT_VALIDATION",
            entry,
            f"Only {len(validation)} validation samples; minimum required is {min_validation}.",
        )

    train_profit = train[profit_col].astype(float)
    train_adverse = train[adverse_col].astype(float)
    tp1_quantile = float(settings.get("horizon_tp1_quantile", 0.50))
    tp2_quantile = float(settings.get("horizon_tp2_quantile", 0.75))
    adverse_quantile = float(settings.get("horizon_adverse_quantile", 0.80))

    tp1_move = float(train_profit.quantile(tp1_quantile))
    tp2_move = float(train_profit.quantile(tp2_quantile))
    sl_move = float(train_adverse.quantile(adverse_quantile))
    if tp1_move <= 0 or tp2_move <= 0 or sl_move <= 0:
        return _empty_horizon(
            horizon,
            side,
            "NO_EDGE",
            entry,
            "Training sample does not produce positive TP/SL move sizes.",
        )

    validation_profit = validation[profit_col].astype(float)
    validation_adverse = validation[adverse_col].astype(float)

    # Conservative candle rule: if TP and SL are both inside one candle, count SL first.
    sl_hit = validation_adverse >= sl_move
    tp1_hit = (~sl_hit) & (validation_profit >= tp1_move)
    tp2_hit = (~sl_hit) & (validation_profit >= tp2_move)

    tp1_rate = float(tp1_hit.mean())
    tp2_rate = float(tp2_hit.mean())
    sl_rate = float(sl_hit.mean())

    cost_buffer = float(settings.get("scalp_cost_buffer", 0.40))
    tp2_bonus_weight = float(settings.get("scalp_tp2_bonus_weight", 0.25))
    scalp_edge = (
        tp1_rate * tp1_move
        + tp2_bonus_weight * tp2_rate * max(tp2_move - tp1_move, 0.0)
        - sl_rate * sl_move
        - cost_buffer
    )

    if side == "BUY":
        stop_loss = entry - sl_move
        take_profit_1 = entry + tp1_move
        take_profit_2 = entry + tp2_move
    else:
        stop_loss = entry + sl_move
        take_profit_1 = entry - tp1_move
        take_profit_2 = entry - tp2_move

    return ScalpHorizon(
        horizon=horizon,
        side=side,
        status="VALID",
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        tp1_move=tp1_move,
        tp2_move=tp2_move,
        sl_move=sl_move,
        scalp_edge=scalp_edge,
        tp1_hit_rate_pct=tp1_rate * 100.0,
        tp2_hit_rate_pct=tp2_rate * 100.0,
        sl_hit_rate_pct=sl_rate * 100.0,
        train_samples=int(len(train)),
        validation_samples=int(len(validation)),
        reason="Walk-forward validation: older samples set TP/SL, recent samples test hit rates. Same-candle TP/SL counts as SL first.",
    )


def _passes_kc_guard(side: str, latest_row: pd.Series) -> tuple[bool, str]:
    momentum = _finite_float(latest_row.get("kc_momentum"), 0.0) or 0.0
    rising = bool(latest_row.get("kc_momentum_rising", False))
    falling = bool(latest_row.get("kc_momentum_falling", False))
    if side == "SELL" and momentum > 0 and rising:
        return False, "KC momentum is rising against SELL scalp."
    if side == "BUY" and momentum < 0 and falling:
        return False, "KC momentum is falling against BUY scalp."
    return True, "KC momentum is not against the scalp side."


def _passes_rsi_guard(side: str, latest_row: pd.Series, settings: dict) -> tuple[bool, str]:
    rsi = _finite_float(latest_row.get("rsi"))
    if rsi is None:
        return True, "RSI unavailable; guard skipped."
    sell_floor = float(settings.get("scalp_rsi_sell_floor", 35.0))
    buy_ceiling = float(settings.get("scalp_rsi_buy_ceiling", 65.0))
    if side == "SELL" and rsi <= sell_floor:
        return False, f"RSI {rsi:.1f} is already oversold for SELL scalp."
    if side == "BUY" and rsi >= buy_ceiling:
        return False, f"RSI {rsi:.1f} is already overbought for BUY scalp."
    return True, "RSI is not in the danger zone."


def scalp_gate(bars: pd.DataFrame, market, scores: dict, macro: dict, kc: dict, settings: dict) -> ScalpDecision:
    latest_row = bars.iloc[-1] if not bars.empty else pd.Series(dtype=float)
    entry = float(market.price) if market is not None else None
    side = _engine_side(scores)
    reasons: list[str] = []
    horizons: list[ScalpHorizon] = []

    rr = 0.0
    rsi = _finite_float(latest_row.get("rsi"))
    kc_momentum = _finite_float(latest_row.get("kc_momentum"), 0.0)
    kc_state = str(kc.get("state", "unknown")) if isinstance(kc, dict) else "unknown"

    available_horizons = horizons_for_interval(str(settings.get("price_interval", "15m")))
    required_horizons = {label: bars_ahead for label, bars_ahead in available_horizons.items() if label in {"5m", "15m"}}

    if side == "NONE":
        reasons.append("Engine bias is mixed; sniper scalp requires clear bullish or bearish bias.")
    if "5m" not in required_horizons or "15m" not in required_horizons:
        reasons.append("Set Timeframe to 5m. Scalp Gate needs exact 5m and 15m forward outcomes.")
    if bool(macro.get("blocked", False)):
        reasons.append("Manual news/event block is active.")
    if bool(getattr(market, "atr_shock", False)):
        reasons.append("Shock candle detected; no scalp immediately after abnormal movement.")

    if side in {"BUY", "SELL"}:
        plan_action = "BUY PLAN" if side == "BUY" else "SELL PLAN"
        plan_settings = dict(settings)
        plan_settings["short_plans_enabled"] = True
        plan = build_trade_plan(plan_action, market, plan_settings)
        rr = room_ratio(plan_action, plan, market)
        room_min = float(settings.get("scalp_room_ratio_min", settings.get("min_reward_risk", 1.5)))
        if rr < room_min:
            reasons.append(f"Room ratio {rr:.2f} is below scalp minimum {room_min:.2f}.")
        if side == "SELL" and bool(getattr(market, "near_support", False)):
            reasons.append("Price is near support; SELL scalp blocked.")
        if side == "BUY" and bool(getattr(market, "near_resistance", False)):
            reasons.append("Price is near resistance; BUY scalp blocked.")

        kc_ok, kc_reason = _passes_kc_guard(side, latest_row)
        if not kc_ok:
            reasons.append(kc_reason)
        rsi_ok, rsi_reason = _passes_rsi_guard(side, latest_row, settings)
        if not rsi_ok:
            reasons.append(rsi_reason)

    if side in {"BUY", "SELL"} and required_horizons:
        labeled = _add_scalp_setup_columns(add_forward_outcomes(bars, required_horizons), settings)
        setup = _current_setup(latest_row, market, settings)
        for label, bars_ahead in required_horizons.items():
            horizons.append(_horizon_edge(labeled, label, bars_ahead, side, setup, float(entry), settings))
    elif side in {"BUY", "SELL"}:
        for label in ["5m", "15m"]:
            horizons.append(_empty_horizon(label, side, "UNAVAILABLE", entry, "Selected timeframe cannot create this horizon exactly."))

    horizon_map = {h.horizon: h for h in horizons}
    five = horizon_map.get("5m")
    fifteen = horizon_map.get("15m")
    min_edge = float(settings.get("scalp_min_edge", 0.0))

    if five is None or five.status != "VALID":
        reasons.append("5m scalp edge is unavailable or invalid.")
    elif five.scalp_edge is None or five.scalp_edge <= min_edge:
        reasons.append(f"5m scalp edge {five.scalp_edge:.2f} is not above minimum {min_edge:.2f}.")
    elif (five.tp1_hit_rate_pct or 0.0) <= (five.sl_hit_rate_pct or 0.0):
        reasons.append("5m TP1 hit rate is not better than SL hit rate.")

    if fifteen is None or fifteen.status != "VALID":
        reasons.append("15m danger check is unavailable or invalid.")
    elif fifteen.scalp_edge is None or fifteen.scalp_edge < 0:
        reasons.append(f"15m danger check is negative: {fifteen.scalp_edge:.2f}.")

    recommendation = "NO SCALP"
    if side in {"BUY", "SELL"} and not reasons:
        recommendation = f"{side} SCALP"

    return ScalpDecision(
        recommendation=recommendation,
        side=side,
        room_ratio=rr,
        rsi=rsi,
        kc_momentum=kc_momentum,
        kc_state=kc_state,
        reasons=reasons,
        horizons=horizons,
    )


def scalp_horizon_table(decision: ScalpDecision) -> pd.DataFrame:
    return pd.DataFrame([h.to_dict() for h in decision.horizons])
