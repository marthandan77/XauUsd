from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.forecast_engine import choose_action, score_forecast
from src.forecast_labels import interval_minutes
from src.indicators import add_indicators
from src.kc_squeeze_engine import kc_squeeze_summary
from src.market_map import build_market_map
from src.regime_engine import classify_regime


@dataclass
class TimeframeView:
    timeframe: str
    role: str
    status: str
    bias: str
    action: str
    regime: str
    confidence: int
    kc_state: str
    kc_momentum: float | None
    rsi: float | None
    price: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


TARGET_TIMEFRAMES: list[tuple[str, str]] = [
    ("5m", "trigger"),
    ("15m", "danger_check"),
    ("1h", "structure"),
]


def _empty_view(timeframe: str, role: str, reason: str) -> TimeframeView:
    return TimeframeView(
        timeframe=timeframe,
        role=role,
        status="UNAVAILABLE",
        bias="mixed",
        action="HOLD",
        regime="unknown",
        confidence=0,
        kc_state="unknown",
        kc_momentum=None,
        rsi=None,
        price=None,
        reason=reason,
    )


def _resample_ohlc(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, errors="coerce")
        work = work[work.index.notna()].copy()
    if work.empty:
        return work
    rule = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}.get(target, target)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in work.columns:
        agg["volume"] = "sum"
    out = work.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    if "volume" not in out.columns:
        out["volume"] = 1
    return out


def _bars_for_timeframe(bars: pd.DataFrame, source_interval: str, target: str) -> tuple[pd.DataFrame, str]:
    source_min = interval_minutes(source_interval)
    target_min = interval_minutes(target)
    if source_min == target_min:
        return bars.copy(), "source candles"
    if target_min > source_min and target_min % source_min == 0:
        return _resample_ohlc(bars, target), f"resampled from {source_interval}"
    return pd.DataFrame(), f"unavailable from {source_interval}; select Timeframe = 5m for full MTF confirmation"


def _view_for_timeframe(bars: pd.DataFrame, source_interval: str, target: str, role: str, settings: dict, macro: dict) -> TimeframeView:
    tf_bars, source_note = _bars_for_timeframe(bars, source_interval, target)
    if tf_bars.empty or len(tf_bars) < 80:
        return _empty_view(target, role, f"Not enough {target} candles after {source_note}.")

    tf_settings = dict(settings)
    tf_settings["price_interval"] = target
    enriched = add_indicators(tf_bars, tf_settings)
    clean = enriched.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
    if clean.empty or len(clean) < 50:
        return _empty_view(target, role, f"Not enough clean {target} candles after indicator warm-up.")

    market = build_market_map(clean, tf_settings)
    regime = classify_regime(clean, market, tf_settings)
    kc = kc_squeeze_summary(clean, tf_settings)
    latest = clean.iloc[-1].to_dict()
    latest["kc_state"] = kc.get("state", "unknown")
    latest["kc_reason"] = kc.get("reason", "")
    scores = score_forecast(latest, market, regime, macro, tf_settings)
    if kc.get("state") != "insufficient_data":
        scores["kc_state"] = kc.get("state", "unknown")
        scores["kc_reason"] = kc.get("reason", "")
    action = choose_action(scores, tf_settings)

    return TimeframeView(
        timeframe=target,
        role=role,
        status="VALID",
        bias=str(scores.get("bias", "mixed")),
        action=action,
        regime=regime,
        confidence=int(scores.get("confidence", 0)),
        kc_state=str(kc.get("state", "unknown")),
        kc_momentum=latest.get("kc_momentum"),
        rsi=latest.get("rsi"),
        price=latest.get("close"),
        reason=str(kc.get("reason", source_note)),
    )


def multi_timeframe_confirmation(bars: pd.DataFrame, settings: dict, macro: dict) -> dict:
    source_interval = str(settings.get("price_interval", "15m"))
    views = [
        _view_for_timeframe(bars, source_interval, timeframe, role, settings, macro)
        for timeframe, role in TARGET_TIMEFRAMES
    ]

    valid = {view.timeframe: view for view in views if view.status == "VALID"}
    trigger = valid.get("5m")
    danger = valid.get("15m")
    structure = valid.get("1h")

    verdict = "INCOMPLETE"
    reasons: list[str] = []
    side = "NONE"
    if trigger is None:
        reasons.append("5m trigger is unavailable.")
    else:
        side = "BUY" if trigger.bias == "bullish" else "SELL" if trigger.bias == "bearish" else "NONE"
        if side == "NONE":
            reasons.append("5m trigger bias is mixed.")

    if side != "NONE" and danger is not None:
        if danger.bias == "mixed":
            reasons.append("15m danger check is mixed; treat as caution.")
        elif (side == "BUY" and danger.bias == "bearish") or (side == "SELL" and danger.bias == "bullish"):
            reasons.append("15m danger check is against the 5m trigger.")
    elif danger is None:
        reasons.append("15m danger check is unavailable.")

    if side != "NONE" and structure is not None:
        if structure.bias == "mixed":
            reasons.append("1h structure is mixed; trade only scalp-sized.")
        elif (side == "BUY" and structure.bias == "bearish") or (side == "SELL" and structure.bias == "bullish"):
            reasons.append("1h structure is against the 5m trigger.")
    elif structure is None:
        reasons.append("1h structure is unavailable.")

    if side != "NONE" and not any("against" in reason for reason in reasons) and trigger is not None and danger is not None:
        if structure is not None and structure.bias in {trigger.bias, "mixed"}:
            verdict = "ALIGNED"
        else:
            verdict = "SCALP_ONLY"
    elif any("against" in reason for reason in reasons):
        verdict = "CONFLICT"

    return {
        "source_interval": source_interval,
        "side": side,
        "verdict": verdict,
        "reasons": reasons,
        "views": [view.to_dict() for view in views],
    }


def multi_timeframe_table(summary: dict) -> pd.DataFrame:
    table = pd.DataFrame(summary.get("views", []) if isinstance(summary, dict) else [])
    if table.empty:
        return table
    for column in ["kc_momentum", "rsi", "price"]:
        if column in table:
            table[column] = table[column].apply(lambda value: None if pd.isna(value) else round(float(value), 2))
    return table
