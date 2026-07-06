# XAU/USD Forecast Manager

Simple Streamlit dashboard for manual XAU/USD forecasting.

Purpose: forecast market regime, directional bias, expected range, clean entry zone, stop level, target levels, and veto reasons. The main design rule is:

> Veto first. Signal second. No trade is better than a weak setup.

## What this app does

- Loads XAU/USD-style OHLC data from live provider when available.
- Falls back to generated sample data so the dashboard always opens.
- Accepts CSV upload with `time`, `open`, `high`, `low`, `close`, and optional `volume`.
- Calculates EMA, VWAP, ATR, RSI, ADX, range position, support, resistance, and swing levels.
- Classifies regime: bull trend, bear trend, range, or shock.
- Creates an ATR-based expected range forecast.
- Produces advisory states: BUY PLAN, SELL PLAN, WAIT, or HOLD.
- Applies strict veto filters before any actionable plan appears.
- Uses manual settings and presets only. No auto-learning.

## What it does not do

- No broker execution.
- No auto-trading.
- No martingale.
- No automatic parameter tuning.
- No AI override of the formula.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

- Repository: `marthandan77/XauUsd`
- Branch: `main`
- Main file path: `app.py`

## Recommended workflow

1. Run in dashboard mode.
2. Keep Balanced Strict preset first.
3. Log at least 50 forecasts before changing parameters.
4. Judge forecasts by regime, not by one or two trades.
5. Keep WAIT/HOLD normal. The system is designed to reject weak setups.
