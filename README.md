# XAU/USD Forecast Manager

Simple Streamlit dashboard for manual XAU/USD forecasting.

Purpose: forecast market regime, directional bias, expected range, clean entry zone, stop level, target levels, veto reasons, and KC Squeeze breakout context. The main design rule is:

> Veto first. Signal second. No trade is better than a weak setup.

## What this app does

- Loads XAU/USD-style OHLC data from live provider when available.
- Falls back to generated sample data so the dashboard always opens.
- Accepts CSV upload with `time`, `open`, `high`, `low`, `close`, and optional `volume`.
- Calculates EMA, VWAP, ATR, RSI, ADX, realized volatility, Bollinger Bands, Keltner Channels, squeeze status, squeeze release, KC momentum, support, resistance, and swing levels.
- Classifies regime: bull trend, bear trend, range, shock, compression, bullish squeeze breakout, or bearish squeeze breakout.
- Creates an ATR-based expected range forecast.
- Produces advisory states: BUY PLAN, EXIT LONG / AVOID BUY, SELL PLAN only if shorts are enabled, WAIT, or HOLD.
- Applies strict veto filters before any actionable plan appears.
- Uses manual settings and presets only. No auto-learning.
- Preserves the original QuantConnect KC Squeeze strategy in `quantconnect_reference/linear_alpha_kc_squeeze.py` for reference only.

## What it does not do

- No broker execution.
- No auto-trading.
- No backtesting.
- No martingale.
- No automatic parameter tuning.
- No AI override of the formula.
- No hidden live-order logic.

## KC Squeeze conversion

The original QuantConnect strategy used:

- XAU/USD CFD on hourly data.
- Bollinger Bands inside Keltner Channels to detect compression.
- Squeeze fired when compression ends.
- Linear-regression momentum to confirm breakout direction.
- 200-period trend SMA.
- ATR-based stop and take-profit logic.

In this Streamlit project, the same idea is converted into advisory modules:

- `src/indicators.py` calculates BB, KC, squeeze status, squeeze fired, ATR, trend SMA, and KC momentum.
- `src/kc_squeeze_engine.py` summarizes the KC Squeeze state.
- `src/regime_engine.py` can classify compression and squeeze-breakout regimes.
- `src/forecast_engine.py` blends KC Squeeze information into bull/bear scoring.
- `src/trade_plan.py` creates advisory entry, guard, and target levels only.
- `app.py` displays the KC Squeeze page and Plotly chart overlays.

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
3. Keep shorts disabled unless you specifically want short advisory plans.
4. Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches.
5. Log at least 50 forecasts before changing parameters.
6. Judge forecasts by regime, not by one or two trades.
7. Keep WAIT/HOLD normal. The system is designed to reject weak setups.
