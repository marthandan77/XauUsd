# XAU/USD Forecast Manager

Streamlit dashboard for manual XAU/USD forecasting using Twelve Data, uploaded CSV data, or deterministic sample data.

The main design rule is:

> Veto first. Signal second. No trade is better than a weak setup.

## What this app does

- Loads XAU/USD OHLC data from Twelve Data using `TWELVE_DATA_API_KEY`.
- Falls back to generated sample data when the live feed is unavailable and sample fallback is enabled.
- Accepts CSV uploads with `time` or `datetime`, `open`, `high`, `low`, `close`, and optional `volume` columns.
- Keeps sidebar changes as drafts until **Apply Settings** is pressed.
- Saves applied settings in the active Streamlit browser session.
- Runs data loading, indicators, KC Squeeze, regime classification, forecast scoring, trade planning, range bands, macro context, and veto logic only when **Scan Market** is pressed.
- Calculates EMA, VWAP, ATR, RSI, ADX, realized volatility, Bollinger Bands, Keltner Channels, squeeze status, squeeze release, KC momentum, support, resistance, and swing levels.
- Classifies bull trend, bear trend, range, shock, compression, bullish squeeze breakout, or bearish squeeze breakout regimes.
- Creates ATR-based expected-range forecasts.
- Produces advisory states: BUY PLAN, EXIT LONG / AVOID BUY, SELL PLAN only when shorts are enabled, WAIT, or HOLD.
- Applies veto filters before any actionable plan appears.
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

## Twelve Data configuration

For Streamlit Community Cloud, open the app settings and add this secret:

```toml
TWELVE_DATA_API_KEY = "your-api-key"
```

The default symbol is `XAU/USD`. It can be changed from the app sidebar or in `config/default_settings.yaml`.

## Apply and scan workflow

1. Adjust the sidebar controls or load a preset.
2. Press **Apply Settings**. This activates the draft values and clears the previous scan.
3. Press **Scan Market** on the dashboard.
4. The scan processes the active data source and every strategy module once.
5. Review the Forecast Manager, KC Squeeze, Market Map, Macro / News, Settings, and Log Snapshot pages.
6. After changing any setting, apply it and run a new scan.

Applied settings persist for the active browser session. Use the Settings page to download the active configuration as YAML.

## KC Squeeze conversion

The original QuantConnect strategy used:

- XAU/USD CFD on hourly data.
- Bollinger Bands inside Keltner Channels to detect compression.
- Squeeze fired when compression ends.
- Linear-regression momentum to confirm breakout direction.
- 200-period trend SMA.
- ATR-based stop and take-profit logic.

In this Streamlit project:

- `src/indicators.py` calculates BB, KC, squeeze status, squeeze fired, ATR, trend SMA, and KC momentum.
- `src/kc_squeeze_engine.py` summarizes the KC Squeeze state.
- `src/regime_engine.py` classifies compression and squeeze-breakout regimes.
- `src/forecast_engine.py` blends KC Squeeze information into bull/bear scoring.
- `src/trade_plan.py` creates advisory entry, guard, and target levels only.
- `app.py` controls settings application, scan execution, dashboard rendering, and Plotly chart overlays.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m compileall -q app.py src tests
pytest -q
```

The test suite covers core veto and range logic, timeframe-correct market windows, CSV validation, Twelve Data interval mapping, Apply/Scan control presence, a full sample-data scan, and Streamlit startup.

## Deploy on Streamlit Cloud

- Repository: `marthandan77/XauUsd`
- Branch: `main`
- Main file path: `app.py`
- Required secret: `TWELVE_DATA_API_KEY`

## Recommended workflow

1. Keep Balanced Strict as the initial preset.
2. Apply settings before each scan.
3. Press Scan Market only when a fresh analysis is required.
4. Keep shorts disabled unless short advisory plans are specifically required.
5. Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches.
6. Log at least 50 forecasts before changing parameters.
7. Judge forecasts by regime, not by one or two trades.
8. Keep WAIT/HOLD normal. The system is designed to reject weak setups.
