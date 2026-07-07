# XAU/USD Forecast Manager

Manual Streamlit advisory dashboard for XAU/USD. It is designed for decision support only.

> Veto first. Signal second. No trade is better than a weak setup.

## Current engine status

The engine is now a multi-layer XAU/USD advisory system:

1. Loads XAU/USD OHLC data from Twelve Data, CSV upload, or sample fallback.
2. Calculates EMA, VWAP, ATR, RSI, ADX, realized volatility, Bollinger Bands, Keltner Channels, squeeze state, squeeze release, and KC momentum.
3. Builds structural market levels: swing high/low, previous-day high/low, Asian session high/low, London session high/low, round-number support/resistance, and liquidity sweep flags.
4. Classifies regime: bull trend, bear trend, range, shock, compression, bullish squeeze breakout, or bearish squeeze breakout.
5. Scores bull/bear pressure with a component breakdown: macro, trend, momentum, location, structure, and volatility/risk.
6. Applies veto filters before any actionable plan appears.
7. Validates bull/bear bias against similar historical setups.
8. Confirms 5m trigger, 15m danger check, and 1h structure through the MTF confirmation module.
9. Runs Scalp Gate for manual BUY/SELL scalp advisory only.
10. Logs manual scalp scans to `data/scan_log.csv`.

## What this app does

- Produces advisory states: `BUY PLAN`, `SELL PLAN`, `EXIT LONG / AVOID BUY`, `WAIT`, or `HOLD`.
- Shows candidate entry, stop loss, TP1, TP2, room ratio, veto reasons, and quality status.
- Shows `SCALP SIGNAL`, `NEAR MISS / WAIT`, or `NO GOOD SIGNAL` when `Scan scalp now` is clicked.
- Shows confidence as a quality label only. Confidence does not override Scalp Gate.
- Keeps TP2 in the empirical scalp edge calculation.
- Uses KC and RSI as live scalp guards.
- Saves sidebar settings only when `Save sidebar settings` is clicked.
- Provides a local smoke-check script: `python scripts/smoke_check.py`.

## What it does not do

- No broker execution.
- No auto-trading.
- No backtesting engine.
- No martingale.
- No automatic parameter tuning.
- No AI override of the formula.
- No hidden live-order logic.

## Main pages

- `Sniper Dashboard` — main decision screen: trade now or wait, final action, MTF status, exhaustion state, blockers, scan button.
- `Forecast Manager` — full forecast, veto, candidate plan, and scalp scan.
- `Bull/Bear Validation` — checks whether the current bull/bear setup historically followed through.
- `MTF Confirmation` — 5m trigger, 15m danger check, 1h structure.
- `Structure Levels` — support/resistance source, previous-day/session/swing/round-number levels.
- `Score Breakdown` — component-level bull/bear scoring.
- `Exhaustion Guard` — reversal/exhaustion warnings and BUY/SELL block state.
- `Scalp Gate` — empirical 5m/15m scalp edge table.
- `Scan History` — logged manual scans from `data/scan_log.csv`.

## Completed build phases

### Phase 1 — Bull/Bear Validation

Added `src/bias_validation.py`.

It matches the current setup against similar historical setups and reports:

- sample count
- follow-through percentage
- average favorable move
- average adverse move
- edge score
- verdict: `STRONG`, `WEAK`, or `NO EDGE`

### Phase 2 / 3 — Scalp Gate correction

Updated `src/scalp_gate.py` and dashboard scan output.

Changes:

- scalp room ratio now uses scalp SL distance, not full trade-plan risk
- 15m danger check is graded with hard block at `scalp_15m_block_edge: -0.25`
- default `scalp_min_samples` reduced from 60 to 40
- confidence removed as hard blocker
- added `NEAR MISS / WAIT`

### Phase 4 — Multi-Timeframe Confirmation

Added `src/multi_timeframe.py`.

Roles:

- 5m = trigger
- 15m = danger check
- 1h = structure

Verdicts:

- `ALIGNED`
- `SCALP_ONLY`
- `CONFLICT`
- `INCOMPLETE`

### Phase 5 — Structure Levels

Updated `src/market_map.py`.

Support/resistance now considers:

- lookback high/low
- recent swing high/low
- previous-day high/low
- Asian session high/low
- London session high/low
- round-number levels
- liquidity sweep detection

### Phase 6 — Score Breakdown

Updated `src/forecast_engine.py`.

Bull/bear scoring now separates:

- macro
- trend
- momentum
- location
- structure
- volatility/risk

### Phase 7 — Exhaustion Guard

Added `src/exhaustion_engine.py` and updated `src/veto_engine.py`.

The guard can warn or block:

- exhausted BUY chase
- exhausted SELL chase
- upside liquidity sweep
- downside liquidity sweep
- RSI/KC momentum reversal at extreme range location

### Phase 8 — Sniper Dashboard

Updated `app.py` with a main decision page:

- Can trade now?
- Final action
- MTF verdict
- Exhaustion state
- Bull/Bear validation
- Blockers
- Scan scalp now
- Entry/SL/TP when active

### Phase 9 — Scan Logging

Added `src/scan_logger.py`.

Manual scalp scans are logged to:

```text
data/scan_log.csv
```

### Phase 10 — Smoke Check

Added:

```text
scripts/smoke_check.py
```

Run it before judging the dashboard after updates.

## Run locally

```bash
pip install -r requirements.txt
python scripts/smoke_check.py
streamlit run app.py
```

## Streamlit Cloud

- Repository: `marthandan77/XauUsd`
- Branch: `main`
- Main file path: `app.py`

## Recommended workflow

1. Pull latest code.
2. Run `python scripts/smoke_check.py`.
3. Start Streamlit.
4. Use `Timeframe = 5m` for Scalp Gate and MTF confirmation.
5. Click `Reset saved sidebar settings` once after major engine updates.
6. Use `Sniper Dashboard` as the main decision page.
7. Treat `WAIT`, `NO GOOD SIGNAL`, and `NEAR MISS / WAIT` as valid outcomes.
8. Log at least 50 scans before changing parameters.
9. Judge the engine by sample behavior, not by one or two trades.
