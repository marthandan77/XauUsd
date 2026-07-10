# New Direction: Diagnostics First, No Recommendation Tuning

## Why this change

The machine was moving toward overfitting because it could generate challenger/recommended settings from the same recent samples it was evaluating. That can make history look better while weakening future performance.

## Removed direction

- No automatic recommended settings.
- No challenger promotion from recent backfill alone.
- No tuning many sliders after every scan.
- No treating backfill accuracy as proof of future edge.

## Current machine role

The app is an XAU/USD advisory scanner. It does not place orders.

The engine should now answer three separate questions:

1. **Signal Engine** — Is there directional pressure?
2. **Risk Engine** — Is there enough room for entry, stop, and target?
3. **Validation Engine** — Has this side recently worked in walk-forward testing?

## Signal stages

The machine should classify every scan into one stage:

```text
NO EDGE
SCALP WATCH
SCALP SETUP
TRADE PERMISSION
WAIT - FORCED
```

Only `TRADE PERMISSION` may show an active BUY PLAN or SELL PLAN.

## Gate breakdown

Every scan should be explainable by gates:

```text
Data
Forced wait
Score
Location
Room / reward-risk
5m entry model
Higher timeframe filters
Final veto
```

A no-trade result is acceptable if the gate table clearly states what blocked it.

## Fixed champion protocol

Use one fixed settings set as the champion.

Change settings only when all conditions are true:

1. A specific gate is proven to be too strict or too loose.
2. The change is made to one setting group only.
3. The change improves out-of-sample outcomes, not just the latest backfill.
4. False positives do not increase.
5. Average realized points do not worsen.

## Backfill rule

Backfill is for diagnosis, not automatic optimization.

Acceptable use:

```text
Backfill → inspect gate failures → form hypothesis → test out-of-sample
```

Unacceptable use:

```text
Backfill → tune many settings → immediately trust the new result
```

## Professional priority

No trade is acceptable. Bad trade permission is not acceptable.

The next development priority is not higher firing frequency. The priority is better gate explanation and cleaner separation of signal, risk, and validation.
