# GitHub Learning Setup

The app supports GitHub-only learning storage.

## Required Streamlit secrets

Add these in Streamlit `secrets.toml` or Streamlit Cloud secrets:

```toml
TWELVE_DATA_API_KEY = "your_twelve_data_key"
GITHUB_TOKEN = "your_fine_grained_github_token"
GITHUB_REPO = "marthandan77/XauUsd"
```

Optional overrides:

```toml
LEARNING_GITHUB_TOKEN = "your_learning_token"
LEARNING_GITHUB_REPO = "marthandan77/XauUsd"
LEARNING_GITHUB_BRANCH = "main"
```

The app accepts `LEARNING_GITHUB_TOKEN`, `GITHUB_TOKEN`, or `GH_TOKEN`.

## What gets written

Prediction events:

```text
learning/predictions/YYYY-MM-DD/<scan_id>.json
```

Outcome events:

```text
learning/outcomes/YYYY-MM-DD/<scan_id>_<horizon>.json
```

Challenger settings:

```text
config/challenger_settings.yaml
learning/reports/latest_learning_report.md
```

## Workflow

1. Press `SCAN NOW`.
2. The app logs the current prediction to GitHub.
3. On later scans, expired predictions are evaluated against real market data.
4. Open the `Learning` page to refresh outcome metrics.
5. Use `Generate challenger settings` to create a manually reviewable challenger config.

Live settings are not overwritten automatically.
