# Global Macro Asset Allocation Dashboard

An AI-augmented portfolio analysis tool built with Streamlit. Pulls live market data, runs Monte Carlo portfolio optimisation with NLP sentiment overlay, and validates strategies through walk-forward backtesting.

## Features

| Tab | Description |
|-----|-------------|
| **Live Market Pulse** | Treemap of trailing 5-day sector performance across 15 ETFs |
| **Correlation Heatmap** | Dynamic correlation matrix built from 1-year daily returns |
| **Portfolio Sandbox** | Monte Carlo optimiser (2,000 simulations) with Finnhub news sentiment alpha |
| **Walk-Forward Backtest** | Rolling out-of-sample backtest — optimise on train window, evaluate on test window, no look-ahead bias |
| **Risk/Return Database** | Sortable table of annualised return, volatility, and Sharpe ratio for all assets |

## Asset Universe

15 tickers covering US equity sectors, fixed income, and commodities:

`XLK` `XLV` `XLF` `XLY` `XLP` `XLE` `XLI` `XLB` `XLU` `XLRE` `XLC` `GLD` `TLT` `SMH` `SKYY`

## Project Structure

```
├── app.py          # Streamlit UI — tabs, charts, user inputs
├── data.py         # Data fetching, NLP model, sentiment scoring
├── optimizer.py    # Monte Carlo optimiser + walk-forward backtester
├── requirements.txt
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Optional: News Sentiment (Finnhub)

The Portfolio Sandbox tab can overlay real NLP sentiment alpha on expected returns. To enable it, add your free [Finnhub](https://finnhub.io) API key:

```toml
# .streamlit/secrets.toml
FINNHUB_API_KEY = "your_key_here"
```

Without the key the app runs normally — sentiment defaults to neutral (0.0).

## Walk-Forward Backtest

The backtest uses a rolling non-overlapping window scheme:

1. **Train** — optimise max-Sharpe portfolio weights on the training window (3–12 months)
2. **Test** — apply those weights to the next out-of-sample period (1–3 months), record returns
3. **Repeat** — slide both windows forward and repeat until data is exhausted

Output includes OOS CAGR, annualised volatility, Sharpe ratio, max drawdown, a cumulative growth curve, per-window return bars, and an allocation heatmap showing how weights shift over time.

## Deployment

Hosted on [Streamlit Community Cloud](https://share.streamlit.io). Any push to `main` triggers an automatic redeploy.
