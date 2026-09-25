"""
data.py — Market data fetching, NLP model loading, and shared constants.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
from datetime import datetime, timedelta
import streamlit as st

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MARKET_TICKERS: dict[str, str] = {
    'Technology':       'XLK',
    'Healthcare':       'XLV',
    'Financials':       'XLF',
    'Consumer Discr':   'XLY',
    'Consumer Staples': 'XLP',
    'Energy':           'XLE',
    'Industrials':      'XLI',
    'Materials':        'XLB',
    'Utilities':        'XLU',
    'Real Estate':      'XLRE',
    'Communications':   'XLC',
    'Gold':             'GLD',
    'Long Bonds':       'TLT',
    'Semiconductors':   'SMH',
    'Cloud':            'SKYY',
}

RISK_FREE_RATE = 0.02
TRADING_DAYS   = 252
NEWS_LOOKBACK_DAYS = 3


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_and_calculate_live_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Download 1-year daily closes; return (daily_returns, summary_df, update_time)."""
    tickers = list(MARKET_TICKERS.values())
    raw_data = yf.download(tickers, period="1y")['Close']
    daily_returns = raw_data.pct_change().dropna(how="all")

    ann_return     = daily_returns.mean() * TRADING_DAYS
    ann_volatility = daily_returns.std()  * np.sqrt(TRADING_DAYS)

    summary_df = pd.DataFrame({
        'Sector':     list(MARKET_TICKERS.keys()),
        'Ticker':     tickers,
        'Ann_Return': ann_return.reindex(tickers).values,
        'Volatility': ann_volatility.reindex(tickers).values,
    })
    summary_df['Sharpe_Ratio'] = (
        (summary_df['Ann_Return'] - RISK_FREE_RATE) / summary_df['Volatility']
    )

    update_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    return daily_returns, summary_df, update_time


# ---------------------------------------------------------------------------
# News sentiment
# ---------------------------------------------------------------------------

@st.cache_resource
def load_nlp_model() -> SentimentIntensityAnalyzer:
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()


@st.cache_data(ttl=1800)
def fetch_ticker_headlines(ticker: str, api_key: str) -> list[str]:
    """Return up to 5 recent headlines for *ticker* via Finnhub."""
    end   = datetime.now().date()
    start = end - timedelta(days=NEWS_LOOKBACK_DAYS)
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": ticker,
            "from":   start.isoformat(),
            "to":     end.isoformat(),
            "token":  api_key,
        },
        timeout=10,
    )
    resp.raise_for_status()
    articles = resp.json()
    return [a["headline"] for a in articles[:5] if a.get("headline")]


def compute_sentiment_alpha(
    tickers: list[str],
    sia: SentimentIntensityAnalyzer,
    api_key: str,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Fetch headlines for each ticker and return
    (sentiments_by_ticker, headlines_by_ticker).
    """
    sentiments: dict[str, float]       = {}
    headlines:  dict[str, list[str]]   = {}

    for ticker in tickers:
        raw_headlines: list[str] = []
        if api_key:
            try:
                raw_headlines = fetch_ticker_headlines(ticker, api_key)
            except Exception:
                pass
        headlines[ticker] = raw_headlines
        scores = [sia.polarity_scores(h)['compound'] for h in raw_headlines]
        sentiments[ticker] = float(np.mean(scores)) if scores else 0.0

    return sentiments, headlines
