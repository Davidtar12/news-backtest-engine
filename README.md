# news-backtest-engine

News-driven stock backtesting engine: Finnhub news API, IBKR & MT5 integration, HMM regime detection, Monte Carlo validation, pairs trading.

## Scripts

| Script | Description |
|--------|-------------|
| `news_trade_orchestrator.py` | Main orchestrator: pulls news events, scores sentiment, triggers trades via IBKR/MT5 |
| `backtest_new_clusters.py` | Clusters news events by topic/sentiment and backtests each cluster's market impact |
| `ibkr_news_backtest_adaptive.py` | IBKR-connected adaptive backtest: adjusts position sizing based on news signal strength |
| `hmm_regime_ml_strategy.py` | Hidden Markov Model regime detector — classifies market as bull/bear/neutral and filters trades |
| `keyword-algo.py` | Keyword-based trading algorithm: triggers positions when specific news keywords appear |
| `sma_enhanced_strategy.py` | SMA crossover strategy enhanced with news sentiment as a secondary filter |

## Prerequisites

- Python 3.9+
- **Interactive Brokers TWS or Gateway** running locally (port 7497 paper / 7496 live)
- **MetaTrader 5** terminal installed (for MT5 integration)
- Finnhub API key
- StockNews API key

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in FINNHUB_API_KEY, STOCKNEWS_API_KEY, IBKR_HOST, IBKR_PORT
```

## Usage

```bash
# Run full news-driven orchestrator (paper trading)
python news_trade_orchestrator.py --mode paper

# Backtest news clusters for a ticker
python backtest_new_clusters.py --ticker AAPL --start 2022-01-01 --end 2024-01-01

# HMM regime detection
python hmm_regime_ml_strategy.py --ticker SPY

# Keyword-triggered algorithm
python keyword-algo.py --keywords "earnings,beat,guidance"
```

## Notes

- All live trading scripts default to **paper trading mode**. Change `--mode live` only after thorough testing.
- IBKR scripts require `ib_insync` and an active TWS session.
- Monte Carlo validation is built into the backtest scripts — runs 1,000 simulations by default.

## Built with

Python · ib_insync · MetaTrader5 · Finnhub API · scikit-learn (HMM) · pandas  
AI-assisted development (Claude, GitHub Copilot) — architecture, requirements, QA validation and debugging by me.
