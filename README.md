# Macro Rotations Dashboard

A quantitative macro analytics platform that combines macroeconomic regime detection with sector rotation analysis, options pricing, and portfolio optimization. Generates multi-page PDF reports with visual dashboards, strategy backtests, and actionable investment playbooks.

## Features

- **Macro Regime Detection** -- GMM clustering, yield curve analysis, recession probability models
- **Sector Rotation** -- Relative Rotation Graphs (RRG), transition matrices, seasonality analysis
- **Capital Flows Plumbing** -- Net liquidity (Fed - TGA - RRP), SOFR/overnight rates, credit spreads
- **Quantitative Lab** -- GARCH volatility, Monte Carlo (GBM), Black-Scholes Greeks, Heston stochastic vol
- **Portfolio Optimization** -- Mean-variance efficient frontier with macro-adjusted forward returns
- **Risk Analytics** -- VaR/CVaR, PCA absorption ratio, correlation surprise, anti-fragility metrics
- **Advanced Models** -- Markov regime switching, Ornstein-Uhlenbeck mean reversion, Merton jump diffusion, Hawkes processes
- **Scenario Analysis** -- Monte Carlo probability tables, contingency playbooks

## Project Structure

```
Macro-Rotations-New/
├── main.py                    # Entry point: orchestrates the full pipeline
├── core/
│   ├── config.py              # Tickers, FRED codes, and settings
│   └── logger.py              # Centralized logging
├── data/
│   └── loader.py              # Yahoo Finance + FRED data fetching with caching
├── analytics/
│   ├── macro_models.py        # Regime detection, capital flows, macro scoring
│   ├── rotations.py           # Sector rotation, RRG, seasonality
│   ├── alpha_models.py        # Net liquidity, VIX term structure, tail risk
│   ├── quant.py               # Options, GARCH, GBM, backtesting, optimization
│   ├── stochastic.py          # Heston, Merton jump diffusion, Hawkes
│   ├── econometrics.py        # Markov regime switching, OU processes
│   └── scenarios.py           # Monte Carlo probabilities, contingency planning
├── plotting/
│   ├── charts.py              # All visualization functions (matplotlib)
│   ├── charts_pkg/            # Modular chart subpackages
│   │   ├── macro.py           # Macro regime & risk charts
│   │   ├── quant.py           # Quant lab & stochastic model charts
│   │   ├── portfolio.py       # Backtest, alpha, & optimization charts
│   │   └── global_macro.py    # FX, valuation, & inflation charts
│   └── report.py              # PDF report generation
├── scripts/
│   └── app.py                 # Streamlit interactive dashboard
├── requirements.txt           # Pinned dependencies
└── .github/workflows/ci.yml   # CI pipeline
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Andrespm12/Macro-Rotations-New.git
cd Macro-Rotations-New

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Generate the full PDF report
python main.py
```

The pipeline will:
1. Fetch price data (Yahoo Finance) and macro data (FRED) with daily caching
2. Run the full analytics pipeline (regime detection, rotations, alpha factors)
3. Generate 18+ visualization pages
4. Output a multi-page PDF report

## Configuration

Edit `core/config.py` to customize:
- **Tickers** -- Add/remove assets from the tracking universe
- **FRED codes** -- Add macro indicators
- **Simulation parameters** -- Monte Carlo sims, default ticker, risk-free rate
- **Moving averages** -- Short/long MA periods and regime thresholds

## Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| Yahoo Finance | Asset prices, options chains, fundamentals | Daily |
| FRED (Federal Reserve) | Yields, spreads, CPI, M2, Fed balance sheet | Daily/Weekly/Monthly |

Data is cached locally in `data/cache/` (date-versioned) to minimize API calls.

## Key Analytics

### Macro Score
A composite score (-1 to +1) derived from 14 rotation ratios (value/growth, cyclical/defensive, small/large, etc.) relative to their 200-day moving averages.

### Net Liquidity Model
Tracks the "Howell/Dale" equation: `Fed Assets - TGA - RRP` to measure actual liquidity available in markets.

### Recession Probability
Uses the Estrella/Mishkin probit model on the 10Y-3M yield spread to estimate 12-month forward recession probability.

## License

This project is for educational and research purposes.
