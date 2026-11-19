# Monte Carlo Stock Analysis

[Slovenská verzia](README.md)

An interactive Streamlit web application for scenario analysis of stock prices and portfolios. It downloads adjusted daily prices from Yahoo Finance, converts foreign listings historically into the account currency, and compares five return models. Outputs include interactive charts, risk metrics, portfolio strategies, stress tests, a walk-forward backtest, and CSV/PDF exports.

**Live demo:** [Open the application](https://omgcotopeterspravil.streamlit.app/)

<!-- Screenshot placeholders: add the exported images at these paths. -->

![Fan chart of simulated price scenarios](assets/fan_chart.png)

![Portfolio outcome comparison](assets/portfolio_comparison.png)

## Overview

- **Technology:** Python, Streamlit, NumPy, pandas, yfinance, Plotly, and ReportLab.
- **Simulation:** five return models with asset correlations, historical data, and currency conversion.
- **Risk and portfolios:** VaR/CVaR, Sharpe Ratio, drawdown, stress scenarios, and four weighting strategies.
- **Validation and exports:** walk-forward model comparison, CSV outputs, and a summary PDF report.

## Run on Windows PowerShell

From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run on Linux or macOS (bash/zsh)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In either case, open `http://localhost:8501`. Stop the server with `Ctrl+C` in the terminal; on later runs, only the final command is needed. If the virtual environment and dependencies are already set up, skip their installation. Internet access is required to download prices and FX rates. The first run may take longer because the app also evaluates a backtest for all five models. Prices and FX rates are cached by Streamlit for 30 minutes.

## Using the application

- The watchlist is empty when the app opens. Search for and select stocks from the catalog or add a custom ticker, then run the simulation from the sidebar.
- The catalog contains **2,340** entries. Search by company name or ticker and filter by exchange/region; custom symbols can also be entered. `Visa (V)`, `Visa`, and `V` are recognized. A simulation supports up to 10 tickers.
- The catalog is a local snapshot dated November 18, 2026, based on [FinanceDatabase](https://github.com/JerBouma/FinanceDatabase), with Anthropic (`ANT`) added after its Nasdaq IPO on November 19, 2026. The base snapshot contains active primary listings classified as Mega Cap or Large Cap. The source categories are not an exact numerical market-cap ranking, so the catalog does not claim to be the exact 1,000 largest stocks by current market capitalization. See [`data/README.md`](data/README.md) for details. The catalog does not update itself.
- The app uses Yahoo Finance symbols and downloads prices when a simulation runs. Historical coverage varies by listing. ANT is a new listing with a short history, so long horizons may be unavailable or especially uncertain.
- Select a benchmark (including the S&P 500, Nasdaq-100, NYSE Composite `^NYA`, and other indices/ETFs), horizon, account currency, and initial investment. The minimum investment is €10 or $10; custom benchmarks cannot be entered manually.
- Choose 1,000, 5,000, or 10,000 simulations and one of the five models.
- Advanced settings include the risk-free rate, historical drift shrinkage, drift-estimate uncertainty, risk limit, weight constraints, and costs. Costs can include commission, a user estimate of spread/slippage, an ongoing fee, and annual turnover as a rebalancing cost proxy. The optional simplified tax is deducted only from a positive terminal gain.
- Click **Run simulation**. After changing settings, run the simulation again to recalculate the results.
- Tabs show stock paths, four portfolio strategies, hypothetical and historical stress scenarios, model backtest comparisons, and exports.

## Models and estimates

The models simulate joint asset returns and use a regularized correlation matrix:

1. **GBM (Geometric Brownian Motion)** — constant volatility and correlated normal shocks.
2. **Block bootstrap** — randomly samples joint 20-day blocks of historical log returns, preserving their short-term time dependence.
3. **Student-t GBM** — a GBM model with 8 degrees of freedom for heavier tails.
4. **GARCH(1,1) with Student-t shocks** — conditional volatility responds to recent shocks; parameters are selected from a small constrained grid.
5. **Regime switching** — calm and stress regimes are estimated from 20-day volatility and a historical transition matrix.

Daily log returns are winsorized at the 1st and 99th percentiles when estimating the mean (drift), so individual extremes do not dominate the expected-return estimate. The drift estimate is shrunk toward the risk-free rate. Volatility and correlation use the original, untrimmed returns to preserve historical tails; the Student-t and GARCH-t models also use untrimmed returns. The uncertainty setting adds a path-specific random drift shift based on the estimated standard error. This is an approximate sensitivity method, not a full Bayesian model. Foreign prices are converted using daily historical Yahoo FX rates; future FX movements remain uncertain.

The fan chart displays up to 1,000 gray sample paths, the 10th, 50th, and 90th percentiles, and the initial price. Percentiles are calculated from all simulations. The band shows scenarios, not guaranteed price boundaries.

## Metrics, portfolio, and backtest

- Mean and median returns after configured transaction and ongoing costs.
- **VaR 95%:** loss at the 5th percentile of terminal returns. **CVaR 95%:** average loss in the worst 5% of scenarios.
- Probability of profit and paired probability of outperforming the benchmark in joint simulations.
- Sharpe Ratio and the probability that the maximum drawdown over the full simulated path exceeds the user-defined limit.
- Four portfolio strategies: equal weights, approximate maximum historical Sharpe, minimum historical volatility, and minimum historical CVaR. Weights are selected from 50,000 randomly generated feasible long-only candidates; this is not a guaranteed global optimum.
- Walk-forward comparison for each model uses three years of training data (756 trading days), a 63-trading-day forecast, and a 21-day step. For every estimate, the training window ends before the test period. The backtest uses 2,000 simulations per estimate and reports 10th–90th percentile coverage, directional accuracy, median error, and band width. Windows overlap, and the comparison also includes a zero-return forecast.
- Sensitivity of terminal returns to drift, volatility, and additional costs, as well as portfolio volatility under different correlation assumptions. These tables are approximations, not additional full Monte Carlo runs.
- A hypothetical one-time decline and the worst historical joint 63-day block. After the historical block, the model simulates the remainder of the selected horizon.
- Comparison of an equally weighted portfolio before and after adding a selected stock, using the same simulated paths.
- Data source, freshness, and coverage summary for each ticker; stale data or low coverage triggers a warning.
- CSV exports for assets, weights, portfolios, and backtests; a summary PDF report; and a JSON settings file with the seed and SHA-256 data fingerprint.

## Costs and methodological limitations

The cost model includes a one-time commission on purchase and sale, a user estimate of spread and slippage per trade, an annual ongoing fee (TER), a cost proxy for the configured annual rebalancing turnover, and simplified tax on a positive terminal capital gain. Annual turnover is a proxy: the app does not replay actual rebalancing trades on each path. Tax is a flat user-defined approximation, not an individual tax-liability calculation. Spread and slippage are user estimates, and liquidity is not checked against live trading volumes.

Yahoo Finance is the app's data source, not a guaranteed production feed; data availability and quality can change. Five years of data may not capture every market regime. At a 63-day historical stress-test horizon, the observed stress block covers the entire horizon. Walk-forward test windows overlap and are therefore not independent. A catalog of currently active companies does not remove survivorship bias. The heuristic verdict and simulations are not investment advice or guarantees of future returns.

## Tests

Run the tests on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run the tests on Linux or macOS after activating the virtual environment:

```bash
python -m unittest discover -s tests -v
```

Helper-function tests use synthetic data or the local catalog and do not download Yahoo Finance data. They include Visa lookup by name/ticker and manual entry as `Visa (V)`.

## Codex agents

`.codex/agents/` contains project-specific read-only profiles for quantitative analysis, data science, UX review, code review, and concise context handoff. They are derived from [VoltAgent/awesome-codex-subagents](https://github.com/VoltAgent/awesome-codex-subagents) and can be used for explicit delegation in Codex. The profiles do not launch other agents automatically. Their MIT license and provenance are recorded in [`.codex/agents/README.md`](.codex/agents/README.md).

## Original batch script

The original batch tool is retained:

```powershell
.\.venv\Scripts\python.exe monte_carlo_akcie.py
```

On Linux/macOS, after activating the virtual environment:

```bash
python monte_carlo_akcie.py
```

The script writes its outputs to `mc_vysledky/`.
