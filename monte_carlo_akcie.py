"""Korelovaná Monte Carlo simulácia akcií pomocou geometrického Brownovho pohybu.

Inštalácia:
    python -m pip install -r requirements.txt

Spustenie:
    python monte_carlo_akcie.py

Upravte sekciu NASTAVENIA pred spustením. Historické ceny s dividendami a
splitmi upravené cez yfinance sú iba vstupom do scenárov, nie predpoveďou.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# NASTAVENIA – upravte podľa svojho zadania
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
BENCHMARK = "SPY"
HISTORY_PERIOD = "5y"
HORIZON_DAYS = 252
N_SIMULATIONS = 10_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # Ročný bezrizikový výnos v desatinnom tvare; nastavte podľa meny.
MAX_ACCEPTABLE_DRAWDOWN = 0.20
MAX_DRAWDOWN_EXCEEDANCE_PROB = 0.10  # Tolerovaná pravdepodobnosť prekročenia limitu.
MIN_PROBABILITY_OF_PROFIT = 0.60
MIN_ANNUALIZED_SHARPE = 0.50
MIN_PROBABILITY_BEAT_BENCHMARK = 0.55
CORRELATION_SHRINKAGE = 0.05  # Mierne zmenšenie korelácií kvôli numerickej stabilite.
RANDOM_SEED = 42
MAX_PORTFOLIO_WEIGHT = 0.40  # Long-only portfólio; žiadna váha nad 40 %.
N_PORTFOLIO_CANDIDATES = 30_000
OUTPUT_DIR = Path("mc_vysledky")


def download_adjusted_prices(tickers: list[str], period: str) -> pd.DataFrame:
    """Stiahne upravené denné ceny a vráti spoločný, úplný dátový panel."""
    downloaded = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    if downloaded is None or downloaded.empty:
        raise RuntimeError("Yahoo Finance nevrátilo žiadne dáta. Skontrolujte pripojenie a tickery.")

    # yfinance vracia pri jednom aj viacerých tickeroch rôznu podobu stĺpcov.
    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)
        price_field = "Close" if "Close" in level0 else ("Close" if "Close" in level1 else None)
        if price_field is None:
            raise RuntimeError("V stiahnutých dátach sa nenašiel stĺpec Close.")
        if price_field in level0:
            prices = downloaded[price_field].copy()
        else:
            prices = downloaded.xs(price_field, axis=1, level=1).copy()
    else:
        if "Close" not in downloaded.columns:
            raise RuntimeError("V stiahnutých dátach sa nenašiel stĺpec Close.")
        prices = downloaded[["Close"]].copy()
        prices.columns = [tickers[0]]

    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise RuntimeError(f"Pre tieto tickery chýbajú ceny: {', '.join(missing)}")
    prices = prices[tickers].apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index().dropna(how="any")
    prices = prices.loc[(prices > 0).all(axis=1)]
    if len(prices) < 60:
        raise RuntimeError(f"Po zosúladení zostalo len {len(prices)} spoločných dní; potrebných je aspoň 60.")
    return prices


def regularized_correlation(returns: pd.DataFrame, shrinkage: float) -> np.ndarray:
    """Odhadne koreláciu a opraví ju na pozitívne definitnú maticu."""
    corr = returns.corr().to_numpy(dtype=float)
    corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0])
    corr = (corr + corr.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    corr = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return corr


def simulate_paths(
    start_prices: np.ndarray,
    annual_drift: np.ndarray,
    annual_volatility: np.ndarray,
    correlation: np.ndarray,
    horizon_days: int,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Vráti denné cenové dráhy a najväčší peak-to-trough pokles každej dráhy."""
    n_assets = len(start_prices)
    if n_assets == 0 or horizon_days <= 0 or n_simulations <= 0:
        raise ValueError("Počet aktív, horizont a počet simulácií musia byť kladné.")
    if annual_drift.shape != (n_assets,) or annual_volatility.shape != (n_assets,):
        raise ValueError("Vektory driftu a volatility musia mať jednu hodnotu na každé aktívum.")
    if correlation.shape != (n_assets, n_assets):
        raise ValueError("Rozmer korelačnej matice nezodpovedá počtu aktív.")
    if (
        not np.all(np.isfinite(start_prices))
        or not np.all(np.isfinite(annual_drift))
        or not np.all(np.isfinite(annual_volatility))
        or not np.all(np.isfinite(correlation))
        or np.any(start_prices <= 0)
        or np.any(annual_volatility < 0)
    ):
        raise ValueError("Vstupy simulácie musia byť konečné; ceny kladné a volatilita nezáporná.")
    dt = 1.0 / TRADING_DAYS_PER_YEAR
    chol = np.linalg.cholesky(correlation)
    daily_vol = annual_volatility * np.sqrt(dt)
    daily_log_drift = (annual_drift - 0.5 * annual_volatility**2) * dt

    # float32 obmedzí pamäť potrebnú pre fan chart; výpočty výnosov ostávajú float64.
    paths = np.empty((n_simulations, horizon_days + 1, n_assets), dtype=np.float32)
    paths[:, 0, :] = start_prices
    max_drawdowns = np.empty((n_simulations, n_assets), dtype=np.float32)
    batch_size = max(1, min(1_000, n_simulations))

    for first in range(0, n_simulations, batch_size):
        last = min(first + batch_size, n_simulations)
        n_batch = last - first
        independent = rng.standard_normal((n_batch, horizon_days, n_assets))
        correlated = independent @ chol.T
        log_steps = daily_log_drift + correlated * daily_vol
        cumulative_log_returns = np.cumsum(log_steps, axis=1)
        batch_prices = start_prices[None, None, :] * np.exp(cumulative_log_returns)
        paths[first:last, 1:, :] = batch_prices.astype(np.float32)

        batch_with_initial = np.concatenate(
            [np.broadcast_to(start_prices, (n_batch, 1, n_assets)), batch_prices], axis=1
        )
        peaks = np.maximum.accumulate(batch_with_initial, axis=1)
        drawdowns = batch_with_initial / peaks - 1.0
        max_drawdowns[first:last, :] = np.min(drawdowns, axis=1).astype(np.float32)

    return paths, max_drawdowns


def tail_metrics(returns: np.ndarray) -> tuple[float, float]:
    """Vráti 95 % VaR a CVaR v znamienkovom vyjadrení straty."""
    cutoff = np.quantile(returns, 0.05)
    var95 = -float(cutoff)
    cvar95 = -float(np.mean(returns[returns <= cutoff]))
    return var95, cvar95


def make_metrics(
    name: str,
    start_value: float,
    terminal_values: np.ndarray,
    max_drawdowns: np.ndarray,
    annual_drift: float,
    annual_volatility: float,
    benchmark_returns: np.ndarray | None,
    horizon_years: float,
) -> dict[str, float | str]:
    terminal_returns = terminal_values / start_value - 1.0
    var95, cvar95 = tail_metrics(terminal_returns)
    # Sharpe je ročný, modelom implikovaný pomer driftu a volatility.
    sharpe = (annual_drift - RISK_FREE_RATE) / annual_volatility if annual_volatility > 0 else np.nan
    prob_exceed_dd = float(np.mean(max_drawdowns < -MAX_ACCEPTABLE_DRAWDOWN))
    prob_beat = (
        float(np.mean(terminal_returns > benchmark_returns))
        if benchmark_returns is not None
        else np.nan
    )
    beat_benchmark_ok = (
        benchmark_returns is None
        or float(np.mean(terminal_returns > benchmark_returns)) >= MIN_PROBABILITY_BEAT_BENCHMARK
    )
    suitable = (
        np.median(terminal_returns) > 0
        and float(np.mean(terminal_returns > 0)) >= MIN_PROBABILITY_OF_PROFIT
        and sharpe >= MIN_ANNUALIZED_SHARPE
        and prob_exceed_dd <= MAX_DRAWDOWN_EXCEEDANCE_PROB
        and beat_benchmark_ok
    )
    return {
        "ticker": name,
        "horizon_years": horizon_years,
        "start_value": start_value,
        "mean_terminal_value": float(np.mean(terminal_values)),
        "median_terminal_value": float(np.median(terminal_values)),
        "mean_return_pct": float(np.mean(terminal_returns) * 100),
        "median_return_pct": float(np.median(terminal_returns) * 100),
        "VaR_95_loss_pct": var95 * 100,
        "CVaR_95_loss_pct": cvar95 * 100,
        "prob_profit_pct": float(np.mean(terminal_returns > 0) * 100),
        "prob_beat_benchmark_pct": prob_beat * 100 if np.isfinite(prob_beat) else np.nan,
        "annual_drift_pct": annual_drift * 100,
        "annual_volatility_pct": annual_volatility * 100,
        "annualized_sharpe": float(sharpe),
        "median_max_drawdown_pct": float(np.median(max_drawdowns) * 100),
        "prob_max_drawdown_over_limit_pct": prob_exceed_dd * 100,
        "screen_pass": "ÁNO" if suitable else "NIE",
    }


def optimize_historical_sharpe(
    annual_returns: np.ndarray,
    annual_covariance: np.ndarray,
    max_weight: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vyberie najvyšší historický Sharpe z náhodne vzorkovaných prípustných váh."""
    n_assets = len(annual_returns)
    if n_assets == 0 or n_candidates <= 0:
        raise ValueError("Počet aktív a kandidátnych portfólií musia byť kladné.")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight musí byť v intervale (0, 1].")
    if max_weight < 1.0 / n_assets - 1e-12:
        raise ValueError(
            f"MAX_PORTFOLIO_WEIGHT={max_weight:.2f} je pre {n_assets} aktív príliš nízky; "
            f"minimum pre plne investované portfólio je {1.0 / n_assets:.2f}."
        )

    equal_weight = np.full(n_assets, 1.0 / n_assets)
    draws = rng.dirichlet(np.ones(n_assets), size=n_candidates)
    excess = np.max(draws, axis=1) - 1.0 / n_assets
    scale = np.ones(n_candidates)
    above_equal = excess > 0
    scale[above_equal] = np.minimum(
        1.0,
        (max_weight - 1.0 / n_assets) / excess[above_equal],
    )
    feasible_draws = equal_weight + (draws - equal_weight) * scale[:, None]
    weights = np.vstack([equal_weight, feasible_draws])

    expected_returns = weights @ annual_returns
    variances = np.einsum("ij,jk,ik->i", weights, annual_covariance, weights)
    volatilities = np.sqrt(np.maximum(variances, 0.0))
    sharpes = np.divide(
        expected_returns - RISK_FREE_RATE,
        volatilities,
        out=np.full_like(expected_returns, -np.inf),
        where=volatilities > 0,
    )
    return weights[int(np.argmax(sharpes))]


def plot_fan_chart(paths: np.ndarray, tickers: list[str], output_path: Path) -> None:
    quantiles = np.quantile(paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    n_assets = len(tickers)
    ncols = min(3, n_assets)
    nrows = int(np.ceil(n_assets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    days = np.arange(paths.shape[1])
    for idx, ticker in enumerate(tickers):
        ax = axes.flat[idx]
        ax.fill_between(days, quantiles[0, :, idx], quantiles[4, :, idx], color="#8bbbe8", alpha=0.25, label="5.–95. percentil")
        ax.fill_between(days, quantiles[1, :, idx], quantiles[3, :, idx], color="#4d90c8", alpha=0.35, label="25.–75. percentil")
        ax.plot(days, quantiles[2, :, idx], color="#123f68", linewidth=1.8, label="Medián")
        ax.set(title=ticker, xlabel="Obchodné dni", ylabel="Upravená cena")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[n_assets:]:
        ax.set_visible(False)
    fig.suptitle("Monte Carlo fan chart – korelovaný GBM", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_distributions(paths: np.ndarray, tickers: list[str], output_path: Path) -> None:
    terminals = paths[:, -1, :]
    n_assets = len(tickers)
    ncols = min(3, n_assets)
    nrows = int(np.ceil(n_assets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for idx, ticker in enumerate(tickers):
        ax = axes.flat[idx]
        values = terminals[:, idx]
        ax.hist(values, bins=55, color="#4d90c8", alpha=0.8, edgecolor="white")
        ax.axvline(np.median(values), color="#123f68", linewidth=2, label=f"Medián {np.median(values):.2f}")
        ax.axvline(paths[0, 0, idx], color="#c4513c", linestyle="--", linewidth=1.5, label="Počiatočná cena")
        ax.set(title=ticker, xlabel="Simulovaná cena na konci horizontu", ylabel="Počet scenárov")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes.flat[n_assets:]:
        ax.set_visible(False)
    fig.suptitle("Rozdelenie simulovaných konečných cien", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tickers = list(dict.fromkeys(t.upper().strip() for t in TICKERS if t.strip()))
    benchmark = BENCHMARK.upper().strip()
    if not tickers:
        raise ValueError("Zadajte aspoň jeden ticker do TICKERS.")
    if not benchmark:
        raise ValueError("Zadajte ticker benchmarku do BENCHMARK.")
    if HORIZON_DAYS <= 0 or N_SIMULATIONS <= 0:
        raise ValueError("HORIZON_DAYS a N_SIMULATIONS musia byť kladné.")
    if not 0 <= CORRELATION_SHRINKAGE < 1:
        raise ValueError("CORRELATION_SHRINKAGE musí byť v intervale [0, 1).")
    if MAX_PORTFOLIO_WEIGHT <= 0 or MAX_PORTFOLIO_WEIGHT > 1:
        raise ValueError("MAX_PORTFOLIO_WEIGHT musí byť v intervale (0, 1].")
    if N_PORTFOLIO_CANDIDATES <= 0:
        raise ValueError("N_PORTFOLIO_CANDIDATES musí byť kladné.")
    if not np.isfinite(RISK_FREE_RATE):
        raise ValueError("RISK_FREE_RATE musí byť konečné číslo.")
    if not 0 < MAX_ACCEPTABLE_DRAWDOWN < 1:
        raise ValueError("MAX_ACCEPTABLE_DRAWDOWN musí byť v intervale (0, 1).")
    for label, value in (
        ("MAX_DRAWDOWN_EXCEEDANCE_PROB", MAX_DRAWDOWN_EXCEEDANCE_PROB),
        ("MIN_PROBABILITY_OF_PROFIT", MIN_PROBABILITY_OF_PROFIT),
        ("MIN_PROBABILITY_BEAT_BENCHMARK", MIN_PROBABILITY_BEAT_BENCHMARK),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{label} musí byť v intervale [0, 1].")
    if not np.isfinite(MIN_ANNUALIZED_SHARPE):
        raise ValueError("MIN_ANNUALIZED_SHARPE musí byť konečné číslo.")

    all_tickers = list(dict.fromkeys(tickers + [benchmark]))
    print(f"Sťahujem {HISTORY_PERIOD} denných dát pre: {', '.join(all_tickers)}")
    prices = download_adjusted_prices(all_tickers, HISTORY_PERIOD)
    daily_returns = prices.pct_change().dropna(how="any")
    if len(daily_returns) < 59:
        raise RuntimeError("Po výpočte výnosov zostalo príliš málo dát na odhad parametrov.")

    annual_drift = daily_returns.mean().to_numpy() * TRADING_DAYS_PER_YEAR
    annual_volatility = daily_returns.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if np.any(annual_volatility <= 0):
        raise RuntimeError("Aspoň jeden ticker má nulovú odhadnutú volatilitu.")
    correlation = regularized_correlation(daily_returns, CORRELATION_SHRINKAGE)
    start_prices = prices.iloc[-1].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    paths, max_drawdowns = simulate_paths(
        start_prices, annual_drift, annual_volatility, correlation,
        HORIZON_DAYS, N_SIMULATIONS, rng,
    )

    ticker_index = {ticker: i for i, ticker in enumerate(all_tickers)}
    benchmark_idx = ticker_index[benchmark]
    benchmark_returns = paths[:, -1, benchmark_idx] / start_prices[benchmark_idx] - 1.0
    horizon_years = HORIZON_DAYS / TRADING_DAYS_PER_YEAR
    rows = []
    for ticker in tickers:
        idx = ticker_index[ticker]
        comparison = None if ticker == benchmark else benchmark_returns
        row = make_metrics(
            ticker,
            start_prices[idx],
            paths[:, -1, idx],
            max_drawdowns[:, idx],
            annual_drift[idx],
            annual_volatility[idx],
            comparison,
            horizon_years,
        )
        if ticker == benchmark:
            row["screen_pass"] = "BENCHMARK"
        rows.append(row)
    metrics = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "metriky_akcii.csv", index=False, encoding="utf-8-sig")
    plot_fan_chart(paths, all_tickers, OUTPUT_DIR / "fan_chart.png")
    plot_terminal_distributions(paths, all_tickers, OUTPUT_DIR / "rozdelenie_konecnych_cien.png")

    display_cols = [
        "ticker", "mean_return_pct", "median_return_pct", "VaR_95_loss_pct",
        "CVaR_95_loss_pct", "prob_profit_pct", "prob_beat_benchmark_pct",
        "annualized_sharpe", "prob_max_drawdown_over_limit_pct", "screen_pass",
    ]
    print("\nVýsledky akcií (percentá sú v %; záporný VaR/CVaR znamená zisk aj v dolnom chvoste):")
    print(metrics[display_cols].round(2).to_string(index=False))

    if MAX_PORTFOLIO_WEIGHT >= 1.0 / len(tickers) - 1e-12:
        # Portfólio s najvyšším historickým Sharpeho pomerom z kandidátnych tickerov.
        candidate_indices = np.array([ticker_index[t] for t in tickers])
        candidate_returns = daily_returns[tickers]
        candidate_mu = annual_drift[candidate_indices]
        candidate_cov = candidate_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
        weights = optimize_historical_sharpe(
            candidate_mu, candidate_cov, MAX_PORTFOLIO_WEIGHT,
            N_PORTFOLIO_CANDIDATES, rng,
        )
        portfolio_path = np.sum(
            paths[:, :, candidate_indices] / start_prices[candidate_indices][None, None, :]
            * weights[None, None, :],
            axis=2,
        )
        portfolio_max_drawdown = np.min(
            portfolio_path / np.maximum.accumulate(portfolio_path, axis=1) - 1.0,
            axis=1,
        )
        portfolio_volatility = float(np.sqrt(weights @ candidate_cov @ weights))
        portfolio_drift = float(weights @ candidate_mu)
        portfolio_row = make_metrics(
            "PORTFÓLIO_MAX_SHARPE", 1.0, portfolio_path[:, -1],
            portfolio_max_drawdown, portfolio_drift, portfolio_volatility,
            benchmark_returns, horizon_years,
        )
        portfolio_row["weights"] = "; ".join(
            f"{ticker}={weight:.1%}" for ticker, weight in zip(tickers, weights)
        )
        portfolio_metrics = pd.DataFrame([portfolio_row])
        portfolio_metrics.to_csv(OUTPUT_DIR / "portfolio.csv", index=False, encoding="utf-8-sig")
        print("\nPortfólio s najvyšším historickým Sharpeho pomerom (orientačný výber):")
        print(portfolio_metrics[["weights", "mean_return_pct", "median_return_pct", "VaR_95_loss_pct", "CVaR_95_loss_pct", "annualized_sharpe", "prob_max_drawdown_over_limit_pct", "screen_pass"]].round(2).to_string(index=False))
    else:
        print(
            "\nPortfólio sa neoptimalizovalo: MAX_PORTFOLIO_WEIGHT je nižší než "
            f"minimálna prípustná váha 1/{len(tickers)} = {1.0 / len(tickers):.1%}. "
            "Upravte limit alebo pridajte ďalšie tickery."
        )
    print(f"\nSúbory s výsledkami: {OUTPUT_DIR.resolve()}")
    print("Poznámka: benchmark neprekonáva sám seba; jeho pravdepodobnosť prekonania je prázdna.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # Jasná chyba pre používateľa aj pri spustení dvojklikom.
        print(f"Chyba: {exc}", file=sys.stderr)
        raise
