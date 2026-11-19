"""Streamlit dashboard for correlated Monte Carlo stock and portfolio scenarios."""

from __future__ import annotations

import math
import re
import io
import html
import json
import hashlib
from datetime import datetime
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from stock_catalog import add_ticker_symbols, load_stock_catalog, search_stock_catalog


TRADING_DAYS_PER_YEAR = 252
HISTORY_PERIOD = "5y"
CORRELATION_SHRINKAGE = 0.05
MAX_TICKERS = 10
MAX_PORTFOLIO_CANDIDATES = 50_000
BOOTSTRAP_BLOCK_DAYS = 20
BACKTEST_HORIZON_DAYS = 63
BACKTEST_TRAIN_DAYS = 756
BACKTEST_STEP_DAYS = 21
BACKTEST_SIMULATIONS = 2_000
FAN_CHART_PATH_LIMIT = 1_000
ANALYSIS_SCHEMA_VERSION = 3
HORIZONS = {
    "3 mesiace": 63,
    "6 mesiacov": 126,
    "1 rok": 252,
    "3 roky": 756,
    "5 rokov": 1_260,
}
SIMULATION_COUNTS = [1_000, 5_000, 10_000]
MODELS = {
    "GBM (konštantná volatilita)": "gbm",
    "Blokový bootstrap (historické výnosy)": "bootstrap",
    "GBM s Student-t šokmi": "student_t",
    "GARCH(1,1) s Student-t šokmi": "garch_t",
    "Prepínanie režimov (pokoj/stres)": "regime_switch",
}
CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$"}
PORTFOLIO_STRATEGIES = {
    "equal": "Rovnomerné váhy",
    "max_sharpe": "Max. Sharpe",
    "min_volatility": "Min. volatilita",
    "min_cvar": "Min. CVaR (historický)",
}


st.set_page_config(
    page_title="Monte Carlo akcií | Scenárová analýza",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _extract_close(downloaded: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Handle the different yfinance Close layouts for one or many symbols."""
    if downloaded is None or downloaded.empty:
        raise RuntimeError("Yahoo Finance nevrátilo žiadne dáta.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)
        if "Close" in level0:
            prices = downloaded["Close"].copy()
        elif "Close" in level1:
            prices = downloaded.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("V stiahnutých dátach sa nenašli záverečné ceny.")
    else:
        if "Close" not in downloaded.columns:
            raise RuntimeError("V stiahnutých dátach sa nenašli záverečné ceny.")
        prices = downloaded[["Close"]].copy()
        if len(symbols) == 1:
            prices.columns = [symbols[0]]

    prices.columns = [str(column).upper() for column in prices.columns]
    wanted = tuple(symbol.upper() for symbol in symbols)
    missing = [symbol for symbol in wanted if symbol not in prices.columns]
    if missing:
        raise RuntimeError("Chýbajú ceny pre: " + ", ".join(missing))
    prices = prices.loc[:, list(wanted)].apply(pd.to_numeric, errors="coerce")
    source_rows = int(len(prices))
    duplicate_dates = int(prices.index.duplicated(keep="last").sum())
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
    missing_by_ticker = prices.isna().sum().astype(int).to_dict()
    incomplete_dates = int(prices.isna().any(axis=1).sum())
    invalid_by_ticker = (prices.le(0) & prices.notna()).sum().astype(int).to_dict()
    complete_prices = prices.dropna(how="any")
    complete_prices = complete_prices.loc[(complete_prices > 0).all(axis=1)]
    complete_prices.attrs["source_rows"] = source_rows
    complete_prices.attrs["duplicate_dates"] = duplicate_dates
    complete_prices.attrs["incomplete_dates"] = incomplete_dates
    complete_prices.attrs["missing_by_ticker"] = missing_by_ticker
    complete_prices.attrs["invalid_by_ticker"] = invalid_by_ticker
    return complete_prices


@st.cache_data(show_spinner=False)
def get_stock_catalog() -> pd.DataFrame:
    """Read the bundled global large-cap catalog once per Streamlit process."""
    return load_stock_catalog()


@st.cache_data(ttl=30 * 60, max_entries=32, show_spinner=False)
def download_yahoo_prices(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance."""
    downloaded = yf.download(
        tickers=list(symbols),
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    return _extract_close(downloaded, symbols)


@st.cache_data(ttl=30 * 60, max_entries=32, show_spinner=False)
def download_quote_currencies(tickers: tuple[str, ...]) -> dict[str, str]:
    """Read quote currencies from Yahoo Finance fast_info."""
    currencies: dict[str, str] = {}
    for ticker in tickers:
        try:
            currency = yf.Ticker(ticker).fast_info.get("currency")
        except Exception as exc:  # Yahoo metadata can fail independently of price download.
            raise RuntimeError(f"Nepodarilo sa zistiť menu tickeru {ticker}: {exc}") from exc
        if not currency:
            raise RuntimeError(
                f"Yahoo Finance neuviedlo menu tickeru {ticker}. "
                "Skontroluj ticker alebo zvoľ iný titul."
            )
        currencies[ticker] = str(currency)
    return currencies


def normalize_currency(currency: str) -> tuple[str, float]:
    """Return ISO-like currency and quote-unit scale for currencies such as GBp."""
    minor_units = {
        "GBp": ("GBP", 0.01),
        "GBX": ("GBP", 0.01),
        "ZAc": ("ZAR", 0.01),
        "ILA": ("ILS", 0.01),
    }
    if currency in minor_units:
        return minor_units[currency]
    return currency.upper(), 1.0


@st.cache_data(ttl=30 * 60, max_entries=32, show_spinner=False)
def download_usd_per_currency(currency: str) -> pd.Series:
    """Return historical USD value of one unit of a currency."""
    currency = currency.upper()
    if currency == "USD":
        return pd.Series(dtype=float, name="USD")

    direct_symbol = f"{currency}USD=X"
    try:
        direct = download_yahoo_prices((direct_symbol,))[direct_symbol].dropna()
        if not direct.empty:
            return direct.rename(currency)
    except Exception:
        pass

    inverse_symbol = f"USD{currency}=X"
    try:
        inverse = download_yahoo_prices((inverse_symbol,))[inverse_symbol].dropna()
        if not inverse.empty:
            return (1.0 / inverse).rename(currency)
    except Exception as exc:
        raise RuntimeError(
            f"Nepodarilo sa stiahnuť historický kurz {currency}/USD z Yahoo Finance."
        ) from exc
    raise RuntimeError(
        f"Yahoo Finance nevrátilo použiteľný kurz {currency}/USD."
    )


@st.cache_data(ttl=30 * 60, max_entries=16, show_spinner=False)
def download_market_prices(
    tickers: tuple[str, ...],
    account_currency: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, float], dict[str, dict[str, float]]]:
    """Download prices and convert them through daily FX rates to account currency.

    The conversion makes both the historical return estimates and simulated prices
    include observed FX changes. Future FX behavior is still modelled from this short
    historical sample, so it remains uncertain.
    """
    account_currency = account_currency.upper()
    native_prices = download_yahoo_prices(tickers)
    currencies = download_quote_currencies(tickers)
    normalized = {ticker: normalize_currency(currencies[ticker]) for ticker in tickers}
    bases = {base for base, _ in normalized.values()} | {account_currency}

    usd_rates: dict[str, pd.Series] = {}
    for base in bases:
        if base != "USD":
            usd_rates[base] = download_usd_per_currency(base)

    aligned: dict[str, pd.Series] = {}
    fx_forward_filled_by_currency: dict[str, int] = {}
    for base in bases:
        if base == "USD":
            aligned[base] = pd.Series(1.0, index=native_prices.index)
        else:
            raw_rate = usd_rates[base].reindex(native_prices.index)
            rate = raw_rate.ffill()
            fx_forward_filled_by_currency[base] = int(
                (raw_rate.isna() & rate.notna()).sum()
            )
            aligned[base] = rate

    account_usd = aligned[account_currency]
    converted = pd.DataFrame(index=native_prices.index)
    spot_rates: dict[str, float] = {}
    fx_stats: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        base, quote_scale = normalized[ticker]
        fx_rate = aligned[base] / account_usd
        conversion = quote_scale * fx_rate
        converted[ticker] = native_prices[ticker] * conversion
        spot_rates[ticker] = float(conversion.dropna().iloc[-1])
        fx_returns = fx_rate.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        fx_stats[ticker] = {
            "annual_return": float(fx_returns.mean() * TRADING_DAYS_PER_YEAR),
            "annual_volatility": float(fx_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if len(fx_returns) > 1
            else 0.0,
        }

    converted = converted.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    converted = converted.loc[(converted > 0).all(axis=1)]
    if len(converted) < 61:
        raise RuntimeError(
            f"Po zosúladení cien a menových kurzov zostalo iba {len(converted)} dní; "
            "potrebných je aspoň 61. Skús odstrániť ticker s krátkou históriou."
        )
    converted.attrs["data_quality"] = {
        "source_rows": int(native_prices.attrs.get("source_rows", len(native_prices))),
        "duplicate_dates": int(native_prices.attrs.get("duplicate_dates", 0)),
        "incomplete_dates": int(native_prices.attrs.get("incomplete_dates", 0)),
        "missing_by_ticker": native_prices.attrs.get("missing_by_ticker", {}),
        "invalid_by_ticker": native_prices.attrs.get("invalid_by_ticker", {}),
        "fx_forward_filled_by_currency": fx_forward_filled_by_currency,
        "usable_rows": int(len(converted)),
    }
    return converted, currencies, spot_rates, fx_stats


def regularized_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Estimate a stable positive-definite correlation matrix."""
    corr = returns.corr().to_numpy(dtype=float)
    corr = (1.0 - CORRELATION_SHRINKAGE) * corr + CORRELATION_SHRINKAGE * np.eye(
        corr.shape[0]
    )
    corr = (corr + corr.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    corr = (eigenvectors * np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return corr


def _project_bounded_simplex(
    values: np.ndarray,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Project rows onto weights summing to one within per-asset bounds."""
    rows, assets = values.shape
    if assets == 1:
        return np.ones((rows, 1), dtype=float)
    lower = np.full(rows, -1.0)
    upper = np.full(rows, 1.0)
    for _ in range(48):
        threshold = (lower + upper) / 2.0
        weights = np.clip(values - threshold[:, None], min_weight, max_weight)
        too_large = weights.sum(axis=1) > 1.0
        lower = np.where(too_large, threshold, lower)
        upper = np.where(too_large, upper, threshold)
    threshold = (lower + upper) / 2.0
    return np.clip(values - threshold[:, None], min_weight, max_weight)


def optimize_portfolio_weights(
    prices: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
    min_weight: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Find four bounded long-only strategies among common random candidates."""
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    asset_count = prices.shape[1]
    equal = np.full(asset_count, 1.0 / asset_count)
    if asset_count == 1:
        return {name: equal.copy() for name in PORTFOLIO_STRATEGIES}, {
            "historical_sharpe": 0.0,
            "minimum_volatility": 0.0,
            "minimum_daily_cvar": 0.0,
        }

    cap = max(float(max_weight), 1.0 / asset_count)
    floor = min(max(0.0, float(min_weight)), 1.0 / asset_count)
    annual_means = returns.mean().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR
    covariance = returns.cov().to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR
    covariance = (1 - CORRELATION_SHRINKAGE) * covariance + CORRELATION_SHRINKAGE * np.diag(
        np.diag(covariance)
    )
    rng = np.random.default_rng(2026)
    raw = rng.dirichlet(np.ones(asset_count), size=MAX_PORTFOLIO_CANDIDATES)
    candidates = _project_bounded_simplex(raw, floor, cap)
    candidates = np.vstack([equal, candidates])
    expected = candidates @ annual_means
    variances = np.einsum("bi,ij,bj->b", candidates, covariance, candidates)
    volatilities = np.sqrt(np.maximum(variances, 1e-16))
    sharpes = (expected - risk_free_rate) / volatilities
    max_sharpe_index = int(np.nanargmax(sharpes))
    min_volatility_index = int(np.nanargmin(volatilities))

    daily_returns = returns.to_numpy(dtype=float)
    candidate_cvars = np.empty(len(candidates), dtype=float)
    batch_size = 500
    for first in range(0, len(candidates), batch_size):
        last = min(first + batch_size, len(candidates))
        portfolio_daily_returns = daily_returns @ candidates[first:last].T
        lower_tail_cutoff = np.quantile(portfolio_daily_returns, 0.05, axis=0)
        tail_losses = np.where(
            portfolio_daily_returns <= lower_tail_cutoff[None, :],
            -portfolio_daily_returns,
            np.nan,
        )
        candidate_cvars[first:last] = np.nanmean(tail_losses, axis=0)
    min_cvar_index = int(np.nanargmin(candidate_cvars))

    strategies = {
        "equal": equal,
        "max_sharpe": candidates[max_sharpe_index],
        "min_volatility": candidates[min_volatility_index],
        "min_cvar": candidates[min_cvar_index],
    }
    scores = {
        "historical_sharpe": float(sharpes[max_sharpe_index]),
        "minimum_volatility": float(volatilities[min_volatility_index]),
        "minimum_daily_cvar": float(candidate_cvars[min_cvar_index]),
    }
    return strategies, scores


def _drawdowns(values: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(values, axis=1)
    return np.min(values / peaks - 1.0, axis=1)


def _winsorize_returns(values: np.ndarray) -> np.ndarray:
    lower = np.quantile(values, 0.01, axis=0)
    upper = np.quantile(values, 0.99, axis=0)
    return np.clip(values, lower, upper)


def _fit_garch_t(values: np.ndarray, df: float = 8.0) -> dict[str, float | np.ndarray]:
    """Fit GARCH(1,1) by a small constrained grid search with Student-t likelihood."""
    center = float(np.mean(values))
    residuals = values - center
    long_run_variance = max(float(np.var(residuals, ddof=1)), 1e-10)
    best_score = float("inf")
    best: tuple[float, float, float, np.ndarray] | None = None
    alpha_grid = (0.03, 0.05, 0.08, 0.12, 0.18, 0.25)
    beta_grid = (0.60, 0.70, 0.78, 0.84, 0.89, 0.93, 0.96)
    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha + beta >= 0.99:
                continue
            omega = (1.0 - alpha - beta) * long_run_variance
            conditional_variance = np.empty(len(values), dtype=float)
            conditional_variance[0] = long_run_variance
            for day in range(1, len(values)):
                conditional_variance[day] = (
                    omega
                    + alpha * residuals[day - 1] ** 2
                    + beta * conditional_variance[day - 1]
                )
            standardized = residuals / np.sqrt(conditional_variance)
            score = float(
                0.5 * np.log(conditional_variance).sum()
                + 0.5 * (df + 1.0) * np.log1p(standardized**2 / (df - 2.0)).sum()
            )
            if score < best_score:
                best_score = score
                best = (alpha, beta, omega, conditional_variance)
    if best is None:
        alpha, beta, omega = 0.08, 0.90, 0.02 * long_run_variance
        conditional_variance = np.full(len(values), long_run_variance)
    else:
        alpha, beta, omega, conditional_variance = best
    next_variance = (
        omega + alpha * residuals[-1] ** 2 + beta * conditional_variance[-1]
    )
    standardized_residuals = residuals / np.sqrt(conditional_variance)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "omega": float(omega),
        "next_variance": float(max(next_variance, 1e-10)),
        "standardized_residuals": standardized_residuals,
        "df": float(df),
    }


def _prepare_model(
    daily_returns: np.ndarray,
    model: str,
    risk_free_rate: float,
    drift_shrinkage: float,
) -> dict[str, object]:
    """Estimate robust, shrunk drifts and model-specific volatility parameters."""
    simple_returns = np.clip(np.asarray(daily_returns, dtype=float), -0.999999, None)
    log_returns = np.log1p(simple_returns)
    robust_log_returns = _winsorize_returns(log_returns)
    robust_mean = robust_log_returns.mean(axis=0)
    shrinkage = float(np.clip(drift_shrinkage, 0.0, 1.0))
    risk_free_daily = math.log1p(max(risk_free_rate, -0.99) / TRADING_DAYS_PER_YEAR)
    daily_drift = (1.0 - shrinkage) * robust_mean + shrinkage * risk_free_daily
    daily_volatility = np.maximum(robust_log_returns.std(axis=0, ddof=1), 1e-6)
    correlation = regularized_correlation(pd.DataFrame(robust_log_returns))
    prepared: dict[str, object] = {
        "model": model,
        "log_returns": log_returns,
        "daily_drift": daily_drift,
        "daily_volatility": daily_volatility,
        "drift_se": daily_volatility / math.sqrt(max(1, len(log_returns))),
        "correlation": correlation,
        "chol": np.linalg.cholesky(correlation),
        "df": 8.0,
        "risk_free_rate": risk_free_rate,
    }
    asset_count = simple_returns.shape[1]

    if model == "garch_t":
        fits = [_fit_garch_t(log_returns[:, idx]) for idx in range(asset_count)]
        standardized = np.column_stack([fit["standardized_residuals"] for fit in fits])
        standardized_correlation = regularized_correlation(pd.DataFrame(standardized))
        prepared.update(
            {
                "garch_fits": fits,
                "correlation": standardized_correlation,
                "chol": np.linalg.cholesky(standardized_correlation),
                "df": 8.0,
            }
        )

    if model == "regime_switch":
        market_volatility = np.sqrt(np.mean(np.square(simple_returns), axis=1))
        rolling_volatility = pd.Series(market_volatility).rolling(20, min_periods=5).mean()
        threshold = float(rolling_volatility.quantile(0.75))
        states = (rolling_volatility.fillna(rolling_volatility.median()).to_numpy() > threshold).astype(int)
        transition_counts = np.full((2, 2), 0.5, dtype=float)
        for left, right in zip(states[:-1], states[1:]):
            transition_counts[int(left), int(right)] += 1.0
        transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        state_means = np.empty((2, asset_count), dtype=float)
        state_volatilities = np.empty((2, asset_count), dtype=float)
        state_cholesky: list[np.ndarray] = []
        for state in range(2):
            mask = states == state
            if int(mask.sum()) < 40:
                state_returns = robust_log_returns
            else:
                state_returns = robust_log_returns[mask]
            state_mean = state_returns.mean(axis=0)
            state_means[state] = (1.0 - shrinkage) * state_mean + shrinkage * risk_free_daily
            state_volatilities[state] = np.maximum(state_returns.std(axis=0, ddof=1), 1e-6)
            state_corr = regularized_correlation(pd.DataFrame(state_returns))
            state_cholesky.append(np.linalg.cholesky(state_corr))
        prepared.update(
            {
                "regime_states": states,
                "regime_transition": transition,
                "regime_initial_state": int(states[-1]),
                "regime_means": state_means,
                "regime_volatilities": state_volatilities,
                "regime_cholesky": state_cholesky,
                "regime_threshold": threshold,
            }
        )
    return prepared


def _student_t_shocks(
    rng: np.random.Generator,
    rows: int,
    assets: int,
    cholesky: np.ndarray,
    df: float = 8.0,
) -> np.ndarray:
    normal = rng.standard_normal((rows, assets)) @ cholesky.T
    scale = np.sqrt(rng.chisquare(df, size=(rows, 1)) / df)
    return normal / scale * math.sqrt((df - 2.0) / df)


def _generate_model_log_steps(
    prepared: dict[str, object],
    n_paths: int,
    horizon_days: int,
    rng: np.random.Generator,
    parameter_uncertainty: float,
) -> np.ndarray:
    model = str(prepared["model"])
    historical_log_returns = prepared["log_returns"]
    daily_drift = prepared["daily_drift"]
    daily_volatility = prepared["daily_volatility"]
    cholesky = prepared["chol"]
    asset_count = len(daily_drift)
    drift_draw = rng.normal(size=(n_paths, asset_count)) * prepared["drift_se"] * parameter_uncertainty

    if model in {"gbm", "student_t"}:
        if model == "gbm":
            shocks = rng.standard_normal((n_paths, horizon_days, asset_count)) @ cholesky.T
        else:
            shocks = _student_t_shocks(
                rng, n_paths * horizon_days, asset_count, cholesky, float(prepared["df"])
            ).reshape(n_paths, horizon_days, asset_count)
        return (
            daily_drift[None, None, :]
            + drift_draw[:, None, :]
            + shocks * daily_volatility[None, None, :]
        )

    if model == "bootstrap":
        steps = np.empty((n_paths, horizon_days, asset_count), dtype=float)
        offset = 0
        while offset < horizon_days:
            width = min(BOOTSTRAP_BLOCK_DAYS, horizon_days - offset)
            starts = rng.integers(0, len(historical_log_returns), size=n_paths)
            positions = (starts[:, None] + np.arange(width)[None, :]) % len(historical_log_returns)
            sampled = historical_log_returns[positions]
            shift = daily_drift - historical_log_returns.mean(axis=0)
            steps[:, offset : offset + width, :] = sampled + shift + drift_draw[:, None, :]
            offset += width
        return steps

    if model == "garch_t":
        fits = prepared["garch_fits"]
        omega = np.array([fit["omega"] for fit in fits], dtype=float)
        alpha = np.array([fit["alpha"] for fit in fits], dtype=float)
        beta = np.array([fit["beta"] for fit in fits], dtype=float)
        variance = np.broadcast_to(
            np.array([fit["next_variance"] for fit in fits], dtype=float),
            (n_paths, asset_count),
        ).copy()
        steps = np.empty((n_paths, horizon_days, asset_count), dtype=float)
        for day in range(horizon_days):
            shocks = _student_t_shocks(rng, n_paths, asset_count, cholesky)
            innovations = shocks * np.sqrt(variance)
            steps[:, day, :] = daily_drift[None, :] + drift_draw + innovations
            variance = omega + alpha * np.square(innovations) + beta * variance
        return steps

    if model == "regime_switch":
        transition = prepared["regime_transition"]
        states = np.full(n_paths, prepared["regime_initial_state"], dtype=int)
        means = prepared["regime_means"]
        volatilities = prepared["regime_volatilities"]
        choleskies = prepared["regime_cholesky"]
        steps = np.empty((n_paths, horizon_days, asset_count), dtype=float)
        for day in range(horizon_days):
            probabilities = transition[states, 1]
            states = (rng.random(n_paths) < probabilities).astype(int)
            shocks = np.empty((n_paths, asset_count), dtype=float)
            for state in (0, 1):
                mask = states == state
                if np.any(mask):
                    shocks[mask] = rng.standard_normal((int(mask.sum()), asset_count)) @ choleskies[state].T
            steps[:, day, :] = (
                means[states]
                + drift_draw
                + shocks * volatilities[states]
            )
        return steps

    raise ValueError("Neznámy simulačný model.")


def find_worst_historical_block(
    prices: pd.DataFrame,
    portfolio_tickers: list[str],
    window_days: int = BACKTEST_HORIZON_DAYS,
) -> dict[str, object]:
    """Find the worst observed equal-weight portfolio block and return all asset returns."""
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < window_days:
        raise RuntimeError("Na historický stresový blok nie je dosť denných dát.")
    portfolio_returns = returns[portfolio_tickers].mean(axis=1).to_numpy(dtype=float)
    cumulative = np.log1p(np.clip(portfolio_returns, -0.999999, None))
    rolling = np.convolve(cumulative, np.ones(window_days), mode="valid")
    start = int(np.argmin(rolling))
    end = start + window_days
    stress_returns = returns.iloc[start:end].to_numpy(dtype=float)
    stress_dates = returns.index[start:end]
    equal_weight_total = float(np.exp(rolling[start]) - 1.0)
    return {
        "returns": stress_returns,
        "start_date": stress_dates[0].strftime("%Y-%m-%d"),
        "end_date": stress_dates[-1].strftime("%Y-%m-%d"),
        "equal_weight_return": equal_weight_total,
        "tickers": list(returns.columns),
    }


def simulate(
    prices: pd.DataFrame,
    horizon_days: int,
    n_simulations: int,
    chart_ticker: str,
    portfolio_tickers: list[str],
    portfolio_weights: dict[str, np.ndarray],
    model: str,
    stress_shock: float,
    trade_cost_rate: float,
    annual_fee_rate: float,
    risk_free_rate: float = 0.04,
    drift_shrinkage: float = 0.50,
    parameter_uncertainty: float = 1.0,
    historical_stress: dict[str, object] | None = None,
) -> dict[str, object]:
    """Simulate correlated GBM, bootstrap, Student-t, GARCH-t, or regime paths."""
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 60:
        raise RuntimeError("Na odhad modelu treba aspoň 60 denných výnosov.")

    tickers = list(prices.columns)
    prepared_model = _prepare_model(
        returns.to_numpy(dtype=float), model, risk_free_rate, drift_shrinkage
    )
    annual_drift = np.asarray(prepared_model["daily_drift"]) * TRADING_DAYS_PER_YEAR
    annual_volatility = np.asarray(prepared_model["daily_volatility"]) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if not np.all(np.isfinite(annual_volatility)) or np.any(annual_volatility <= 0):
        raise RuntimeError("Aspoň jeden ticker nemá použiteľnú historickú volatilitu.")

    start_prices = prices.iloc[-1].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    n_assets = len(tickers)
    chart_index = tickers.index(chart_ticker)
    indices = [tickers.index(ticker) for ticker in portfolio_tickers]
    full_strategy_weights: dict[str, np.ndarray] = {}
    for strategy, weights in portfolio_weights.items():
        full_weights = np.zeros(n_assets)
        full_weights[indices] = weights
        full_strategy_weights[strategy] = full_weights

    terminal_prices = np.empty((n_simulations, n_assets), dtype=np.float32)
    max_drawdowns = np.empty((n_simulations, n_assets), dtype=np.float32)
    stress_drawdowns = np.empty((n_simulations, n_assets), dtype=np.float32)
    chart_paths = np.empty((n_simulations, horizon_days + 1), dtype=np.float32)
    portfolio_drawdowns = {
        strategy: np.empty(n_simulations, dtype=np.float32)
        for strategy in portfolio_weights
    }
    stress_portfolio_drawdowns = {
        strategy: np.empty(n_simulations, dtype=np.float32)
        for strategy in portfolio_weights
    }
    day_factors = np.ones(horizon_days + 1, dtype=float)
    if horizon_days > 0:
        days = np.arange(1, horizon_days + 1)
        day_factors[1:] = (1.0 - trade_cost_rate) * (1.0 - annual_fee_rate) ** (
            days / TRADING_DAYS_PER_YEAR
        )
        day_factors[-1] *= 1.0 - trade_cost_rate
    terminal_cost_factor = (1.0 - trade_cost_rate) ** 2 * (
        1.0 - annual_fee_rate
    ) ** (horizon_days / TRADING_DAYS_PER_YEAR)

    batch_size = min(256, n_simulations)
    historical_returns = returns.to_numpy(dtype=float)

    historical_stress_factors = None
    historical_stress_block = None
    historical_stress_length = 0
    historical_stress_drawdowns = None
    historical_stress_portfolio_drawdowns = None
    historical_stress_terminal_relative = None
    if historical_stress is not None:
        stress_tickers = historical_stress["tickers"]
        stress_block = historical_stress["returns"]
        block_by_asset = stress_block[:, [stress_tickers.index(ticker) for ticker in tickers]]
        historical_stress_length = min(horizon_days, len(block_by_asset))
        historical_stress_block = block_by_asset[:historical_stress_length]
        historical_stress_factors = np.vstack(
            [np.ones((1, n_assets)), np.cumprod(1.0 + historical_stress_block, axis=0)]
        )
        historical_stress_drawdowns = np.empty((n_simulations, n_assets), dtype=np.float32)
        historical_stress_terminal_relative = np.empty(
            (n_simulations, n_assets), dtype=np.float32
        )
        historical_stress_portfolio_drawdowns = {
            strategy: np.empty(n_simulations, dtype=np.float32)
            for strategy in portfolio_weights
        }

    for first in range(0, n_simulations, batch_size):
        last = min(first + batch_size, n_simulations)
        batch_count = last - first
        log_steps = _generate_model_log_steps(
            prepared_model, batch_count, horizon_days, rng, parameter_uncertainty
        )

        cumulative_log_returns = np.cumsum(log_steps, axis=1)
        batch_prices = start_prices[None, None, :] * np.exp(
            np.clip(cumulative_log_returns, -50.0, 50.0)
        )
        paths_with_start = np.concatenate(
            [np.broadcast_to(start_prices, (batch_count, 1, n_assets)), batch_prices],
            axis=1,
        )
        relative_paths = paths_with_start / start_prices[None, None, :]

        net_relative_paths = relative_paths * day_factors[None, :, None]
        stressed_relative = net_relative_paths.copy()
        if horizon_days > 0 and stress_shock > 0:
            stressed_relative[:, 1:, :] *= 1.0 - stress_shock
        batch_stress_drawdowns = _drawdowns(stressed_relative)
        historical_stressed_relative = None
        if historical_stress_block is not None:
            historical_stressed_relative = np.empty_like(net_relative_paths)
            historical_stressed_relative[:, : historical_stress_length + 1, :] = (
                historical_stress_factors[None, :, :]
            )
            remaining_days = horizon_days - historical_stress_length
            if remaining_days > 0:
                continuation_log_steps = _generate_model_log_steps(
                    prepared_model,
                    batch_count,
                    remaining_days,
                    rng,
                    parameter_uncertainty,
                )
                continuation_factors = np.exp(
                    np.cumsum(continuation_log_steps, axis=1)
                ) * historical_stress_factors[-1][None, None, :]
                historical_stressed_relative[:, historical_stress_length + 1 :, :] = (
                    continuation_factors
                )
            historical_stressed_relative *= day_factors[None, :, None]

        terminal_prices[first:last] = batch_prices[:, -1, :].astype(np.float32)
        max_drawdowns[first:last] = _drawdowns(net_relative_paths).astype(np.float32)
        stress_drawdowns[first:last] = batch_stress_drawdowns.astype(np.float32)
        if historical_stress_drawdowns is not None and historical_stressed_relative is not None:
            historical_stress_terminal_relative[first:last] = (
                historical_stressed_relative[:, -1, :]
            )
            historical_stress_drawdowns[first:last] = _drawdowns(
                historical_stressed_relative
            ).astype(np.float32)
        chart_paths[first:last] = paths_with_start[:, :, chart_index].astype(np.float32)
        for strategy, weights in full_strategy_weights.items():
            portfolio_values = net_relative_paths @ weights
            stress_values = stressed_relative @ weights
            portfolio_drawdowns[strategy][first:last] = _drawdowns(
                portfolio_values
            ).astype(np.float32)
            stress_portfolio_drawdowns[strategy][first:last] = _drawdowns(
                stress_values
            ).astype(np.float32)
            if historical_stress_portfolio_drawdowns is not None:
                historical_stress_portfolio_drawdowns[strategy][first:last] = _drawdowns(
                    historical_stressed_relative @ weights
                ).astype(np.float32)

    return {
        "tickers": tickers,
        "start_prices": start_prices,
        "annual_drift": annual_drift,
        "annual_volatility": annual_volatility,
        "terminal_prices": terminal_prices,
        "max_drawdowns": max_drawdowns,
        "stress_drawdowns": stress_drawdowns,
        "chart_paths": chart_paths,
        "portfolio_tickers": portfolio_tickers,
        "portfolio_weights": portfolio_weights,
        "portfolio_drawdowns": portfolio_drawdowns,
        "stress_portfolio_drawdowns": stress_portfolio_drawdowns,
        "historical_stress_drawdowns": historical_stress_drawdowns,
        "historical_stress_portfolio_drawdowns": historical_stress_portfolio_drawdowns,
        "historical_stress_terminal_relative": historical_stress_terminal_relative,
        "historical_stress_days": historical_stress_length,
        "historical_stress": historical_stress,
        "stress_shock": stress_shock,
        "model": model,
        "drift_shrinkage": drift_shrinkage,
        "parameter_uncertainty": parameter_uncertainty,
        "net_cost_factor": terminal_cost_factor,
        "trade_cost_rate": trade_cost_rate,
        "annual_fee_rate": annual_fee_rate,
    }


def verdict(
    median_return: float,
    probability_profit: float,
    sharpe: float,
    probability_drawdown_over_limit: float,
    var95_loss: float,
    max_drawdown_limit: float,
) -> tuple[str, str]:
    """Apply visible screening thresholds; this is a heuristic, not advice."""
    if (
        median_return > 0
        and probability_profit >= 0.60
        and sharpe >= 0.50
        and probability_drawdown_over_limit <= 0.10
    ):
        return "Splnené priaznivé prahy", "success"
    if probability_drawdown_over_limit >= 0.25 or var95_loss > max_drawdown_limit:
        return "Vysoké riziko", "error"
    return "Neutrálne", "warning"


def explain_verdict(
    median_return: float,
    probability_profit: float,
    sharpe: float,
    probability_drawdown_over_limit: float,
    var95_loss: float,
    max_drawdown_limit: float,
) -> tuple[str, str, list[str]]:
    """Return a screening label and the exact heuristic checks behind it."""
    label, level = verdict(
        median_return, probability_profit, sharpe,
        probability_drawdown_over_limit, var95_loss, max_drawdown_limit,
    )
    checks = [
        f"Mediánový výnos {median_return:+.1%} (priaznivý prah: nad 0 %): {'splnené' if median_return > 0 else 'nesplnené'}.",
        f"Pravdepodobnosť zisku {probability_profit:.1%} (priaznivý prah: aspoň 60 %): {'splnené' if probability_profit >= 0.60 else 'nesplnené'}.",
        f"Sharpe {sharpe:.2f} (priaznivý prah: aspoň 0,50): {'splnené' if sharpe >= 0.50 else 'nesplnené'}.",
        f"Pravdepodobnosť poklesu nad limit {probability_drawdown_over_limit:.1%} (priaznivý prah: najviac 10 %): {'splnené' if probability_drawdown_over_limit <= 0.10 else 'nesplnené'}.",
        f"VaR 95 % {var95_loss:.1%}; limit {max_drawdown_limit:.1%}: "
        f"{'prekročený' if var95_loss > max_drawdown_limit else 'neprekročený'} "
        "(VaR a maximálny pokles sú odlišné ukazovatele).",
    ]
    return label, level, checks


def apply_terminal_tax(returns: np.ndarray, tax_rate: float) -> np.ndarray:
    """Apply a simplified tax only to positive terminal gains."""
    values = np.asarray(returns, dtype=float)
    return values - np.maximum(values, 0.0) * float(tax_rate)


def make_sensitivity_table(
    gross_returns: np.ndarray,
    cost_factor: float,
    investment: float,
    horizon_days: int,
    tax_rate: float = 0.0,
) -> pd.DataFrame:
    """Approximate terminal-return sensitivity around the selected simulation."""
    base_log = np.log1p(np.clip(np.asarray(gross_returns, dtype=float), -0.999999, None))
    center = float(np.mean(base_log))
    years = horizon_days / TRADING_DAYS_PER_YEAR
    cases = [
        ("Základ", 0.00, 1.00, 0.00),
        ("Drift −5 p. b.", -0.05, 1.00, 0.00),
        ("Drift +5 p. b.", 0.05, 1.00, 0.00),
        ("Volatilita −20 %", 0.00, 0.80, 0.00),
        ("Volatilita +20 %", 0.00, 1.20, 0.00),
        ("Náklady +25 bps za smer", 0.00, 1.00, 25.0),
    ]
    rows = []
    for name, drift_shift, vol_scale, added_cost_bps in cases:
        adjusted_log = center + (base_log - center) * vol_scale + drift_shift * years
        adjusted_gross = np.expm1(np.clip(adjusted_log, -50, 50))
        factor = cost_factor * (1.0 - added_cost_bps / 10_000.0) ** 2
        net = apply_terminal_tax((1.0 + adjusted_gross) * factor - 1.0, tax_rate)
        var95, cvar95 = _tail_metrics(net)
        rows.append({
            "Scenár": name,
            "Mediánový výnos (%)": float(np.median(net) * 100),
            "Priemerný výnos (%)": float(np.mean(net) * 100),
            "VaR 95 % (%)": var95 * 100,
            "CVaR 95 % (%)": cvar95 * 100,
            "P. zisku (%)": float(np.mean(net > 0) * 100),
            "Mediánová hodnota investície": float(np.median(1 + net) * investment),
        })
    return pd.DataFrame(rows)


def make_correlation_sensitivity(
    annual_means: np.ndarray,
    annual_vols: np.ndarray,
    correlation: np.ndarray,
    weights: np.ndarray,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Show portfolio risk if historical pairwise correlations weaken or strengthen."""
    rows = []
    for name, multiplier in (("Bez korelácií", 0.0), ("Polovica historických", 0.5), ("Historická", 1.0), ("Vyššia (+25 %)", 1.25)):
        stressed_corr = np.eye(len(annual_vols)) + (correlation - np.eye(len(annual_vols))) * multiplier
        stressed_corr = (stressed_corr + stressed_corr.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(stressed_corr)
        stressed_corr = (eigenvectors * np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
        scales = np.sqrt(np.diag(stressed_corr))
        stressed_corr = stressed_corr / np.outer(scales, scales)
        np.fill_diagonal(stressed_corr, 1.0)
        covariance = np.outer(annual_vols, annual_vols) * stressed_corr
        annual_return = float(np.asarray(weights) @ annual_means)
        annual_volatility = float(np.sqrt(max(np.asarray(weights) @ covariance @ np.asarray(weights), 0.0)))
        rows.append({
            "Korelačný predpoklad": name,
            "Odhad výnosu p. a. (%)": annual_return * 100,
            "Odhad volatility p. a. (%)": annual_volatility * 100,
            "Sharpe": (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def format_money(value: float, symbol: str) -> str:
    return f"{symbol}{value:,.2f}"


def format_delta_money(value: float, symbol: str) -> str:
    """Format metric deltas with the sign first so Streamlit chooses the right arrow."""
    amount = abs(float(value))
    if value < 0:
        return f"-{symbol}{amount:,.2f}"
    if value > 0:
        return f"+{symbol}{amount:,.2f}"
    return f"{symbol}{amount:,.2f}"


def _tail_metrics(returns: np.ndarray) -> tuple[float, float]:
    q05 = float(np.quantile(returns, 0.05))
    tail = returns[returns <= q05]
    return -q05, -float(np.mean(tail))


def calculate_net_cost_factor(
    trade_cost_bps: float,
    annual_fee_rate: float,
    horizon_days: int,
) -> float:
    """Apply one buy, one sell, and compounded annual fees to terminal value."""
    one_way_cost = float(trade_cost_bps) / 10_000.0
    if one_way_cost >= 1.0 or annual_fee_rate >= 1.0:
        raise ValueError("Náklady musia byť nižšie ako 100 %.")
    years = horizon_days / TRADING_DAYS_PER_YEAR
    return (1.0 - one_way_cost) ** 2 * (1.0 - annual_fee_rate) ** years


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    ticker: str,
    model: str,
    net_cost_factor: float,
    risk_free_rate: float = 0.04,
    drift_shrinkage: float = 0.50,
    parameter_uncertainty: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Forecast rolling 63-day returns from trailing three-year samples."""
    series = prices[ticker].dropna()
    if len(series) < BACKTEST_TRAIN_DAYS + BACKTEST_HORIZON_DAYS:
        raise RuntimeError(
            "Na spätný test treba aspoň 3 roky tréningových dát a ďalšie 3 mesiace "
            "na vyhodnotenie."
        )

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(431)
    final_origin = len(series) - BACKTEST_HORIZON_DAYS
    for origin in range(BACKTEST_TRAIN_DAYS, final_origin + 1, BACKTEST_STEP_DAYS):
        training_prices = series.iloc[origin - BACKTEST_TRAIN_DAYS : origin]
        training_returns = training_prices.pct_change(fill_method=None).dropna().to_numpy()
        if len(training_returns) < 60:
            continue
        prepared = _prepare_model(
            training_returns.reshape(-1, 1), model, risk_free_rate, drift_shrinkage
        )
        simulated_log_steps = _generate_model_log_steps(
            prepared,
            BACKTEST_SIMULATIONS,
            BACKTEST_HORIZON_DAYS,
            rng,
            parameter_uncertainty,
        )
        forecast_returns = np.exp(simulated_log_steps.sum(axis=1)[:, 0]) - 1.0

        forecast_returns = (1.0 + forecast_returns) * net_cost_factor - 1.0
        q10, median_forecast, q90 = np.quantile(forecast_returns, [0.10, 0.50, 0.90])
        start_price = float(series.iloc[origin - 1])
        end_price = float(series.iloc[origin + BACKTEST_HORIZON_DAYS - 1])
        actual_return = (end_price / start_price) * net_cost_factor - 1.0
        rows.append(
            {
                "Dátum odhadu": series.index[origin - 1],
                "Dátum výsledku": series.index[origin + BACKTEST_HORIZON_DAYS - 1],
                "10. percentil (%)": float(q10 * 100),
                "Medián odhadu (%)": float(median_forecast * 100),
                "90. percentil (%)": float(q90 * 100),
                "Skutočný výnos (%)": float(actual_return * 100),
                "Nulová prognóza (%)": 0.0,
                "Pokrytie pásma": bool(q10 <= actual_return <= q90),
                "Správne určený smer": bool((median_forecast >= 0) == (actual_return >= 0)),
                "Správny smer nuly": bool(actual_return >= 0),
                "Absolútna chyba nuly (p. b.)": abs(float(actual_return * 100)),
            }
        )

    backtest = pd.DataFrame(rows)
    if backtest.empty:
        raise RuntimeError("Nedostatok údajov na vytvorenie spätného testu.")
    metrics = {
        "coverage": float(backtest["Pokrytie pásma"].mean()),
        "direction_accuracy": float(backtest["Správne určený smer"].mean()),
        "median_absolute_error_pp": float(
            np.median(
                np.abs(backtest["Skutočný výnos (%)"] - backtest["Medián odhadu (%)"])
            )
        ),
        "origins": float(len(backtest)),
        "zero_direction_accuracy": float(backtest["Správny smer nuly"].mean()),
        "zero_median_absolute_error_pp": float(
            np.median(backtest["Absolútna chyba nuly (p. b.)"])
        ),
    }
    return backtest, metrics


def run_walk_forward_model_comparison(
    prices: pd.DataFrame,
    ticker: str,
    net_cost_factor: float,
    risk_free_rate: float,
    drift_shrinkage: float,
    parameter_uncertainty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every scenario model on identical rolling forecast origins."""
    forecast_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for label, model in MODELS.items():
        forecast, metrics = run_walk_forward_backtest(
            prices,
            ticker,
            model,
            net_cost_factor,
            risk_free_rate,
            drift_shrinkage,
            parameter_uncertainty,
        )
        forecast.insert(2, "Model", label)
        forecast_frames.append(forecast)
        summary_rows.append(
            {
                "Model": label,
                "Pokrytie 10.–90. (%)": metrics["coverage"] * 100,
                "Správny smer mediánu (%)": metrics["direction_accuracy"] * 100,
                "Mediánová absolútna chyba (p. b.)": metrics["median_absolute_error_pp"],
                "Priemerná šírka pásma (p. b.)": float(
                    (forecast["90. percentil (%)"] - forecast["10. percentil (%)"]).mean()
                ),
                "Presnosť smeru nuly (%)": metrics["zero_direction_accuracy"] * 100,
                "Chyba nulovej prognózy (p. b.)": metrics["zero_median_absolute_error_pp"],
                "Počet období": int(metrics["origins"]),
            }
        )
    return pd.concat(forecast_frames, ignore_index=True), pd.DataFrame(summary_rows)


def make_backtest_chart(backtest: pd.DataFrame, ticker: str) -> go.Figure:
    dates = backtest["Dátum odhadu"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=backtest["10. percentil (%)"],
            mode="lines",
            line={"color": "#d55e00", "width": 1.5, "dash": "dash"},
            name="10. percentil",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=backtest["90. percentil (%)"],
            mode="lines",
            line={"color": "#009e73", "width": 1.5, "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(86,180,233,0.14)",
            name="90. percentil",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=backtest["Medián odhadu (%)"],
            mode="lines",
            line={"color": "#0072b2", "width": 2},
            name="Medián odhadu",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=backtest["Skutočný výnos (%)"],
            mode="markers",
            marker={"color": "#222222", "size": 7, "symbol": "x"},
            name="Skutočný výnos",
        )
    )
    figure.add_hline(y=0, line_color="rgba(128,128,128,0.55)", line_width=1)
    figure.update_layout(
        title=f"Walk-forward test: 3-mesačný výnos {ticker}",
        xaxis_title="Dátum odhadu",
        yaxis_title="Výnos po nákladoch (%)",
        legend={"orientation": "h", "y": 1.16, "x": 0},
        margin={"t": 90, "b": 50},
        height=420,
        hovermode="x unified",
    )
    figure.update_yaxes(ticksuffix="%", gridcolor="rgba(128,128,128,0.2)")
    return figure


def make_results(
    simulation: dict[str, object],
    tracked_tickers: list[str],
    benchmark_ticker: str,
    investment: float,
    risk_free_rate: float,
    max_drawdown_limit: float,
    currencies: dict[str, str],
    spot_rates: dict[str, float],
    fx_stats: dict[str, dict[str, float]],
    horizon_days: int,
    tax_rate: float = 0.0,
) -> pd.DataFrame:
    tickers = simulation["tickers"]
    start_prices = simulation["start_prices"]
    annual_volatility = simulation["annual_volatility"]
    terminal_prices = simulation["terminal_prices"]
    max_drawdowns = simulation["max_drawdowns"]
    stress_drawdowns = simulation["stress_drawdowns"]
    stress_shock = simulation["stress_shock"]
    cost_factor = simulation["net_cost_factor"]
    benchmark_idx = tickers.index(benchmark_ticker)
    benchmark_gross_returns = terminal_prices[:, benchmark_idx] / start_prices[benchmark_idx] - 1.0
    benchmark_returns = apply_terminal_tax(
        (1.0 + benchmark_gross_returns) * cost_factor - 1.0, tax_rate
    )
    years = horizon_days / TRADING_DAYS_PER_YEAR
    rows = []

    for ticker in tracked_tickers:
        idx = tickers.index(ticker)
        gross_returns = terminal_prices[:, idx] / start_prices[idx] - 1.0
        simulated_returns = apply_terminal_tax(
            (1.0 + gross_returns) * cost_factor - 1.0, tax_rate
        )
        var95_loss, cvar95_loss = _tail_metrics(simulated_returns)
        stress_returns = apply_terminal_tax(
            (1.0 - stress_shock) * (1.0 + gross_returns) * cost_factor - 1.0,
            tax_rate,
        )
        stress_var95, stress_cvar95 = _tail_metrics(stress_returns)
        probability_profit = float(np.mean(simulated_returns > 0))
        annualized_net_mean = float(np.mean(simulated_returns) / years) if years > 0 else 0.0
        sharpe = (
            (annualized_net_mean - risk_free_rate) / annual_volatility[idx]
            if annual_volatility[idx] > 0
            else float("nan")
        )
        probability_drawdown_over_limit = float(
            np.mean(max_drawdowns[:, idx] < -max_drawdown_limit)
        )
        probability_stress_drawdown_over_limit = float(
            np.mean(stress_drawdowns[:, idx] < -max_drawdown_limit)
        )
        label, _ = verdict(
            float(np.median(simulated_returns)),
            probability_profit,
            float(sharpe),
            probability_drawdown_over_limit,
            var95_loss,
            max_drawdown_limit,
        )
        probability_beat = (
            float(np.mean(simulated_returns > benchmark_returns))
            if ticker != benchmark_ticker
            else float("nan")
        )
        rows.append(
            {
                "Ticker": ticker,
                "Mena kótovania": currencies[ticker],
                "Kurz do meny účtu": spot_rates[ticker],
                "FX historický výnos (ročný, %)": fx_stats[ticker]["annual_return"] * 100,
                "FX volatilita (ročná, %)": fx_stats[ticker]["annual_volatility"] * 100,
                "Priemerný výnos pred nákladmi (%)": float(np.mean(gross_returns) * 100),
                "Priemerný výnos (%)": float(np.mean(simulated_returns) * 100),
                "Priemerný zisk/strata": float(np.mean(simulated_returns) * investment),
                "Mediánový výnos (%)": float(np.median(simulated_returns) * 100),
                "Mediánový zisk/strata": float(np.median(simulated_returns) * investment),
                "VaR 95 % (%)": var95_loss * 100,
                "CVaR 95 % (%)": cvar95_loss * 100,
                "Pravdepodobnosť zisku (%)": probability_profit * 100,
                f"P. prekonať {benchmark_ticker} (%)": probability_beat * 100,
                "Sharpe": float(sharpe),
                "P. prekročenia poklesu (%)": probability_drawdown_over_limit * 100,
                "Stresová p. prekročenia poklesu (%)": probability_stress_drawdown_over_limit * 100,
                "Stresový medián výnosu (%)": float(np.median(stress_returns) * 100),
                "Stresový VaR 95 % (%)": stress_var95 * 100,
                "Stresový CVaR 95 % (%)": stress_cvar95 * 100,
                "Verdikt": label,
            }
        )
    return pd.DataFrame(rows)


def make_portfolio_summary(
    simulation: dict[str, object],
    weights: np.ndarray,
    investment: float,
    risk_free_rate: float,
    horizon_days: int,
    max_drawdown_limit: float,
    benchmark_ticker: str,
    stress: bool = False,
    portfolio_kind: str = "equal",
    tax_rate: float = 0.0,
) -> dict[str, float | str]:
    tickers = simulation["tickers"]
    start_prices = simulation["start_prices"]
    terminal_prices = simulation["terminal_prices"]
    portfolio_tickers = simulation["portfolio_tickers"]
    indices = [tickers.index(ticker) for ticker in portfolio_tickers]
    asset_returns = terminal_prices[:, indices] / start_prices[indices] - 1.0
    gross_portfolio_returns = asset_returns @ weights
    cost_factor = simulation["net_cost_factor"]
    portfolio_returns = (1.0 + gross_portfolio_returns) * cost_factor - 1.0
    stress_shock = simulation["stress_shock"]
    if stress:
        portfolio_returns = (1.0 - stress_shock) * (1.0 + gross_portfolio_returns) * cost_factor - 1.0
    portfolio_returns = apply_terminal_tax(portfolio_returns, tax_rate)

    drawdown_key = "stress_portfolio_drawdowns" if stress else "portfolio_drawdowns"
    max_drawdowns = simulation[drawdown_key][portfolio_kind]
    var95_loss, cvar95_loss = _tail_metrics(portfolio_returns)
    probability_profit = float(np.mean(portfolio_returns > 0))
    years = horizon_days / TRADING_DAYS_PER_YEAR
    annualized_mean = float(np.mean(portfolio_returns) / years)
    annualized_vol = float(np.std(portfolio_returns, ddof=1) / np.sqrt(years))
    sharpe = (annualized_mean - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0
    dd_probability = float(np.mean(max_drawdowns < -max_drawdown_limit))
    label, _ = verdict(
        float(np.median(portfolio_returns)),
        probability_profit,
        sharpe,
        dd_probability,
        var95_loss,
        max_drawdown_limit,
    )

    benchmark_idx = tickers.index(benchmark_ticker)
    benchmark_gross_returns = terminal_prices[:, benchmark_idx] / start_prices[benchmark_idx] - 1.0
    benchmark_returns = apply_terminal_tax(
        (1.0 + benchmark_gross_returns) * cost_factor - 1.0, tax_rate
    )
    if stress:
        benchmark_returns = apply_terminal_tax(
            (1.0 - stress_shock) * (1.0 + benchmark_gross_returns) * cost_factor - 1.0,
            tax_rate,
        )
    probability_beat = (
        float(np.mean(portfolio_returns > benchmark_returns))
        if benchmark_ticker not in portfolio_tickers
        else float("nan")
    )
    return {
        "mean_return": float(np.mean(portfolio_returns)),
        "median_return": float(np.median(portfolio_returns)),
        "mean_gain": float(np.mean(portfolio_returns) * investment),
        "median_gain": float(np.median(portfolio_returns) * investment),
        "var95": var95_loss,
        "cvar95": cvar95_loss,
        "probability_profit": probability_profit,
        "probability_beat": probability_beat,
        "sharpe": float(sharpe),
        "drawdown_probability": dd_probability,
        "verdict": label,
        "median_terminal_value": float(np.median(1.0 + portfolio_returns) * investment),
        "investment": investment,
    }


def make_fan_chart(
    paths: np.ndarray,
    ticker: str,
    horizon_days: int,
    currency_symbol: str,
) -> go.Figure:
    p10, p50, p90 = np.quantile(paths, [0.10, 0.50, 0.90], axis=0)
    days = np.arange(horizon_days + 1)
    sample_count = min(FAN_CHART_PATH_LIMIT, len(paths))
    sample_indices = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    sample_paths = paths[sample_indices]
    path_x = np.tile(np.append(days, np.nan), sample_count)
    path_y = np.column_stack(
        [sample_paths, np.full(sample_count, np.nan, dtype=sample_paths.dtype)]
    ).ravel()
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=path_x,
            y=path_y,
            mode="lines",
            line={"width": 0.65, "color": "rgba(185,190,198,0.09)"},
            name="Simulované cesty",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p10,
            mode="lines",
            line={"width": 3, "color": "#e64ac2", "dash": "dash"},
            name="10. percentil",
            hovertemplate=f"Deň %{{x}}<br>Cena {currency_symbol}%{{y:,.2f}}<extra>10. percentil</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p90,
            mode="lines",
            line={"width": 3, "color": "#f0a23b", "dash": "dash"},
            name="90. percentil",
            hovertemplate=f"Deň %{{x}}<br>Cena {currency_symbol}%{{y:,.2f}}<extra>90. percentil</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=days,
            y=p50,
            mode="lines",
            line={"width": 3.5, "color": "#4f83ff"},
            name="Medián (50. percentil)",
            hovertemplate=f"Deň %{{x}}<br>Cena {currency_symbol}%{{y:,.2f}}<extra>Medián</extra>",
        )
    )
    figure.update_layout(
        title=f"Monte Carlo simulácia ceny {ticker} ({sample_count:,} ukážkových ciest)",
        xaxis_title=f"Obchodné dni (0 až {horizon_days:,})",
        yaxis_title=f"Cena akcie ({currency_symbol})",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.18, "x": 0, "xanchor": "left"},
        margin={"t": 110, "b": 60, "l": 75, "r": 25},
        height=450,
    )
    figure.update_xaxes(
        nticks=8,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.12)",
        zeroline=False,
    )
    figure.update_yaxes(
        tickprefix=currency_symbol,
        tickformat=",.0f",
        gridcolor="rgba(128,128,128,0.2)",
    )
    return figure


def make_distribution_chart(
    terminal_values: np.ndarray,
    investment: float,
    ticker: str,
    symbol: str,
) -> go.Figure:
    median_value = float(np.median(terminal_values))
    figure = go.Figure(
        go.Histogram(
            x=terminal_values,
            nbinsx=40,
            marker={"color": "#4c9be8", "line": {"color": "rgba(255,255,255,0.35)", "width": 0.5}},
            opacity=0.88,
            name="Simulované hodnoty",
            hovertemplate=f"Hodnota {symbol}%{{x:,.0f}}<br>Scenáre %{{y}}<extra></extra>",
        )
    )
    figure.add_vline(
        x=investment,
        line_dash="dash",
        line_color="#d55e00",
    )
    figure.add_vline(x=median_value, line_color="#0072b2", line_width=2)
    figure.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line={"color": "#d55e00", "dash": "dash", "width": 2},
            name="Počiatočný vklad",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line={"color": "#0072b2", "width": 2},
            name="Medián",
        )
    )
    figure.update_layout(
        title=f"Konečná hodnota investície do {ticker}",
        xaxis_title="Hodnota investície",
        yaxis_title="Počet scenárov",
        legend={"orientation": "h", "y": 1.16, "x": 0, "xanchor": "left"},
        margin={"t": 100, "b": 60, "l": 65, "r": 25},
        height=400,
    )
    figure.update_xaxes(tickprefix=symbol, tickformat=",.0f", nticks=8, showgrid=False)
    figure.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return figure


def make_portfolio_distribution(
    equal_returns: np.ndarray,
    comparison_returns: np.ndarray,
    investment: float,
    symbol: str,
    comparison_label: str,
    baseline_label: str = "Rovnomerné váhy",
    chart_title: str = "Konečná hodnota portfólia",
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=investment * (1.0 + equal_returns),
            nbinsx=55,
            name=baseline_label,
            opacity=0.65,
        )
    )
    figure.add_trace(
        go.Histogram(
            x=investment * (1.0 + comparison_returns),
            nbinsx=55,
            name=comparison_label,
            opacity=0.65,
        )
    )
    figure.add_vline(
        x=investment,
        line_dash="dash",
        line_color="#d55e00",
    )
    figure.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line={"color": "#d55e00", "dash": "dash", "width": 2},
            name="Počiatočný vklad",
        )
    )
    figure.update_layout(
        title=chart_title,
        xaxis_title=f"Hodnota ({symbol})",
        yaxis_title="Počet scenárov",
        barmode="overlay",
        legend={"orientation": "h", "y": 1.14, "x": 0, "xanchor": "left"},
        margin={"t": 95, "b": 60, "l": 65, "r": 25},
        height=400,
    )
    figure.update_xaxes(tickprefix=symbol, tickformat=",.0f", nticks=8, showgrid=False)
    figure.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return figure


def _format_results_table(results: pd.DataFrame, symbol: str) -> pd.DataFrame:
    display = results.copy()
    for column in ["Priemerný zisk/strata", "Mediánový zisk/strata"]:
        display[column] = display[column].map(lambda value: format_money(float(value), symbol))
    display["Kurz do meny účtu"] = display["Kurz do meny účtu"].map(
        lambda value: f"{float(value):.6f}"
    )
    return display.round(2)


def make_pdf_report(
    parameters: dict[str, str],
    asset_results: pd.DataFrame,
    portfolio_results: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    backtest_metrics: dict[str, float] | pd.DataFrame,
    backtest_error: str | None,
) -> bytes:
    """Build a compact downloadable PDF summary using a Unicode font when available."""
    buffer = io.BytesIO()
    regular_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    font_candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
    ]
    for regular_path, bold_path in font_candidates:
        if regular_path.exists():
            try:
                pdfmetrics.registerFont(TTFont("MonteCarloUnicode", str(regular_path)))
                regular_font = "MonteCarloUnicode"
                if bold_path.exists():
                    pdfmetrics.registerFont(TTFont("MonteCarloUnicodeBold", str(bold_path)))
                    bold_font = "MonteCarloUnicodeBold"
                else:
                    bold_font = regular_font
                break
            except Exception:
                continue

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AppTitle", parent=styles["Title"], fontName=bold_font, fontSize=19, leading=23
    )
    heading_style = ParagraphStyle(
        "AppHeading", parent=styles["Heading2"], fontName=bold_font, fontSize=12, leading=15
    )
    body_style = ParagraphStyle(
        "AppBody", parent=styles["BodyText"], fontName=regular_font, fontSize=8.5, leading=12
    )
    cell_style = ParagraphStyle(
        "AppCell", parent=body_style, fontSize=7, leading=9, alignment=TA_LEFT
    )
    header_style = ParagraphStyle(
        "AppHeader", parent=cell_style, fontName=bold_font, textColor=colors.white
    )

    def safe_text(value: object) -> str:
        if pd.isna(value):
            return "—"
        text = str(value)
        if regular_font == "Helvetica":
            text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return html.escape(text).replace("\n", "<br/>")

    def report_table(frame: pd.DataFrame, widths: list[float] | None = None) -> Table:
        data = [[Paragraph(safe_text(column), header_style) for column in frame.columns]]
        for row in frame.itertuples(index=False, name=None):
            data.append([Paragraph(safe_text(value), cell_style) for value in row])
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234c70")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Monte Carlo report",
    )
    story = [
        Paragraph("Monte Carlo simulácia akcií — report", title_style),
        Spacer(1, 4 * mm),
        Paragraph("Parametre simulácie", heading_style),
    ]
    parameter_frame = pd.DataFrame(
        [(key, value) for key, value in parameters.items()],
        columns=["Parameter", "Hodnota"],
    )
    story.extend([report_table(parameter_frame, [55 * mm, 205 * mm]), Spacer(1, 4 * mm)])
    asset_columns = [
        "Ticker",
        "Priemerný výnos (%)",
        "Priemerný výnos pred nákladmi (%)",
        "VaR 95 % (%)",
        "CVaR 95 % (%)",
        "FX historický výnos (ročný, %)",
        "FX volatilita (ročná, %)",
        "Pravdepodobnosť zisku (%)",
        "Sharpe",
        "Verdikt",
    ]
    asset_frame = asset_results[[c for c in asset_columns if c in asset_results.columns]].round(2)
    story.extend([Paragraph("Výsledky akcií", heading_style), report_table(asset_frame), Spacer(1, 4 * mm)])
    story.extend(
        [Paragraph("Porovnanie portfólií", heading_style), report_table(portfolio_results.round(2)), Spacer(1, 4 * mm)]
    )
    story.extend(
        [Paragraph("Váhy portfólia", heading_style), report_table(portfolio_weights.round(2)), Spacer(1, 4 * mm)]
    )
    story.append(Paragraph("Walk-forward spätný test", heading_style))
    if isinstance(backtest_metrics, pd.DataFrame) and not backtest_metrics.empty:
        story.append(report_table(backtest_metrics.round(2)))
    elif isinstance(backtest_metrics, dict) and backtest_metrics:
        backtest_frame = pd.DataFrame(
            [
                ("Počet období", int(backtest_metrics["origins"])),
                ("Pokrytie pásma 10.–90. percentil", f"{backtest_metrics['coverage']:.1%}"),
                ("Správnosť smeru mediánu", f"{backtest_metrics['direction_accuracy']:.1%}"),
                ("Mediánová absolútna chyba", f"{backtest_metrics['median_absolute_error_pp']:.2f} p. b."),
            ],
            columns=["Metrika", "Hodnota"],
        )
        story.append(report_table(backtest_frame, [90 * mm, 65 * mm]))
    else:
        story.append(Paragraph(safe_text(backtest_error or "Spätný test nie je dostupný."), body_style))
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                "Modelové scenáre a spätný test vychádzajú z historických dát. Výsledky nie sú "
                "predpoveďou ani investičným odporúčaním; spätné obdobia sa prekrývajú a "
                "nepredstavujú nezávislé pozorovania.",
                body_style,
            ),
        ]
    )
    document.build(story)
    return buffer.getvalue()


def _render_portfolio_metrics(
    title: str,
    summary: dict[str, float | str],
    symbol: str,
    benchmark_ticker: str,
) -> None:
    st.markdown(f"#### {title}")
    st.caption(str(summary["verdict"]))
    columns = st.columns(4)
    columns[0].metric(
        "Očakávaný výnos (priemer)",
        f"{float(summary['mean_return']):+.2%}",
        format_delta_money(float(summary["mean_gain"]), symbol),
    )
    columns[1].metric(
        "Mediánový výnos",
        f"{float(summary['median_return']):+.2%}",
        format_delta_money(float(summary["median_gain"]), symbol),
    )
    columns[2].metric(
        "VaR 95 %",
        f"{float(summary['var95']):.2%}",
        format_delta_money(-float(summary["var95"]) * float(summary["investment"]), symbol),
    )
    columns[3].metric(
        "CVaR 95 %",
        f"{float(summary['cvar95']):.2%}",
        format_delta_money(-float(summary["cvar95"]) * float(summary["investment"]), symbol),
    )

    columns = st.columns(4)
    columns[0].metric("Sharpe Ratio", f"{float(summary['sharpe']):.2f}")
    columns[1].metric("Pravdepodobnosť zisku", f"{float(summary['probability_profit']):.1%}")
    beat = float(summary["probability_beat"])
    columns[2].metric(
        f"P. prekonať {benchmark_ticker}", "—" if math.isnan(beat) else f"{beat:.1%}"
    )
    columns[3].metric(
        "Pravdepodobnosť prekročenia poklesu",
        f"{float(summary['drawdown_probability']):.1%}",
    )
    st.metric(
        "Mediánová konečná hodnota",
        format_money(float(summary["median_terminal_value"]), symbol),
    )


if "tickers" not in st.session_state:
    st.session_state.tickers = ["AAPL", "NVDA", "MSFT"]

catalog = get_stock_catalog()
with st.container(border=True):
    st.caption("SCENÁRE • RIZIKO • PORTFÓLIO")
    st.title("Monte Carlo analýza akcií")
    st.write(
        "Porovnaj možné výsledky investície, riziko a modely na historických dátach. "
        "Vyber si tickery a nastav vlastné parametre simulácie."
    )

with st.container(border=True):
    header_col, count_col = st.columns([3, 1], vertical_alignment="center")
    with header_col:
        st.subheader("Sledované akcie")
    with count_col:
        st.caption(f"{len(st.session_state.tickers)} / {MAX_TICKERS} tickerov")
    st.caption(
        f"Katalóg obsahuje {len(catalog):,} veľkých a mega-cap akcií. Hľadaj názov alebo "
        "ticker, prípadne zadaj symbol ručne."
    )

    with st.expander("Vyhľadať akciu v katalógu", expanded=False):
        catalog_query = st.text_input(
            "Názov spoločnosti alebo ticker",
            placeholder="Napr. Visa alebo V",
            key="catalog_search_query",
        )
        if catalog_query.strip():
            catalog_matches = search_stock_catalog(catalog, catalog_query, limit=50)
            if catalog_matches.empty:
                st.info("V katalógu sa nič nenašlo. Skús celý názov alebo ticker Yahoo Finance.")
            else:
                selected_match = st.selectbox(
                    "Výsledky vyhľadávania",
                    catalog_matches["ticker"].tolist(),
                    format_func=lambda ticker: (
                        f"{catalog_matches.loc[catalog_matches['ticker'] == ticker, 'name'].iloc[0]} "
                        f"({ticker}) — "
                        f"{catalog_matches.loc[catalog_matches['ticker'] == ticker, 'country'].iloc[0]}"
                    ),
                )
                already_selected = selected_match in st.session_state.tickers
                if st.button(
                    "Pridať vybranú akciu",
                    key="add_catalog_ticker",
                    disabled=already_selected or len(st.session_state.tickers) >= MAX_TICKERS,
                ):
                    st.session_state.tickers, _ = add_ticker_symbols(
                        selected_match,
                        st.session_state.tickers,
                        catalog=catalog,
                        max_tickers=MAX_TICKERS,
                    )
                    st.rerun()
                if already_selected:
                    st.caption("Táto akcia už je v zozname sledovaných titulov.")
                elif len(st.session_state.tickers) >= MAX_TICKERS:
                    st.caption("Limit je 10 sledovaných tickerov. Najprv niektorý odober.")
        else:
            st.caption("Katalóg vyhľadá názov spoločnosti aj symbol, napríklad Visa alebo V.")

    st.caption("Ručne môžeš zadať viac tickerov alebo formát názov (ticker), napríklad Visa (V).")
    with st.form("add_tickers_form", clear_on_submit=True):
        ticker_input = st.text_input(
            "Tickery na pridanie",
            placeholder="Napr. AAPL, NVDA, Visa (V)",
            label_visibility="collapsed",
        )
        add_clicked = st.form_submit_button("Pridať tickery", icon=":material/add:")

    if add_clicked:
        updated, invalid_symbols = add_ticker_symbols(
            ticker_input,
            st.session_state.tickers,
            catalog=catalog,
            max_tickers=MAX_TICKERS,
        )
        st.session_state.tickers = updated
        if invalid_symbols:
            st.warning(
                "Niektoré položky neboli pridané alebo prekročili limit 10 tickerov: "
                + ", ".join(invalid_symbols)
            )
        elif not ticker_input.strip():
            st.info("Najprv zadaj ticker alebo zoznam tickerov.")

    if st.session_state.tickers:
        ticker_columns = st.columns(min(5, len(st.session_state.tickers)))
        for idx, ticker in enumerate(list(st.session_state.tickers)):
            with ticker_columns[idx % len(ticker_columns)]:
                if st.button(
                    f"Odobrať {ticker}",
                    key=f"remove_{ticker}",
                    icon=":material/close:",
                    help=f"Odobrať {ticker} zo sledovaných akcií",
                ):
                    st.session_state.tickers.remove(ticker)
                    if st.session_state.tickers:
                        st.session_state.chart_ticker = st.session_state.tickers[0]
                    st.rerun()
    else:
        st.warning("Pridaj aspoň jeden ticker, aby sa dala spustiť simulácia.")

tickers = list(st.session_state.tickers)
if tickers:
    if st.session_state.get("chart_ticker") not in tickers:
        st.session_state.chart_ticker = tickers[0]
    ticker_col, hint_col = st.columns([1, 2], vertical_alignment="bottom")
    with ticker_col:
        chart_ticker = st.selectbox(
            "Akcia zobrazená v grafe",
            tickers,
            key="chart_ticker",
        )
    with hint_col:
        st.caption("Zmena akcie v grafe vyžaduje nové spustenie simulácie.")
else:
    chart_ticker = ""

min_feasible_max_pct = (
    min(100, math.ceil(100 / len(tickers) / 5) * 5) if tickers else 10
)
max_min_weight_pct = (
    min(100, math.floor(100 / len(tickers) / 5) * 5) if tickers else 0
)
default_weight_pct = min(100, max(min_feasible_max_pct, 40))

with st.sidebar:
    st.header("Nastavenia")
    st.caption("Základné parametre simulácie")
    with st.form("simulation_settings", border=False):
        horizon_label = st.selectbox(
            "Investičný horizont",
            list(HORIZONS),
            help="Počet obchodných dní, ktoré bude model simulovať.",
        )
        n_simulations = st.selectbox(
            "Počet scenárov",
            SIMULATION_COUNTS,
            index=2,
            format_func=lambda value: f"{value:,}",
        )
        investment = st.number_input(
            "Počiatočná investícia",
            min_value=100.0,
            max_value=10_000_000.0,
            value=10_000.0,
            step=500.0,
        )
        currency = st.selectbox("Mena účtu", ["EUR", "USD"])
        benchmark_ticker = st.text_input("Benchmark", value="SPY").strip().upper()
        model_label = st.selectbox("Model scenárov", list(MODELS))
        run_clicked = st.form_submit_button(
            "Spustiť simuláciu",
            type="primary",
            icon=":material/monitoring:",
            width="stretch",
        )

    with st.expander("Pokročilé nastavenia", expanded=False):
        risk_free_pct = st.slider(
            "Bezriziková sadzba pre Sharpe (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.0,
            step=0.25,
        )
        drift_shrinkage_pct = st.slider(
            "Zmenšenie driftu k bezrizikovej sadzbe (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            help="Znižuje vplyv hlučného historického priemeru na očakávaný výnos.",
        )
        parameter_uncertainty_pct = st.slider(
            "Neistota odhadu driftu (%)",
            min_value=0,
            max_value=200,
            value=100,
            step=25,
            help="Zohľadňuje, že drift odhadujeme z konečnej historickej vzorky.",
        )
        max_drawdown_limit = st.selectbox(
            "Prijateľný maximálny pokles",
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            index=2,
            format_func=lambda value: f"{value:.0%}",
        )
        if len(tickers) > 1:
            max_weight_pct = st.slider(
                "Maximálna váha jednej akcie",
                min_value=min_feasible_max_pct,
                max_value=100,
                value=default_weight_pct,
                step=5,
            )
            min_weight_pct = st.slider(
                "Minimálna váha jednej akcie",
                min_value=0,
                max_value=max_min_weight_pct,
                value=0,
                step=5,
            )
        else:
            max_weight_pct = 100
            min_weight_pct = 0
            st.caption("Pri jednej akcii je jej portfóliová váha automaticky 100 %.")
        trade_cost_bps = st.number_input(
            "Poplatok za obchod (bps za jeden smer)",
            min_value=0,
            max_value=1_000,
            value=10,
            step=5,
            help="Uplatní sa raz pri nákupe a raz pri predaji.",
        )
        annual_fee_pct = st.slider(
            "Ročný priebežný poplatok (%)",
            min_value=0.0,
            max_value=3.0,
            value=0.2,
            step=0.1,
        )
        spread_slippage_bps = st.number_input(
            "Spread a sklz (bps za jeden smer)", min_value=0, max_value=500,
            value=5, step=1,
            help="Hrubý odhad dodatočného nákladu pri nákupe aj predaji; nie je to živý spread.",
        )
        annual_turnover_pct = st.slider(
            "Odhadovaný obrat pri ročnom rebalansovaní (%)",
            min_value=0, max_value=200, value=0, step=10,
            help="Nákladový proxy: percento portfólia obchodované za rok. 0 % znamená buy-and-hold.",
        )
        tax_pct = st.slider(
            "Zjednodušená daň zo zisku pri predaji (%)",
            min_value=0.0, max_value=50.0, value=0.0, step=1.0,
            help="Uplatní sa len na kladný konečný zisk. Skutočné zdanenie závisí od krajiny a situácie.",
        )
        stress_shock_pct = st.slider(
            "Jednorazový stresový šok (%)", 0, 60, 20, 5
        )
    st.caption("Zmenené pokročilé parametre sa použijú pri ďalšom spustení.")

currency_symbol = CURRENCY_SYMBOLS[currency]
current_signature = (
    ANALYSIS_SCHEMA_VERSION,
    tuple(tickers),
    benchmark_ticker,
    HORIZONS[horizon_label],
    n_simulations,
    float(investment),
    currency,
    float(risk_free_pct / 100.0),
    float(drift_shrinkage_pct / 100.0),
    float(parameter_uncertainty_pct / 100.0),
    float(max_drawdown_limit),
    float(max_weight_pct / 100.0),
    float(min_weight_pct / 100.0),
    float(trade_cost_bps),
    float(annual_fee_pct / 100.0),
    float(spread_slippage_bps),
    float(annual_turnover_pct),
    float(tax_pct / 100.0),
    MODELS[model_label],
    float(stress_shock_pct / 100.0),
    chart_ticker,
)

if run_clicked:
    if not tickers:
        st.error("Pridaj aspoň jeden ticker.")
    elif not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", benchmark_ticker):
        st.error("Benchmark musí byť jeden platný symbol Yahoo Finance, napríklad SPY.")
    elif min_weight_pct > max_weight_pct:
        st.error("Minimálna váha akcie nesmie byť vyššia ako maximálna váha.")
    else:
        all_tickers = list(dict.fromkeys(tickers + [benchmark_ticker]))
        try:
            with st.spinner("Sťahujem ceny a menové kurzy, počítam scenáre…"):
                market_prices, quote_currencies, spot_rates, fx_stats = download_market_prices(
                    tuple(all_tickers), currency
                )
                historical_portfolio_returns = market_prices.loc[:, tickers].pct_change(
                    fill_method=None
                ).dropna(how="any")
                annual_asset_means = historical_portfolio_returns.mean().to_numpy() * TRADING_DAYS_PER_YEAR
                annual_asset_vols = historical_portfolio_returns.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS_PER_YEAR)
                historical_corr = historical_portfolio_returns.corr().to_numpy()
                historical_stress = find_worst_historical_block(market_prices, tickers)
                portfolio_weights, optimization_scores = (
                    optimize_portfolio_weights(
                        market_prices.loc[:, tickers],
                        float(risk_free_pct / 100.0),
                        float(max_weight_pct / 100.0),
                        float(min_weight_pct / 100.0),
                    )
                )
                horizon_days = HORIZONS[horizon_label]
                effective_trade_bps = float(trade_cost_bps + spread_slippage_bps)
                trade_cost_rate = effective_trade_bps / 10_000.0
                annual_fee_rate = float(annual_fee_pct / 100.0)
                rebalance_cost_rate = (effective_trade_bps / 10_000.0) * (annual_turnover_pct / 100.0)
                annual_fee_rate += rebalance_cost_rate
                simulation = simulate(
                    market_prices,
                    horizon_days,
                    n_simulations,
                    chart_ticker,
                    tickers,
                    portfolio_weights,
                    MODELS[model_label],
                    float(stress_shock_pct / 100.0),
                    trade_cost_rate,
                    annual_fee_rate,
                    float(risk_free_pct / 100.0),
                    float(drift_shrinkage_pct / 100.0),
                    float(parameter_uncertainty_pct / 100.0),
                    historical_stress,
                )
                result_table = make_results(
                    simulation,
                    tickers,
                    benchmark_ticker,
                    float(investment),
                    float(risk_free_pct / 100.0),
                    float(max_drawdown_limit),
                    quote_currencies,
                    spot_rates,
                    fx_stats,
                    horizon_days,
                    float(tax_pct / 100.0),
                )
                backtest_cost_factor = calculate_net_cost_factor(
                    effective_trade_bps,
                    annual_fee_rate,
                    BACKTEST_HORIZON_DAYS,
                )
                try:
                    backtest_table, backtest_metrics = run_walk_forward_model_comparison(
                        market_prices,
                        chart_ticker,
                        backtest_cost_factor,
                        float(risk_free_pct / 100.0),
                        float(drift_shrinkage_pct / 100.0),
                        float(parameter_uncertainty_pct / 100.0),
                    )
                    backtest_error = None
                except RuntimeError as exc:
                    backtest_table = pd.DataFrame()
                    backtest_metrics = pd.DataFrame()
                    backtest_error = str(exc)
                st.session_state.analysis = {
                    "signature": current_signature,
                    "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "data_sha256": hashlib.sha256(
                        pd.util.hash_pandas_object(market_prices, index=True).values.tobytes()
                    ).hexdigest(),
                    "simulation": simulation,
                    "results": result_table,
                    "prices_as_of": market_prices.index[-1].strftime("%Y-%m-%d"),
                    "prices_from": market_prices.index[0].strftime("%Y-%m-%d"),
                    "price_observations": int(len(market_prices)),
                    "price_tickers": tuple(all_tickers),
                    "price_data_quality": market_prices.attrs.get("data_quality", {}),
                    "quote_currencies": quote_currencies,
                    "spot_rates": spot_rates,
                    "fx_stats": fx_stats,
                    "historical_stress": historical_stress,
                    "portfolio_weights": portfolio_weights,
                    "optimization_scores": optimization_scores,
                    "annual_asset_means": annual_asset_means,
                    "annual_asset_vols": annual_asset_vols,
                    "historical_correlation": historical_corr,
                    "backtest": backtest_table,
                    "backtest_metrics": backtest_metrics,
                    "backtest_error": backtest_error,
                    "trade_cost_bps": trade_cost_bps,
                    "spread_slippage_bps": spread_slippage_bps,
                    "annual_turnover_pct": annual_turnover_pct,
                    "tax_pct": tax_pct,
                    "effective_trade_bps": effective_trade_bps,
                    "annual_fee_pct": annual_fee_pct,
                    "drift_shrinkage_pct": drift_shrinkage_pct,
                    "parameter_uncertainty_pct": parameter_uncertainty_pct,
                }
        except Exception as exc:
            st.session_state.pop("analysis", None)
            st.error(f"Simuláciu sa nepodarilo dokončiť: {exc}")


analysis = st.session_state.get("analysis")
if analysis and analysis["signature"] != current_signature:
    st.info("Nastavenia sa zmenili. Klikni na **Spustiť simuláciu**, aby sa prepočítali výsledky.")
elif analysis:
    simulation = analysis["simulation"]
    results = analysis["results"]
    selected_idx = simulation["tickers"].index(chart_ticker)
    selected_row = results.loc[results["Ticker"] == chart_ticker].iloc[0]
    terminal_prices = simulation["terminal_prices"][:, selected_idx]
    start_price = float(simulation["start_prices"][selected_idx])
    gross_simulated_returns = terminal_prices / start_price - 1.0
    cost_factor = float(simulation["net_cost_factor"])
    simulated_returns = apply_terminal_tax(
        (1.0 + gross_simulated_returns) * cost_factor - 1.0,
        float(analysis.get("tax_pct", 0.0)) / 100.0,
    )
    terminal_values = float(investment) * (1.0 + simulated_returns)
    estimated_cost = float(investment) * (1.0 - cost_factor)
    var95_loss = float(selected_row["VaR 95 % (%)"]) / 100.0
    median_return = float(selected_row["Mediánový výnos (%)"]) / 100.0
    profit_probability = float(selected_row["Pravdepodobnosť zisku (%)"]) / 100.0
    sharpe = float(selected_row["Sharpe"])
    dd_probability = float(selected_row["P. prekročenia poklesu (%)"]) / 100.0
    label, level, verdict_checks = explain_verdict(
        median_return,
        profit_probability,
        sharpe,
        dd_probability,
        var95_loss,
        float(max_drawdown_limit),
    )
    st.subheader(f"Výsledky pre {chart_ticker}")
    model_description = next(
        label for label, model_key in MODELS.items() if model_key == simulation["model"]
    )
    st.caption(
        f"{HORIZONS[horizon_label]:,} obchodných dní · {n_simulations:,} scenárov · "
        f"{model_description} · {currency} · dáta do {analysis['prices_as_of']}"
    )
    mean_return = float(selected_row["Priemerný výnos (%)"]) / 100.0
    mean_gain = float(selected_row["Priemerný zisk/strata"])
    median_gain = float(selected_row["Mediánový zisk/strata"])
    beat_value = float(selected_row[f"P. prekonať {benchmark_ticker} (%)"]) / 100.0
    with st.container(border=True):
        if level == "success":
            st.success(f"**{label}** — ticker splnil orientačné výnosové a rizikové prahy.")
        elif level == "error":
            st.error(f"**{label}** — výsledok prekračuje aspoň jeden nastavený rizikový limit.")
        else:
            st.warning(f"**{label}** — priaznivé ani vysokorizikové prahy nie sú splnené.")
        st.caption("Skríningový signál podľa zvolených prahov, nie investičné odporúčanie.")

    kpi = st.columns(5)
    kpi[0].metric(
        "Výnos — medián",
        f"{median_return:+.1%}",
        format_delta_money(median_gain, currency_symbol),
        help="Polovica scenárov skončila nad touto hodnotou a polovica pod ňou.",
    )
    kpi[1].metric(
        "Výnos — priemer",
        f"{mean_return:+.1%}",
        format_delta_money(mean_gain, currency_symbol),
        help="Priemerný výsledok všetkých scenárov po zadaných poplatkoch.",
    )
    kpi[2].metric(
        "VaR 95 %",
        f"{var95_loss:.1%}",
        format_delta_money(-var95_loss * investment, currency_symbol),
        help="Strata na 5. percentile konečného výnosu. Nie je to najväčší pokles počas cesty.",
    )
    kpi[3].metric(
        "Sharpe ratio",
        f"{sharpe:.2f}",
        help="Odhad výnosu nad bezrizikovú sadzbu na jednotku volatility.",
    )
    kpi[4].metric(
        "Pravdepodobnosť zisku",
        f"{profit_probability:.1%}",
        help="Podiel simulovaných scenárov s kladným čistým výnosom.",
    )

    with st.expander("Ďalšie metriky a náklady"):
        extra = st.columns(3)
        extra[0].metric(
            "CVaR 95 %",
            f"{float(selected_row['CVaR 95 % (%)']) / 100.0:.1%}",
            help="Priemerná strata v najhorších 5 % simulovaných scenárov.",
        )
        extra[1].metric(
            f"Pravdepodobnosť prekonať {benchmark_ticker}",
            "—" if math.isnan(beat_value) else f"{beat_value:.1%}",
        )
        extra[2].metric(
            "Pravdepodobnosť prekročiť limit poklesu",
            f"{dd_probability:.1%}",
            f"Limit {max_drawdown_limit:.0%}",
            delta_color="inverse",
            help="Pravdepodobnosť, že maximálny prepad od predchádzajúceho vrcholu prekročí limit.",
        )
        st.caption(
            f"Výsledky zohľadňujú {analysis['trade_cost_bps']} bps provízie + "
            f"{analysis['spread_slippage_bps']} bps odhad spreadu/sklzu za smer a "
            f"ročný poplatok {analysis['annual_fee_pct']:.1f} %. Pri nulovom pohybe cien "
            f"by odhadované náklady predstavovali približne {format_money(estimated_cost, currency_symbol)}. "
            f"Drift je zmenšený o {analysis['drift_shrinkage_pct']:.0f} % k bezrizikovej sadzbe; "
            f"neistota jeho odhadu je {analysis['parameter_uncertainty_pct']:.0f} %."
        )
    with st.expander("Prečo vyšlo toto hodnotenie?"):
        st.markdown("**„Splnené priaznivé prahy“ je iba orientačný filter modelu, nie pokyn na nákup.**")
        for check in verdict_checks:
            st.write(f"- {check}")
        st.caption(
            "Model nepozná fundamenty, budúce správy ani tvoju finančnú situáciu. Rozhodnutie treba "
            "overiť aj mimo tejto simulácie."
        )

    st.subheader("Citlivosť na predpoklady")
    sensitivity = make_sensitivity_table(
        gross_simulated_returns,
        cost_factor,
        float(investment),
        HORIZONS[horizon_label],
        float(analysis.get("tax_pct", 0.0)) / 100.0,
    )
    st.dataframe(sensitivity.round(2), width="stretch", hide_index=True)
    st.caption(
        "Ide o približnú citlivosť konečných výnosov: mení drift, rozptyl výnosov a náklady na "
        "spready/sklz. Nejde o nové plné simulácie; neistotu korelácií sleduj najmä v porovnaní "
        "portfóliových stratégií a stresových scenároch."
    )

    with st.expander("Zdroj a kvalita dát"):
        data_quality = analysis.get("price_data_quality", {})
        data_cols = st.columns(3)
        data_cols[0].metric("Spoločné denné pozorovania", f"{analysis['price_observations']:,}")
        data_cols[1].metric("Začiatok dát", analysis["prices_from"])
        data_cols[2].metric("Posledný dostupný deň", analysis["prices_as_of"])
        st.caption(f"Dáta stiahnuté: {analysis['retrieved_at']} · identifikátor dát SHA-256: {analysis['data_sha256'][:16]}…")
        latest_data_day = pd.Timestamp(analysis["prices_as_of"]).date()
        local_today = datetime.now().date()
        weekday_lag = max(0, len(pd.bdate_range(latest_data_day, local_today)) - 1)
        if weekday_lag > 5:
            st.warning(
                f"Posledný cenový deň je pred {weekday_lag} pracovnými dňami. Skontroluj dostupnosť "
                "zdroja alebo ticker; sviatky a časové pásma môžu dátum ovplyvniť."
            )
        st.markdown(
            "- **Ceny a meny:** upravené denné záverečné ceny a historické FX kurzy z Yahoo Finance.\n"
            "- **Katalóg spoločností:** lokálna snímka FinanceDatabase z 18. 11. 2026; nemení sa automaticky.\n"
            "- **Úplnosť:** simulácia používa iba spoločné dátumy s použiteľnými cenami; chýbajúce alebo "
            "neprekrývajúce sa cenové pozorovania sa pri zosúladení vynechajú. Dostupné FX medzery sa "
            "dopĺňajú posledným známym kurzom.\n"
            "- **Obnova cien:** odpovede Yahoo Finance sú v cache najviac 30 minút. Dátum posledného "
            "pozorovania je uvedený vyššie."
        )
        if data_quality:
            source_rows = int(data_quality.get("source_rows", 0))
            incomplete_dates = int(data_quality.get("incomplete_dates", 0))
            duplicate_dates = int(data_quality.get("duplicate_dates", 0))
            if source_rows:
                st.caption(
                    f"Zdroj vrátil {source_rows:,} kalendárnych dátumov; "
                    f"{incomplete_dates:,} dátumov s neúplnými cenami a "
                    f"{duplicate_dates:,} duplicitných dátumov sa vynechalo alebo zlúčilo. "
                    "Neúplné dátumy zahŕňajú aj rozdielne sviatky a obchodné kalendáre búrz."
                )
            missing_prices = {
                ticker: int(count)
                for ticker, count in data_quality.get("missing_by_ticker", {}).items()
                if int(count) > 0
            }
            if source_rows:
                coverage_rows = []
                for ticker, count in data_quality.get("missing_by_ticker", {}).items():
                    count = int(count)
                    coverage_rows.append(
                        {
                            "Ticker": ticker,
                            "Chýbajúce cenové dni": count,
                            "Použiteľné zdrojové dni": max(0, source_rows - count),
                            "Pokrytie zdrojových dát (%)": (
                                100.0 * (source_rows - count) / source_rows if source_rows else float("nan")
                            ),
                        }
                    )
                st.caption("Úplnosť cien pred zosúladením; rozdielne sviatky búrz môžu vyzerať ako chýbajúce dni.")
                st.dataframe(
                    pd.DataFrame(coverage_rows).round(1),
                    width="stretch",
                    hide_index=True,
                )
                if any((source_rows - int(count)) / source_rows < 0.90 for count in data_quality.get("missing_by_ticker", {}).values()):
                    st.warning("Aspoň jeden ticker má pod 90 % pokrytie zdrojových dát; over jeho históriu a obchodný kalendár.")
            invalid_prices = {
                ticker: int(count)
                for ticker, count in data_quality.get("invalid_by_ticker", {}).items()
                if int(count) > 0
            }
            if invalid_prices:
                st.warning(
                    "Nulové alebo neplatné ceny sa vylúčili: "
                    + ", ".join(f"{ticker}: {count}" for ticker, count in invalid_prices.items())
                )
            fx_filled = {
                currency_code: int(count)
                for currency_code, count in data_quality.get(
                    "fx_forward_filled_by_currency", {}
                ).items()
                if int(count) > 0
            }
            if fx_filled:
                st.caption(
                    "Počet doplnených chýbajúcich denných FX kurzov podľa meny: "
                    + ", ".join(f"{currency_code}: {count}" for currency_code, count in fx_filled.items())
                )

    validation_metrics = analysis["backtest_metrics"]
    with st.expander("Ako obstál model v spätnom teste a kde môže zlyhať"):
        if not validation_metrics.empty:
            selected_validation = validation_metrics.loc[
                validation_metrics["Model"] == model_description
            ]
            if not selected_validation.empty:
                validation_row = selected_validation.iloc[0]
                validation_cols = st.columns(4)
                validation_cols[0].metric(
                    "Pokrytie pásma 10.–90.",
                    f"{validation_row['Pokrytie 10.–90. (%)']:.1f}%",
                    "cieľ približne 80 %",
                )
                validation_cols[1].metric(
                    "Správny smer",
                    f"{validation_row['Správny smer mediánu (%)']:.1f}%",
                )
                validation_cols[2].metric(
                    "Mediánová chyba",
                    f"{validation_row['Mediánová absolútna chyba (p. b.)']:.2f} p. b.",
                )
                validation_cols[3].metric(
                    "Počet období",
                    f"{int(validation_row['Počet období'])}",
                )
                st.caption(
                    "Spätný test používa predchádzajúce tri roky na odhad a kontroluje ďalších "
                    f"{BACKTEST_HORIZON_DAYS} obchodných dní. Odhady sa posúvajú o "
                    f"{BACKTEST_STEP_DAYS} dní, preto sa testovacie obdobia prekrývajú. "
                    "Nižšie pokrytie môže znamenať príliš úzke pásmo; vysoké pokrytie môže vzniknúť "
                    "aj pri veľmi širokom pásme. Samotné pokrytie preto neznamená presnú predpoveď."
                )
        else:
            st.warning(analysis["backtest_error"] or "Pre tento ticker sa spätný test nepodarilo vypočítať.")
        st.markdown(
            "Model nepozná budúce výsledky firiem, správy, zmeny regulácie ani náhle zmeny "
            "trhového režimu. Päťročná história nemusí obsahovať relevantnú krízu a katalóg "
            "zameraný na aktívne spoločnosti môže vytvárať survivorship bias. FX kurzy, "
            "poplatky a kvalita externého zdroja tiež môžu ovplyvniť výstup."
        )

    portfolio_tickers = list(simulation["portfolio_tickers"])
    if chart_ticker in portfolio_tickers and len(portfolio_tickers) > 1:
        without_ticker = [ticker for ticker in portfolio_tickers if ticker != chart_ticker]
        asset_index = [simulation["tickers"].index(ticker) for ticker in portfolio_tickers]
        asset_returns = (
            simulation["terminal_prices"][:, asset_index]
            / simulation["start_prices"][asset_index]
            - 1.0
        )
        net_returns_with = (1.0 + asset_returns.mean(axis=1)) * cost_factor - 1.0
        without_indices = [portfolio_tickers.index(ticker) for ticker in without_ticker]
        net_returns_without = (
            1.0 + asset_returns[:, without_indices].mean(axis=1)
        ) * cost_factor - 1.0
        without_var, without_cvar = _tail_metrics(net_returns_without)
        with_var, with_cvar = _tail_metrics(net_returns_with)
        comparison_rows = [
            {
                "Ukazovateľ": "Priemerný výnos",
                f"Bez {chart_ticker}": f"{np.mean(net_returns_without):+.1%}",
                f"S {chart_ticker}": f"{np.mean(net_returns_with):+.1%}",
                "Zmena po pridaní": f"{np.mean(net_returns_with) - np.mean(net_returns_without):+.1%}",
            },
            {
                "Ukazovateľ": "Mediánový výnos",
                f"Bez {chart_ticker}": f"{np.median(net_returns_without):+.1%}",
                f"S {chart_ticker}": f"{np.median(net_returns_with):+.1%}",
                "Zmena po pridaní": f"{np.median(net_returns_with) - np.median(net_returns_without):+.1%}",
            },
            {
                "Ukazovateľ": "VaR 95 %",
                f"Bez {chart_ticker}": f"{without_var:.1%}",
                f"S {chart_ticker}": f"{with_var:.1%}",
                "Zmena po pridaní": f"{with_var - without_var:+.1%}",
            },
            {
                "Ukazovateľ": "CVaR 95 %",
                f"Bez {chart_ticker}": f"{without_cvar:.1%}",
                f"S {chart_ticker}": f"{with_cvar:.1%}",
                "Zmena po pridaní": f"{with_cvar - without_cvar:+.1%}",
            },
            {
                "Ukazovateľ": "Pravdepodobnosť zisku",
                f"Bez {chart_ticker}": f"{np.mean(net_returns_without > 0):.1%}",
                f"S {chart_ticker}": f"{np.mean(net_returns_with > 0):.1%}",
                "Zmena po pridaní": (
                    f"{np.mean(net_returns_with > 0) - np.mean(net_returns_without > 0):+.1%}"
                ),
            },
            {
                "Ukazovateľ": "Priemerná konečná hodnota",
                f"Bez {chart_ticker}": format_money(
                    investment * (1.0 + float(np.mean(net_returns_without))), currency_symbol
                ),
                f"S {chart_ticker}": format_money(
                    investment * (1.0 + float(np.mean(net_returns_with))), currency_symbol
                ),
                "Zmena po pridaní": format_money(
                    investment * float(np.mean(net_returns_with) - np.mean(net_returns_without)),
                    currency_symbol,
                ),
            },
        ]
        with st.container(border=True):
            st.subheader(f"Čo sa zmení po pridaní {chart_ticker}?")
            st.caption(
                "Porovnávam rovnomerne vážené portfólio bez vybraného tickeru a s ním "
                "v tých istých simulovaných scenároch a po rovnakých odhadovaných nákladoch. "
                "Je to porovnanie váh, nie garancia ani optimalizovaná alokácia."
            )
            if with_cvar < without_cvar:
                st.success(f"V tomto modeli pridanie {chart_ticker} znížilo priemernú stratu v najhorších 5 % scenárov.")
            elif with_cvar > without_cvar:
                st.warning(f"V tomto modeli pridanie {chart_ticker} zvýšilo priemernú stratu v najhorších 5 % scenárov.")
            else:
                st.info("Pridanie akcie v tomto modeli nezmenilo CVaR 95 % portfólia.")
            st.caption("Porovnanie platí len pre zvolený model, historické dáta a rovnaké váhy; výsledok sa môže zmeniť pri inom modeli.")
            st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
            st.plotly_chart(
                make_portfolio_distribution(
                    net_returns_without,
                    net_returns_with,
                    float(investment),
                    currency_symbol,
                    comparison_label=f"S {chart_ticker}",
                    baseline_label=f"Bez {chart_ticker}",
                    chart_title=f"Rozdelenie hodnoty portfólia bez a s {chart_ticker}",
                ),
                width="stretch",
                config={"displaylogo": False},
            )

    assets_tab, portfolio_tab, stress_tab, backtest_tab, export_tab = st.tabs(
        ["Akcie", "Portfólio", "Stresový scenár", "Backtest", "Export"]
    )
    with assets_tab:
        chart_tab, comparison_tab = st.tabs(["Vývoj a rozdelenie", "Porovnanie tickerov"])
        with chart_tab:
            st.plotly_chart(
                make_fan_chart(
                    simulation["chart_paths"],
                    chart_ticker,
                    HORIZONS[horizon_label],
                    currency_symbol,
                ),
                width="stretch",
                config={"displaylogo": False},
            )
            st.caption(
                f"Sivé čiary sú {min(FAN_CHART_PATH_LIMIT, len(simulation['chart_paths'])):,} "
                "ukážkové simulované cesty. Prerušované čiary označujú 10. a 90. percentil; "
                "modrá čiara je medián (50. percentil) vypočítaný zo všetkých simulácií. "
                "Ide o scenáre, nie garantované cenové hranice. Fan chart zobrazuje ceny pred poplatkami."
            )
            st.plotly_chart(
                make_distribution_chart(
                    terminal_values,
                    float(investment),
                    chart_ticker,
                    currency_symbol,
                ),
                width="stretch",
                config={"displaylogo": False},
            )
        with comparison_tab:
            st.dataframe(
                _format_results_table(results, currency_symbol),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Kurz v tabuľke vyjadruje hodnotu jednej jednotky kótovanej meny v mene účtu "
                "podľa posledného dostupného historického kurzu."
            )

    with portfolio_tab:
        portfolio_tickers = simulation["portfolio_tickers"]
        portfolio_weights = simulation["portfolio_weights"]
        portfolio_summaries: dict[str, dict[str, float | str]] = {}
        for strategy, weights in portfolio_weights.items():
            portfolio_summaries[strategy] = make_portfolio_summary(
                simulation,
                weights,
                float(investment),
                float(risk_free_pct / 100.0),
                HORIZONS[horizon_label],
                float(max_drawdown_limit),
                benchmark_ticker,
                portfolio_kind=strategy,
                tax_rate=float(analysis.get("tax_pct", 0.0)) / 100.0,
            )
        portfolio_table = pd.DataFrame(
            [
                {
                    "Portfólio": PORTFOLIO_STRATEGIES[strategy],
                    "Priemerný výnos (%)": float(summary["mean_return"]) * 100,
                    "Priemerný zisk/strata": float(summary["mean_gain"]),
                    "Mediánový výnos (%)": float(summary["median_return"]) * 100,
                    "VaR 95 % (%)": float(summary["var95"]) * 100,
                    "CVaR 95 % (%)": float(summary["cvar95"]) * 100,
                    "Sharpe": float(summary["sharpe"]),
                    "Pravdepodobnosť zisku (%)": float(summary["probability_profit"]) * 100,
                    f"P. prekonať {benchmark_ticker} (%)": float(summary["probability_beat"]) * 100,
                    "P. prekročenia poklesu (%)": float(summary["drawdown_probability"]) * 100,
                    "Verdikt": summary["verdict"],
                }
                for strategy, summary in portfolio_summaries.items()
            ]
        )
        st.info(
            f"Porovnávam 4 long-only stratégie z {MAX_PORTFOLIO_CANDIDATES:,} spoločných "
            f"kandidátov. Limit váhy: {min_weight_pct}% až {max_weight_pct}% na ticker. "
            f"Historické metriky: max Sharpe {analysis['optimization_scores']['historical_sharpe']:.2f}, "
            f"min volatilita {analysis['optimization_scores']['minimum_volatility']:.2%}, "
            f"min denný CVaR {analysis['optimization_scores']['minimum_daily_cvar']:.2%}. "
            "Váhy sú približné a nie sú zárukou budúcich výsledkov."
        )
        st.caption(
            f"Odhad nákladov obsahuje províziu a spread/sklz za obchod; ročný rebalansovací "
            f"obrat {analysis['annual_turnover_pct']} % sa účtuje ako približný nákladový proxy. "
            "Aplikácia nevykonáva skutočné obchody ani neoveruje živé objemy a likviditu titulov."
        )
        st.dataframe(portfolio_table.round(2), width="stretch", hide_index=True)
        weights_table = pd.DataFrame(
            {
                "Ticker": portfolio_tickers,
                "Mena kótovania": [analysis["quote_currencies"][t] for t in portfolio_tickers],
            }
        )
        for strategy, weights in portfolio_weights.items():
            weights_table[f"{PORTFOLIO_STRATEGIES[strategy]} (%)"] = weights * 100
        st.dataframe(weights_table.round(2), width="stretch", hide_index=True)
        corr_strategy = st.selectbox(
            "Korelácia použitá v citlivostnom prehľade",
            list(PORTFOLIO_STRATEGIES),
            format_func=lambda key: PORTFOLIO_STRATEGIES[key],
            key="correlation_sensitivity_strategy",
        )
        corr_sensitivity = make_correlation_sensitivity(
            analysis["annual_asset_means"],
            analysis["annual_asset_vols"],
            analysis["historical_correlation"],
            portfolio_weights[corr_strategy],
            float(risk_free_pct / 100.0),
        )
        st.caption("Citlivosť portfóliového rizika pri rôznej sile historických korelácií (kovariančný odhad, nie nová simulácia).")
        st.dataframe(corr_sensitivity.round(2), width="stretch", hide_index=True)
        portfolio_indices = [simulation["tickers"].index(t) for t in portfolio_tickers]
        gross_asset_return_matrix = (
            simulation["terminal_prices"][:, portfolio_indices]
            / simulation["start_prices"][portfolio_indices]
            - 1.0
        )
        net_asset_return_matrix = apply_terminal_tax(
            (1.0 + gross_asset_return_matrix) * cost_factor - 1.0,
            float(analysis.get("tax_pct", 0.0)) / 100.0,
        )
        comparison_strategy = st.selectbox(
            "Porovnať rozdelenie s rovnomerným portfóliom",
            list(PORTFOLIO_STRATEGIES)[1:],
            format_func=lambda key: PORTFOLIO_STRATEGIES[key],
        )
        st.plotly_chart(
            make_portfolio_distribution(
                net_asset_return_matrix @ portfolio_weights["equal"],
                net_asset_return_matrix @ portfolio_weights[comparison_strategy],
                float(investment),
                currency_symbol,
                PORTFOLIO_STRATEGIES[comparison_strategy],
            ),
            width="stretch",
            config={"displaylogo": False},
        )

    with stress_tab:
        shock = float(stress_shock_pct / 100.0)
        st.info(
            f"Stresový scenár vloží jednorazový pokles {shock:.0%} na prvý deň horizontu "
            f"a potom pokračuje modelom {model_description}. Je to jednoduchý "
            "hypotetický šok, nie konkrétna historická kríza."
        )
        stressed_asset_returns = apply_terminal_tax(
            (1.0 - shock) * (1.0 + gross_simulated_returns) * cost_factor - 1.0,
            float(analysis.get("tax_pct", 0.0)) / 100.0,
        )
        stressed_var, stressed_cvar = _tail_metrics(stressed_asset_returns)
        stress_cols = st.columns(4)
        stress_cols[0].metric("Stresový mediánový výnos", f"{np.median(stressed_asset_returns):+.2%}")
        stress_cols[1].metric("Stresový VaR 95 %", f"{stressed_var:.2%}")
        stress_cols[2].metric("Stresový CVaR 95 %", f"{stressed_cvar:.2%}")
        stress_cols[3].metric(
            "Stresová pravdepodobnosť zisku", f"{np.mean(stressed_asset_returns > 0):.1%}"
        )
        stress_table = results[
            [
                "Ticker",
                "Stresový medián výnosu (%)",
                "Stresový VaR 95 % (%)",
                "Stresový CVaR 95 % (%)",
                "Stresová p. prekročenia poklesu (%)",
            ]
        ].copy()
        st.dataframe(stress_table.round(2), width="stretch", hide_index=True)
        stress_summaries = {
            strategy: make_portfolio_summary(
                simulation,
                weights,
                float(investment),
                float(risk_free_pct / 100.0),
                HORIZONS[horizon_label],
                float(max_drawdown_limit),
                benchmark_ticker,
                stress=True,
                portfolio_kind=strategy,
                tax_rate=float(analysis.get("tax_pct", 0.0)) / 100.0,
            )
            for strategy, weights in portfolio_weights.items()
        }
        stress_portfolio_table = pd.DataFrame(
            [
                {
                    "Portfólio": PORTFOLIO_STRATEGIES[strategy],
                    "Priemerný výnos (%)": float(summary["mean_return"]) * 100,
                    "Mediánový výnos (%)": float(summary["median_return"]) * 100,
                    "VaR 95 % (%)": float(summary["var95"]) * 100,
                    "CVaR 95 % (%)": float(summary["cvar95"]) * 100,
                    "Sharpe": float(summary["sharpe"]),
                    "P. prekročenia poklesu (%)": float(summary["drawdown_probability"]) * 100,
                    "Verdikt": summary["verdict"],
                }
                for strategy, summary in stress_summaries.items()
            ]
        )
        st.subheader("Portfóliá po stresovom šoku")
        st.dataframe(stress_portfolio_table.round(2), width="stretch", hide_index=True)

        historical_stress = analysis["historical_stress"]
        st.subheader("Historický najhorší 3-mesačný scenár")
        st.info(
            f"Prehrávam skutočné denné výnosy od {historical_stress['start_date']} do "
            f"{historical_stress['end_date']}. Rovnomerné portfólio sledovaných tickerov "
            f"vtedy stratilo {historical_stress['equal_weight_return']:.1%}. "
            f"Historický blok zaberá prvých {simulation['historical_stress_days']} dní; "
            "zvyšok horizontu pokračuje zvoleným Monte Carlo modelom."
        )
        historical_asset_returns = (
            simulation["historical_stress_terminal_relative"][:, selected_idx] - 1.0
        )
        historical_asset_var, historical_asset_cvar = _tail_metrics(historical_asset_returns)
        historical_drawdowns = simulation["historical_stress_drawdowns"][:, selected_idx]
        historical_cols = st.columns(4)
        historical_cols[0].metric("Medián po strese", f"{np.median(historical_asset_returns):+.2%}")
        historical_cols[1].metric("VaR 95 %", f"{historical_asset_var:.2%}")
        historical_cols[2].metric("CVaR 95 %", f"{historical_asset_cvar:.2%}")
        historical_cols[3].metric(
            "P. prekročenia poklesu",
            f"{np.mean(historical_drawdowns < -max_drawdown_limit):.1%}",
        )
        stress_indices = [
            simulation["tickers"].index(ticker)
            for ticker in simulation["portfolio_tickers"]
        ]
        historical_portfolio_rows = []
        for strategy, weights in simulation["portfolio_weights"].items():
            stressed_asset_values = simulation["historical_stress_terminal_relative"][:, stress_indices]
            portfolio_returns = stressed_asset_values @ weights - 1.0
            hist_var, hist_cvar = _tail_metrics(portfolio_returns)
            drawdowns = simulation["historical_stress_portfolio_drawdowns"][strategy]
            historical_portfolio_rows.append(
                {
                    "Portfólio": PORTFOLIO_STRATEGIES[strategy],
                    "Mediánový výnos (%)": float(np.median(portfolio_returns) * 100),
                    "VaR 95 % (%)": hist_var * 100,
                    "CVaR 95 % (%)": hist_cvar * 100,
                    "P. prekročenia poklesu (%)": float(
                        np.mean(drawdowns < -max_drawdown_limit) * 100
                    ),
                }
            )
        st.dataframe(pd.DataFrame(historical_portfolio_rows).round(2), width="stretch", hide_index=True)

    with backtest_tab:
        backtest = analysis["backtest"]
        backtest_metrics = analysis["backtest_metrics"]
        if not backtest_metrics.empty:
            st.caption(
                f"Každý model používa predchádzajúce 3 roky a predikuje ďalších "
                f"{BACKTEST_HORIZON_DAYS} obchodných dní. Tréningové okno končí pred "
                "začiatkom testu, takže budúce ceny nevstupujú do odhadu. Odhady sa posúvajú o "
                f"{BACKTEST_STEP_DAYS} dní; obdobia sa preto prekrývajú."
            )
            st.subheader("Porovnanie modelov na rovnakých dátumoch")
            st.dataframe(backtest_metrics.round(2), width="stretch", hide_index=True)
            st.caption(
                "Tabuľka ukazuje citlivosť odhadu na voľbu modelu: porovnaj pokrytie pásma a chybu "
                "mediánu aj s nulovou prognózou (bez zmeny). Širšie pásmo môže zvýšiť pokrytie, "
                "ale samo osebe neznamená lepší odhad."
            )
            backtest_model = st.selectbox(
                "Model zobrazený v grafe spätného testu",
                list(MODELS),
                index=list(MODELS).index(model_description),
                key="backtest_model",
            )
            selected_backtest = backtest.loc[backtest["Model"] == backtest_model].copy()
            selected_metrics = backtest_metrics.loc[
                backtest_metrics["Model"] == backtest_model
            ].iloc[0]
            columns = st.columns(4)
            columns[0].metric(
                "Pokrytie 10.–90. percentilu",
                f"{selected_metrics['Pokrytie 10.–90. (%)']:.1f}%",
                "cieľ: približne 80 %",
            )
            columns[1].metric(
                "Správny smer mediánu",
                f"{selected_metrics['Správny smer mediánu (%)']:.1f}%",
            )
            columns[2].metric(
                "Mediánová absolútna chyba",
                f"{selected_metrics['Mediánová absolútna chyba (p. b.)']:.2f} p. b.",
            )
            columns[3].metric("Počet období", f"{int(selected_metrics['Počet období'])}")
            st.plotly_chart(
                make_backtest_chart(selected_backtest, chart_ticker),
                width="stretch",
                config={"displaylogo": False},
            )
            st.dataframe(selected_backtest.round(2), width="stretch", hide_index=True)
            st.caption(
                "Pokrytie je podiel skutočných výsledkov v odhadovanom pásme. Prekrývajúce "
                "sa obdobia nie sú nezávislé. Katalóg obsahuje dnešné aktívne firmy, nie historické "
                "zloženie trhu, preto backtest neodstraňuje survivorship bias. Výsledky nezaručujú budúcu presnosť."
            )
        else:
            st.warning(analysis["backtest_error"] or "Spätný test nie je dostupný.")

    with export_tab:
        st.subheader("Stiahnuť výsledky")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "CSV — metriky akcií",
                data=results.to_csv(index=False).encode("utf-8-sig"),
                file_name="monte_carlo_akcie.csv",
                mime="text/csv",
            )
            st.download_button(
                "CSV — stratégie portfólia",
                data=portfolio_table.to_csv(index=False).encode("utf-8-sig"),
                file_name="monte_carlo_portfolia.csv",
                mime="text/csv",
            )
        with export_col2:
            st.download_button(
                "CSV — portfóliové váhy",
                data=weights_table.to_csv(index=False).encode("utf-8-sig"),
                file_name="monte_carlo_vahy.csv",
                mime="text/csv",
            )
            st.download_button(
                "CSV — spätný test",
                data=analysis["backtest"].to_csv(index=False).encode("utf-8-sig"),
                file_name="monte_carlo_backtest.csv",
                mime="text/csv",
                disabled=analysis["backtest"].empty,
            )
            st.download_button(
                "CSV — porovnanie modelov",
                data=analysis["backtest_metrics"].to_csv(index=False).encode("utf-8-sig"),
                file_name="monte_carlo_porovnanie_modelov.csv",
                mime="text/csv",
                disabled=analysis["backtest_metrics"].empty,
            )
            reproducibility = {
                "app": "Monte Carlo analýza akcií",
                "model_version": f"{MODELS[model_label]}-v1 / schema-{ANALYSIS_SCHEMA_VERSION}",
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "retrieved_at": analysis["retrieved_at"],
                "source": "Yahoo Finance; upravené denné ceny a historické FX",
                "data_sha256": analysis["data_sha256"],
                "price_data_from": analysis["prices_from"],
                "price_data_through": analysis["prices_as_of"],
                "random_seeds": {
                    "simulation": 42,
                    "walk_forward_backtest": 431,
                    "portfolio_candidates": 2026,
                },
                "tickers": list(tickers),
                "chart_ticker": chart_ticker,
                "benchmark": benchmark_ticker,
                "horizon_days": HORIZONS[horizon_label],
                "simulations": n_simulations,
                "model": MODELS[model_label],
                "account_currency": currency,
                "investment": float(investment),
                "risk_free_rate": float(risk_free_pct / 100.0),
                "drift_shrinkage": float(drift_shrinkage_pct / 100.0),
                "parameter_uncertainty": float(parameter_uncertainty_pct / 100.0),
                "commission_bps_per_side": float(trade_cost_bps),
                "spread_slippage_bps_per_side": float(spread_slippage_bps),
                "annual_fee_percent": float(annual_fee_pct),
                "annual_turnover_assumption_percent": int(annual_turnover_pct),
                "simplified_positive_gain_tax_percent": float(tax_pct),
                "max_drawdown_limit": float(max_drawdown_limit),
                "max_ticker_weight": float(max_weight_pct / 100.0),
                "min_ticker_weight": float(min_weight_pct / 100.0),
                "stress_shock_percent": float(stress_shock_pct),
                "portfolio_weights": {
                    PORTFOLIO_STRATEGIES[key]: np.asarray(value, dtype=float).tolist()
                    for key, value in analysis["portfolio_weights"].items()
                },
                "limitations": [
                    "Sensitivity rows are terminal-return approximations, not rerun full simulations.",
                    "Tax is a user-defined flat rate on positive terminal gains, not tax advice.",
                    "Turnover is a cost proxy; the model does not simulate actual rebalancing trades.",
                ],
            }
            st.download_button(
                "JSON — nastavenia a pôvod výpočtu",
                data=json.dumps(reproducibility, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="monte_carlo_nastavenia.json",
                mime="application/json",
            )
        report_parameters = {
            "Tickery": ", ".join(tickers),
            "Benchmark": benchmark_ticker,
            "Horizont": f"{horizon_label} ({HORIZONS[horizon_label]} obchodných dní)",
            "Model": model_description,
            "Počet simulácií": f"{n_simulations:,}",
            "Zmenšenie driftu": f"{analysis['drift_shrinkage_pct']:.0f} % k bezrizikovej sadzbe",
            "Neistota odhadu driftu": f"{analysis['parameter_uncertainty_pct']:.0f} %",
            "Investícia": format_money(float(investment), currency_symbol),
            "Mena účtu": currency,
            "Dáta do": analysis["prices_as_of"],
            "Poplatok za obchod": f"{analysis['trade_cost_bps']} bps jedným smerom",
            "Spread a sklz": f"{analysis['spread_slippage_bps']} bps jedným smerom (odhad)",
            "Ročný poplatok": f"{analysis['annual_fee_pct']:.1f} %",
            "Rebalansovací obrat": f"{analysis['annual_turnover_pct']} % ročne (nákladový proxy)",
            "Zjednodušená daň zo zisku": f"{analysis['tax_pct']:.1f} %",
            "Stiahnuté": analysis["retrieved_at"],
            "Seed simulácie": "42",
            "Limit váhy tickeru": f"{min_weight_pct}–{max_weight_pct} %",
        }
        pdf_bytes = make_pdf_report(
            report_parameters,
            results,
            portfolio_table,
            weights_table,
            analysis["backtest_metrics"],
            analysis["backtest_error"],
        )
        st.download_button(
            "PDF — stručný report",
            data=pdf_bytes,
            file_name=f"monte_carlo_report_{datetime.now():%Y%m%d}.pdf",
            mime="application/pdf",
        )

    with st.expander("Metodika, metriky a obmedzenia"):
        st.markdown(
            "**VaR 95 %** je strata na 5. percentile konečného výnosu; v približne 5 % "
            "scenárov bol výsledok horší. **CVaR 95 %** je priemerná strata v tomto "
            "najhoršom 5 % chvoste. **Maximálny pokles** meria prepad od predchádzajúceho "
            "vrcholu počas celej simulovanej cesty a jeho pravdepodobnosť prekročenia limitu.\n\n"
            "**Dáta a drift.** Aplikácia sťahuje päť rokov upravených denných cien Yahoo Finance. "
            "Pri zahraničných tituloch ich prepočítava denným historickým FX kurzom do meny účtu. "
            "Denné logaritmické výnosy sa pred odhadom drifta orežú na 1. a 99. percentile. "
            "Odhadovaný priemer sa zmenší smerom k bezrizikovej sadzbe podľa nastavenia v bočnom "
            "paneli. Neistota drifta sa pridá ako samostatný náhodný posun pre každú simulovanú "
            "cestu; nejde o úplný Bayesov posterior.\n\n"
            "**Modely scenárov.** GBM používa konštantnú volatilitu a korelované normálne šoky. "
            "Blokový bootstrap vzorkuje spoločné 20-dňové bloky historických výnosov. Student-t "
            "GBM používa ťažšie chvosty (8 stupňov voľnosti). GARCH(1,1) odhaduje zhlukovanie "
            "volatility malou mriežkou parametrov a Student-t šokmi. Prepínanie režimov rozlišuje "
            "pokojný a stresový stav pomocou 20-dňovej volatility a historickej prechodovej matice. "
            "Korelácie aktív sa regularizujú, aby boli vhodné na spoločné simulácie. Fan chart "
            "zobrazuje až 1 000 ukážkových ciest; percentily sa počítajú zo všetkých simulácií.\n\n"
            "Výsledky po nákladoch odpočítajú zadanú jednorazovú províziu pri nákupe aj predaji "
            "a priebežný ročný poplatok. Ide o zjednodušený nákladový model; nezahŕňa dane, "
            "spread, sklz ani individuálne poplatky sprostredkovateľa. Walk-forward spätný test "
            "porovnáva všetkých päť modelov na rovnakých dátumoch: pri každom odhade použije iba "
            "predchádzajúce tri roky a vyhodnotí nasledujúcich 63 obchodných dní; ďalší odhad "
            "posunie o 21 dní. Odhady sa prekrývajú, takže obdobia nie sú nezávislé.\n\n"
            "**Stresové testy.** Hypotetický šok je okamžitý pokles zadaný používateľom. Historický "
            "stresový test nájde najhorší spoločný 63-dňový výnos rovnomerného portfólia v dostupnej "
            "histórii, prehrá ten istý denný blok pre všetky aktíva a potom modeluje zvyšok zvoleného "
            "horizontu. Ak horizont trvá iba 63 dní, historický blok vyplní celý horizont a výsledok "
            "je deterministický.\n\n"
            "Portfólio porovnáva rovnomerné váhy, približné maximum historického Sharpe, "
            "minimum historickej volatility a minimum historického denného CVaR. Optimalizácia "
            "preverí 50 000 náhodných prípustných long-only váh s nastaveným minimom a maximom "
            "na jednu akciu; nejde o garantované globálne optimum. Modely aj výber váh závisia "
            "od minulých dát a nemusia zachytiť zmenu režimu ani budúce FX pohyby.\n\n"
            "Verdikt používa orientačné prahy: mediánový výnos nad nulou, pravdepodobnosť "
            "zisku aspoň 60 %, Sharpe aspoň 0,50 a pravdepodobnosť prekročenia limitu "
            "poklesu najviac 10 %. Je to skríningový signál, nie investičné odporúčanie. "
            "Všetky modely predpokladajú, že určité vlastnosti minulosti sú pre budúcnosť aspoň "
            "čiastočne informatívne; nevedia predvídať nové správy, zmeny firmy, likviditu ani "
            "náhle zmeny menovej politiky. Simulácia nie je predpoveď ani prísľub výnosu."
        )
else:
    with st.container(border=True):
        st.subheader("Prvé spustenie")
        st.markdown(
            "1. Skontroluj alebo uprav tickery.\n"
            "2. V bočnom paneli nastav horizont, sumu a počet scenárov.\n"
            "3. Stlač **Spustiť simuláciu**; grafy a metriky sa zobrazia nižšie."
        )
        st.caption(
            "Ceny a menové kurzy sa načítavajú z Yahoo Finance; výsledky sú scenáre, nie záruka výnosu."
        )
