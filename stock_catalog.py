"""Search helpers for the bundled global large-cap stock catalog."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TICKER_PATTERN = re.compile(r"[A-Z0-9.^=-]{1,20}")
CATALOG_COLUMNS = [
    "ticker",
    "name",
    "country",
    "cap_tier",
    "sector",
    "market",
    "exchange",
]


def load_stock_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Load and normalize the bundled catalog without requiring network access."""
    catalog_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent / "data" / "global_stocks_largecap.csv"
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Katalóg akcií sa nenašiel: {catalog_path}")

    catalog = pd.read_csv(catalog_path, dtype={"ticker": "string"})
    missing = {"ticker", "name"} - set(catalog.columns)
    if missing:
        raise ValueError("Katalógu chýbajú stĺpce: " + ", ".join(sorted(missing)))

    for column in CATALOG_COLUMNS:
        if column not in catalog:
            catalog[column] = ""

    catalog["ticker"] = catalog["ticker"].fillna("").astype(str).str.strip().str.upper()
    catalog["name"] = catalog["name"].fillna("").astype(str).str.strip()
    for column in CATALOG_COLUMNS[2:]:
        catalog[column] = catalog[column].fillna("").astype(str).str.strip()

    valid = catalog["ticker"].str.fullmatch(TICKER_PATTERN, na=False)
    catalog = catalog.loc[valid & catalog["name"].ne("")]
    return catalog.drop_duplicates("ticker", keep="first")[CATALOG_COLUMNS].reset_index(drop=True)


def search_stock_catalog(
    catalog: pd.DataFrame,
    query: str,
    limit: int = 50,
) -> pd.DataFrame:
    """Search by ticker, company name, country, or a ``Company (TICKER)`` label."""
    if not query or not query.strip() or catalog.empty:
        return catalog.iloc[0:0].copy()

    search = query.strip()
    parenthesized = re.fullmatch(r".*\(([A-Z0-9.^=-]{1,20})\)", search, flags=re.IGNORECASE)
    if parenthesized:
        search = parenthesized.group(1)
    needle = search.casefold()

    tickers = catalog["ticker"].fillna("").astype(str).str.casefold()
    names = catalog["name"].fillna("").astype(str).str.casefold()
    countries = catalog["country"].fillna("").astype(str).str.casefold()

    exact_ticker = tickers.eq(needle)
    exact_name = names.eq(needle)
    ticker_prefix = tickers.str.startswith(needle)
    name_prefix = names.str.startswith(needle)
    ticker_match = tickers.str.contains(re.escape(needle), regex=True)
    name_match = names.str.contains(re.escape(needle), regex=True)
    country_match = countries.str.contains(re.escape(needle), regex=True)
    matched = exact_ticker | exact_name | ticker_prefix | name_prefix | ticker_match | name_match | country_match
    if not matched.any():
        return catalog.iloc[0:0].copy()

    ranked = catalog.loc[matched].copy()
    ranked_tickers = ranked["ticker"].astype(str).str.casefold()
    ranked_names = ranked["name"].astype(str).str.casefold()
    ranked["_rank"] = 6
    ranked.loc[ranked_names.str.contains(re.escape(needle), regex=True), "_rank"] = 5
    ranked.loc[ranked_tickers.str.contains(re.escape(needle), regex=True), "_rank"] = 4
    ranked.loc[ranked_names.str.startswith(needle), "_rank"] = 3
    ranked.loc[ranked_tickers.str.startswith(needle), "_rank"] = 2
    ranked.loc[ranked_names.eq(needle), "_rank"] = 1
    ranked.loc[ranked_tickers.eq(needle), "_rank"] = 0
    ranked = ranked.sort_values(["_rank", "name", "ticker"], kind="stable")
    return ranked.drop(columns="_rank").head(max(1, int(limit))).reset_index(drop=True)


def add_ticker_symbols(
    raw: str,
    current: list[str],
    catalog: pd.DataFrame | None = None,
    max_tickers: int = 10,
) -> tuple[list[str], list[str]]:
    """Resolve ticker lists and company labels such as ``Visa (V)`` safely."""
    additions: list[str] = []
    invalid: list[str] = []

    for chunk in re.split(r"[,;\n]+", raw.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue

        symbol_match = re.search(r"\(([A-Z0-9.^=-]{1,20})\)\s*$", chunk, flags=re.IGNORECASE)
        if symbol_match:
            candidates = [symbol_match.group(1).upper()]
        else:
            exact_name = pd.DataFrame()
            exact_ticker = False
            if catalog is not None and "name" in catalog:
                exact_ticker = bool(
                    catalog["ticker"].fillna("").astype(str).str.casefold().eq(chunk.casefold()).any()
                )
                exact_name = catalog.loc[
                    catalog["name"].fillna("").astype(str).str.casefold().eq(chunk.casefold())
                ]

            if exact_ticker:
                candidates = [chunk.upper()]
            elif len(exact_name) == 1:
                candidates = [str(exact_name.iloc[0]["ticker"]).upper()]
            else:
                name_prefixes = pd.DataFrame()
                if catalog is not None and "name" in catalog:
                    name_prefixes = catalog.loc[
                        catalog["name"].fillna("").astype(str).str.casefold().str.startswith(chunk.casefold())
                    ]
                words = chunk.split()
                if len(name_prefixes) == 1:
                    candidates = [str(name_prefixes.iloc[0]["ticker"]).upper()]
                elif len(words) > 1 and all(TICKER_PATTERN.fullmatch(word.upper()) for word in words):
                    candidates = [word.upper() for word in words]
                else:
                    candidates = [chunk.upper()]

        for candidate in candidates:
            if not TICKER_PATTERN.fullmatch(candidate):
                invalid.append(candidate)
            elif candidate not in current and candidate not in additions:
                additions.append(candidate)

    available = max(0, max_tickers - len(current))
    return current + additions[:available], invalid + additions[available:]
