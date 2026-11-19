import unittest

from stock_catalog import (
    add_ticker_symbols,
    filter_stock_catalog,
    load_stock_catalog,
    search_stock_catalog,
)


class StockCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_stock_catalog()

    def test_catalog_has_large_universe_and_visa(self):
        self.assertGreaterEqual(len(self.catalog), 1_000)
        visa = self.catalog.loc[self.catalog["ticker"] == "V"].iloc[0]
        self.assertEqual(visa["name"], "Visa Inc.")

    def test_catalog_includes_nyse_listings(self):
        nyse = filter_stock_catalog(self.catalog, "NYSE")
        self.assertGreaterEqual(len(nyse), 100)
        self.assertTrue(nyse["exchange"].eq("NYQ").all())
        self.assertIn("V", nyse["ticker"].tolist())
        self.assertIn("JPM", nyse["ticker"].tolist())

    def test_search_by_company_name_and_ticker(self):
        for query in ("Visa", "V", "Visa (V)"):
            with self.subTest(query=query):
                result = search_stock_catalog(self.catalog, query)
                self.assertFalse(result.empty)
                self.assertEqual(result.iloc[0]["ticker"], "V")

    def test_manual_entry_resolves_visa_and_multiple_symbols(self):
        result, invalid = add_ticker_symbols(
            "Visa (V), AAPL NVDA", [], catalog=self.catalog
        )
        self.assertEqual(result, ["V", "AAPL", "NVDA"])
        self.assertEqual(invalid, [])

    def test_manual_entry_resolves_company_name_without_parentheses(self):
        result, invalid = add_ticker_symbols("Visa", [], catalog=self.catalog)
        self.assertEqual(result, ["V"])
        self.assertEqual(invalid, [])

    def test_limit_and_duplicates_are_enforced(self):
        current = ["AAPL"]
        result, invalid = add_ticker_symbols(
            "AAPL, MSFT, NVDA", current, catalog=self.catalog, max_tickers=2
        )
        self.assertEqual(result, ["AAPL", "MSFT"])
        self.assertEqual(invalid, ["NVDA"])


if __name__ == "__main__":
    unittest.main()
