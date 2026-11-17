import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import monte_carlo_akcie as mc


class DownloadTests(unittest.TestCase):
    def test_download_multi_ticker_close_panel(self):
        dates = pd.bdate_range("2024-01-01", periods=65)
        columns = pd.MultiIndex.from_product([["Close"], ["AAA", "BBB"]])
        raw = pd.DataFrame(
            np.column_stack([np.arange(100.0, 165.0), np.arange(80.0, 145.0)]),
            index=dates,
            columns=columns,
        )
        with patch.object(mc.yf, "download", return_value=raw):
            result = mc.download_adjusted_prices(["AAA", "BBB"], "1y")
        self.assertEqual(list(result.columns), ["AAA", "BBB"])
        self.assertEqual(len(result), 65)
        self.assertTrue((result > 0).all().all())

    def test_download_single_ticker_close_column(self):
        dates = pd.bdate_range("2024-01-01", periods=65)
        raw = pd.DataFrame({"Close": np.arange(50.0, 115.0)}, index=dates)
        with patch.object(mc.yf, "download", return_value=raw):
            result = mc.download_adjusted_prices(["AAA"], "1y")
        self.assertEqual(list(result.columns), ["AAA"])


class SimulationTests(unittest.TestCase):
    def test_regularized_correlation_is_positive_definite(self):
        rng = np.random.default_rng(7)
        base = rng.normal(size=300)
        returns = pd.DataFrame({
            "A": base,
            "B": base + rng.normal(scale=0.01, size=300),
            "C": rng.normal(size=300),
        })
        corr = mc.regularized_correlation(returns, 0.05)
        self.assertTrue(np.allclose(corr, corr.T))
        self.assertTrue(np.allclose(np.diag(corr), 1.0))
        self.assertGreater(np.linalg.eigvalsh(corr).min(), 0.0)

    def test_simulation_shapes_positive_prices_and_drawdowns(self):
        paths, max_drawdowns = mc.simulate_paths(
            np.array([100.0, 50.0]),
            np.array([0.08, 0.04]),
            np.array([0.20, 0.30]),
            np.array([[1.0, 0.4], [0.4, 1.0]]),
            horizon_days=20,
            n_simulations=100,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(paths.shape, (100, 21, 2))
        self.assertEqual(max_drawdowns.shape, (100, 2))
        self.assertTrue(np.all(paths > 0))
        self.assertTrue(np.allclose(paths[:, 0, :], [100.0, 50.0]))
        self.assertTrue(np.all(max_drawdowns <= 0))

    def test_simulation_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            mc.simulate_paths(
                np.array([100.0]), np.array([0.08]), np.array([-0.2]),
                np.array([[1.0]]), 10, 10, np.random.default_rng(1),
            )

    def test_tail_metrics_use_lower_five_percent_tail(self):
        var95, cvar95 = mc.tail_metrics(np.array([0.10, 0.05, 0.0, -0.10, -0.50]))
        self.assertAlmostEqual(var95, 0.42)
        self.assertAlmostEqual(cvar95, 0.50)

    def test_portfolio_weights_respect_long_only_cap(self):
        weights = mc.optimize_historical_sharpe(
            np.array([0.12, 0.08]),
            np.array([[0.04, 0.01], [0.01, 0.02]]),
            max_weight=0.65,
            n_candidates=500,
            rng=np.random.default_rng(12),
        )
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertLessEqual(weights.max(), 0.65 + 1e-12)


class MetricsAndPlotTests(unittest.TestCase):
    def test_make_metrics_reports_monte_carlo_summary(self):
        returns = np.full(100, 0.10)
        metrics = mc.make_metrics(
            "AAA", 100.0, 100.0 * (1.0 + returns), np.full(100, -0.05),
            annual_drift=0.15, annual_volatility=0.20,
            benchmark_returns=np.zeros(100), horizon_years=1.0,
        )
        self.assertEqual(metrics["ticker"], "AAA")
        self.assertAlmostEqual(metrics["mean_return_pct"], 10.0)
        self.assertAlmostEqual(metrics["prob_profit_pct"], 100.0)
        self.assertAlmostEqual(metrics["prob_beat_benchmark_pct"], 100.0)
        self.assertEqual(metrics["screen_pass"], "ÁNO")

    def test_plot_functions_write_png_files(self):
        paths, _ = mc.simulate_paths(
            np.array([100.0, 50.0]), np.array([0.05, 0.03]),
            np.array([0.15, 0.20]), np.eye(2), 10, 30,
            np.random.default_rng(22),
        )
        with TemporaryDirectory(dir=Path(mc.__file__).resolve().parent) as temp_dir:
            fan = Path(temp_dir) / "fan.png"
            distribution = Path(temp_dir) / "distribution.png"
            mc.plot_fan_chart(paths, ["AAA", "BBB"], fan)
            mc.plot_terminal_distributions(paths, ["AAA", "BBB"], distribution)
            self.assertGreater(fan.stat().st_size, 0)
            self.assertGreater(distribution.stat().st_size, 0)


class MainIntegrationTests(unittest.TestCase):
    def test_main_creates_reports_from_synthetic_prices(self):
        tickers = ["AAA", "BBB", "SPY"]
        dates = pd.bdate_range("2023-01-02", periods=180)
        rng = np.random.default_rng(31)
        base = rng.normal(0.0004, 0.009, size=len(dates))
        asset_returns = np.column_stack([
            base + rng.normal(0.0001, 0.006, size=len(dates)),
            0.5 * base + rng.normal(0.0002, 0.012, size=len(dates)),
            base,
        ])
        price_values = 100.0 * np.cumprod(1.0 + asset_returns, axis=0)
        columns = pd.MultiIndex.from_product([["Close"], tickers])
        raw = pd.DataFrame(price_values, index=dates, columns=columns)

        with TemporaryDirectory(dir=Path(mc.__file__).resolve().parent) as temp_dir:
            output_dir = Path(temp_dir) / "results"
            patched = {
                "TICKERS": tickers,
                "BENCHMARK": "SPY",
                "HISTORY_PERIOD": "1y",
                "HORIZON_DAYS": 20,
                "N_SIMULATIONS": 100,
                "N_PORTFOLIO_CANDIDATES": 500,
                "MAX_PORTFOLIO_WEIGHT": 0.80,
                "RISK_FREE_RATE": 0.0,
                "RANDOM_SEED": 10,
                "OUTPUT_DIR": output_dir,
            }
            with patch.multiple(mc, **patched), patch.object(mc.yf, "download", return_value=raw):
                mc.main()
            self.assertTrue((output_dir / "metriky_akcii.csv").is_file())
            self.assertTrue((output_dir / "portfolio.csv").is_file())
            self.assertTrue((output_dir / "fan_chart.png").is_file())
            self.assertTrue((output_dir / "rozdelenie_konecnych_cien.png").is_file())
            report = pd.read_csv(output_dir / "metriky_akcii.csv")
            self.assertEqual(set(report["ticker"]), set(tickers))


if __name__ == "__main__":
    unittest.main()
