import argparse
import os
import sys
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock
import matplotlib
import pandas as pd
import requests

# Force headless backend during automated tests
matplotlib.use("Agg")

from api_client import EvdsApiClient, EvdsApiError
from config import ConfigurationError, EvdsConfig
from data_processor import (
    DataProcessor,
    DataRepository,
)
from visualizer import Visualizer


class TestEvdsConfigAndEnv(unittest.TestCase):
    """Tests .env parsing, quote stripping, and configuration validation."""

    def setUp(self) -> None:
        self._orig_env = os.environ.copy()
        os.environ.pop("EVDS_API_KEY", None)
        os.environ.pop("EVDS_BASE_URL", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_loads_quoted_api_key_from_dotenv_file(self) -> None:
        """Verifies that EVDS_API_KEY=\"88WtBZBjW8\" is loaded with quotes stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = os.path.join(tmpdir, ".env")
            with open(env_file, "w", encoding="utf-8") as f:
                f.write('# EVDS Configuration\nEVDS_API_KEY="88WtBZBjW8"  \n')

            cfg = EvdsConfig(BASE_DIR=tmpdir, ENV_FILE=env_file)
            self.assertEqual(cfg.API_KEY, "88WtBZBjW8")
            self.assertEqual(cfg.BASE_API_URL, "https://evds3.tcmb.gov.tr/igmevdsms-dis/")

    def test_raises_configuration_error_when_key_missing(self) -> None:
        """Verifies explicit ConfigurationError when EVDS_API_KEY is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_env = os.path.join(tmpdir, ".env")
            with open(empty_env, "w", encoding="utf-8") as f:
                f.write("# Empty env\n")

            with self.assertRaises(ConfigurationError) as ctx:
                EvdsConfig(BASE_DIR=tmpdir, ENV_FILE=empty_env)
            self.assertIn("EVDS_API_KEY is missing", str(ctx.exception))


class TestEvdsApiClientDiagnostics(unittest.TestCase):
    """Tests EVDS3 URL construction, serieList discovery, and failure diagnostics."""

    def setUp(self) -> None:
        self.cfg = EvdsConfig(API_KEY="88WtBZBjW8")
        self.mock_session = MagicMock(spec=requests.Session)
        self.mock_session.headers = {}
        self.client = EvdsApiClient(self.cfg, session=self.mock_session)

    def test_header_and_url_formatting(self) -> None:
        """Ensures API key is placed in headers and parameters use EVDS path format."""
        self.assertEqual(self.mock_session.headers.get("key"), "88WtBZBjW8")
        url = self.client.build_endpoint_url(
            series=("TP.AB.A02", "TP.AB.A10"),
            start_date="01-01-2026",
            end_date="15-01-2026",
        )
        expected = (
            "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
            "series=TP.AB.A02-TP.AB.A10&startDate=01-01-2026&endDate=15-01-2026"
            "&type=json&aggregationTypes=last-last&formulas=0-0"
        )
        self.assertEqual(url, expected)

    def test_detects_html_response_from_legacy_or_waf_endpoint(self) -> None:
        """Pinpoints when EVDS returns an HTML page instead of JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = "https://evds2.tcmb.gov.tr/service/evds/"
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.text = "<!DOCTYPE html><html><head><title>EVDS</title></head><body></body></html>"
        mock_resp.raise_for_status.return_value = None
        self.mock_session.get.return_value = mock_resp

        with self.assertRaises(EvdsApiError) as ctx:
            self.client.fetch_series(["TP.AB.A02"])
        self.assertIn("HTML response received instead of JSON", str(ctx.exception))

    def test_detects_empty_response_body(self) -> None:
        """Pinpoints empty HTTP 200 body caused by invalid API key or empty query."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = "   "
        mock_resp.raise_for_status.return_value = None
        self.mock_session.get.return_value = mock_resp

        with self.assertRaises(EvdsApiError) as ctx:
            self.client.fetch_series(["TP.AB.A02"])
        self.assertIn("Empty response body received from EVDS", str(ctx.exception))

    def test_detects_malformed_json_with_preview(self) -> None:
        """Pinpoints corrupted JSON and includes raw payload preview in error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.text = "System maintenance in progress"
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("Expecting value")
        self.mock_session.get.return_value = mock_resp

        with self.assertRaises(EvdsApiError) as ctx:
            self.client.fetch_series(["TP.AB.A02"])
        self.assertIn("Invalid JSON received from EVDS", str(ctx.exception))
        self.assertIn("System maintenance in progress", str(ctx.exception))

    def test_detects_missing_items_key(self) -> None:
        """Pinpoints valid JSON that lacks the required EVDS 'items' list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = '{"error": "Invalid series code"}'
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"error": "Invalid series code"}
        self.mock_session.get.return_value = mock_resp

        with self.assertRaises(EvdsApiError) as ctx:
            self.client.fetch_series(["TP.AB.A02"])
        self.assertIn("missing 'items' list", str(ctx.exception))

    def test_discovers_portfolio_flow_codes_from_serielist(self) -> None:
        """Verifies automatic discovery of Equity and DIBS net flow codes from bie_mknethar."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = "[...]"
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {"SERIE_CODE": "TP.MKNETHAR.S1", "SERIE_NAME": "1.1.1. Hisse Senedi (Stok)"},
            {"SERIE_CODE": "TP.MKNETHAR.S2", "SERIE_NAME": "1.1.2. DİBS (Kesin Alım) (Stok)"},
            {"SERIE_CODE": "TP.MKNETHAR.N1", "SERIE_NAME": "2.1.1. Hisse Senedi"},
            {"SERIE_CODE": "TP.MKNETHAR.N2", "SERIE_NAME": "2.1.2. DİBS (Kesin Alım)"},
        ]
        self.mock_session.get.return_value = mock_resp

        discovered = self.client.discover_portfolio_flow_codes()
        self.assertEqual(discovered.get("Foreign_Equity_Net_Flow"), "TP.MKNETHAR.N1")
        self.assertEqual(discovered.get("Foreign_Bond_Net_Flow"), "TP.MKNETHAR.N2")


class TestDataProcessorAndAlignment(unittest.TestCase):
    """Tests mathematical formulas, monthly/weekly date parsing, merge_asof, and ffill."""

    def setUp(self) -> None:
        self.cfg = EvdsConfig(API_KEY="88WtBZBjW8")
        self.processor = DataProcessor(self.cfg)

    def test_net_position_math_and_weekly_alignment(self) -> None:
        """Verifies core Net Position math, 'YYYY-M' swap date parsing, and weekly ffill."""
        raw_daily: Dict[str, Any] = {
            "items": [
                {
                    "Tarih": "02-01-2026",
                    "TP_AB_A02": "5,000,000,000",
                    "TP_AB_A10": "3,225,000,000",
                    "TP_DK_USD_A_YTL": "35.50",
                },
                {
                    "Tarih": "05-01-2026",
                    "TP_AB_A02": "5,355,000,000",
                    "TP_AB_A10": "3,219,000,000",
                    "TP_DK_USD_A_YTL": "35.60",
                },
                {
                    "Tarih": "06-01-2026",
                    "TP_AB_A02": None,  # Holiday / non-trading day row
                    "TP_AB_A10": None,
                    "TP_DK_USD_A_YTL": None,
                },
            ]
        }
        raw_swap: Dict[str, Any] = {
            "items": [
                {"Tarih": "2026-1", "TP_DOVVARNC_K14": "-12,500"},
            ]
        }
        extra_metrics: Dict[str, Dict[str, Any]] = {
            "Gross_FX_Reserves": {
                "series_code": "TP.AB.C2",
                "column_key": "TP_AB_C2",
                "items": [{"Tarih": "02-01-2026", "TP_AB_C2": "88,250.40"}],
            },
            "Gold_Reserves": {
                "series_code": "TP.AB.C1",
                "column_key": "TP_AB_C1",
                "items": [{"Tarih": "02-01-2026", "TP_AB_C1": "64,120.10"}],
            },
            "Foreign_Equity_Net_Flow": {
                "series_code": "TP.MKNETHAR.N1",
                "column_key": "TP_MKNETHAR_N1",
                "items": [{"Tarih": "02-01-2026", "TP_MKNETHAR_N1": "145.20"}],
            },
            "Foreign_Bond_Net_Flow": {
                "series_code": "TP.MKNETHAR.N2",
                "column_key": "TP_MKNETHAR_N2",
                "items": [{"Tarih": "02-01-2026", "TP_MKNETHAR_N2": "-55.30"}],
            },
        }

        df = self.processor.build_dataset(raw_daily, raw_swap, extra_metrics)
        self.assertEqual(len(df), 2)

        self.assertAlmostEqual(df.loc[0, "Net_Reserves_Bn_USD"], 50.00, places=2)
        self.assertAlmostEqual(df.loc[0, "Swap_Debt_Bn_USD"], -12.50, places=2)
        self.assertAlmostEqual(df.loc[0, "Net_Position_Without_Swap"], 37.50, places=2)

        self.assertAlmostEqual(df.loc[1, "Gross_FX_Reserves"], 88250.40, places=2)
        self.assertAlmostEqual(df.loc[1, "Gold_Reserves"], 64120.10, places=2)
        self.assertAlmostEqual(df.loc[1, "Foreign_Equity_Net_Flow"], 145.20, places=2)
        self.assertAlmostEqual(df.loc[1, "Foreign_Bond_Net_Flow"], -55.30, places=2)


class TestRepositoryAndVisualizer(unittest.TestCase):
    """Tests TSV persistence, deduplication, and Matplotlib chart rendering."""

    def test_save_load_and_plot_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvdsConfig(BASE_DIR=tmpdir, API_KEY="88WtBZBjW8")
            repo = DataRepository(cfg)
            viz = Visualizer(cfg)

            sample_df = pd.DataFrame(
                {
                    "Date": ["02-01-2026", "05-01-2026"],
                    "Net_Reserves_Bn_USD": [50.0, 60.0],
                    "Swap_Debt_Bn_USD": [-12.5, -10.0],
                    "Net_Position_Without_Swap": [37.5, 50.0],
                    "Gross_FX_Reserves": [88250.4, 89100.0],
                    "Gold_Reserves": [64120.1, 64500.0],
                    "Foreign_Equity_Net_Flow": [145.2, 210.5],
                    "Foreign_Bond_Net_Flow": [-55.3, 85.0],
                }
            )

            added = repo.save_dataset(sample_df)
            self.assertEqual(added, 2)

            added_again = repo.save_dataset(sample_df)
            self.assertEqual(added_again, 0)

            loaded_df = repo.load_dataset()
            self.assertEqual(len(loaded_df), 2)

            chart_path = viz.plot_dashboard(loaded_df, show_interactive=False)
            self.assertTrue(os.path.exists(chart_path))
            self.assertGreater(os.path.getsize(chart_path), 1000)


def run_live_diagnostics() -> int:
    """Runs live pre-flight diagnostics against TCMB EVDS3 using the local .env file."""
    print("=" * 75)
    print("         TCMB EVDS3 LIVE DIAGNOSTIC & CONNECTIVITY CHECK")
    print("=" * 75)

    try:
        cfg = EvdsConfig()
        masked_key = f"{cfg.API_KEY[:3]}***{cfg.API_KEY[-2:]}" if len(cfg.API_KEY) >= 5 else "***"
        print(f"[OK] .env loaded successfully. API Key: {masked_key} (len={len(cfg.API_KEY)})")
        print(f"[OK] Target Base URL: {cfg.BASE_API_URL}")
    except ConfigurationError as exc:
        print(f"[FAIL] Configuration Error: {exc}")
        return 1

    client = EvdsApiClient(cfg)
    end_date = pd.Timestamp.now().strftime("%d-%m-%Y")

    try:
        # 1. Core Daily Series
        print(f"\n[TEST 1] Fetching Daily Balance Sheet & USD: {', '.join(cfg.DAILY_SERIES)} ...")
        daily_payload = client.fetch_series(
            series=cfg.DAILY_SERIES,
            start_date="01-01-2026",
            end_date=end_date,
        )
        daily_items = daily_payload.get("items", [])
        print(f"  -> [OK] Received {len(daily_items)} record(s).")

        # 2. Core Swap Series
        print(f"\n[TEST 2] Fetching Central Bank Swaps: {', '.join(cfg.SWAP_SERIES)} ...")
        swap_payload = client.fetch_series(
            series=cfg.SWAP_SERIES,
            start_date="01-01-2026",
            end_date=end_date,
        )
        swap_items = swap_payload.get("items", [])
        print(f"  -> [OK] Received {len(swap_items)} record(s).")

        # 3. Inspect Datagroup Metadata & Fetch Individual Reserve/Flow Metrics
        print(
            f"\n[TEST 3] Discovering & Fetching Reserves ({cfg.RESERVES_DATAGROUP}) "
            f"and Foreign Flows ({cfg.PORTFOLIO_FLOWS_DATAGROUP}) Individually ..."
        )
        extra_results = client.fetch_extra_metrics_individually(
            start_date="01-01-2026",
            end_date=end_date,
        )
        for col in cfg.extra_metric_columns:
            info = extra_results.get(col)
            if info and info.get("items"):
                last_row = info["items"][-1]
                print(
                    f"  -> [OK] {col}: Resolved to '{info['series_code']}' "
                    f"({len(info['items'])} records, latest date={last_row.get('Tarih')})"
                )
            else:
                print(f"  -> [WARN] {col}: Could not resolve a valid EVDS3 series code.")

    except EvdsApiError as exc:
        print(f"\n[FAIL] Live EVDS API Check Failed:\n  {exc}")
        return 1
    finally:
        client.close()

    print(f"\n{'=' * 75}")
    print("[SUCCESS] Live EVDS3 diagnostic completed!")
    print(f"{'=' * 75}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unit tests and optional live EVDS diagnostics.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live connectivity checks against TCMB EVDS3 after unit tests.",
    )
    args, _ = parser.parse_known_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    if args.live:
        sys.exit(run_live_diagnostics())