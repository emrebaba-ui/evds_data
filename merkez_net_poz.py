import os
import sys
import traceback
import pandas as pd
from playwright.sync_api import sync_playwright, BrowserContext

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class EvdsConfig:
    """Configuration variables and API endpoints."""

    BASE_URL = "https://evds3.tcmb.gov.tr/"
    FRONTEND_API_URL = (
        "https://evds3.tcmb.gov.tr/igmevdsms-dis/fe"  # Bypass instead of API-key
    )
    START_DATE = "01-01-2026"
    END_DATE = "31-12-2026"
    OUTPUT_FILE = "central_bank_net_position.tsv"  # .tsv for clarity


class EvdsClient:
    """Handles EVDS authentication (WAF bypass) and internal frontend API requests."""

    def __init__(self, context: BrowserContext):
        self.context = context

    def bypass_waf(self) -> None:
        """Visits homepage to acquire required WAF session cookies."""
        page = self.context.new_page()
        page.goto(EvdsConfig.BASE_URL)
        page.wait_for_timeout(3000)
        page.close()

    def fetch_series(
        self,
        series: str,
        agg_types: str,
        formulas: str,
        frequency: str = "2",
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """Fetches time-series data using the provided parameters."""
        payload = {
            "type": "json",
            "series": series,
            "aggregationTypes": agg_types,
            "formulas": formulas,
            "startDate": start_date or EvdsConfig.START_DATE,
            "endDate": end_date or EvdsConfig.END_DATE,
            "frequency": frequency,
            "decimalSeperator": ".",
            "decimal": "2",
            "dateFormat": "0",
            "lang": "tr",
            "yon": "0",
            "sira": "0",
            "ozelFormuller": [],
            "groupSeperator": True,
            "isRaporSayfasi": False,
        }

        response = self.context.request.post(EvdsConfig.FRONTEND_API_URL, data=payload)
        if not response.ok:
            raise ConnectionError(f"API request failed. Status Code: {response.status}")

        return response.json()


class PositionCalculator:
    """Processes raw EVDS data and calculates core financial metrics."""

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """Removes formatting commas and safely casts string series to float."""
        return (
            series.astype(str)
            .str.replace(",", "")
            .apply(pd.to_numeric, errors="coerce")
        )

    def calculate(self, raw_daily: dict, raw_swap: dict) -> pd.DataFrame:
        """Merges datasets (as-of logic) and applies net position mathematical formulas."""
        if "items" not in raw_daily or "items" not in raw_swap:
            raise ValueError("Invalid JSON structure: 'items' key is missing.")

        df_daily = pd.DataFrame(raw_daily["items"])
        df_swap = pd.DataFrame(raw_swap["items"])

        if df_daily.empty or df_swap.empty:
            return pd.DataFrame()

        df_daily.rename(columns={"Tarih": "Date"}, inplace=True)
        df_swap.rename(columns={"Tarih": "Date"}, inplace=True)

        required_cols = ["TP_AB_A02", "TP_AB_A10", "TP_DK_USD_A_YTL"]
        if (
            not all(col in df_daily.columns for col in required_cols)
            or "TP_DOVVARNC_K14" not in df_swap.columns
        ):
            return pd.DataFrame()

        # Format dates for precise temporal matching (as-of merge)
        df_daily["Date_Obj"] = pd.to_datetime(df_daily["Date"], format="mixed", dayfirst=True)
        df_swap["Date_Obj"] = pd.to_datetime(df_swap["Date"], format="mixed", dayfirst=True)

        df_daily.sort_values("Date_Obj", inplace=True)
        df_swap.sort_values("Date_Obj", inplace=True)

        # Merge asynchronous datasets. Pulls the latest available swap data for each active trading day
        df = pd.merge_asof(
            df_daily,
            df_swap[["Date_Obj", "TP_DOVVARNC_K14"]],
            on="Date_Obj",
            direction="backward",
        )

        ext_assets = self._clean_numeric(df["TP_AB_A02"])
        fx_liabilities = self._clean_numeric(df["TP_AB_A10"])
        usd_rate = self._clean_numeric(df["TP_DK_USD_A_YTL"])
        raw_swap_val = self._clean_numeric(df["TP_DOVVARNC_K14"]).fillna(0)

        net_reserves_try = ext_assets - fx_liabilities
        net_reserves_usd_bn = (net_reserves_try / usd_rate) / 1_000_000
        swap_debt_usd_bn = raw_swap_val / 1_000
        net_position_usd_bn = net_reserves_usd_bn + swap_debt_usd_bn

        df["Net_Reserves_Bn_USD"] = net_reserves_usd_bn.round(2)
        df["Swap_Debt_Bn_USD"] = swap_debt_usd_bn.round(2)
        df["Net_Position_Without_Swap"] = net_position_usd_bn.round(2)

        columns_to_keep = [
            "Date",
            "Net_Reserves_Bn_USD",
            "Swap_Debt_Bn_USD",
            "Net_Position_Without_Swap",
        ]
        return df[columns_to_keep].dropna()


class ReportGenerator:
    """Handles data output formatting for the console and filesystem."""

    @staticmethod
    def print_console(df: pd.DataFrame, tail_count: int = 10) -> None:
        """Prints a tab-separated, formatted table to the standard output."""
        print(f"\n{'='*80}")
        print("Date\t\tNet Reserves\tTotal Swaps\tNet Position w/o Swap")
        print("-" * 80)
        for _, row in df.tail(tail_count).iterrows():
            print(
                f"{row['Date']}\t{row['Net_Reserves_Bn_USD']} bn $\t"
                f"{row['Swap_Debt_Bn_USD']} bn $\t{row['Net_Position_Without_Swap']} bn $"
            )
        print(f"{'='*80}\n")

    @staticmethod
    def save_csv(df: pd.DataFrame, filename: str) -> None:
        """Saves new unique records to the TSV file."""
        if df.empty:
            print("[-] No new data to save.")
            return

        file_exists = os.path.exists(filename)
        if file_exists:
            try:
                existing_df = pd.read_csv(filename, sep="\t", encoding="utf-8-sig")
                existing_dates = set(existing_df["Date"].tolist())
                df = df[~df["Date"].isin(existing_dates)]
            except Exception:
                pass

        if df.empty:
            print("[-] All fetched records are already in cache.")
            return

        df.to_csv(
            filename,
            mode="a" if file_exists else "w",
            index=False,
            header=not file_exists,
            sep="\t",
            encoding="utf-8-sig",
        )
        action = "Appended to" if file_exists else "Created"
        print(f"[+] {action} {filename} ({len(df)} records)")


def main():
    print("1. Checking local cache...")
    start_date = "01-01-2026"
    end_date = pd.Timestamp.now().strftime("%d-%m-%Y")

    if os.path.exists(EvdsConfig.OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(
                EvdsConfig.OUTPUT_FILE, sep="\t", encoding="utf-8-sig"
            )
            if not existing_df.empty and "Date" in existing_df.columns:
                existing_dates = pd.to_datetime(existing_df["Date"], dayfirst=True)
                latest_date = existing_dates.max()
                start_dt = latest_date + pd.Timedelta(days=1)

                if start_dt.date() > pd.Timestamp.now().date():
                    print("[*] Data is already up-to-date.")
                    return
                start_date = start_dt.strftime("%d-%m-%Y")
        except Exception as e:
            print(f"[!] Error reading cache: {e}. Fetching all data.")

    EvdsConfig.START_DATE = start_date
    EvdsConfig.END_DATE = end_date
    print(f"    Target range: {EvdsConfig.START_DATE} to {EvdsConfig.END_DATE}")

    print("2. Initializing Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        client = EvdsClient(context)

        print("3. Bypassing WAF (Web Application Firewall)...")
        client.bypass_waf()

        print("4. Fetching Market Data...")
        daily_data = client.fetch_series(
            series="TP.AB.A02-TP.AB.A10-TP.DK.USD.A.YTL",
            agg_types="last-last-last",
            formulas="0-0-0",
        )
        swap_data = client.fetch_series(
            series="TP.DOVVARNC.K14",
            agg_types="last",
            formulas="0",
            start_date="01-01-2026",
        )
        browser.close()

    print("5. Calculating Net Positions...")
    calculator = PositionCalculator()
    df_result = calculator.calculate(daily_data, swap_data)

    if df_result.empty:
        print("[*] No new data records to process.")
        return

    print("6. Generating Reports...")
    ReportGenerator.print_console(df_result)
    ReportGenerator.save_csv(df_result, EvdsConfig.OUTPUT_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*50}\nFATAL ERROR CAUGHT!\n{e}\n{'='*50}")
        traceback.print_exc()
        input("\n[Press Enter to exit...]")
