import logging
import os
import re
from typing import Any, Dict, List
import pandas as pd

from config import EvdsConfig

logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data cleaning and merging errors."""


class DataProcessor:
    """Cleans EVDS payloads, aligns multi-frequency series, and computes metrics."""

    def __init__(self, config: EvdsConfig) -> None:
        self._config = config

    @staticmethod
    def clean_numeric(series: pd.Series) -> pd.Series:
        """Strips formatting commas and safely casts values to float."""
        return (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .apply(pd.to_numeric, errors="coerce")
        )

    @staticmethod
    def parse_evds_dates(date_series: pd.Series) -> pd.Series:
        """Parses EVDS daily ('DD-MM-YYYY'), weekly, and monthly ('YYYY-M') date strings."""
        def _normalize_date_str(val: Any) -> str:
            text = str(val).strip()
            # Match monthly EVDS format 'YYYY-M' or 'YYYY-MM' (e.g. '2026-9' -> '01-09-2026')
            match = re.match(r"^(\d{4})-(\d{1,2})$", text)
            if match:
                year, month = int(match.group(1)), int(match.group(2))
                return f"01-{month:02d}-{year}"
            return text

        normalized = date_series.apply(_normalize_date_str)
        return pd.to_datetime(normalized, format="mixed", dayfirst=True, errors="coerce")

    def _payload_to_dataframe(
        self,
        payload: Dict[str, Any],
        expected_cols: List[str],
        apply_ffill: bool = False,
    ) -> pd.DataFrame:
        """Converts raw EVDS JSON payload into a date-sorted, numeric DataFrame."""
        items = payload.get("items")
        if not isinstance(items, list):
            raise DataProcessingError("Invalid EVDS payload: 'items' list is missing.")

        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        if "Tarih" not in df.columns:
            raise DataProcessingError("Invalid EVDS payload: 'Tarih' field is missing.")

        df.rename(columns={"Tarih": self._config.COL_DATE}, inplace=True)
        df[self._config.COL_DATE_OBJ] = self.parse_evds_dates(df[self._config.COL_DATE])
        df.dropna(subset=[self._config.COL_DATE_OBJ], inplace=True)
        df.sort_values(self._config.COL_DATE_OBJ, inplace=True)

        for col in expected_cols:
            if col in df.columns:
                df[col] = self.clean_numeric(df[col])
            else:
                df[col] = float("nan")

        if apply_ffill:
            df[expected_cols] = df[expected_cols].ffill()

        df.reset_index(drop=True, inplace=True)
        return df

    def build_dataset(
        self,
        raw_daily: Dict[str, Any],
        raw_swap: Dict[str, Any],
        extra_metrics_data: Dict[str, Dict[str, Any]] | Dict[str, Any],
    ) -> pd.DataFrame:
        """Merges daily, swap, and weekly reserve/flow datasets and computes positions."""
        daily_cols = ["TP_AB_A02", "TP_AB_A10", "TP_DK_USD_A_YTL"]
        swap_cols = ["TP_DOVVARNC_K14"]

        df_daily = self._payload_to_dataframe(raw_daily, daily_cols, apply_ffill=False)
        df_swap = self._payload_to_dataframe(raw_swap, swap_cols, apply_ffill=True)

        if df_daily.empty or df_swap.empty:
            return pd.DataFrame()

        df_daily.dropna(subset=daily_cols, inplace=True)
        if df_daily.empty:
            return pd.DataFrame()

        date_key = self._config.COL_DATE_OBJ

        # Align swap series onto active daily dates using backward as-of join
        df = pd.merge_asof(
            df_daily,
            df_swap[[date_key] + swap_cols],
            on=date_key,
            direction="backward",
        )

        # Core Net Position & Swap formulas (preserved unchanged)
        ext_assets = df["TP_AB_A02"]
        fx_liabilities = df["TP_AB_A10"]
        usd_rate = df["TP_DK_USD_A_YTL"]
        raw_swap_val = df["TP_DOVVARNC_K14"].fillna(0)

        net_reserves_try = ext_assets - fx_liabilities
        net_reserves_usd_bn = (net_reserves_try / usd_rate) / 1_000_000
        swap_debt_usd_bn = raw_swap_val / 1_000
        net_position_usd_bn = net_reserves_usd_bn + swap_debt_usd_bn

        df[self._config.COL_NET_RESERVES] = net_reserves_usd_bn.round(2)
        df[self._config.COL_SWAP_DEBT] = swap_debt_usd_bn.round(2)
        df[self._config.COL_NET_POS_EX_SWAP] = net_position_usd_bn.round(2)

        # Support both individual metric dict structure and legacy combined 'items' dict
        if "items" in extra_metrics_data and isinstance(extra_metrics_data["items"], list):
            legacy_map = {
                "TP_UR_A02": self._config.COL_GROSS_FX_RESERVES,
                "TP_AB_C2": self._config.COL_GROSS_FX_RESERVES,
                "TP_UR_A03": self._config.COL_GOLD_RESERVES,
                "TP_AB_C1": self._config.COL_GOLD_RESERVES,
                "TP_MK_F_YI_G1": self._config.COL_FOREIGN_EQUITY_FLOW,
                "TP_MK_F_YI_G2": self._config.COL_FOREIGN_BOND_FLOW,
            }
            raw_cols = list(legacy_map.keys())
            df_extra = self._payload_to_dataframe(extra_metrics_data, raw_cols, apply_ffill=True)
            for col in self._config.extra_metric_columns:
                df[col] = float("nan")
            if not df_extra.empty:
                merged_extra = pd.merge_asof(
                    df[[date_key]],
                    df_extra[[date_key] + raw_cols],
                    on=date_key,
                    direction="backward",
                )
                for raw_col, target_col in legacy_map.items():
                    if raw_col in merged_extra.columns and not merged_extra[raw_col].dropna().empty:
                        df[target_col] = merged_extra[raw_col].ffill().round(2)
        else:
            for target_col in self._config.extra_metric_columns:
                metric_info = extra_metrics_data.get(target_col)
                if not isinstance(metric_info, dict) or not metric_info.get("items"):
                    df[target_col] = float("nan")
                    continue

                col_key = str(metric_info.get("column_key", ""))
                df_metric = self._payload_to_dataframe(
                    {"items": metric_info["items"]},
                    expected_cols=[col_key],
                    apply_ffill=True,
                )
                if df_metric.empty or col_key not in df_metric.columns:
                    df[target_col] = float("nan")
                    continue

                df_metric = df_metric.dropna(subset=[col_key]).sort_values(date_key)
                if df_metric.empty:
                    df[target_col] = float("nan")
                    continue

                aligned = pd.merge_asof(
                    df[[date_key]],
                    df_metric[[date_key, col_key]],
                    on=date_key,
                    direction="backward",
                )
                df[target_col] = aligned[col_key].ffill().round(2)

        output_columns = [
            self._config.COL_DATE,
            self._config.COL_NET_RESERVES,
            self._config.COL_SWAP_DEBT,
            self._config.COL_NET_POS_EX_SWAP,
            self._config.COL_GROSS_FX_RESERVES,
            self._config.COL_GOLD_RESERVES,
            self._config.COL_FOREIGN_EQUITY_FLOW,
            self._config.COL_FOREIGN_BOND_FLOW,
        ]

        core_subset = [
            self._config.COL_NET_RESERVES,
            self._config.COL_SWAP_DEBT,
            self._config.COL_NET_POS_EX_SWAP,
        ]
        result_df = df[output_columns].dropna(subset=core_subset).reset_index(drop=True)
        return result_df


class DataRepository:
    """Manages reading, incremental updating, and persisting the TSV dataset."""

    def __init__(self, config: EvdsConfig) -> None:
        self._config = config

    def load_dataset(self, file_path: str | None = None) -> pd.DataFrame:
        """Loads and sorts the saved TSV dataset by date."""
        target = file_path or self._config.output_tsv_path
        if not os.path.exists(target):
            raise FileNotFoundError(f"Dataset file not found: {target}")

        df = pd.read_csv(target, sep="\t", encoding="utf-8-sig")
        df[self._config.COL_DATE_OBJ] = DataProcessor.parse_evds_dates(
            df[self._config.COL_DATE]
        )
        df.sort_values(self._config.COL_DATE_OBJ, inplace=True)

        extra_cols = [
            col for col in self._config.extra_metric_columns if col in df.columns
        ]
        if extra_cols:
            df[extra_cols] = df[extra_cols].ffill()

        df.reset_index(drop=True, inplace=True)
        return df

    def save_dataset(self, df: pd.DataFrame, file_path: str | None = None) -> int:
        """Merges new records with existing cache and writes updated TSV file."""
        if df.empty:
            logger.info("No new records to save.")
            return 0

        target = file_path or self._config.output_tsv_path
        if os.path.exists(target):
            try:
                existing_df = pd.read_csv(target, sep="\t", encoding="utf-8-sig")
                combined = pd.concat([existing_df, df], ignore_index=True)
                combined.drop_duplicates(
                    subset=[self._config.COL_DATE], keep="last", inplace=True
                )
                combined["_sort_dt"] = DataProcessor.parse_evds_dates(
                    combined[self._config.COL_DATE]
                )
                combined.sort_values("_sort_dt", inplace=True)
                combined.drop(columns=["_sort_dt"], inplace=True)

                extra_cols = [
                    col
                    for col in self._config.extra_metric_columns
                    if col in combined.columns
                ]
                if extra_cols:
                    combined[extra_cols] = combined[extra_cols].ffill()

                new_count = max(0, len(combined) - len(existing_df))
                combined.to_csv(target, index=False, sep="\t", encoding="utf-8-sig")
                return new_count
            except Exception as exc:
                logger.warning("Failed to merge existing cache (%s); overwriting.", exc)

        df.to_csv(target, index=False, sep="\t", encoding="utf-8-sig")
        return len(df)


class PositionAnalyzer:
    """Computes descriptive summary statistics across tracked metrics."""

    def __init__(self, config: EvdsConfig) -> None:
        self._config = config

    def print_console_table(self, df: pd.DataFrame, tail_count: int = 10) -> None:
        """Prints recent rows of core and extended metrics to stdout."""
        if df.empty:
            return

        print(f"\n{'=' * 115}")
        print(
            f"{'Date':<12} | {'Net Res (Bn$)':>13} | {'Swaps (Bn$)':>12} | "
            f"{'Net Pos w/o Swap':>16} | {'Gross FX':>11} | {'Gold Res':>11} | "
            f"{'Eq Flow':>10} | {'Bond Flow':>10}"
        )
        print("-" * 115)

        for _, row in df.tail(tail_count).iterrows():
            print(
                f"{str(row[self._config.COL_DATE]):<12} | "
                f"{row[self._config.COL_NET_RESERVES]:>13.2f} | "
                f"{row[self._config.COL_SWAP_DEBT]:>12.2f} | "
                f"{row[self._config.COL_NET_POS_EX_SWAP]:>16.2f} | "
                f"{row.get(self._config.COL_GROSS_FX_RESERVES, float('nan')):>11.2f} | "
                f"{row.get(self._config.COL_GOLD_RESERVES, float('nan')):>11.2f} | "
                f"{row.get(self._config.COL_FOREIGN_EQUITY_FLOW, float('nan')):>10.2f} | "
                f"{row.get(self._config.COL_FOREIGN_BOND_FLOW, float('nan')):>10.2f}"
            )
        print(f"{'=' * 115}\n")

    def print_summary_stats(self, df: pd.DataFrame) -> None:
        """Prints min, max, mean, and current statistics for each metric."""
        if df.empty:
            return

        date_col = self._config.COL_DATE
        print(f"{'=' * 75}")
        print("             CENTRAL BANK RESERVES & FLOWS DATA ANALYSIS")
        print(f"{'=' * 75}")
        print(f"Data Range : {df[date_col].iloc[0]} to {df[date_col].iloc[-1]}")
        print(f"Total Days : {len(df)}")
        print(f"{'-' * 75}")

        metrics = [
            (self._config.COL_NET_RESERVES, "bn $"),
            (self._config.COL_SWAP_DEBT, "bn $"),
            (self._config.COL_NET_POS_EX_SWAP, "bn $"),
            (self._config.COL_GROSS_FX_RESERVES, "mn $"),
            (self._config.COL_GOLD_RESERVES, "mn $"),
            (self._config.COL_FOREIGN_EQUITY_FLOW, "mn $"),
            (self._config.COL_FOREIGN_BOND_FLOW, "mn $"),
        ]

        for col, unit in metrics:
            if col not in df.columns or df[col].dropna().empty:
                continue

            valid = df.dropna(subset=[col])
            title = col.replace("_", " ")
            current_val = valid[col].iloc[-1]
            mean_val = valid[col].mean()
            min_val = valid[col].min()
            max_val = valid[col].max()

            min_date = valid.loc[valid[col] == min_val, date_col].iloc[0]
            max_date = valid.loc[valid[col] == max_val, date_col].iloc[0]

            print(f"{title}:")
            print(f"  Current  : {current_val:,.2f} {unit}")
            print(f"  Average  : {mean_val:,.2f} {unit}")
            print(f"  Minimum  : {min_val:,.2f} {unit} (on {min_date})")
            print(f"  Maximum  : {max_val:,.2f} {unit} (on {max_date})")
            print(f"{'-' * 75}")