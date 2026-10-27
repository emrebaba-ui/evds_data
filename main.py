import logging
import os
import sys
import pandas as pd

from api_client import EvdsApiClient, EvdsApiError
from config import ConfigurationError, EvdsConfig
from data_processor import (
    DataProcessingError,
    DataProcessor,
    DataRepository,
    PositionAnalyzer,
)
from visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ApplicationOrchestrator:
    """Coordinates EVDS data retrieval, processing, persistence, and plotting."""

    def __init__(
        self,
        config: EvdsConfig,
        api_client: EvdsApiClient,
        processor: DataProcessor,
        repository: DataRepository,
        analyzer: PositionAnalyzer,
        visualizer: Visualizer,
    ) -> None:
        self._config = config
        self._client = api_client
        self._processor = processor
        self._repository = repository
        self._analyzer = analyzer
        self._visualizer = visualizer

    def _resolve_start_date(self) -> str:
        """Determines incremental start date from local TSV cache if all columns exist."""
        tsv_path = self._config.output_tsv_path
        if not os.path.exists(tsv_path):
            return self._config.START_DATE

        try:
            existing_df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
            required_cols = set(self._config.extra_metric_columns)
            if not required_cols.issubset(existing_df.columns):
                logger.info("New reserve/flow columns missing in cache. Performing full refresh.")
                return self._config.START_DATE

            # If any of the extra columns are completely NaN in cache, trigger full refresh
            if any(existing_df[col].dropna().empty for col in required_cols):
                logger.info("One or more reserve/flow columns are empty in cache. Performing full refresh.")
                return self._config.START_DATE

            if not existing_df.empty and self._config.COL_DATE in existing_df.columns:
                dates = DataProcessor.parse_evds_dates(existing_df[self._config.COL_DATE])
                latest_date = dates.max()
                next_day = latest_date + pd.Timedelta(days=1)
                return next_day.strftime("%d-%m-%Y")
        except Exception as exc:
            logger.warning("Could not read existing cache (%s). Fetching full range.", exc)

        return self._config.START_DATE

    def run(self, show_interactive_chart: bool = True) -> None:
        """Executes end-to-end ETL, statistical summary, and chart generation."""
        start_date = self._resolve_start_date()
        end_date = pd.Timestamp.now().strftime("%d-%m-%Y")

        start_dt = pd.to_datetime(start_date, dayfirst=True)
        end_dt = pd.to_datetime(end_date, dayfirst=True)

        if start_dt.date() <= end_dt.date():
            logger.info("Fetching EVDS data from %s to %s...", start_date, end_date)

            # 1. Fetch daily balance sheet & USD rate
            raw_daily = self._client.fetch_series(
                series=self._config.DAILY_SERIES,
                start_date=start_date,
                end_date=end_date,
            )

            # 2. Fetch swap series from initial baseline for as-of lookback
            raw_swap = self._client.fetch_series(
                series=self._config.SWAP_SERIES,
                start_date=self._config.START_DATE,
                end_date=end_date,
            )

            # 3. Fetch weekly reserve & foreign portfolio flow metrics individually
            # (Prevents multi-series frequency/code mismatch HTTP 400 errors)
            extra_metrics_data = self._client.fetch_extra_metrics_individually(
                start_date=self._config.START_DATE,
                end_date=end_date,
            )

            df_new = self._processor.build_dataset(
                raw_daily=raw_daily,
                raw_swap=raw_swap,
                extra_metrics_data=extra_metrics_data,
            )

            if not df_new.empty:
                added = self._repository.save_dataset(df_new)
                logger.info(
                    "Saved %d new/updated record(s) to %s.",
                    added,
                    self._config.OUTPUT_FILE,
                )
            else:
                logger.info("No new trading day records returned from EVDS.")
        else:
            logger.info("Local dataset is already up to date.")

        # Load full dataset for reporting and visualization
        full_df = self._repository.load_dataset()
        self._analyzer.print_console_table(full_df)
        self._analyzer.print_summary_stats(full_df)

        chart_path = self._visualizer.plot_dashboard(
            full_df,
            show_interactive=show_interactive_chart,
        )
        logger.info("Pipeline completed. Chart exported to: %s", chart_path)


def main() -> int:
    """Application entry point."""
    try:
        config = EvdsConfig()
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    client = EvdsApiClient(config)
    processor = DataProcessor(config)
    repository = DataRepository(config)
    analyzer = PositionAnalyzer(config)
    visualizer = Visualizer(config)

    orchestrator = ApplicationOrchestrator(
        config=config,
        api_client=client,
        processor=processor,
        repository=repository,
        analyzer=analyzer,
        visualizer=visualizer,
    )

    try:
        orchestrator.run()
        return 0
    except (EvdsApiError, DataProcessingError, FileNotFoundError) as exc:
        logger.error("Operation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected fatal error: %s", exc)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())