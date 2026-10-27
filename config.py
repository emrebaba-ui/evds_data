import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


class ConfigurationError(Exception):
    """Raised when required configuration or environment variables are missing or invalid."""


def load_env_file(env_path: Optional[str] = None) -> None:
    """Loads key-value pairs from a .env file into os.environ without external dependencies."""
    target_path = Path(
        env_path
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    )

    try:
        from dotenv import load_dotenv  # type: ignore

        if target_path.exists():
            load_dotenv(dotenv_path=target_path, override=False)
            return
    except ImportError:
        pass

    if not target_path.exists():
        return

    for raw_line in target_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        elif "#" in value:
            value = value.split("#", 1)[0].strip()

        if key and key not in os.environ:
            os.environ[key] = value


def _sanitize_env_value(val: Optional[str]) -> str:
    """Strips whitespace and accidental wrapping quotes from an environment variable."""
    if not val:
        return ""
    cleaned = val.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


@dataclass(frozen=True)
class EvdsConfig:
    """Configuration settings, EVDS3 endpoints, and series definitions."""

    # Paths
    BASE_DIR: str = field(
        default_factory=lambda: os.path.dirname(os.path.abspath(__file__))
    )
    ENV_FILE: Optional[str] = None
    OUTPUT_FILE: str = "central_bank_net_position.tsv"
    OUTPUT_IMAGE: str = "net_position_chart.png"

    # Authentication & Official EVDS3 Endpoint
    API_KEY: str = field(default="")
    BASE_API_URL: str = field(
        default_factory=lambda: _sanitize_env_value(os.getenv("EVDS_BASE_URL"))
        or "https://evds3.tcmb.gov.tr/igmevdsms-dis/"
    )
    REQUEST_TIMEOUT: int = 30

    # Default Date Range
    START_DATE: str = "01-01-2026"
    END_DATE: str = "31-12-2026"

    # Core Daily Series (Analytical Balance Sheet & FX Rate)
    DAILY_SERIES: Tuple[str, ...] = (
        "TP.AB.A02",       # Analytical Balance Sheet: Foreign Assets (TRY)
        "TP.AB.A10",       # Analytical Balance Sheet: Total FX Liabilities (TRY)
        "TP.DK.USD.A.YTL", # USD/TRY Buying Exchange Rate
    )

    # Core Swap Series
    SWAP_SERIES: Tuple[str, ...] = (
        "TP.DOVVARNC.K14", # Central Bank Swap Transactions
    )

    # Output Column Names
    COL_DATE: str = "Date"
    COL_DATE_OBJ: str = "Date_Obj"
    COL_NET_RESERVES: str = "Net_Reserves_Bn_USD"
    COL_SWAP_DEBT: str = "Swap_Debt_Bn_USD"
    COL_NET_POS_EX_SWAP: str = "Net_Position_Without_Swap"
    COL_GROSS_FX_RESERVES: str = "Gross_FX_Reserves"
    COL_GOLD_RESERVES: str = "Gold_Reserves"
    COL_FOREIGN_EQUITY_FLOW: str = "Foreign_Equity_Net_Flow"
    COL_FOREIGN_BOND_FLOW: str = "Foreign_Bond_Net_Flow"

    # Candidate EVDS3 Series Codes per Target Metric (tried in order before datagroup discovery)
    # Note: In EVDS3, Gross FX Reserves is TP.AB.C2 and Gold Reserves is TP.AB.C1
    # under 'Merkez Bankası Rezervleri' (bie_abres2), while Foreign Portfolio Flows
    # are under 'bie_mknethar' (TP.MKNETHAR.*). Legacy/prompt codes are also kept as candidates.
    EXTRA_METRIC_CANDIDATES: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "Gross_FX_Reserves": ("TP.AB.C2", "TP.UR.A02"),
            "Gold_Reserves": ("TP.AB.C1", "TP.UR.A03"),
            "Foreign_Equity_Net_Flow": (
                "TP.MKNETHAR.M6",
                "TP.MKNETHAR.M7",
                "TP.MK.F.YI.G1",
            ),
            "Foreign_Bond_Net_Flow": (
                "TP.MKNETHAR.M7",
                "TP.MKNETHAR.M8",
                "TP.MK.F.YI.G2",
            ),
        }
    )

    # Datagroups used for dynamic series code discovery via EVDS3 serieList endpoint
    RESERVES_DATAGROUP: str = "bie_abres2"
    PORTFOLIO_FLOWS_DATAGROUP: str = "bie_mknethar"

    def __post_init__(self) -> None:
        """Loads .env file and validates the EVDS API key and base URL."""
        env_target = self.ENV_FILE or os.path.join(self.BASE_DIR, ".env")
        load_env_file(env_target)

        resolved_key = _sanitize_env_value(self.API_KEY or os.getenv("EVDS_API_KEY"))
        if not resolved_key:
            raise ConfigurationError(
                f"EVDS_API_KEY is missing. Define EVDS_API_KEY=\"...\" in {env_target} "
                "or set it as an environment variable."
            )

        object.__setattr__(self, "API_KEY", resolved_key)

        if not self.BASE_API_URL.endswith("/"):
            object.__setattr__(self, "BASE_API_URL", f"{self.BASE_API_URL}/")

    @property
    def extra_metric_columns(self) -> Tuple[str, ...]:
        """Returns the tuple of additional metric column names."""
        return (
            self.COL_GROSS_FX_RESERVES,
            self.COL_GOLD_RESERVES,
            self.COL_FOREIGN_EQUITY_FLOW,
            self.COL_FOREIGN_BOND_FLOW,
        )

    @property
    def output_tsv_path(self) -> str:
        """Returns absolute path for the TSV data file."""
        return os.path.join(self.BASE_DIR, self.OUTPUT_FILE)

    @property
    def output_chart_path(self) -> str:
        """Returns absolute path for the chart image."""
        return os.path.join(self.BASE_DIR, self.OUTPUT_IMAGE)