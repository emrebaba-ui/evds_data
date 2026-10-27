import logging
from typing import Any, Dict, List, Optional, Sequence
import requests

from config import EvdsConfig

logger = logging.getLogger(__name__)


class EvdsApiError(Exception):
    """Custom exception for EVDS API communication and response errors."""


class EvdsApiClient:
    """Fetches time-series and metadata from the official TCMB EVDS3 REST API."""

    def __init__(
        self,
        config: EvdsConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "key": self._config.API_KEY,
                "Accept": "application/json, text/plain, */*",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            }
        )

    def build_endpoint_url(
        self,
        series: Sequence[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agg_types: Optional[Sequence[str]] = None,
        formulas: Optional[Sequence[str]] = None,
        frequency: Optional[str] = None,
    ) -> str:
        """Constructs the EVDS3 path-parameter endpoint URL."""
        if not series:
            raise ValueError("At least one series code must be provided.")

        series_count = len(series)
        series_str = "-".join(series)
        start = start_date or self._config.START_DATE
        end = end_date or self._config.END_DATE
        agg_str = "-".join(agg_types) if agg_types else "-".join(["last"] * series_count)
        formula_str = "-".join(formulas) if formulas else "-".join(["0"] * series_count)

        query_parts = [
            f"series={series_str}",
            f"startDate={start}",
            f"endDate={end}",
            "type=json",
            f"aggregationTypes={agg_str}",
            f"formulas={formula_str}",
        ]
        if frequency:
            query_parts.append(f"frequency={frequency}")

        return f"{self._config.BASE_API_URL}{'&'.join(query_parts)}"

    def _get_json(self, url: str, context_label: str) -> Any:
        """Executes HTTP GET and validates JSON response structure."""
        logger.debug("Requesting EVDS URL: %s", url)
        try:
            response = self._session.get(
                url,
                timeout=self._config.REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise EvdsApiError(
                f"EVDS request timed out after {self._config.REQUEST_TIMEOUT}s "
                f"for '{context_label}'."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "Unknown"
            body_preview = (
                exc.response.text[:200].strip() if exc.response is not None else ""
            )
            raise EvdsApiError(
                f"EVDS HTTP error ({status}) for '{context_label}'. "
                f"Response preview: {body_preview!r}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise EvdsApiError(
                f"Network error while querying EVDS for '{context_label}': {exc}"
            ) from exc

        raw_text = (response.text or "").strip()
        content_type = response.headers.get("Content-Type", "unknown")

        if not raw_text:
            raise EvdsApiError(
                f"Empty response body received from EVDS (HTTP {response.status_code}, "
                f"Content-Type: {content_type}) for '{context_label}'."
            )

        if raw_text.lstrip().startswith("<"):
            snippet = " ".join(raw_text[:200].split())
            raise EvdsApiError(
                f"HTML response received instead of JSON from {response.url} "
                f"(HTTP {response.status_code}, Content-Type: {content_type}) "
                f"for '{context_label}'. Snippet: {snippet!r}"
            )

        try:
            return response.json()
        except ValueError as exc:
            snippet = " ".join(raw_text[:200].split())
            raise EvdsApiError(
                f"Invalid JSON received from EVDS (HTTP {response.status_code}, "
                f"Content-Type: {content_type}) for '{context_label}'. "
                f"Raw response preview: {snippet!r}"
            ) from exc

    def fetch_series(
        self,
        series: Sequence[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agg_types: Optional[Sequence[str]] = None,
        formulas: Optional[Sequence[str]] = None,
        frequency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Requests one or more EVDS series and returns validated JSON payload."""
        endpoint = self.build_endpoint_url(
            series=series,
            start_date=start_date,
            end_date=end_date,
            agg_types=agg_types,
            formulas=formulas,
            frequency=frequency,
        )
        series_str = "-".join(series)
        payload = self._get_json(endpoint, context_label=f"series '{series_str}'")

        if (
            not isinstance(payload, dict)
            or "items" not in payload
            or not isinstance(payload["items"], list)
        ):
            raise EvdsApiError(
                f"Unexpected EVDS response schema for series '{series_str}': "
                f"missing 'items' list."
            )

        return payload

    def fetch_datagroup_series_list(self, datagroup_code: str) -> List[Dict[str, Any]]:
        """Fetches official series metadata list for a datagroup via EVDS3 serieList API."""
        url = f"{self._config.BASE_API_URL}serieList/type=json&code={datagroup_code}"
        data = self._get_json(url, context_label=f"serieList '{datagroup_code}'")
        if isinstance(data, list):
            return data
        return []

    def discover_portfolio_flow_codes(self) -> Dict[str, str]:
        """Discovers exact Equity and Bond (DIBS) Net Flow series codes from bie_mknethar."""
        discovered: Dict[str, str] = {}
        try:
            series_meta = self.fetch_datagroup_series_list(
                self._config.PORTFOLIO_FLOWS_DATAGROUP
            )
        except EvdsApiError as exc:
            logger.warning("Could not query serieList for %s: %s", self._config.PORTFOLIO_FLOWS_DATAGROUP, exc)
            return discovered

        equity_candidates: List[str] = []
        bond_candidates: List[str] = []

        for item in series_meta:
            code = str(item.get("SERIE_CODE", "")).strip()
            name = str(item.get("SERIE_NAME", "")).strip()
            name_upper = name.upper()
            if not code:
                continue

            # In bie_mknethar:
            # 1.1.1 Hisse Senedi (Stok), 2.1.1 Hisse Senedi (Net Değişim)
            # 1.1.2 DİBS (Kesin Alım) (Stok), 2.1.2 DİBS (Kesin Alım) (Net Değişim)
            if "2.1.1" in name or ("NET" in name_upper and "HİSSE" in name_upper):
                discovered[self._config.COL_FOREIGN_EQUITY_FLOW] = code
            elif "HİSSE SENEDİ" in name_upper or "HISSE SENEDI" in name_upper:
                equity_candidates.append(code)

            if "2.1.2" in name and "2.1.2.i" not in name.lower():
                discovered[self._config.COL_FOREIGN_BOND_FLOW] = code
            elif (
                ("DİBS (KESİN ALIM)" in name_upper or "DIBS (KESIN ALIM)" in name_upper)
                and "ŞUBE" not in name_upper
            ):
                bond_candidates.append(code)

        # If section numbers (2.1.1 / 2.1.2) are not in SERIE_NAME, the 2nd occurrence
        # in bie_mknethar corresponds to 'Net Değişim' (after 'Stok')
        if self._config.COL_FOREIGN_EQUITY_FLOW not in discovered and len(equity_candidates) >= 2:
            discovered[self._config.COL_FOREIGN_EQUITY_FLOW] = equity_candidates[1]
        elif self._config.COL_FOREIGN_EQUITY_FLOW not in discovered and equity_candidates:
            discovered[self._config.COL_FOREIGN_EQUITY_FLOW] = equity_candidates[0]

        if self._config.COL_FOREIGN_BOND_FLOW not in discovered and len(bond_candidates) >= 2:
            discovered[self._config.COL_FOREIGN_BOND_FLOW] = bond_candidates[1]
        elif self._config.COL_FOREIGN_BOND_FLOW not in discovered and bond_candidates:
            discovered[self._config.COL_FOREIGN_BOND_FLOW] = bond_candidates[0]

        return discovered

    def fetch_extra_metrics_individually(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetches each additional metric individually with automatic code discovery and fallback."""
        results: Dict[str, Dict[str, Any]] = {}
        discovered_flows = self.discover_portfolio_flow_codes()

        for target_col, default_candidates in self._config.EXTRA_METRIC_CANDIDATES.items():
            candidates: List[str] = []
            if target_col in discovered_flows:
                candidates.append(discovered_flows[target_col])
            for code in default_candidates:
                if code not in candidates:
                    candidates.append(code)

            fetched_payload: Optional[Dict[str, Any]] = None
            used_code: Optional[str] = None

            for code in candidates:
                try:
                    payload = self.fetch_series(
                        series=[code],
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if payload.get("items"):
                        fetched_payload = payload
                        used_code = code
                        logger.info(
                            "Fetched metric '%s' using EVDS series '%s' (%d records).",
                            target_col,
                            code,
                            len(payload["items"]),
                        )
                        break
                except EvdsApiError as exc:
                    logger.debug(
                        "Candidate series '%s' for '%s' failed: %s",
                        code,
                        target_col,
                        exc,
                    )

            if fetched_payload and used_code:
                results[target_col] = {
                    "series_code": used_code,
                    "column_key": used_code.replace(".", "_"),
                    "items": fetched_payload["items"],
                }
            else:
                logger.warning(
                    "Could not fetch valid data for '%s' (tried: %s).",
                    target_col,
                    ", ".join(candidates),
                )

        return results

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self._session.close()