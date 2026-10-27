import logging
from typing import Any, List, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from config import EvdsConfig

logger = logging.getLogger(__name__)


class Visualizer:
    """Renders static and interactive Matplotlib charts for CBRT metrics."""

    def __init__(self, config: EvdsConfig) -> None:
        self._config = config

    def plot_dashboard(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        show_interactive: bool = True,
    ) -> str:
        """Plots CBRT net position, reserves, and foreign flows, and saves PNG."""
        if df.empty:
            raise ValueError("Cannot plot an empty DataFrame.")

        plot_df = df.copy()
        if self._config.COL_DATE_OBJ not in plot_df.columns:
            plot_df[self._config.COL_DATE_OBJ] = pd.to_datetime(
                plot_df[self._config.COL_DATE], format="mixed", dayfirst=True
            )

        output_path = save_path or self._config.output_chart_path

        fig, (ax_main, ax_res, ax_flows) = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(13, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.4, 1.4]},
        )

        # Panel 1: Core Net Position & Swap Debt (Billion USD)
        line_reserves, = ax_main.plot(
            plot_df[self._config.COL_DATE_OBJ],
            plot_df[self._config.COL_NET_RESERVES],
            label="Net Reserves",
            color="#10B981",
            linewidth=2.0,
        )
        line_swap, = ax_main.plot(
            plot_df[self._config.COL_DATE_OBJ],
            plot_df[self._config.COL_SWAP_DEBT].abs(),
            label="Swap Debt (Absolute)",
            color="#EF4444",
            linewidth=1.5,
            linestyle="--",
        )
        line_net_pos, = ax_main.plot(
            plot_df[self._config.COL_DATE_OBJ],
            plot_df[self._config.COL_NET_POS_EX_SWAP],
            label="Net Position (excl. Swap)",
            color="#3B82F6",
            linewidth=2.5,
        )

        ax_main.set_title(
            "CBRT Net Position, Gross Reserves & Foreign Portfolio Flows",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax_main.set_ylabel("Billion USD ($)", fontsize=10, labelpad=8)
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")

        # Panel 2: Gross FX & Gold Reserves
        if self._config.COL_GROSS_FX_RESERVES in plot_df.columns:
            ax_res.plot(
                plot_df[self._config.COL_DATE_OBJ],
                plot_df[self._config.COL_GROSS_FX_RESERVES],
                label="Gross FX Reserves (TP.UR.A02)",
                color="#0EA5E9",
                linewidth=1.8,
            )
        if self._config.COL_GOLD_RESERVES in plot_df.columns:
            ax_res.plot(
                plot_df[self._config.COL_DATE_OBJ],
                plot_df[self._config.COL_GOLD_RESERVES],
                label="Gold Reserves (TP.UR.A03)",
                color="#F59E0B",
                linewidth=1.8,
            )
        ax_res.set_ylabel("Million USD ($)", fontsize=10, labelpad=8)
        ax_res.grid(True, linestyle=":", alpha=0.6)
        ax_res.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")

        # Panel 3: Foreign Equity & Bond (DIBS) Net Flows
        if self._config.COL_FOREIGN_EQUITY_FLOW in plot_df.columns:
            ax_flows.plot(
                plot_df[self._config.COL_DATE_OBJ],
                plot_df[self._config.COL_FOREIGN_EQUITY_FLOW],
                label="Foreign Equity Net Flow (TP.MK.F.YI.G1)",
                color="#8B5CF6",
                linewidth=1.6,
            )
        if self._config.COL_FOREIGN_BOND_FLOW in plot_df.columns:
            ax_flows.plot(
                plot_df[self._config.COL_DATE_OBJ],
                plot_df[self._config.COL_FOREIGN_BOND_FLOW],
                label="Foreign Bond/DIBS Net Flow (TP.MK.F.YI.G2)",
                color="#EC4899",
                linewidth=1.6,
            )
        ax_flows.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
        ax_flows.set_xlabel("Date", fontsize=11, labelpad=8)
        ax_flows.set_ylabel("Million USD ($)", fontsize=10, labelpad=8)
        ax_flows.grid(True, linestyle=":", alpha=0.6)
        ax_flows.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logger.info("Chart saved to %s", output_path)

        self._attach_hover_tooltip(
            fig=fig,
            ax=ax_main,
            lines=[line_reserves, line_swap, line_net_pos],
            names=["Net Reserves", "Swap Debt", "Net Position"],
        )

        if show_interactive:
            try:
                plt.show()
            except Exception as exc:
                logger.debug("Interactive GUI display skipped: %s", exc)
        else:
            plt.close(fig)

        return output_path

    @staticmethod
    def _attach_hover_tooltip(
        fig: plt.Figure,
        ax: plt.Axes,
        lines: List[Line2D],
        names: List[str],
    ) -> None:
        """Attaches interactive mouse-hover annotations to the main chart axis."""
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.5", "fc": "#1F2937", "ec": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "#4B5563"},
            color="white",
            fontsize=10,
            fontweight="semibold",
        )
        annot.set_visible(False)

        def update_annot(line: Line2D, idx: int, name: str) -> None:
            x_data, y_data = line.get_data()
            date_val = x_data[idx]
            y_val = y_data[idx]
            date_str = pd.to_datetime(date_val).strftime("%d-%m-%Y")

            annot.xy = (date_val, y_val)
            annot.set_text(f"Date: {date_str}\n{name}: {y_val:,.2f} bn $")
            annot.get_bbox_patch().set_edgecolor(line.get_color())
            annot.get_bbox_patch().set_linewidth(1.5)

        def on_hover(event: Any) -> None:
            is_visible = annot.get_visible()
            if event.inaxes == ax:
                for line, name in zip(lines, names):
                    contains, ind = line.contains(event)
                    if contains and len(ind.get("ind", [])) > 0:
                        update_annot(line, int(ind["ind"][0]), name)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if is_visible:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)