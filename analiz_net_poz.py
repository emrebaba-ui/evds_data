import os
import sys
import time
import traceback
import pandas as pd
import matplotlib.pyplot as plt

# Lock working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class AnalysisConfig:
    """Configuration for data analysis and visualization."""

    INPUT_FILE = "central_bank_net_position.tsv"
    OUTPUT_IMAGE = "net_position_chart.png"

    COL_DATE = "Date"
    COL_RESERVES = "Net_Reserves_Bn_USD"
    COL_SWAP = "Swap_Debt_Bn_USD"
    COL_NET_POS = "Net_Position_Without_Swap"


class DataLoader:
    """Handles file reading and data preprocessing."""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file '{file_path}' not found in {os.getcwd()}"
            )

        df = pd.read_csv(file_path, sep="\t", encoding="utf-8-sig")
        df["Date_Obj"] = pd.to_datetime(df[AnalysisConfig.COL_DATE], dayfirst=True)
        df.sort_values("Date_Obj", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


class PositionAnalyzer:
    """Calculates statistics and detects data anomalies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns_to_analyze = [
            AnalysisConfig.COL_RESERVES,
            AnalysisConfig.COL_SWAP,
            AnalysisConfig.COL_NET_POS,
        ]

    def print_summary(self) -> None:
        print(f"{'='*70}")
        print("          CENTRAL BANK NET POSITION DATA ANALYSIS")
        print(f"{'='*70}")
        print(
            f"Data Range : {self.df[AnalysisConfig.COL_DATE].iloc[0]} to {self.df[AnalysisConfig.COL_DATE].iloc[-1]}"
        )
        print(f"Total Days : {len(self.df)}")
        print(f"{'-'*70}")

        for col in self.columns_to_analyze:
            self._print_column_stats(col)

    def _print_column_stats(self, col: str) -> None:
        title = col.replace("_", " ")
        current_val = self.df[col].iloc[-1]
        mean_val = self.df[col].mean()
        min_val = self.df[col].min()
        max_val = self.df[col].max()

        min_date = self.df.loc[self.df[col] == min_val, AnalysisConfig.COL_DATE].values[
            0
        ]
        max_date = self.df.loc[self.df[col] == max_val, AnalysisConfig.COL_DATE].values[
            0
        ]

        print(f"{title}:")
        print(f"  Current  : {current_val:,.2f} bn $")
        print(f"  Average  : {mean_val:,.2f} bn $")
        print(f"  Minimum  : {min_val:,.2f} bn $ (on {min_date})")
        print(f"  Maximum  : {max_val:,.2f} bn $ (on {max_date})")
        print(f"{'-'*70}")


class Visualizer:
    """Generates charts using matplotlib."""

    @staticmethod
    def plot_data(df: pd.DataFrame, save_path: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot lines. Note: Swap debt is plotted as an absolute (positive) value visually.
        line_reserves, = ax.plot(
            df["Date_Obj"],
            df[AnalysisConfig.COL_RESERVES],
            label="Net Reserves",
            color="#10B981",
            linewidth=2,
        )
        line_swap, = ax.plot(
            df["Date_Obj"],
            df[AnalysisConfig.COL_SWAP].abs(),
            label="Swap Debt (Absolute)",
            color="#EF4444",
            linewidth=1.5,
            linestyle="--",
        )
        line_net_pos, = ax.plot(
            df["Date_Obj"],
            df[AnalysisConfig.COL_NET_POS],
            label="Net Position (excl. Swap)",
            color="#3B82F6",
            linewidth=2.5,
        )

        # Formatting
        ax.set_title("CBRT Net Position Analysis", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=11, labelpad=10)
        ax.set_ylabel("Billion USD ($)", fontsize=11, labelpad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb")
        plt.tight_layout()

        # Save static image
        plt.savefig(save_path, dpi=300)
        print(f"[+] Chart saved as: {save_path}")

        # Setup custom hover tooltip
        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#1F2937", ec="none", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#4B5563"),
            color="white",
            fontsize=10,
            fontweight="semibold",
        )
        annot.set_visible(False)

        lines = [line_reserves, line_swap, line_net_pos]
        names = ["Net Reserves", "Swap Debt", "Net Position"]

        def update_annot(line, idx, name):
            x, y = line.get_data()
            date_val = x[idx]
            y_val = y[idx]
            date_str = pd.to_datetime(date_val).strftime("%d-%m-%Y")
            
            annot.xy = (date_val, y_val)
            text = f"Date: {date_str}\n{name}: {y_val:,.2f} bn $"
            annot.set_text(text)
            annot.get_bbox_patch().set_edgecolor(line.get_color())
            annot.get_bbox_patch().set_linewidth(1.5)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                for line, name in zip(lines, names):
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(line, ind["ind"][0], name)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        print("\n[*] Displaying interactive window. Hover to inspect values, close window to exit.")
        try:
            plt.show()
        except Exception as e:
            print(f"[!] Could not open GUI window: {e}")


def main():
    print("Loading data...")
    df = DataLoader.load_data(AnalysisConfig.INPUT_FILE)

    analyzer = PositionAnalyzer(df)
    analyzer.print_summary()

    Visualizer.plot_data(df, AnalysisConfig.OUTPUT_IMAGE)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*50}\nFATAL ERROR CAUGHT!\n{'='*50}")
        traceback.print_exc()
        input("\n[Press Enter to exit...]")
