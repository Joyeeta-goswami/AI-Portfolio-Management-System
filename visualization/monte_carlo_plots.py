"""
Monte Carlo Visualization

Creates visualizations for the Monte Carlo portfolio simulation.

Outputs:
1. Terminal wealth distributions
2. Maximum drawdown distributions
3. Probability of loss comparison
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TERMINAL_WEALTH_FILE = (
    DATA_DIR / "monte_carlo_terminal_wealth.csv"
)

DRAWDOWN_FILE = (
    DATA_DIR / "monte_carlo_drawdowns.csv"
)

RESULTS_FILE = (
    DATA_DIR / "monte_carlo_results.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "visualization" / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    terminal_wealth = pd.read_csv(
        TERMINAL_WEALTH_FILE
    )

    drawdowns = pd.read_csv(
        DRAWDOWN_FILE
    )

    results = pd.read_csv(
        RESULTS_FILE
    )

    return (
        terminal_wealth,
        drawdowns,
        results
    )


# ============================================================
# TERMINAL WEALTH DISTRIBUTION
# ============================================================

def plot_terminal_wealth(
    terminal_wealth
):

    strategies = [
        "Max Sharpe",
        "Min Variance",
        "Risk Parity"
    ]

    plt.figure(
        figsize=(11, 6)
    )

    for strategy in strategies:

        data = terminal_wealth.loc[
            terminal_wealth["Strategy"] == strategy,
            "Terminal Wealth"
        ]

        plt.hist(
            data,
            bins=60,
            alpha=0.35,
            density=True,
            label=strategy
        )

    plt.axvline(
        100,
        linestyle="--",
        linewidth=2,
        label="Initial Wealth = 100"
    )

    plt.title(
        "Monte Carlo Distribution of Terminal Wealth"
    )

    plt.xlabel(
        "Terminal Wealth"
    )

    plt.ylabel(
        "Density"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "terminal_wealth_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# MAXIMUM DRAWDOWN DISTRIBUTION
# ============================================================

def plot_drawdown_distribution(
    drawdowns
):

    strategies = [
        "Max Sharpe",
        "Min Variance",
        "Risk Parity"
    ]

    plt.figure(
        figsize=(11, 6)
    )

    for strategy in strategies:

        data = drawdowns.loc[
            drawdowns["Strategy"] == strategy,
            "Maximum Drawdown"
        ]

        plt.hist(
            data * 100,
            bins=60,
            alpha=0.35,
            density=True,
            label=strategy
        )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2
    )

    plt.title(
        "Monte Carlo Distribution of Maximum Drawdown"
    )

    plt.xlabel(
        "Maximum Drawdown (%)"
    )

    plt.ylabel(
        "Density"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "maximum_drawdown_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# PROBABILITY OF LOSS
# ============================================================

def plot_probability_of_loss(
    results
):

    plot_data = results[
        [
            "Strategy",
            "Probability of Loss",
            "Probability of Loss > 10%"
        ]
    ].copy()

    plot_data["Probability of Loss"] *= 100

    plot_data["Probability of Loss > 10%"] *= 100

    strategies = plot_data[
        "Strategy"
    ]

    x = np.arange(
        len(strategies)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        plot_data["Probability of Loss"],
        width,
        label="Probability of Loss"
    )

    plt.bar(
        x + width / 2,
        plot_data["Probability of Loss > 10%"],
        width,
        label="Probability of Loss > 10%"
    )

    plt.xticks(
        x,
        strategies
    )

    plt.ylabel(
        "Probability (%)"
    )

    plt.title(
        "Monte Carlo Downside Risk Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "monte_carlo_downside_risk.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results):

    print("\n")
    print("=" * 75)
    print("MONTE CARLO VISUALIZATION SUMMARY")
    print("=" * 75)

    columns = [
        "Strategy",
        "Expected Terminal Wealth",
        "Median Terminal Wealth",
        "5th Percentile Terminal Wealth",
        "95th Percentile Terminal Wealth",
        "Probability of Loss",
        "Probability of Loss > 10%",
        "Expected Maximum Drawdown"
    ]

    summary = results[
        columns
    ].copy()

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("MONTE CARLO VISUALIZATION")
    print("=" * 75)

    (
        terminal_wealth,
        drawdowns,
        results
    ) = load_data()

    print(
        f"\nTerminal wealth observations: "
        f"{len(terminal_wealth):,}"
    )

    print(
        f"Drawdown observations: "
        f"{len(drawdowns):,}"
    )

    print_summary(
        results
    )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    print("\nCreating terminal wealth distribution...")
    plot_terminal_wealth(
        terminal_wealth
    )

    print("\nCreating maximum drawdown distribution...")
    plot_drawdown_distribution(
        drawdowns
    )

    print("\nCreating downside-risk comparison...")
    plot_probability_of_loss(
        results
    )

    print("\n")
    print("=" * 75)
    print("ALL MONTE CARLO PLOTS CREATED SUCCESSFULLY")
    print("=" * 75)

    print(
        f"\nPlots saved in:\n{OUTPUT_DIR}"
    )