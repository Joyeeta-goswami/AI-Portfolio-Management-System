"""
Monte Carlo Simulation for Risk-Aware Regime Switching

Simulates the adaptive regime-switching strategy by
resampling the historical out-of-sample daily returns
according to the strategy actually selected on each day.

Strategies used by the regime-switching system:
- Max Sharpe
- Min Variance
- Risk Parity

Outputs:
- Terminal wealth distribution
- Probability of loss
- Probability of loss > 10%
- Maximum drawdown distribution
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"

REGIME_FILE = (
    DATA_DIR / "regime_strategy_test_returns.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "monte_carlo_regime_results.csv"
)

TERMINAL_WEALTH_FILE = (
    DATA_DIR / "monte_carlo_regime_terminal_wealth.csv"
)

DRAWDOWN_FILE = (
    DATA_DIR / "monte_carlo_regime_drawdowns.csv"
)

INITIAL_WEALTH = 100.0

SIMULATIONS = 10_000

HORIZON_DAYS = 252

RANDOM_SEED = 42

RISK_FREE_RATE = 0.065

TRADING_DAYS = 252


# ============================================================
# LOAD REGIME-SWITCHING RETURNS
# ============================================================

def load_regime_returns():
    """
    Load the historical out-of-sample regime-switching
    daily returns.

    The existing file contains:
    - date
    - regime
    - Selected Strategy
    - Regime Strategy Return
    """

    if not REGIME_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{REGIME_FILE}"
        )

    data = pd.read_csv(
        REGIME_FILE,
        parse_dates=["date"]
    )

    required_columns = [
        "date",
        "regime",
        "Selected Strategy",
        "Regime Strategy Return"
    ]

    missing = [
        col
        for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in regime file: {missing}"
        )

    data = (
        data[
            required_columns
        ]
        .dropna(
            subset=["Regime Strategy Return"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(data) == 0:
        raise ValueError(
            "No valid regime-switching returns found."
        )

    return data


# ============================================================
# SIMULATION
# ============================================================

def simulate_regime_strategy(
    historical_returns,
    simulations=SIMULATIONS,
    horizon=HORIZON_DAYS,
    initial_wealth=INITIAL_WEALTH,
    seed=RANDOM_SEED
):
    """
    Bootstrap daily regime-switching returns.

    The empirical distribution of realized regime-switching
    returns is sampled with replacement.

    This preserves the observed return distribution of the
    adaptive strategy without assuming normality.
    """

    historical_returns = (
        pd.Series(historical_returns)
        .dropna()
        .to_numpy()
    )

    if len(historical_returns) == 0:
        raise ValueError(
            "Historical return series is empty."
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Bootstrap sample
    # --------------------------------------------------------

    simulated_returns = rng.choice(
        historical_returns,
        size=(simulations, horizon),
        replace=True
    )

    # --------------------------------------------------------
    # Wealth paths
    # --------------------------------------------------------

    wealth_paths = (
        initial_wealth
        * np.cumprod(
            1 + simulated_returns,
            axis=1
        )
    )

    # Add initial wealth
    wealth_paths = np.column_stack(
        [
            np.full(
                simulations,
                initial_wealth
            ),
            wealth_paths
        ]
    )

    # --------------------------------------------------------
    # Terminal wealth
    # --------------------------------------------------------

    terminal_wealth = (
        wealth_paths[:, -1]
    )

    # --------------------------------------------------------
    # Drawdowns
    # --------------------------------------------------------

    running_max = np.maximum.accumulate(
        wealth_paths,
        axis=1
    )

    drawdowns = (
        wealth_paths / running_max
    ) - 1

    maximum_drawdowns = (
        drawdowns.min(axis=1)
    )

    return (
        terminal_wealth,
        maximum_drawdowns,
        wealth_paths
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    terminal_wealth,
    maximum_drawdowns
):
    """
    Calculate Monte Carlo summary statistics.
    """

    terminal_returns = (
        terminal_wealth / INITIAL_WEALTH
    ) - 1

    probability_of_loss = np.mean(
        terminal_wealth < INITIAL_WEALTH
    )

    probability_of_profit = np.mean(
        terminal_wealth > INITIAL_WEALTH
    )

    probability_loss_gt_10 = np.mean(
        terminal_returns < -0.10
    )

    expected_terminal_wealth = np.mean(
        terminal_wealth
    )

    median_terminal_wealth = np.median(
        terminal_wealth
    )

    metrics = {
        "Strategy":
            "Risk-Aware Regime Switching",

        "Expected Terminal Wealth":
            expected_terminal_wealth,

        "Median Terminal Wealth":
            median_terminal_wealth,

        "5th Percentile Terminal Wealth":
            np.percentile(
                terminal_wealth,
                5
            ),

        "95th Percentile Terminal Wealth":
            np.percentile(
                terminal_wealth,
                95
            ),

        "Minimum Terminal Wealth":
            np.min(terminal_wealth),

        "Maximum Terminal Wealth":
            np.max(terminal_wealth),

        "Probability of Loss":
            probability_of_loss,

        "Probability of Profit":
            probability_of_profit,

        "Probability of Loss > 10%":
            probability_loss_gt_10,

        "Expected Maximum Drawdown":
            np.mean(
                maximum_drawdowns
            ),

        "Median Maximum Drawdown":
            np.median(
                maximum_drawdowns
            ),

        "5th Percentile Maximum Drawdown":
            np.percentile(
                maximum_drawdowns,
                5
            ),

        "95th Percentile Maximum Drawdown":
            np.percentile(
                maximum_drawdowns,
                95
            )
    }

    return metrics


# ============================================================
# STRATEGY USAGE SUMMARY
# ============================================================

def print_strategy_usage(data):
    """
    Show how frequently each underlying portfolio strategy
    was selected during the historical test period.
    """

    print("\n")
    print("=" * 70)
    print("REGIME-SWITCHING STRATEGY USAGE")
    print("=" * 70)

    counts = (
        data["Selected Strategy"]
        .value_counts()
        .rename("Observations")
    )

    percentages = (
        data["Selected Strategy"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
    )

    usage = pd.concat(
        [
            counts,
            percentages
        ],
        axis=1
    )

    print(
        usage.to_string()
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_monte_carlo_regime():

    print("=" * 70)
    print("MONTE CARLO: RISK-AWARE REGIME SWITCHING")
    print("=" * 70)

    print(
        f"\nNumber of simulations : "
        f"{SIMULATIONS:,}"
    )

    print(
        f"Horizon               : "
        f"{HORIZON_DAYS} trading days"
    )

    print(
        f"Initial wealth        : "
        f"{INITIAL_WEALTH:.2f}"
    )

    print(
        f"Random seed           : "
        f"{RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_regime_returns()

    print(
        f"\nHistorical observations: "
        f"{len(data)}"
    )

    print(
        f"Historical period: "
        f"{data['date'].min().date()} "
        f"→ "
        f"{data['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Display underlying strategy usage
    # --------------------------------------------------------

    print_strategy_usage(
        data
    )

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RUNNING BOOTSTRAP SIMULATION")
    print("=" * 70)

    (
        terminal_wealth,
        maximum_drawdowns,
        wealth_paths
    ) = simulate_regime_strategy(
        historical_returns=(
            data["Regime Strategy Return"]
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        terminal_wealth,
        maximum_drawdowns
    )

    results_df = pd.DataFrame(
        [metrics]
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save terminal wealth
    # --------------------------------------------------------

    terminal_df = pd.DataFrame({
        "Strategy":
            "Risk-Aware Regime Switching",

        "Simulation":
            np.arange(
                1,
                SIMULATIONS + 1
            ),

        "Terminal Wealth":
            terminal_wealth
    })

    terminal_df.to_csv(
        TERMINAL_WEALTH_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save drawdowns
    # --------------------------------------------------------

    drawdown_df = pd.DataFrame({
        "Strategy":
            "Risk-Aware Regime Switching",

        "Simulation":
            np.arange(
                1,
                SIMULATIONS + 1
            ),

        "Maximum Drawdown":
            maximum_drawdowns
    })

    drawdown_df.to_csv(
        DRAWDOWN_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)

    print(
        f"\nExpected terminal wealth : "
        f"{metrics['Expected Terminal Wealth']:.2f}"
    )

    print(
        f"Median terminal wealth   : "
        f"{metrics['Median Terminal Wealth']:.2f}"
    )

    print(
        f"5th percentile           : "
        f"{metrics['5th Percentile Terminal Wealth']:.2f}"
    )

    print(
        f"95th percentile          : "
        f"{metrics['95th Percentile Terminal Wealth']:.2f}"
    )

    print(
        f"Probability of loss      : "
        f"{metrics['Probability of Loss']:.2%}"
    )

    print(
        f"Probability loss > 10%   : "
        f"{metrics['Probability of Loss > 10%']:.2%}"
    )

    print(
        f"Expected maximum DD      : "
        f"{metrics['Expected Maximum Drawdown']:.2%}"
    )

    print(
        f"Median maximum DD        : "
        f"{metrics['Median Maximum Drawdown']:.2%}"
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )

    print(
        TERMINAL_WEALTH_FILE
    )

    print(
        DRAWDOWN_FILE
    )

    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_monte_carlo_regime()