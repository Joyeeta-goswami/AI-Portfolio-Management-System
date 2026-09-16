"""
Monte Carlo Portfolio Simulation

Simulates future portfolio wealth using bootstrap sampling
from the training-period historical strategy returns.

Strategies:
- Max Sharpe
- Min Variance
- Risk Parity

The simulation produces:
- Terminal wealth distribution
- Probability of loss
- Expected terminal wealth
- Median terminal wealth
- 5th and 95th percentile outcomes
- Simulated maximum drawdown
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports from the project root when this script
# is executed directly.
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# CONFIGURATION
# ============================================================



DATA_DIR = PROJECT_ROOT / "data" / "processed"

RETURNS_FILE = DATA_DIR / "returns.csv"

OUTPUT_FILE = DATA_DIR / "monte_carlo_results.csv"
PATH_OUTPUT_FILE = DATA_DIR / "monte_carlo_terminal_wealth.csv"
DRAWDOWN_OUTPUT_FILE = DATA_DIR / "monte_carlo_drawdowns.csv"

INITIAL_WEALTH = 100.0

SIMULATIONS = 10_000

HORIZON_DAYS = 252

RANDOM_SEED = 42

TRADING_DAYS = 252


# ============================================================
# PORTFOLIO RETURN GENERATION
# ============================================================

def generate_training_strategy_returns():
    """
    Reconstruct the three portfolio strategies using only the
    training-period observations.

    Returns
    -------
    pd.DataFrame
        Daily training-period strategy returns.
    """

    from optimization.backtest import PortfolioBacktester

    print("=" * 70)
    print("GENERATING TRAINING-PERIOD STRATEGY RETURNS")
    print("=" * 70)

    backtester = PortfolioBacktester(
        returns_file=str(RETURNS_FILE),
        train_ratio=0.80
    )

    backtester.load_returns()

    backtester.split_data()

    # --------------------------------------------------------
    # Markowitz portfolios
    # --------------------------------------------------------

    max_sharpe, min_variance = (
        backtester.generate_training_markowitz_portfolios()
    )

    # --------------------------------------------------------
    # Risk Parity portfolio
    # --------------------------------------------------------

    covariance = (
        backtester.calculate_training_covariance()
    )

    risk_parity = (
        backtester.optimize_risk_parity(
            covariance
        )
    )

    # --------------------------------------------------------
    # Extract portfolio weights
    # --------------------------------------------------------

    weights = {
        "Max Sharpe": max_sharpe.x,
        "Min Variance": min_variance.x,
        "Risk Parity": risk_parity.x
    }

    asset_columns = (
        backtester.train_returns.columns[1:]
    )

    train_data = backtester.train_returns.copy()

    train_data = (
        train_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Calculate strategy returns
    # --------------------------------------------------------

    train_data["Max Sharpe"] = (
        train_data[asset_columns].values
        @ weights["Max Sharpe"]
    )

    train_data["Min Variance"] = (
        train_data[asset_columns].values
        @ weights["Min Variance"]
    )

    train_data["Risk Parity"] = (
        train_data[asset_columns].values
        @ weights["Risk Parity"]
    )

    print(
        f"\nTraining observations: "
        f"{len(train_data)}"
    )

    print(
        f"Training period: "
        f"{train_data['date'].min().date()} "
        f"→ "
        f"{train_data['date'].max().date()}"
    )

    return train_data[
        [
            "date",
            "Max Sharpe",
            "Min Variance",
            "Risk Parity"
        ]
    ]


# ============================================================
# BOOTSTRAP SIMULATION
# ============================================================

def simulate_portfolio(
    historical_returns,
    simulations=SIMULATIONS,
    horizon=HORIZON_DAYS,
    initial_wealth=INITIAL_WEALTH,
    seed=RANDOM_SEED
):
    """
    Bootstrap historical daily returns to generate future
    portfolio paths.

    Sampling is performed with replacement.

    Parameters
    ----------
    historical_returns : pd.Series
        Historical strategy daily returns.

    simulations : int
        Number of simulated paths.

    horizon : int
        Number of trading days in each path.

    initial_wealth : float
        Starting portfolio value.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    terminal_wealth : np.ndarray
    maximum_drawdowns : np.ndarray
    paths : np.ndarray
    """

    historical_returns = (
        pd.Series(historical_returns)
        .dropna()
        .values
    )

    if len(historical_returns) == 0:
        raise ValueError(
            "No historical returns available for simulation."
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Bootstrap sample
    # --------------------------------------------------------

    sampled_returns = rng.choice(
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
            1 + sampled_returns,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Add initial wealth
    # --------------------------------------------------------

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
    # Maximum drawdown
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
# SUMMARY STATISTICS
# ============================================================

def calculate_simulation_metrics(
    terminal_wealth,
    maximum_drawdowns,
    initial_wealth=INITIAL_WEALTH
):
    """
    Calculate summary statistics from simulated outcomes.
    """

    terminal_returns = (
        terminal_wealth / initial_wealth
    ) - 1

    probability_of_loss = (
        np.mean(terminal_wealth < initial_wealth)
    )

    probability_of_profit = (
        np.mean(terminal_wealth > initial_wealth)
    )

    probability_of_large_loss = (
        np.mean(
            terminal_returns < -0.10
        )
    )

    metrics = {
        "Expected Terminal Wealth":
            np.mean(terminal_wealth),

        "Median Terminal Wealth":
            np.median(terminal_wealth),

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
            probability_of_large_loss,

        "Expected Maximum Drawdown":
            np.mean(maximum_drawdowns),

        "Median Maximum Drawdown":
            np.median(maximum_drawdowns),

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
# MAIN MONTE CARLO ANALYSIS
# ============================================================

def run_monte_carlo():

    print("\n")
    print("=" * 70)
    print("MONTE CARLO PORTFOLIO SIMULATION")
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
    # Generate training-period strategy returns
    # --------------------------------------------------------

    strategy_returns = (
        generate_training_strategy_returns()
    )

    results = []

    terminal_wealth_all = []

    drawdown_all = []

    # ========================================================
    # SIMULATE EACH STRATEGY
    # ========================================================

    for strategy in [
        "Max Sharpe",
        "Min Variance",
        "Risk Parity"
    ]:

        print("\n")
        print("-" * 70)
        print(f"Simulating: {strategy}")
        print("-" * 70)

        historical_returns = (
            strategy_returns[strategy]
        )

        (
            terminal_wealth,
            maximum_drawdowns,
            wealth_paths
        ) = simulate_portfolio(
            historical_returns=historical_returns,
            simulations=SIMULATIONS,
            horizon=HORIZON_DAYS,
            initial_wealth=INITIAL_WEALTH,
            seed=RANDOM_SEED
        )

        metrics = calculate_simulation_metrics(
            terminal_wealth,
            maximum_drawdowns
        )

        metrics["Strategy"] = strategy

        results.append(metrics)

        # ----------------------------------------------------
        # Save terminal wealth
        # ----------------------------------------------------

        terminal_df = pd.DataFrame({
            "Strategy": strategy,
            "Simulation": np.arange(
                1,
                SIMULATIONS + 1
            ),
            "Terminal Wealth": terminal_wealth
        })

        terminal_wealth_all.append(
            terminal_df
        )

        # ----------------------------------------------------
        # Save drawdowns
        # ----------------------------------------------------

        drawdown_df = pd.DataFrame({
            "Strategy": strategy,
            "Simulation": np.arange(
                1,
                SIMULATIONS + 1
            ),
            "Maximum Drawdown": maximum_drawdowns
        })

        drawdown_all.append(
            drawdown_df
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            f"Expected terminal wealth : "
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
            f"Probability of loss >10% : "
            f"{metrics['Probability of Loss > 10%']:.2%}"
        )

        print(
            f"Expected max drawdown    : "
            f"{metrics['Expected Maximum Drawdown']:.2%}"
        )

        print(
            f"Median max drawdown      : "
            f"{metrics['Median Maximum Drawdown']:.2%}"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # Put Strategy first
    columns = [
        "Strategy"
    ] + [
        col
        for col in results_df.columns
        if col != "Strategy"
    ]

    results_df = results_df[
        columns
    ]

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SAVE TERMINAL WEALTH
    # ========================================================

    terminal_wealth_df = pd.concat(
        terminal_wealth_all,
        ignore_index=True
    )

    terminal_wealth_df.to_csv(
        PATH_OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SAVE DRAWDOWNS
    # ========================================================

    drawdown_df = pd.concat(
        drawdown_all,
        ignore_index=True
    )

    drawdown_df.to_csv(
        DRAWDOWN_OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("MONTE CARLO SUMMARY")
    print("=" * 70)

    display_columns = [
        "Strategy",
        "Expected Terminal Wealth",
        "Median Terminal Wealth",
        "5th Percentile Terminal Wealth",
        "95th Percentile Terminal Wealth",
        "Probability of Loss",
        "Probability of Loss > 10%",
        "Expected Maximum Drawdown"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n")
    print("Files saved:")
    print(
        OUTPUT_FILE
    )
    print(
        PATH_OUTPUT_FILE
    )
    print(
        DRAWDOWN_OUTPUT_FILE
    )

    return results_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_monte_carlo()