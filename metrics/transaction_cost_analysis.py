"""
Transaction Cost Analysis

Calculates the impact of proportional transaction costs
on out-of-sample portfolio performance.

Strategies:
- Max Sharpe
- Min Variance
- Risk Parity
- Risk-Aware Regime Switching
"""

from pathlib import Path

import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure the project root is on sys.path so sibling packages
# (e.g. `optimization`) can be imported regardless of how or
# from where this script is run.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"

RETURNS_FILE = DATA_DIR / "returns.csv"
REGIME_TEST_FILE = DATA_DIR / "regime_strategy_test_returns.csv"

OUTPUT_FILE = DATA_DIR / "transaction_cost_results.csv"
SENSITIVITY_FILE = DATA_DIR / "transaction_cost_sensitivity.csv"

# 0.10% transaction cost per unit turnover
TRANSACTION_COST = 0.001

RISK_FREE_RATE = 0.065
TRADING_DAYS = 252


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(returns):
    """
    Calculate standard portfolio performance metrics.
    """

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "Total Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
            "VaR 95%": np.nan
        }

    total_return = (1 + returns).prod() - 1

    annualized_return = (
        (1 + total_return)
        ** (TRADING_DAYS / len(returns))
        - 1
    )

    annualized_volatility = (
        returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    if annualized_volatility > 0:
        sharpe_ratio = (
            annualized_return - RISK_FREE_RATE
        ) / annualized_volatility
    else:
        sharpe_ratio = np.nan

    wealth = (1 + returns).cumprod()

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max
    ) - 1

    maximum_drawdown = drawdown.min()

    var_95 = returns.quantile(0.05)

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": maximum_drawdown,
        "VaR 95%": var_95
    }


# ============================================================
# LOAD DAILY PORTFOLIO RETURNS
# ============================================================

def load_daily_portfolio_returns():
    """
    Reconstruct the three fixed portfolio daily returns
    using the training-period portfolio weights.

    The weights are estimated from the training period only
    and then applied to the test period.
    """

    from optimization.backtest import PortfolioBacktester

    backtester = PortfolioBacktester(
        returns_file=str(RETURNS_FILE),
        train_ratio=0.80
    )

    backtester.load_returns()
    backtester.split_data()

    # --------------------------------------------------------
    # Generate training portfolios
    # --------------------------------------------------------

    max_sharpe, min_variance = (
        backtester.generate_training_markowitz_portfolios()
    )

    covariance = (
        backtester.calculate_training_covariance()
    )

    risk_parity = (
        backtester.optimize_risk_parity(
            covariance
        )
    )

    # --------------------------------------------------------
    # Extract weights
    # --------------------------------------------------------

    weights = {
        "Max Sharpe": max_sharpe.x,
        "Min Variance": min_variance.x,
        "Risk Parity": risk_parity.x
    }

    asset_columns = backtester.train_returns.columns[1:]

    # --------------------------------------------------------
    # Test-period returns
    # --------------------------------------------------------

    test_data = backtester.test_returns.copy()

    test_data = test_data.sort_values(
        "date"
    ).reset_index(drop=True)

    test_data["Max Sharpe"] = (
        test_data[asset_columns].values
        @ weights["Max Sharpe"]
    )

    test_data["Min Variance"] = (
        test_data[asset_columns].values
        @ weights["Min Variance"]
    )

    test_data["Risk Parity"] = (
        test_data[asset_columns].values
        @ weights["Risk Parity"]
    )

    return test_data[
        [
            "date",
            "Max Sharpe",
            "Min Variance",
            "Risk Parity"
        ]
    ]


# ============================================================
# LOAD REGIME-SWITCHING RETURNS
# ============================================================

def load_regime_switching_returns():
    """
    Load the existing out-of-sample regime-switching results.
    """

    if not REGIME_TEST_FILE.exists():
        return None

    regime = pd.read_csv(
        REGIME_TEST_FILE,
        parse_dates=["date"]
    )

    required_columns = [
        "date",
        "Selected Strategy",
        "Regime Strategy Return"
    ]

    missing = [
        col
        for col in required_columns
        if col not in regime.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in regime test file: "
            f"{missing}"
        )

    regime = regime.sort_values(
        "date"
    ).reset_index(drop=True)

    return regime[
        required_columns
    ]


# ============================================================
# TURNOVER
# ============================================================

def calculate_initial_turnover(index):
    """
    For a fixed portfolio that is established at the beginning
    of the test period:

        Initial turnover = 1

    Thereafter no additional turnover is assumed because the
    portfolio is kept fixed during the test period.
    """

    turnover = pd.Series(
        0.0,
        index=index
    )

    if len(turnover) > 0:
        turnover.iloc[0] = 1.0

    return turnover


def calculate_regime_turnover(strategy_series):
    """
    Calculate turnover for the regime-switching strategy.

    Initial investment:
        turnover = 1

    Strategy change:
        turnover = 1

    No strategy change:
        turnover = 0
    """

    strategy_series = pd.Series(
        strategy_series
    ).reset_index(drop=True)

    turnover = pd.Series(
        0.0,
        index=strategy_series.index
    )

    if len(turnover) == 0:
        return turnover

    # Initial portfolio allocation
    turnover.iloc[0] = 1.0

    for i in range(1, len(strategy_series)):

        previous = strategy_series.iloc[i - 1]
        current = strategy_series.iloc[i]

        if (
            pd.notna(previous)
            and pd.notna(current)
            and previous != current
        ):
            turnover.iloc[i] = 1.0

    return turnover


# ============================================================
# APPLY TRANSACTION COST
# ============================================================

def apply_transaction_cost(
    gross_returns,
    turnover,
    transaction_cost
):
    """
    Net return:

        Net Return =
        Gross Return - Transaction Cost × Turnover
    """

    gross_returns = pd.Series(
        gross_returns
    ).reset_index(drop=True)

    turnover = pd.Series(
        turnover
    ).reset_index(drop=True)

    costs = (
        turnover
        * transaction_cost
    )

    net_returns = (
        gross_returns - costs
    )

    return net_returns


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_transaction_cost_analysis(
    transaction_cost=TRANSACTION_COST
):

    print("=" * 70)
    print("TRANSACTION COST ANALYSIS")
    print("=" * 70)

    print(
        f"\nTransaction cost assumption: "
        f"{transaction_cost:.3%} per unit turnover"
    )

    # --------------------------------------------------------
    # Load daily returns
    # --------------------------------------------------------

    portfolio_data = (
        load_daily_portfolio_returns()
    )

    regime_data = (
        load_regime_switching_returns()
    )

    print(
        f"\nTest observations: "
        f"{len(portfolio_data)}"
    )

    results = []

    # ========================================================
    # FIXED PORTFOLIOS
    # ========================================================

    for strategy in [
        "Max Sharpe",
        "Min Variance",
        "Risk Parity"
    ]:

        gross_returns = (
            portfolio_data[strategy]
            .dropna()
            .reset_index(drop=True)
        )

        turnover = (
            calculate_initial_turnover(
                gross_returns.index
            )
        )

        net_returns = apply_transaction_cost(
            gross_returns,
            turnover,
            transaction_cost
        )

        gross_metrics = calculate_metrics(
            gross_returns
        )

        net_metrics = calculate_metrics(
            net_returns
        )

        total_turnover = turnover.sum()

        total_transaction_cost = (
            total_turnover
            * transaction_cost
        )

        results.append({
            "Strategy": strategy,

            "Gross Total Return":
                gross_metrics["Total Return"],

            "Net Total Return":
                net_metrics["Total Return"],

            "Gross Annualized Return":
                gross_metrics["Annualized Return"],

            "Net Annualized Return":
                net_metrics["Annualized Return"],

            "Gross Volatility":
                gross_metrics["Annualized Volatility"],

            "Net Volatility":
                net_metrics["Annualized Volatility"],

            "Gross Sharpe":
                gross_metrics["Sharpe Ratio"],

            "Net Sharpe":
                net_metrics["Sharpe Ratio"],

            "Gross Maximum Drawdown":
                gross_metrics["Maximum Drawdown"],

            "Net Maximum Drawdown":
                net_metrics["Maximum Drawdown"],

            "Gross VaR 95%":
                gross_metrics["VaR 95%"],

            "Net VaR 95%":
                net_metrics["VaR 95%"],

            "Total Turnover":
                total_turnover,

            "Transaction Cost":
                total_transaction_cost
        })

    # ========================================================
    # REGIME SWITCHING
    # ========================================================

    if regime_data is not None:

        regime_returns = (
            regime_data[
                "Regime Strategy Return"
            ]
            .dropna()
            .reset_index(drop=True)
        )

        selected_strategy = (
            regime_data[
                "Selected Strategy"
            ]
            .dropna()
            .reset_index(drop=True)
        )

        turnover = (
            calculate_regime_turnover(
                selected_strategy
            )
        )

        net_returns = apply_transaction_cost(
            regime_returns,
            turnover,
            transaction_cost
        )

        gross_metrics = calculate_metrics(
            regime_returns
        )

        net_metrics = calculate_metrics(
            net_returns
        )

        total_turnover = turnover.sum()

        total_transaction_cost = (
            total_turnover
            * transaction_cost
        )

        results.append({
            "Strategy":
                "Risk-Aware Regime Switching",

            "Gross Total Return":
                gross_metrics["Total Return"],

            "Net Total Return":
                net_metrics["Total Return"],

            "Gross Annualized Return":
                gross_metrics["Annualized Return"],

            "Net Annualized Return":
                net_metrics["Annualized Return"],

            "Gross Volatility":
                gross_metrics["Annualized Volatility"],

            "Net Volatility":
                net_metrics["Annualized Volatility"],

            "Gross Sharpe":
                gross_metrics["Sharpe Ratio"],

            "Net Sharpe":
                net_metrics["Sharpe Ratio"],

            "Gross Maximum Drawdown":
                gross_metrics["Maximum Drawdown"],

            "Net Maximum Drawdown":
                net_metrics["Maximum Drawdown"],

            "Gross VaR 95%":
                gross_metrics["VaR 95%"],

            "Net VaR 95%":
                net_metrics["VaR 95%"],

            "Total Turnover":
                total_turnover,

            "Transaction Cost":
                total_transaction_cost
        })

    # ========================================================
    # SAVE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\nPerformance after transaction costs")
    print("-" * 70)

    for _, row in results_df.iterrows():

        print(
            f"\n{row['Strategy']}"
        )

        print(
            f"Gross Total Return : "
            f"{row['Gross Total Return']:.4%}"
        )

        print(
            f"Net Total Return   : "
            f"{row['Net Total Return']:.4%}"
        )

        print(
            f"Gross Sharpe       : "
            f"{row['Gross Sharpe']:.3f}"
        )

        print(
            f"Net Sharpe         : "
            f"{row['Net Sharpe']:.3f}"
        )

        print(
            f"Total Turnover     : "
            f"{row['Total Turnover']:.2f}"
        )

        print(
            f"Transaction Cost   : "
            f"{row['Transaction Cost']:.4%}"
        )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )

    return results_df


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity_analysis():

    print("\n")
    print("=" * 70)
    print("TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 70)

    cost_levels = [
        0.0000,   # 0.00%
        0.0005,   # 0.05%
        0.0010,   # 0.10%
        0.0015,   # 0.15%
        0.0020    # 0.20%
    ]

    portfolio_data = (
        load_daily_portfolio_returns()
    )

    regime_data = (
        load_regime_switching_returns()
    )

    sensitivity_results = []

    # ========================================================
    # FIXED STRATEGIES
    # ========================================================

    for cost in cost_levels:

        for strategy in [
            "Max Sharpe",
            "Min Variance",
            "Risk Parity"
        ]:

            gross_returns = (
                portfolio_data[strategy]
                .dropna()
                .reset_index(drop=True)
            )

            turnover = (
                calculate_initial_turnover(
                    gross_returns.index
                )
            )

            net_returns = apply_transaction_cost(
                gross_returns,
                turnover,
                cost
            )

            metrics = calculate_metrics(
                net_returns
            )

            sensitivity_results.append({
                "Transaction Cost":
                    cost,

                "Transaction Cost (%)":
                    cost * 100,

                "Strategy":
                    strategy,

                "Net Total Return":
                    metrics["Total Return"],

                "Net Annualized Return":
                    metrics["Annualized Return"],

                "Net Volatility":
                    metrics["Annualized Volatility"],

                "Net Sharpe":
                    metrics["Sharpe Ratio"],

                "Net Maximum Drawdown":
                    metrics["Maximum Drawdown"]
            })

    # ========================================================
    # REGIME SWITCHING
    # ========================================================

    if regime_data is not None:

        gross_returns = (
            regime_data[
                "Regime Strategy Return"
            ]
            .dropna()
            .reset_index(drop=True)
        )

        selected_strategy = (
            regime_data[
                "Selected Strategy"
            ]
            .dropna()
            .reset_index(drop=True)
        )

        turnover = (
            calculate_regime_turnover(
                selected_strategy
            )
        )

        for cost in cost_levels:

            net_returns = (
                apply_transaction_cost(
                    gross_returns,
                    turnover,
                    cost
                )
            )

            metrics = calculate_metrics(
                net_returns
            )

            sensitivity_results.append({
                "Transaction Cost":
                    cost,

                "Transaction Cost (%)":
                    cost * 100,

                "Strategy":
                    "Risk-Aware Regime Switching",

                "Net Total Return":
                    metrics["Total Return"],

                "Net Annualized Return":
                    metrics["Annualized Return"],

                "Net Volatility":
                    metrics["Annualized Volatility"],

                "Net Sharpe":
                    metrics["Sharpe Ratio"],

                "Net Maximum Drawdown":
                    metrics["Maximum Drawdown"]
            })

    sensitivity_df = pd.DataFrame(
        sensitivity_results
    )

    sensitivity_df.to_csv(
        SENSITIVITY_FILE,
        index=False
    )

    print(
        "\nSensitivity results saved to:"
    )

    print(
        SENSITIVITY_FILE
    )

    print("\n")
    print(
        sensitivity_df.to_string(
            index=False
        )
    )

    return sensitivity_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_transaction_cost_analysis(
        transaction_cost=TRANSACTION_COST
    )

    run_sensitivity_analysis()