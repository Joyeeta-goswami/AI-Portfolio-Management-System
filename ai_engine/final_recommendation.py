"""
Final Integrated Portfolio Recommendation Engine

Combines evidence from:
1. Out-of-sample backtesting
2. NIFTY 50 benchmark comparison
3. Transaction-cost analysis
4. Monte Carlo simulation
5. Regime-switching Monte Carlo

This is an interpretable multi-criteria decision engine.
It is NOT a black-box machine-learning model.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    DATA_DIR / "final_recommendation_results.csv"
)

SUMMARY_FILE = (
    DATA_DIR / "final_recommendation_summary.csv"
)


# ============================================================
# STRATEGIES
# ============================================================

STRATEGIES = [
    "Max Sharpe",
    "Min Variance",
    "Risk Parity",
    "Risk-Aware Regime Switching"
]


# ============================================================
# WEIGHTS
# ============================================================

# Weights represent the importance assigned to each evidence
# category. They are deliberately transparent.

WEIGHTS = {
    "Backtest": 0.25,
    "Benchmark": 0.15,
    "Transaction Cost": 0.15,
    "Monte Carlo": 0.25,
    "Downside Risk": 0.20
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    # --------------------------------------------------------
    # Backtest results
    # --------------------------------------------------------

    backtest_file = (
        DATA_DIR / "backtest_results.csv"
    )

    backtest = pd.read_csv(
        backtest_file
    )

    # --------------------------------------------------------
    # Benchmark-adjusted results
    # --------------------------------------------------------

    benchmark_file = (
        DATA_DIR / "benchmark_adjusted_results.csv"
    )

    benchmark = pd.read_csv(
        benchmark_file
    )

    # --------------------------------------------------------
    # Transaction-cost results
    # --------------------------------------------------------

    transaction_file = (
        DATA_DIR / "transaction_cost_results.csv"
    )

    transaction = pd.read_csv(
        transaction_file
    )

    # --------------------------------------------------------
    # Monte Carlo fixed strategies
    # --------------------------------------------------------

    monte_carlo_file = (
        DATA_DIR / "monte_carlo_results.csv"
    )

    monte_carlo = pd.read_csv(
        monte_carlo_file
    )

    # --------------------------------------------------------
    # Regime Monte Carlo
    # --------------------------------------------------------

    regime_mc_file = (
        DATA_DIR /
        "monte_carlo_regime_results.csv"
    )

    regime_mc = pd.read_csv(
        regime_mc_file
    )

    return (
        backtest,
        benchmark,
        transaction,
        monte_carlo,
        regime_mc
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_strategy_name(name):
    """
    Standardize strategy names between output files.
    """

    mapping = {
        "Maximum Sharpe": "Max Sharpe",
        "Maximum Sharpe Ratio": "Max Sharpe",
        "Minimum Variance": "Min Variance",
        "Risk Parity": "Risk Parity",
        "Regime Switching":
            "Risk-Aware Regime Switching",
        "Risk-Aware Regime Switching":
            "Risk-Aware Regime Switching"
    }

    return mapping.get(
        name,
        name
    )


def min_max_score(
    series,
    higher_is_better=True
):
    """
    Normalize values between 0 and 1.

    1 = strongest observed value
    0 = weakest observed value
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()

    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            0.5,
            index=series.index
        )

    if np.isclose(
        minimum,
        maximum
    ):
        return pd.Series(
            0.5,
            index=series.index
        )

    if higher_is_better:

        return (
            (series - minimum)
            / (maximum - minimum)
        )

    return (
        (maximum - series)
        / (maximum - minimum)
    )


def prepare_strategy_table():

    return pd.DataFrame({
        "Strategy": STRATEGIES
    })


# ============================================================
# BACKTEST SCORE
# ============================================================

def calculate_backtest_score(
    strategy_table,
    backtest
):

    backtest = backtest.copy()

    backtest["Strategy"] = (
        backtest["Strategy"]
        .map(clean_strategy_name)
    )

    # --------------------------------------------------------
    # Keep only our strategies
    # --------------------------------------------------------

    backtest = backtest[
        backtest["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    backtest = backtest.set_index(
        "Strategy"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    annual_return = min_max_score(
        backtest["Annualized Return"],
        higher_is_better=True
    )

    sharpe = min_max_score(
        backtest["Sharpe Ratio"],
        higher_is_better=True
    )

    volatility = min_max_score(
        backtest["Annualized Volatility"],
        higher_is_better=False
    )

    drawdown = min_max_score(
        backtest["Maximum Drawdown"],
        higher_is_better=True
    )

    # Since drawdowns are negative, the less-negative value
    # receives the higher score.

    var_score = min_max_score(
        backtest["VaR (95%)"],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Internal weighted score
    # --------------------------------------------------------

    score = (
        0.30 * annual_return
        + 0.25 * sharpe
        + 0.15 * volatility
        + 0.20 * drawdown
        + 0.10 * var_score
    )

    score = (
        score
        .reindex(STRATEGIES)
        .fillna(0.5)
    )

    strategy_table[
        "Backtest Score"
    ] = strategy_table[
        "Strategy"
    ].map(score)

    return strategy_table


# ============================================================
# BENCHMARK SCORE
# ============================================================

def calculate_benchmark_score(
    strategy_table,
    benchmark
):

    benchmark = benchmark.copy()

    benchmark["Strategy"] = (
        benchmark["Strategy"]
        .map(clean_strategy_name)
    )

    benchmark = benchmark[
        benchmark["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    # --------------------------------------------------------
    # Benchmark-relative return
    # --------------------------------------------------------

    # Extract NIFTY row if available
    nifty_row = benchmark[
        benchmark["Strategy"].str.upper()
        == "NIFTY 50"
    ]

    strategy_rows = benchmark[
        benchmark["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    if len(strategy_rows) == 0:
        strategy_table[
            "Benchmark Score"
        ] = 0.5

        return strategy_table

    # --------------------------------------------------------
    # Use relative annualized return where possible
    # --------------------------------------------------------

    strategy_return = min_max_score(
        strategy_rows["Annualized Return"],
        higher_is_better=True
    )

    strategy_sharpe = min_max_score(
        strategy_rows["Sharpe Ratio"],
        higher_is_better=True
    )

    strategy_volatility = min_max_score(
        strategy_rows["Annualized Volatility"],
        higher_is_better=False
    )

    scores = (
        0.50 * strategy_return
        + 0.30 * strategy_sharpe
        + 0.20 * strategy_volatility
    )

    score_map = (
        pd.Series(
            scores.values,
            index=strategy_rows["Strategy"]
        )
    )

    strategy_table[
        "Benchmark Score"
    ] = strategy_table[
        "Strategy"
    ].map(score_map).fillna(0.5)

    return strategy_table


# ============================================================
# TRANSACTION COST SCORE
# ============================================================

def calculate_transaction_cost_score(
    strategy_table,
    transaction
):

    transaction = transaction.copy()

    transaction["Strategy"] = (
        transaction["Strategy"]
        .map(clean_strategy_name)
    )

    transaction = transaction[
        transaction["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    if len(transaction) == 0:

        strategy_table[
            "Transaction Cost Score"
        ] = 0.5

        return strategy_table

    net_return_score = min_max_score(
        transaction["Net Annualized Return"],
        higher_is_better=True
    )

    turnover_score = min_max_score(
        transaction["Total Turnover"],
        higher_is_better=False
    )

    score = (
        0.70 * net_return_score
        + 0.30 * turnover_score
    )

    score_map = pd.Series(
        score.values,
        index=transaction["Strategy"]
    )

    strategy_table[
        "Transaction Cost Score"
    ] = strategy_table[
        "Strategy"
    ].map(
        score_map
    ).fillna(0.5)

    return strategy_table


# ============================================================
# MONTE CARLO SCORE
# ============================================================

def calculate_monte_carlo_score(
    strategy_table,
    monte_carlo,
    regime_mc
):

    monte_carlo = monte_carlo.copy()

    monte_carlo["Strategy"] = (
        monte_carlo["Strategy"]
        .map(clean_strategy_name)
    )

    regime_mc = regime_mc.copy()

    regime_mc["Strategy"] = (
        regime_mc["Strategy"]
        .map(clean_strategy_name)
    )

    # --------------------------------------------------------
    # Add regime-switching Monte Carlo result
    # --------------------------------------------------------

    combined = pd.concat(
        [
            monte_carlo,
            regime_mc
        ],
        ignore_index=True
    )

    combined = combined[
        combined["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    # --------------------------------------------------------
    # Higher expected wealth = better
    # --------------------------------------------------------

    wealth_score = min_max_score(
        combined[
            "Expected Terminal Wealth"
        ],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Higher median wealth = better
    # --------------------------------------------------------

    median_score = min_max_score(
        combined[
            "Median Terminal Wealth"
        ],
        higher_is_better=True
    )

    # --------------------------------------------------------
    # Lower probability of loss = better
    # --------------------------------------------------------

    loss_score = min_max_score(
        combined[
            "Probability of Loss"
        ],
        higher_is_better=False
    )

    # --------------------------------------------------------
    # Lower probability of severe loss = better
    # --------------------------------------------------------

    severe_loss_score = min_max_score(
        combined[
            "Probability of Loss > 10%"
        ],
        higher_is_better=False
    )

    # --------------------------------------------------------
    # Combined Monte Carlo score
    # --------------------------------------------------------

    score = (
        0.30 * wealth_score
        + 0.20 * median_score
        + 0.30 * loss_score
        + 0.20 * severe_loss_score
    )

    score_map = pd.Series(
        score.values,
        index=combined["Strategy"]
    )

    strategy_table[
        "Monte Carlo Score"
    ] = strategy_table[
        "Strategy"
    ].map(
        score_map
    ).fillna(0.5)

    return strategy_table


# ============================================================
# DOWNSIDE RISK SCORE
# ============================================================

def calculate_downside_score(
    strategy_table,
    transaction,
    monte_carlo,
    regime_mc
):

    # --------------------------------------------------------
    # Transaction-cost drawdown
    # --------------------------------------------------------

    transaction = transaction.copy()

    transaction["Strategy"] = (
        transaction["Strategy"]
        .map(clean_strategy_name)
    )

    transaction = transaction[
        transaction["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    # --------------------------------------------------------
    # Monte Carlo drawdown
    # --------------------------------------------------------

    mc = pd.concat(
        [
            monte_carlo,
            regime_mc
        ],
        ignore_index=True
    )

    mc["Strategy"] = (
        mc["Strategy"]
        .map(clean_strategy_name)
    )

    mc = mc[
        mc["Strategy"].isin(
            STRATEGIES
        )
    ].copy()

    mc_drawdown_score = min_max_score(
        mc["Expected Maximum Drawdown"],
        higher_is_better=True
    )

    severe_loss_score = min_max_score(
        mc["Probability of Loss > 10%"],
        higher_is_better=False
    )

    mc_score_map = pd.Series(
        (
            0.60 * mc_drawdown_score
            + 0.40 * severe_loss_score
        ).values,
        index=mc["Strategy"]
    )

    # --------------------------------------------------------
    # Map to final table
    # --------------------------------------------------------

    strategy_table[
        "Downside Risk Score"
    ] = strategy_table[
        "Strategy"
    ].map(
        mc_score_map
    ).fillna(0.5)

    return strategy_table


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    strategy_table
):

    strategy_table[
        "Final Score"
    ] = (

        WEIGHTS["Backtest"]
        * strategy_table[
            "Backtest Score"
        ]

        +

        WEIGHTS["Benchmark"]
        * strategy_table[
            "Benchmark Score"
        ]

        +

        WEIGHTS["Transaction Cost"]
        * strategy_table[
            "Transaction Cost Score"
        ]

        +

        WEIGHTS["Monte Carlo"]
        * strategy_table[
            "Monte Carlo Score"
        ]

        +

        WEIGHTS["Downside Risk"]
        * strategy_table[
            "Downside Risk Score"
        ]
    )

    return strategy_table


# ============================================================
# RECOMMENDATION EXPLANATION
# ============================================================

def generate_explanation(
    row
):

    explanations = []

    if row["Backtest Score"] >= 0.67:
        explanations.append(
            "strong out-of-sample backtest performance"
        )
    elif row["Backtest Score"] <= 0.33:
        explanations.append(
            "relatively weak out-of-sample backtest performance"
        )

    if row["Benchmark Score"] >= 0.67:
        explanations.append(
            "favorable benchmark-relative performance"
        )

    if row["Transaction Cost Score"] >= 0.67:
        explanations.append(
            "relatively favorable transaction-cost characteristics"
        )

    if row["Monte Carlo Score"] >= 0.67:
        explanations.append(
            "strong Monte Carlo outcome characteristics"
        )

    if row["Downside Risk Score"] >= 0.67:
        explanations.append(
            "relatively strong downside-risk characteristics"
        )

    if len(explanations) == 0:
        return (
            "The strategy has a mixed evidence profile "
            "across the evaluated criteria."
        )

    return (
        "Evidence includes "
        + ", ".join(explanations)
        + "."
    )


# ============================================================
# RISK PROFILE
# ============================================================

def assign_risk_profile(row):

    downside = row[
        "Downside Risk Score"
    ]

    monte_carlo = row[
        "Monte Carlo Score"
    ]

    backtest = row[
        "Backtest Score"
    ]

    if (
        downside >= 0.67
        and monte_carlo >= 0.60
    ):
        return "Lower Downside Orientation"

    if (
        backtest >= 0.67
        and downside < 0.50
    ):
        return "Return-Oriented"

    return "Balanced"


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

def generate_final_recommendation(
    strategy_table
):

    strategy_table = strategy_table.copy()

    # --------------------------------------------------------
    # Select highest integrated score
    # --------------------------------------------------------

    best_index = (
        strategy_table[
            "Final Score"
        ].idxmax()
    )

    recommended_strategy = (
        strategy_table.loc[
            best_index,
            "Strategy"
        ]
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    strategy_table[
        "Evidence Summary"
    ] = strategy_table.apply(
        generate_explanation,
        axis=1
    )

    strategy_table[
        "Risk Profile"
    ] = strategy_table.apply(
        assign_risk_profile,
        axis=1
    )

    strategy_table[
        "Recommended"
    ] = (
        strategy_table["Strategy"]
        == recommended_strategy
    )

    return (
        strategy_table,
        recommended_strategy
    )


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    results,
    recommended_strategy
):

    print("\n")
    print("=" * 80)
    print("FINAL INTEGRATED PORTFOLIO ANALYSIS")
    print("=" * 80)

    display_columns = [
        "Strategy",
        "Backtest Score",
        "Benchmark Score",
        "Transaction Cost Score",
        "Monte Carlo Score",
        "Downside Risk Score",
        "Final Score",
        "Risk Profile"
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("INTEGRATED DECISION OUTPUT")
    print("=" * 80)

    print(
        f"\nRecommended Strategy: "
        f"{recommended_strategy}"
    )

    recommended_row = results[
        results["Strategy"]
        == recommended_strategy
    ].iloc[0]

    print(
        f"Final Integrated Score: "
        f"{recommended_row['Final Score']:.4f}"
    )

    print(
        f"Risk Profile: "
        f"{recommended_row['Risk Profile']}"
    )

    print(
        "\nEvidence Summary:"
    )

    print(
        recommended_row[
            "Evidence Summary"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def run_final_recommendation():

    print("=" * 80)
    print("FINAL PORTFOLIO RECOMMENDATION ENGINE")
    print("=" * 80)

    (
        backtest,
        benchmark,
        transaction,
        monte_carlo,
        regime_mc
    ) = load_data()

    strategy_table = (
        prepare_strategy_table()
    )

    # --------------------------------------------------------
    # Individual evidence scores
    # --------------------------------------------------------

    strategy_table = (
        calculate_backtest_score(
            strategy_table,
            backtest
        )
    )

    strategy_table = (
        calculate_benchmark_score(
            strategy_table,
            benchmark
        )
    )

    strategy_table = (
        calculate_transaction_cost_score(
            strategy_table,
            transaction
        )
    )

    strategy_table = (
        calculate_monte_carlo_score(
            strategy_table,
            monte_carlo,
            regime_mc
        )
    )

    strategy_table = (
        calculate_downside_score(
            strategy_table,
            transaction,
            monte_carlo,
            regime_mc
        )
    )

    # --------------------------------------------------------
    # Final integrated score
    # --------------------------------------------------------

    strategy_table = (
        calculate_final_score(
            strategy_table
        )
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    (
        results,
        recommended_strategy
    ) = generate_final_recommendation(
        strategy_table
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save compact summary
    # --------------------------------------------------------

    summary_columns = [
        "Strategy",
        "Final Score",
        "Recommended",
        "Risk Profile",
        "Evidence Summary"
    ]

    results[
        summary_columns
    ].to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        results,
        recommended_strategy
    )

    print("\n")
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        OUTPUT_FILE
    )

    print(
        SUMMARY_FILE
    )

    return results


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_final_recommendation()