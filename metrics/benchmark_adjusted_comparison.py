import sys
import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class BenchmarkAdjustedComparison:
    """
    Compare portfolio strategies with NIFTY 50
    using exactly the same dates.
    """

    def __init__(
        self,
        portfolio_returns_file=(
            "data/processed/returns.csv"
        ),
        benchmark_file=(
            "data/processed/nifty50_returns.csv"
        ),
        regime_test_file=(
            "data/processed/"
            "regime_strategy_test_returns.csv"
        ),
        train_ratio=0.80
    ):
        self.portfolio_returns_file = Path(
            portfolio_returns_file
        )

        self.benchmark_file = Path(
            benchmark_file
        )

        self.regime_test_file = Path(
            regime_test_file
        )

        self.train_ratio = train_ratio

        self.portfolio_returns = None
        self.benchmark_returns = None
        self.regime_returns = None

        self.comparison_data = None
        self.performance_table = None

    # ==================================================
    # LOAD DATA
    # ==================================================

    def load_data(self):
        """
        Load portfolio asset returns, NIFTY 50 returns,
        and regime-switching test returns.
        """

        self.portfolio_returns = pd.read_csv(
            self.portfolio_returns_file,
            parse_dates=["date"]
        )

        self.benchmark_returns = pd.read_csv(
            self.benchmark_file,
            parse_dates=["date"]
        )

        self.regime_returns = pd.read_csv(
            self.regime_test_file,
            parse_dates=["date"]
        )

        # Ensure chronological ordering
        self.portfolio_returns = (
            self.portfolio_returns
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.benchmark_returns = (
            self.benchmark_returns
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.regime_returns = (
            self.regime_returns
            .sort_values("date")
            .reset_index(drop=True)
        )

        print(
            "Portfolio, benchmark and regime data loaded."
        )

    # ==================================================
    # RECREATE FIXED PORTFOLIO TEST RETURNS
    # ==================================================

    def calculate_fixed_strategy_returns(self):
        """
        Recreate the three fixed portfolio strategy
        test returns using the same frozen training
        weights as the original backtest.
        """

        split_index = int(
            len(self.portfolio_returns)
            * self.train_ratio
        )

        train_data = (
            self.portfolio_returns
            .iloc[:split_index]
            .copy()
        )

        test_data = (
            self.portfolio_returns
            .iloc[split_index:]
            .copy()
        )

        # Use the SAME PortfolioBacktester
        # implementation as the main project.
        from optimization.backtest import (
            PortfolioBacktester
        )

        backtester = PortfolioBacktester(
            returns_file=str(
                self.portfolio_returns_file
            ),
            train_ratio=self.train_ratio
        )

        # These two calls are required before
        # optimization can use train_returns.
        backtester.load_returns()
        backtester.split_data()

        # Generate the same training portfolios
        max_sharpe, min_variance = (
            backtester
            .generate_training_markowitz_portfolios()
        )

        # Generate Risk Parity
        covariance = (
            backtester
            .calculate_training_covariance()
        )

        risk_parity = (
            backtester
            .optimize_risk_parity(
                covariance
            )
        )

        asset_columns = (
            train_data.columns[1:]
        )

        test_assets = (
            test_data[asset_columns]
        )

        fixed_returns = pd.DataFrame(
            {
                "date": test_data["date"],

                "Maximum Sharpe":
                    test_assets.values
                    @ max_sharpe.x,

                "Minimum Variance":
                    test_assets.values
                    @ min_variance.x,

                "Risk Parity":
                    test_assets.values
                    @ risk_parity.x
            }
        )

        return fixed_returns
    # ==================================================
    # ALIGN COMMON DATES
    # ==================================================

    def align_common_dates(self):
        """
        Keep only dates available for all strategies
        and NIFTY 50.
        """

        fixed_returns = (
            self.calculate_fixed_strategy_returns()
        )

        regime_data = (
            self.regime_returns[
                [
                    "date",
                    "Regime Strategy Return"
                ]
            ]
        )

        benchmark_data = (
            self.benchmark_returns[
                [
                    "date",
                    "NIFTY50"
                ]
            ]
        )

        comparison = fixed_returns.merge(
            regime_data,
            on="date",
            how="inner"
        )

        comparison = comparison.merge(
            benchmark_data,
            on="date",
            how="inner"
        )

        comparison = (
            comparison
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.comparison_data = comparison

        print("\nCommon-Date Benchmark Comparison")
        print("-" * 60)

        print(
            f"Common observations : "
            f"{len(comparison)}"
        )

        print(
            f"Start : "
            f"{comparison['date'].iloc[0].date()}"
        )

        print(
            f"End   : "
            f"{comparison['date'].iloc[-1].date()}"
        )

        print("\nLatest observations:")

        print(
            comparison.tail(5).to_string(
                index=False
            )
        )

        return comparison

    # ==================================================
    # PERFORMANCE METRICS
    # ==================================================

    def calculate_metrics(
        self,
        returns
    ):
        """
        Calculate the same portfolio metrics used
        throughout the project.
        """

        trading_days = 252
        risk_free_rate = 0.065

        returns = (
            pd.Series(returns)
            .dropna()
        )

        total_return = (
            (1 + returns).prod() - 1
        )

        annualized_return = (
            (1 + total_return)
            ** (
                trading_days
                / len(returns)
            )
            - 1
        )

        annualized_volatility = (
            returns.std()
            * np.sqrt(trading_days)
        )

        if annualized_volatility == 0:
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = (
                annualized_return
                - risk_free_rate
            ) / annualized_volatility

        wealth = (
            1 + returns
        ).cumprod()

        running_max = (
            wealth.cummax()
        )

        drawdown = (
            wealth / running_max
        ) - 1

        maximum_drawdown = (
            drawdown.min()
        )

        var_95 = (
            returns.quantile(0.05)
        )

        return {
            "Total Return": total_return,
            "Annualized Return":
                annualized_return,
            "Annualized Volatility":
                annualized_volatility,
            "Sharpe Ratio":
                sharpe_ratio,
            "Maximum Drawdown":
                maximum_drawdown,
            "VaR (95%)":
                var_95
        }

    # ==================================================
    # COMPARE
    # ==================================================

    def calculate_comparison(self):
        """
        Calculate performance for all five strategies
        over the exact same dates.
        """

        strategy_columns = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity",
            "Regime Strategy Return",
            "NIFTY50"
        ]

        results = []

        for strategy in strategy_columns:

            metrics = self.calculate_metrics(
                self.comparison_data[
                    strategy
                ]
            )

            if strategy == "Regime Strategy Return":
                display_name = (
                    "Risk-Aware Regime Switching"
                )

            elif strategy == "NIFTY50":
                display_name = "NIFTY 50"

            else:
                display_name = strategy

            results.append(
                {
                    "Strategy": display_name,
                    **metrics
                }
            )

        self.performance_table = (
            pd.DataFrame(results)
            .set_index("Strategy")
        )

        print(
            "\nCommon-Date Performance Comparison"
        )

        print("-" * 85)

        print(
            self.performance_table.to_string(
                float_format=lambda x:
                    f"{x:.4%}"
            )
        )

        return self.performance_table

    # ==================================================
    # NORMALIZED WEALTH
    # ==================================================

    def calculate_normalized_wealth(self):
        """
        Start every strategy at 100 and calculate
        cumulative wealth.
        """

        strategy_columns = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity",
            "Regime Strategy Return",
            "NIFTY50"
        ]

        wealth = pd.DataFrame()

        wealth["date"] = (
            self.comparison_data["date"]
        )

        for strategy in strategy_columns:

            wealth[strategy] = (
                100
                * (
                    1
                    + self.comparison_data[
                        strategy
                    ]
                ).cumprod()
            )

        wealth = wealth.rename(
            columns={
                "Regime Strategy Return":
                    "Risk-Aware Regime Switching",
                "NIFTY50":
                    "NIFTY 50"
            }
        )

        return wealth

    # ==================================================
    # SAVE
    # ==================================================

    def save_outputs(
        self,
        performance_file=(
            "data/processed/"
            "benchmark_adjusted_results.csv"
        ),
        wealth_file=(
            "data/processed/"
            "benchmark_adjusted_wealth.csv"
        )
    ):
        """
        Save common-date performance and wealth data.
        """

        self.performance_table.to_csv(
            performance_file
        )

        wealth = (
            self.calculate_normalized_wealth()
        )

        wealth.to_csv(
            wealth_file,
            index=False
        )

        print(
            "\nBenchmark-adjusted performance saved to:"
        )

        print(performance_file)

        print(
            "\nBenchmark-adjusted wealth saved to:"
        )

        print(wealth_file)


if __name__ == "__main__":

    comparator = (
        BenchmarkAdjustedComparison()
    )

    comparator.load_data()

    comparator.align_common_dates()

    comparator.calculate_comparison()

    comparator.save_outputs()