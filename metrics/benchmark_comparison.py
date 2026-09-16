import pandas as pd
import numpy as np
from pathlib import Path


class BenchmarkComparator:
    """
    Compare portfolio strategies against the NIFTY 50 benchmark
    on the same chronological test period.
    """

    def __init__(
        self,
        returns_file="data/processed/returns.csv",
        benchmark_file="data/processed/nifty50_returns.csv",
        train_ratio=0.80
    ):
        self.returns_file = Path(returns_file)
        self.benchmark_file = Path(benchmark_file)
        self.train_ratio = train_ratio

        self.returns = None
        self.benchmark = None

        self.test_dates = None
        self.test_benchmark = None

    def load_data(self):
        """
        Load portfolio and NIFTY 50 returns.
        """

        self.returns = pd.read_csv(
            self.returns_file,
            parse_dates=["date"]
        )

        self.benchmark = pd.read_csv(
            self.benchmark_file,
            parse_dates=["date"]
        )

        self.returns = (
            self.returns
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.benchmark = (
            self.benchmark
            .sort_values("date")
            .reset_index(drop=True)
        )

        print("Portfolio and benchmark data loaded.")

    def get_test_period(self):
        """
        Identify the original 80/20 portfolio test period.
        """

        split_index = int(
            len(self.returns) * self.train_ratio
        )

        test_returns = self.returns.iloc[
            split_index:
        ].copy()

        self.test_dates = test_returns["date"]

        self.test_benchmark = pd.merge(
            self.test_dates.to_frame(),
            self.benchmark,
            on="date",
            how="inner"
        )

        print("\nBenchmark Test Period")
        print("-" * 50)

        print(
            f"Portfolio test observations : "
            f"{len(self.test_dates)}"
        )

        print(
            f"Aligned NIFTY observations  : "
            f"{len(self.test_benchmark)}"
        )

        print(
            f"Start : "
            f"{self.test_benchmark['date'].iloc[0].date()}"
        )

        print(
            f"End   : "
            f"{self.test_benchmark['date'].iloc[-1].date()}"
        )

    def calculate_metrics(self, returns):
        """
        Calculate the same metrics used for the portfolios.
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
            ** (trading_days / len(returns))
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

        running_max = wealth.cummax()

        drawdown = (
            wealth / running_max
        ) - 1

        maximum_drawdown = drawdown.min()

        var_95 = returns.quantile(0.05)

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility":
                annualized_volatility,
            "Sharpe Ratio":
                sharpe_ratio,
            "Maximum Drawdown":
                maximum_drawdown,
            "VaR (95%)":
                var_95
        }

    def compare_nifty(self):
        """
        Calculate NIFTY 50 test-period performance.
        """

        returns = self.test_benchmark[
            "NIFTY50"
        ]

        metrics = self.calculate_metrics(
            returns
        )

        result = pd.DataFrame(
            [metrics],
            index=["NIFTY 50"]
        )

        print("\nNIFTY 50 Test Performance")
        print("-" * 60)

        print(
            f"Total Return          : "
            f"{metrics['Total Return']:.4%}"
        )

        print(
            f"Annualized Return     : "
            f"{metrics['Annualized Return']:.4%}"
        )

        print(
            f"Annualized Volatility : "
            f"{metrics['Annualized Volatility']:.4%}"
        )

        print(
            f"Sharpe Ratio          : "
            f"{metrics['Sharpe Ratio']:.3f}"
        )

        print(
            f"Maximum Drawdown      : "
            f"{metrics['Maximum Drawdown']:.4%}"
        )

        print(
            f"VaR (95%)             : "
            f"{metrics['VaR (95%)']:.4%}"
        )

        return result


if __name__ == "__main__":

    comparator = BenchmarkComparator()

    comparator.load_data()

    comparator.get_test_period()

    comparator.compare_nifty()