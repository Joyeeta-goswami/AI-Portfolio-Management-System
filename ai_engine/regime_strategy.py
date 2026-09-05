import sys
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import redirect_stdout
import io


# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from optimization.backtest import PortfolioBacktester


class RegimeStrategyAnalyzer:
    """
    Learn a strategy-selection rule from training-period
    regime performance and evaluate it out-of-sample.

    Strategy selection:
        Calm            -> best training strategy
        Defensive       -> best training strategy
        High Volatility -> best training strategy

    The learned mapping is frozen before evaluating the test data.
    """

    def __init__(
        self,
        returns_file="data/processed/returns.csv",
        regime_file="data/processed/market_regimes.csv",
        train_ratio=0.80
    ):
        self.returns_file = Path(returns_file)
        self.regime_file = Path(regime_file)
        self.train_ratio = train_ratio

        self.returns = None
        self.regimes = None

        self.train_returns = None
        self.test_returns = None

        self.train_regimes = None
        self.test_regimes = None

        self.train_data = None
        self.test_data = None

        self.best_strategy_by_regime = None

    def load_data(self):
        """
        Load asset returns and classified market regimes.
        """

        self.returns = pd.read_csv(
            self.returns_file,
            parse_dates=["date"]
        )

        self.regimes = pd.read_csv(
            self.regime_file,
            parse_dates=["date"]
        )

        self.returns = self.returns.sort_values(
            "date"
        ).reset_index(drop=True)

        self.regimes = self.regimes.sort_values(
            "date"
        ).reset_index(drop=True)

        print(
            "Returns and regime data loaded successfully."
        )

    def split_data(self):
        """
        Split returns and regime observations chronologically.
        """

        split_index = int(
            len(self.returns) * self.train_ratio
        )

        split_date = self.returns[
            "date"
        ].iloc[split_index]

        self.train_returns = self.returns[
            self.returns["date"] < split_date
        ].copy()

        self.test_returns = self.returns[
            self.returns["date"] >= split_date
        ].copy()

        self.train_regimes = self.regimes[
            (self.regimes["dataset"] == "train")
            & (self.regimes["date"] < split_date)
        ].copy()

        self.test_regimes = self.regimes[
            (self.regimes["dataset"] == "test")
            & (self.regimes["date"] >= split_date)
        ].copy()

        print("\nData Split")
        print("-" * 55)

        print(
            f"Training returns : "
            f"{len(self.train_returns)}"
        )

        print(
            f"Testing returns  : "
            f"{len(self.test_returns)}"
        )

        print(
            f"Training regimes : "
            f"{len(self.train_regimes)}"
        )

        print(
            f"Testing regimes  : "
            f"{len(self.test_regimes)}"
        )

    def align_data(self):
        """
        Align regime classifications with return observations.
        """

        self.train_data = pd.merge(
            self.train_returns,
            self.train_regimes[
                ["date", "regime"]
            ],
            on="date",
            how="inner"
        )

        self.test_data = pd.merge(
            self.test_returns,
            self.test_regimes[
                ["date", "regime"]
            ],
            on="date",
            how="inner"
        )

        self.train_data = (
            self.train_data
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.test_data = (
            self.test_data
            .sort_values("date")
            .reset_index(drop=True)
        )

        print("\nData Alignment")
        print("-" * 55)

        print(
            f"Aligned training observations : "
            f"{len(self.train_data)}"
        )

        print(
            f"Aligned testing observations  : "
            f"{len(self.test_data)}"
        )

    def calculate_strategy_returns(self):
        """
        Calculate daily returns for the three optimized
        portfolio strategies on both training and test data.

        All portfolio weights are estimated from the training
        period only and then kept fixed.
        """

        backtester = PortfolioBacktester(
            returns_file=str(self.returns_file),
            train_ratio=self.train_ratio
        )

        # Load and split data using the same backtester
        backtester.load_returns()
        backtester.split_data()

        # ---------------------------------------------
        # Generate Markowitz portfolios
        # ---------------------------------------------

        max_sharpe, min_variance = (
            backtester.generate_training_markowitz_portfolios()
        )

        # ---------------------------------------------
        # Generate Risk Parity portfolio
        # ---------------------------------------------

        covariance = (
            backtester.calculate_training_covariance()
        )

        risk_parity = (
            backtester.optimize_risk_parity(
                covariance
            )
        )

        # Extract frozen training weights
        weights = {
            "Maximum Sharpe": max_sharpe.x,
            "Minimum Variance": min_variance.x,
            "Risk Parity": risk_parity.x
        }

        asset_columns = (
            self.train_returns.columns[1:]
        )

        # ---------------------------------------------
        # Training strategy returns
        # ---------------------------------------------

        self.train_data["Maximum Sharpe"] = (
            self.train_data[asset_columns].values
            @ weights["Maximum Sharpe"]
        )

        self.train_data["Minimum Variance"] = (
            self.train_data[asset_columns].values
            @ weights["Minimum Variance"]
        )

        self.train_data["Risk Parity"] = (
            self.train_data[asset_columns].values
            @ weights["Risk Parity"]
        )

        # ---------------------------------------------
        # Test strategy returns
        # ---------------------------------------------

        self.test_data["Maximum Sharpe"] = (
            self.test_data[asset_columns].values
            @ weights["Maximum Sharpe"]
        )

        self.test_data["Minimum Variance"] = (
            self.test_data[asset_columns].values
            @ weights["Minimum Variance"]
        )

        self.test_data["Risk Parity"] = (
            self.test_data[asset_columns].values
            @ weights["Risk Parity"]
        )

        # Store weights for possible later use
        self.training_weights = weights

        print(
            "\nTraining and test strategy returns calculated."
        )
    def analyze_strategy_performance_by_regime(self):
        """
        Calculate average daily strategy return within
        each training regime.
        """

        strategies = [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity"
        ]

        performance = (
            self.train_data
            .groupby("regime")[strategies]
            .mean()
        )

        print(
            "\nAverage Daily Strategy Return by Regime"
        )
        print("-" * 75)

        print(
            performance.to_string(
                float_format=lambda x: f"{x:.6%}"
            )
        )

        self.regime_performance = performance

        return performance

    def learn_best_strategy_by_regime(self):
        """
        Learn the best strategy for each regime using
        training-period average daily returns only.
        """

        self.best_strategy_by_regime = (
            self.regime_performance
            .idxmax(axis=1)
            .to_dict()
        )

        print(
            "\nBest Strategy Learned for Each Regime"
        )
        print("-" * 65)

        for regime, strategy in (
            self.best_strategy_by_regime.items()
        ):
            print(
                f"{regime:<20} → {strategy}"
            )

        return self.best_strategy_by_regime

    def apply_regime_strategy_to_test(self):
        """
        Apply the frozen training regime-to-strategy mapping
        to the unseen test period.
        """

        self.test_data["Selected Strategy"] = (
            self.test_data["regime"].map(
                self.best_strategy_by_regime
            )
        )

        self.test_data[
            "Regime Strategy Return"
        ] = np.nan

        for strategy in [
            "Maximum Sharpe",
            "Minimum Variance",
            "Risk Parity"
        ]:

            mask = (
                self.test_data["Selected Strategy"]
                == strategy
            )

            self.test_data.loc[
                mask,
                "Regime Strategy Return"
            ] = self.test_data.loc[
                mask,
                strategy
            ]

        print(
            "\nRegime-Switching Test Results"
        )
        print("-" * 75)

        print(
            self.test_data[
                [
                    "date",
                    "regime",
                    "Selected Strategy",
                    "Regime Strategy Return"
                ]
            ].head(10).to_string(
                index=False
            )
        )

        print("\nStrategy Selection Counts")
        print("-" * 55)

        print(
            self.test_data["Selected Strategy"]
            .value_counts()
        )

        return self.test_data

    def calculate_regime_strategy_performance(self):
        """
        Calculate out-of-sample performance of the
        regime-switching strategy.
        """

        returns = (
            self.test_data[
                "Regime Strategy Return"
            ]
            .dropna()
        )

        if len(returns) == 0:
            raise ValueError(
                "No valid regime-switching test returns."
            )

        trading_days = 252
        risk_free_rate = 0.065

        cumulative_return = (
            (1 + returns).prod() - 1
        )

        annualized_return = (
            (1 + cumulative_return)
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

        metrics = {
            "Total Return": cumulative_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility":
                annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Maximum Drawdown":
                maximum_drawdown,
            "VaR (95%)": var_95
        }

        self.regime_strategy_metrics = metrics

        print(
            "\nRegime-Switching Strategy Performance"
        )
        print("-" * 65)

        print(
            f"Observations          : {len(returns)}"
        )

        print(
            f"Total Return          : "
            f"{cumulative_return:.4%}"
        )

        print(
            f"Annualized Return     : "
            f"{annualized_return:.4%}"
        )

        print(
            f"Annualized Volatility : "
            f"{annualized_volatility:.4%}"
        )

        print(
            f"Sharpe Ratio          : "
            f"{sharpe_ratio:.3f}"
        )

        print(
            f"Maximum Drawdown      : "
            f"{maximum_drawdown:.4%}"
        )

        print(
            f"VaR (95%)             : "
            f"{var_95:.4%}"
        )

        return metrics

    def save_outputs(
        self,
        regime_results_file=(
            "data/processed/"
            "regime_switching_results.csv"
        ),
        regime_test_file=(
            "data/processed/"
            "regime_strategy_test_returns.csv"
        )
    ):
        """
        Save regime-switching performance and
        test-period selection history.
        """

        metrics_table = pd.DataFrame(
            [self.regime_strategy_metrics],
            index=["Regime Switching"]
        )

        metrics_table.index.name = "Strategy"

        metrics_table.to_csv(
            regime_results_file
        )

        self.test_data[
            [
                "date",
                "regime",
                "Selected Strategy",
                "Regime Strategy Return"
            ]
        ].to_csv(
            regime_test_file,
            index=False
        )

        print(
            "\nRegime-switching results saved to:"
        )
        print(regime_results_file)

        print(
            "\nRegime-switching test history saved to:"
        )
        print(regime_test_file)


if __name__ == "__main__":

    analyzer = RegimeStrategyAnalyzer()

    analyzer.load_data()

    analyzer.split_data()

    analyzer.align_data()

    analyzer.calculate_strategy_returns()

    analyzer.analyze_strategy_performance_by_regime()

    analyzer.learn_best_strategy_by_regime()

    analyzer.apply_regime_strategy_to_test()

    analyzer.calculate_regime_strategy_performance()

    analyzer.save_outputs()