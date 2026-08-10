import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from yfinance import data

from yfinance import data


class PortfolioBacktester:
    """
    Perform chronological out-of-sample backtesting
    for portfolio strategies.
    """

    def __init__(
        self,
        returns_file="data/processed/returns.csv",
        train_ratio=0.80
    ):

        self.returns_file = Path(returns_file)
        self.train_ratio = train_ratio

        self.returns = None

        self.train_returns = None
        self.test_returns = None

    def load_returns(self):
        """
        Load historical daily returns.
        """

        self.returns = pd.read_csv(
            self.returns_file,
            parse_dates=["date"]
        )

        print("Returns loaded successfully.")

    def split_data(self):
        """
        Split returns chronologically into
        training and testing periods.
        """

        split_index = int(
            len(self.returns) * self.train_ratio
        )

        self.train_returns = self.returns.iloc[
            :split_index
        ].copy()

        self.test_returns = self.returns.iloc[
            split_index:
        ].copy()

        print("\nBacktest Data Split")
        print("-" * 40)

        print(
            f"Total observations : "
            f"{len(self.returns)}"
        )

        print(
            f"Training observations : "
            f"{len(self.train_returns)}"
        )

        print(
            f"Testing observations : "
            f"{len(self.test_returns)}"
        )

        print(
            f"\nTraining period:"
        )

        print(
            f"{self.train_returns['date'].iloc[0].date()}"
            f" → "
            f"{self.train_returns['date'].iloc[-1].date()}"
        )

        print(
            f"\nTesting period:"
        )

        print(
            f"{self.test_returns['date'].iloc[0].date()}"
            f" → "
            f"{self.test_returns['date'].iloc[-1].date()}"
        )
    def get_asset_returns(self, data):
        """
        Extract only asset return columns.
        """

        return data.iloc[:, 1:].copy()
    def calculate_training_covariance(self):
        """
        Calculate annualized covariance matrix
        using training data only.
        """

        training_assets = self.get_asset_returns(
        self.train_returns
    )

        covariance = (
        training_assets.cov() * 252
    )

        print("\nTraining Covariance Matrix")
        print("-" * 40)

        print(covariance)

        return covariance
    def calculate_training_expected_returns(self):
        """
        Calculate annualized expected returns
        using training data only.
        """

        training_assets = self.get_asset_returns(
        self.train_returns
    )

        expected_returns = (
        training_assets.mean() * 252
    )

        print("\nTraining Expected Returns")
        print("-" * 40)

        print(expected_returns)

        return expected_returns

    def optimize_markowitz(
    self,
    expected_returns,
    covariance_matrix,
    objective="max_sharpe"
):
        """
        Optimize a Markowitz portfolio using
        training-period estimates only.
        """

        n_assets = len(expected_returns)

        initial_weights = np.ones(n_assets) / n_assets

        bounds = tuple(
        (0, 1)
        for _ in range(n_assets)
    )

        constraints = {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1
    }

        def portfolio_return(weights):
            return np.dot(
            weights,
            expected_returns.values
        )

        def portfolio_risk(weights):
            return np.sqrt(
            weights.T
            @ covariance_matrix.values
            @ weights
        )

        def negative_sharpe(weights):

            risk = portfolio_risk(weights)

            if risk == 0:
                return 0

            return -(
            portfolio_return(weights) - 0.065
        ) / risk

        def portfolio_variance(weights):

            return (
            weights.T
            @ covariance_matrix.values
            @ weights
        )

        if objective == "max_sharpe":

            objective_function = negative_sharpe

        else:

            objective_function = portfolio_variance

        result = minimize(
        objective_function,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

        return result

    def generate_training_markowitz_portfolios(self):
        """
        Generate Maximum Sharpe and Minimum Variance
        portfolios using training data only.
        """

        covariance = self.calculate_training_covariance()

        expected_returns = (
        self.calculate_training_expected_returns()
    )

        max_sharpe = self.optimize_markowitz(
        expected_returns,
        covariance,
        objective="max_sharpe"
    )

        min_variance = self.optimize_markowitz(
        expected_returns,
        covariance,
        objective="min_variance"
    )

        return max_sharpe, min_variance

    def optimize_risk_parity(self, covariance_matrix):
        """
        Generate Risk Parity weights using
        training-period covariance only.
        """

        n_assets = covariance_matrix.shape[0]

        initial_weights = (
        np.ones(n_assets) / n_assets
    )

        def portfolio_risk(weights):

            return np.sqrt(
            weights.T
            @ covariance_matrix.values
            @ weights
        )

        def risk_contributions(weights):

            risk = portfolio_risk(weights)

            marginal_contribution = (
            covariance_matrix.values @ weights
        )

            component_contribution = (
            weights * marginal_contribution
        )

            return (
            component_contribution / risk
        )

        def objective(weights):

            contributions = risk_contributions(
            weights
        )

            target = np.ones(n_assets) / n_assets

            return np.sum(
            (contributions - target) ** 2
        )

        constraints = {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1
    }

        bounds = tuple(
        (0, 1)
        for _ in range(n_assets)
    )

        result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

        return result
      
if __name__ == "__main__":

    backtester = PortfolioBacktester()

    backtester.load_returns()

    backtester.split_data()

    max_sharpe, min_variance = (
        backtester.generate_training_markowitz_portfolios()
    )

    # Maximum Sharpe
    print("\nTraining Maximum Sharpe Weights")
    print("-" * 40)

    for asset, weight in zip(
        backtester.train_returns.columns[1:],
        max_sharpe.x
    ):
        print(
            f"{asset:<12} "
            f"{weight:.4%}"
        )

    # Minimum Variance
    print("\nTraining Minimum Variance Weights")
    print("-" * 40)

    for asset, weight in zip(
        backtester.train_returns.columns[1:],
        min_variance.x
    ):
        print(
            f"{asset:<12} "
            f"{weight:.4%}"
        )

    # Risk Parity
    covariance = (
        backtester.calculate_training_covariance()
    )

    risk_parity = backtester.optimize_risk_parity(
        covariance
    )

    print("\nTraining Risk Parity Weights")
    print("-" * 40)

    for asset, weight in zip(
        backtester.train_returns.columns[1:],
        risk_parity.x
    ):
        print(
            f"{asset:<12} "
            f"{weight:.4%}"
        )

    print("\nWeight Sum")
    print("-" * 40)

    print(
        f"{risk_parity.x.sum():.4%}"
    )