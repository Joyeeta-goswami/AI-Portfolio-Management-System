import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
import plotly.graph_objects as go


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

        print("\nTraining period:")

        print(
            f"{self.train_returns['date'].iloc[0].date()}"
            f" → "
            f"{self.train_returns['date'].iloc[-1].date()}"
        )

        print("\nTesting period:")

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

        if not result.success:
            print("Markowitz optimization failed:")
            print(result.message)

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

        covariance = covariance_matrix.values

        def portfolio_risk(weights):
            """
            Calculate portfolio volatility.
            """

            return np.sqrt(
                weights.T
                @ covariance
                @ weights
            )

        def risk_contributions(weights):
            """
            Calculate each asset's percentage contribution
            to total portfolio risk.
            """

            risk = portfolio_risk(weights)

            marginal_contribution = (
                covariance @ weights
            )

            component_contribution = (
                weights * marginal_contribution
            )

            return (
                component_contribution / risk
            )

        def objective(weights):
            """
            Minimize dispersion in risk contributions.
            """

            contributions = risk_contributions(weights)

            if np.any(contributions <= 0):
                return 1e10

            log_contributions = np.log(contributions)

            return np.sum(
                (
                    log_contributions
                    - log_contributions.mean()
                ) ** 2
            )

        constraints = {
            "type": "eq",
            "fun": lambda weights:
                np.sum(weights) - 1
        }

        bounds = tuple(
            (0.001, 1)
            for _ in range(n_assets)
        )

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            print("Risk Parity optimization failed:")
            print(result.message)

        return result

    def calculate_test_portfolio_returns(self, weights):
        """
        Calculate daily portfolio returns on the test data
        using fixed weights estimated from the training period.
        """

        test_assets = self.get_asset_returns(
            self.test_returns
        )

        portfolio_returns = (
            test_assets.values @ weights
        )

        return pd.Series(
            portfolio_returns,
            index=self.test_returns["date"]
        )

    def calculate_portfolio_metrics(self, portfolio_returns):
        """
        Calculate performance metrics for a portfolio
        using test-period returns.
        """

        trading_days = 252
        risk_free_rate = 0.065

        # Cumulative growth
        cumulative_returns = (
            1 + portfolio_returns
        ).cumprod()

        # Total return
        total_return = (
            cumulative_returns.iloc[-1] - 1
        )

        # Number of observations
        n_days = len(portfolio_returns)

        # Annualized return
        annualized_return = (
            (1 + total_return)
            ** (trading_days / n_days)
            - 1
        )

        # Annualized volatility
        annualized_volatility = (
            portfolio_returns.std()
            * np.sqrt(trading_days)
        )

        # Sharpe ratio
        sharpe_ratio = (
            annualized_return - risk_free_rate
        ) / annualized_volatility

        # Maximum Drawdown
        running_max = cumulative_returns.cummax()

        drawdown = (
            cumulative_returns / running_max
        ) - 1

        max_drawdown = drawdown.min()

        # Historical VaR at 95% confidence
        var_95 = portfolio_returns.quantile(0.05)

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Maximum Drawdown": max_drawdown,
            "VaR (95%)": var_95
        }
    def calculate_cumulative_wealth(self, portfolio_returns, initial_value=100):
        """
        Calculate cumulative portfolio value from daily returns.
        """

        wealth = (
            1 + portfolio_returns
        ).cumprod() * initial_value

        return wealth
    def calculate_drawdown(self, portfolio_returns):
        """
        Calculate the drawdown series from daily portfolio returns.
        """

        cumulative_wealth = (
            1 + portfolio_returns
        ).cumprod()

        running_max = cumulative_wealth.cummax()

        drawdown = (
            cumulative_wealth / running_max
        ) - 1

        return drawdown


if __name__ == "__main__":

    backtester = PortfolioBacktester()

    backtester.load_returns()

    backtester.split_data()

    max_sharpe, min_variance = (
        backtester.generate_training_markowitz_portfolios()
    )

    asset_names = backtester.train_returns.columns[1:]

    # Maximum Sharpe
    print("\nTraining Maximum Sharpe Weights")
    print("-" * 40)

    for asset, weight in zip(asset_names, max_sharpe.x):
        print(f"{asset:<12} {weight:.4%}")

    # Minimum Variance
    print("\nTraining Minimum Variance Weights")
    print("-" * 40)

    for asset, weight in zip(asset_names, min_variance.x):
        print(f"{asset:<12} {weight:.4%}")

    # Risk Parity
    train_covariance = backtester.calculate_training_covariance()

    risk_parity = backtester.optimize_risk_parity(train_covariance)
    rp_weights = risk_parity.x

    print("\nTraining Risk Parity Weights")
    print("-" * 40)

    for asset, weight in zip(asset_names, rp_weights):
        print(f"{asset:<12} {weight:.4%}")

    print("\nWeight Sum")
    print("-" * 40)
    print(f"{rp_weights.sum():.4%}")

    # Calculate and display Risk Parity risk contributions
    train_covariance_matrix = train_covariance.values

    rp_risk = np.sqrt(
        rp_weights.T @ train_covariance_matrix @ rp_weights
    )

    marginal_contribution = train_covariance_matrix @ rp_weights

    component_contribution = rp_weights * marginal_contribution

    risk_contributions = component_contribution / rp_risk

    risk_contribution_percent = (
        risk_contributions / rp_risk
    )

    print("\nTraining Risk Contributions")
    print("-" * 40)

    for asset, contribution, percentage in zip(
        asset_names,
        risk_contributions,
        risk_contribution_percent
    ):
        print(
            f"{asset:<12} "
            f"RC = {contribution:.4%} | "
            f"Risk Share = {percentage:.4%}"
        )

    print("\nTotal Portfolio Risk")
    print("-" * 40)
    print(f"{rp_risk:.4%}")

    print("\nRisk Contribution Sum")
    print("-" * 40)
    print(f"{risk_contributions.sum():.4%}")

    # --------------------------------------------------
    # Out-of-Sample Test Portfolio Returns
    # --------------------------------------------------

    max_sharpe_test_returns = (
        backtester.calculate_test_portfolio_returns(
            max_sharpe.x
        )
    )

    min_variance_test_returns = (
        backtester.calculate_test_portfolio_returns(
            min_variance.x
        )
    )

    risk_parity_test_returns = (
        backtester.calculate_test_portfolio_returns(
            rp_weights
        )
    )

    print("\nTest Portfolio Returns Calculated")
    print("-" * 40)

    print(
        f"Maximum Sharpe : "
        f"{len(max_sharpe_test_returns)} observations"
    )

    print(
        f"Minimum Variance : "
        f"{len(min_variance_test_returns)} observations"
    )

    print(
        f"Risk Parity : "
        f"{len(risk_parity_test_returns)} observations"
    )

    # --------------------------------------------------
    # Calculate Out-of-Sample Performance Metrics
    # --------------------------------------------------

    max_sharpe_metrics = (
        backtester.calculate_portfolio_metrics(
            max_sharpe_test_returns
        )
    )

    min_variance_metrics = (
        backtester.calculate_portfolio_metrics(
            min_variance_test_returns
        )
    )

    risk_parity_metrics = (
        backtester.calculate_portfolio_metrics(
            risk_parity_test_returns
        )
    )

    # Create comparison table
    performance_table = pd.DataFrame(
        {
            "Maximum Sharpe": max_sharpe_metrics,
            "Minimum Variance": min_variance_metrics,
            "Risk Parity": risk_parity_metrics
        }
    )

    print("\nOut-of-Sample Portfolio Performance")
    print("-" * 60)

    print("\nTotal Return")
    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_metrics['Total Return']:.4%}"
    )
    print(
        f"Minimum Variance   : "
        f"{min_variance_metrics['Total Return']:.4%}"
    )
    print(
        f"Risk Parity        : "
        f"{risk_parity_metrics['Total Return']:.4%}"
    )
    print("\nAnnualized Return")
    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_metrics['Annualized Return']:.4%}"
    )
    print(
        f"Minimum Variance   : "
        f"{min_variance_metrics['Annualized Return']:.4%}"
    )
    print(
        f"Risk Parity        : "
        f"{risk_parity_metrics['Annualized Return']:.4%}"
    )
    print("\nAnnualized Volatility")
    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_metrics['Annualized Volatility']:.4%}"
    )
    print(
        f"Minimum Variance   : "
        f"{min_variance_metrics['Annualized Volatility']:.4%}"
    )
    print(
        f"Risk Parity        : "
        f"{risk_parity_metrics['Annualized Volatility']:.4%}"
    )
    print("\nSharpe Ratio")
    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_metrics['Sharpe Ratio']:.3f}"
    )
    print(
        f"Minimum Variance   : "
        f"{min_variance_metrics['Sharpe Ratio']:.3f}"
    )
    print(
        f"Risk Parity        : "
        f"{risk_parity_metrics['Sharpe Ratio']:.3f}"
    )
    print("\nMaximum Drawdown")
    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_metrics['Maximum Drawdown']:.4%}"
    )
    print(
        f"Minimum Variance   : "
        f"{min_variance_metrics['Maximum Drawdown']:.4%}"
    )
    print(
        f"Risk Parity        : "
        f"{risk_parity_metrics['Maximum Drawdown']:.4%}"
    )
    print("\nVaR (95%)")
    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_metrics['VaR (95%)']:.4%}"
    )
    print(
        f"Minimum Variance   : "
        f"{min_variance_metrics['VaR (95%)']:.4%}"
    )
    print(
        f"Risk Parity        : "
        f"{risk_parity_metrics['VaR (95%)']:.4%}"
    )
        # --------------------------------------------------
    # Calculate Cumulative Wealth
    # --------------------------------------------------

    max_sharpe_wealth = (
        backtester.calculate_cumulative_wealth(
            max_sharpe_test_returns
        )
    )

    min_variance_wealth = (
        backtester.calculate_cumulative_wealth(
            min_variance_test_returns
        )
    )

    risk_parity_wealth = (
        backtester.calculate_cumulative_wealth(
            risk_parity_test_returns
        )
    )

    wealth_table = pd.DataFrame(
        {
            "date": max_sharpe_wealth.index,
            "Maximum Sharpe": max_sharpe_wealth.values,
            "Minimum Variance": min_variance_wealth.values,
            "Risk Parity": risk_parity_wealth.values
        }
    )

    print("\nFinal Portfolio Values")
    print("-" * 40)

    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_wealth.iloc[-1]:.2f}"
    )

    print(
        f"Minimum Variance   : "
        f"{min_variance_wealth.iloc[-1]:.2f}"
    )

    print(
        f"Risk Parity        : "
        f"{risk_parity_wealth.iloc[-1]:.2f}"
    )
        # --------------------------------------------------
    # Plot Cumulative Wealth
    # --------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=max_sharpe_wealth.index,
            y=max_sharpe_wealth.values,
            mode="lines",
            name="Maximum Sharpe"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=min_variance_wealth.index,
            y=min_variance_wealth.values,
            mode="lines",
            name="Minimum Variance"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=risk_parity_wealth.index,
            y=risk_parity_wealth.values,
            mode="lines",
            name="Risk Parity"
        )
    )

    fig.update_layout(
        title="Out-of-Sample Cumulative Portfolio Wealth",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode="x unified"
    )

    fig.show()
        # --------------------------------------------------
    # Calculate Drawdown
    # --------------------------------------------------

    max_sharpe_drawdown = (
        backtester.calculate_drawdown(
            max_sharpe_test_returns
        )
    )

    min_variance_drawdown = (
        backtester.calculate_drawdown(
            min_variance_test_returns
        )
    )

    risk_parity_drawdown = (
        backtester.calculate_drawdown(
            risk_parity_test_returns
        )
    )

    print("\nMaximum Drawdown Check")
    print("-" * 40)

    print(
        f"Maximum Sharpe     : "
        f"{max_sharpe_drawdown.min():.4%}"
    )

    print(
        f"Minimum Variance   : "
        f"{min_variance_drawdown.min():.4%}"
    )

    print(
        f"Risk Parity        : "
        f"{risk_parity_drawdown.min():.4%}"
    )
        # --------------------------------------------------
    # Plot Drawdown
    # --------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=max_sharpe_drawdown.index,
            y=max_sharpe_drawdown.values * 100,
            mode="lines",
            name="Maximum Sharpe"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=min_variance_drawdown.index,
            y=min_variance_drawdown.values * 100,
            mode="lines",
            name="Minimum Variance"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=risk_parity_drawdown.index,
            y=risk_parity_drawdown.values * 100,
            mode="lines",
            name="Risk Parity"
        )
    )

    fig.update_layout(
        title="Out-of-Sample Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified"
    )

    fig.show()
        # --------------------------------------------------
    # Save Backtest Performance Results
    # --------------------------------------------------

    results = pd.DataFrame(
        {
            "Maximum Sharpe": max_sharpe_metrics,
            "Minimum Variance": min_variance_metrics,
            "Risk Parity": risk_parity_metrics
        }
    ).T

    results.index.name = "Strategy"

    output_file = Path(
        "data/processed/backtest_results.csv"
    )

    results.to_csv(output_file)

    print("\nBacktest results saved to:")
    print(output_file)
        # --------------------------------------------------
    # Save Cumulative Wealth
    # --------------------------------------------------

    cumulative_wealth = pd.DataFrame(
        {
            "date": max_sharpe_wealth.index,
            "Maximum Sharpe": max_sharpe_wealth.values,
            "Minimum Variance": min_variance_wealth.values,
            "Risk Parity": risk_parity_wealth.values
        }
    )

    wealth_file = Path(
        "data/processed/cumulative_wealth.csv"
    )

    cumulative_wealth.to_csv(
        wealth_file,
        index=False
    )

    print("\nCumulative wealth saved to:")
    print(wealth_file)
    