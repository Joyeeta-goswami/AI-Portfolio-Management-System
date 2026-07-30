import numpy as np
import pandas as pd
from pathlib import Path
import logging

class RiskMetrics:
    """
    Calculate common financial risk and performance metrics.
    """

    def __init__(self, risk_free_rate: float = 0.065):
        # Annual risk-free rate (6.5%)
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate daily percentage returns.
        """
        return prices.pct_change().dropna()

    def cagr(self, prices: pd.Series) -> float:
        """
        Compound Annual Growth Rate.
        """
        years = len(prices) / self.trading_days
        return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1

    def annual_volatility(self, returns: pd.Series) -> float:
        """
        Annualised volatility.
        """
        return returns.std() * np.sqrt(self.trading_days)

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Annualised Sharpe Ratio.
        """

        volatility = self.annual_volatility(returns)

        if volatility == 0:
            return 0.0

        annual_return = returns.mean() * self.trading_days
        excess_return = annual_return - self.risk_free_rate

        return excess_return / volatility

    def max_drawdown(self, prices: pd.Series) -> float:
        """
        Maximum peak-to-trough decline.
        """

        running_max = prices.cummax()
        drawdown = (prices - running_max) / running_max

        return drawdown.min()

    def historical_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Historical Value at Risk.
        """
        return np.percentile(returns, (1 - confidence) * 100)

    def total_return(self, prices: pd.Series) -> float:
        """
        Total percentage return.
        """
        return (prices.iloc[-1] / prices.iloc[0]) - 1

    def summarize(self, prices: pd.Series) -> dict:

        returns = self.calculate_returns(prices)

        return {
            "Total Return (%)":
                round(self.total_return(prices) * 100, 2),

            "CAGR (%)":
                round(self.cagr(prices) * 100, 2),

            "Annual Volatility (%)":
                round(self.annual_volatility(returns) * 100, 2),

            "Sharpe Ratio":
                round(self.sharpe_ratio(returns), 2),

            "Maximum Drawdown (%)":
                round(self.max_drawdown(prices) * 100, 2),

            "95% Daily VaR (%)":
                round(self.historical_var(returns) * 100, 2)
        }

class RiskSummaryGenerator:
    """
    Generate a risk summary report for all stocks in data/raw.
    """

    def __init__(
        self,
        raw_data_path="data/raw",
        output_path="data/processed/risk_summary.csv"
    ):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.engine = RiskMetrics()

    def generate_summary(self):
        """
        Read every stock CSV, calculate risk metrics,
        and save them into a single CSV.
        """

        summary = []

        csv_files = sorted(self.raw_data_path.glob("*.csv"))

        if not csv_files:
            print("No CSV files found.")
            return

        for file in csv_files:

            try:
                df = pd.read_csv(file)

                report = self.engine.summarize(df["close"])

                report["Stock"] = file.stem

                summary.append(report)

                print(f"Processed {file.stem}")

            except Exception as e:
                print(f"Error processing {file.name}: {e}")

        summary_df = pd.DataFrame(summary)

        columns = [
            "Stock",
            "Total Return (%)",
            "CAGR (%)",
            "Annual Volatility (%)",
            "Sharpe Ratio",
            "Maximum Drawdown (%)",
            "95% Daily VaR (%)"
        ]

        summary_df = summary_df[columns]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(self.output_path, index=False)

        print("\nRisk summary saved successfully!")
        print(self.output_path)

        return summary_df

if __name__ == "__main__":

    generator = RiskSummaryGenerator()

    report = generator.generate_summary()

    print("\n")
    print(report)