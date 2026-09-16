import pandas as pd
import yfinance as yf
from pathlib import Path


class Nifty50Benchmark:
    """
    Download NIFTY 50 historical prices and create
    a daily return series aligned with the project data.
    """

    def __init__(
        self,
        start_date="2024-01-01",
        end_date="2026-07-29",
        output_file=(
            "data/processed/nifty50_returns.csv"
        )
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.output_file = Path(output_file)

    def download_data(self):
        """
        Download NIFTY 50 historical data from Yahoo Finance.
        """

        print("Downloading NIFTY 50 data...")

        data = yf.download(
            "^NSEI",
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            raise ValueError(
                "No NIFTY 50 data was downloaded."
            )

        return data

    def create_returns(self, data):
        """
        Create daily NIFTY 50 percentage returns.
        """

        # Handle Yahoo Finance's possible MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].iloc[:, 0]
        else:
            close = data["Close"]

        returns = close.pct_change()

        result = pd.DataFrame(
            {
                "date": returns.index,
                "NIFTY50": returns.values
            }
        )

        result = result.dropna()

        result["date"] = pd.to_datetime(
            result["date"]
        ).dt.tz_localize(None)

        return result

    def save_returns(self, returns):
        """
        Save NIFTY 50 returns.
        """

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        returns.to_csv(
            self.output_file,
            index=False
        )

        print(
            "\nNIFTY 50 returns saved to:"
        )

        print(self.output_file)

    def run(self):
        """
        Download, process and save NIFTY 50 returns.
        """

        data = self.download_data()

        returns = self.create_returns(
            data
        )

        print("\nNIFTY 50 Return Data")
        print("-" * 50)

        print(
            f"Observations : {len(returns)}"
        )

        print(
            f"Start        : "
            f"{returns['date'].iloc[0].date()}"
        )

        print(
            f"End          : "
            f"{returns['date'].iloc[-1].date()}"
        )

        print("\nLatest observations:")

        print(
            returns.tail(5).to_string(
                index=False
            )
        )

        self.save_returns(
            returns
        )


if __name__ == "__main__":

    benchmark = Nifty50Benchmark()

    benchmark.run()