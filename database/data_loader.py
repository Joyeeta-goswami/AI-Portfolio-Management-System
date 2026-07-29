import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

RAW_DATA = Path("data/raw")
PROCESSED_DATA = Path("data/processed")

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)


class DataLoader:

    def __init__(self):
        self.raw_path = RAW_DATA
        self.processed_path = PROCESSED_DATA

    def load_stock(self, filename: str) -> pd.DataFrame:
        """
        Load a single stock CSV.
        """

        file_path = self.raw_path / filename

        df = pd.read_csv(file_path)

        return df

    def load_all_stocks(self) -> dict:
        """
        Load every CSV inside data/raw.
        """

        stocks = {}

        for file in self.raw_path.glob("*.csv"):

            ticker = file.stem

            logger.info(f"Loading {ticker}")

            stocks[ticker] = pd.read_csv(file)

        return stocks
    
    def create_price_matrix(self, stocks: dict) -> pd.DataFrame:
        """
        Merge all stock closing prices into one table.
        """

        merged = None

        for ticker, df in stocks.items():

            temp = df[["date", "close"]].copy()

            temp.rename(
                columns={"close": ticker},
                inplace=True
            )

            if merged is None:

                merged = temp

            else:

                merged = merged.merge(
                    temp,
                    on="date",
                    how="inner"
                )
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values("date").reset_index(drop=True)

        return merged
    
    def create_returns_matrix(
        self,
        price_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate daily returns.
        """

        returns = price_matrix.copy()

        returns.iloc[:, 1:] = (
            returns.iloc[:, 1:]
            .pct_change()
        )

        returns.dropna(inplace=True)

        return returns
    
    def save_processed_data(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame
    ):

        prices.to_csv(
            self.processed_path / "prices.csv",
            index=False
        )

        returns.to_csv(
            self.processed_path / "returns.csv",
            index=False
        )

        logger.info("Processed data saved.")


# -------------------------
# Main Program
# -------------------------

if __name__ == "__main__":
    loader = DataLoader()

    stocks = loader.load_all_stocks()

    prices = loader.create_price_matrix(stocks)

    returns = loader.create_returns_matrix(prices)

    loader.save_processed_data(
        prices,
        returns
    )

    print("\nPRICE MATRIX\n")
    print(prices.head())

    print("\nRETURNS MATRIX\n")
    print(returns.head())