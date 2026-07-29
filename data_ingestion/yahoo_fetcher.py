import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create data directory
DATA_PATH = Path("data/raw")
DATA_PATH.mkdir(parents=True, exist_ok=True)


def fetch_nse_stock(
    ticker: str,
    start: str = "2019-01-01",
    end: str = None
) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Example:
        RELIANCE.NS
        TCS.NS
        HDFCBANK.NS
    """

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    logger.info(f"Fetching {ticker} from {start} to {end}...")

    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    # Convert index to normal column
    df.reset_index(inplace=True)

    # Standardize column names
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

    # Keep only required columns
    df = df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dividends",
            "stock_splits",
        ]
    ]

    # Convert datetime → date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Add ticker column
    df["ticker"] = ticker.replace(".NS", "")

    return df


def save_to_csv(df: pd.DataFrame, ticker: str):
    """
    Save stock data to CSV.
    """

    filename = DATA_PATH / f"{ticker.replace('.NS', '')}.csv"

    df.to_csv(filename, index=False)

    logger.info(f"Saved to {filename}")


def fetch_multiple_stocks(
    tickers: list,
    start: str = "2019-01-01"
):
    """
    Download multiple stocks and save each one as a CSV.
    """

    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}")

            df = fetch_nse_stock(
                ticker=ticker,
                start=start
            )

            save_to_csv(df, ticker)

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")


if __name__ == "__main__":

    NIFTY50_TICKERS = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "LT.NS",
        "ITC.NS"
    ]

    fetch_multiple_stocks(
        tickers=NIFTY50_TICKERS,
        start="2024-01-01"
    )

    logger.info("All downloads completed.")