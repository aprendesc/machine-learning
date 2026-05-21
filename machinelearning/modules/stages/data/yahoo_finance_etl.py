"""YahooFinanceETL — downloads OHLCV data from Yahoo Finance and persists as Dataset."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from eigenframework.modules.core.dataset import Dataset


class YahooFinanceETL:
    """Downloads OHLCV data for a given ticker and saves it as a Dataset.
    
    Follows the Data Stage protocol: run(X, y, metadata) → (X, y, metadata).
    """

    def __init__(self, ticker: str, period: str, dataset_name: str, data_path: str, **kwargs):
        self.ticker = ticker
        self.period = period
        self.dataset_name = dataset_name
        self.data_path = data_path

    def run(self, X=None, y=None, metadata=None, **kwargs) -> tuple:
        """Download OHLCV data and persist as Dataset.
        
        Downloads historical data via yfinance for the configured ticker/period.
        Writes the result to disk in Dataset format at {data_path}/{dataset_name}/.
        """
        metadata = metadata or {}

        # Download data
        ticker_obj = yf.Ticker(self.ticker)
        df = ticker_obj.history(period=self.period)
        df = df.reset_index()

        # Keep relevant columns: Date, Open, High, Low, Close, Volume
        columns_to_keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in columns_to_keep if c in df.columns]
        df = df[available_cols].copy()

        # Ensure Date is string for CSV compatibility
        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)

        # Write as Dataset (resolve path from cwd, consistent with eigenframework ETLDummy)
        output_path = Path.cwd() / self.data_path / self.dataset_name
        Dataset().write(df=df, mmf={}, path=output_path, table_storage_format=".csv")

        return X, y, metadata

