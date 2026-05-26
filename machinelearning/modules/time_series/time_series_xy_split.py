"""TimeSeriesXYSplit — splits temporal DataFrame into X (features) and y (target)."""

import pandas as pd


class TimeSeriesXYSplit:
    """Splits a temporal DataFrame into features X and target y.
    
    Converts Date column to datetime index, extracts target as y,
    and keeps remaining OHLCV columns as X. Orders chronologically.
    
    Follows the Transformer protocol: fit/transform(X, y, metadata) → (X, y, metadata).
    """

    def __init__(self, target_column: str, forecast_horizon: int = 1, **kwargs):
        self.target_column = target_column
        self.forecast_horizon = forecast_horizon

    def fit(self, X: pd.DataFrame, y=None, metadata: dict = None, **kwargs) -> tuple:
        """No-op for stateless transform."""
        return X, y, metadata or {}

    def transform(self, X: pd.DataFrame, y=None, metadata: dict = None, **kwargs) -> tuple:
        """Split DataFrame into X (features) and y (target).
        
        - Converts 'Date' column to datetime index
        - Sorts chronologically
        - Extracts target_column as y (DataFrame)
        - X = remaining numeric columns
        """
        metadata = metadata or {}
        df = X.copy()

        # Convert Date to datetime index if present
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df = df.sort_index()

        # Extract target
        y = df[[self.target_column]].copy()

        # X = all columns except target
        feature_cols = [c for c in df.columns if c != self.target_column]
        X_out = df[feature_cols].copy()

        return X_out, y, metadata

    def inverse_transform(self, X, y, metadata=None, **kwargs) -> tuple:
        return X, y, metadata or {}

