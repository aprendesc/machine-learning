"""TimeSeriesFeatureEngineering — computes configurable technical indicators."""

import numpy as np
import pandas as pd


class TimeSeriesFeatureEngineering:
    """Computes technical indicators as additional features for time series data.
    
    Supported features: sma_N, ema_N, rsi_N, macd, volatility_N, returns_Nd, day_of_week.
    After computing, drops rows with NaN (warmup period) and aligns y accordingly.
    
    Follows the Transformer protocol: fit/transform(X, y, metadata) → (X, y, metadata).
    """

    def __init__(self, features: list, **kwargs):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None, metadata: dict = None, **kwargs) -> tuple:
        """No-op for stateless transform."""
        return X, y, metadata or {}

    def transform(self, X: pd.DataFrame, y=None, metadata: dict = None, **kwargs) -> tuple:
        """Compute technical indicators and drop NaN rows.
        
        Uses 'Close' column from X for price-based indicators.
        Aligns y with remaining rows after NaN removal.
        """
        metadata = metadata or {}
        df = X.copy()

        # Use Close price for computing indicators (must be in X)
        close = df["Close"] if "Close" in df.columns else None

        for feature in self.features:
            if feature.startswith("sma_") and close is not None:
                window = int(feature.split("_")[1])
                df[feature] = close.rolling(window=window).mean()

            elif feature.startswith("ema_") and close is not None:
                span = int(feature.split("_")[1])
                df[feature] = close.ewm(span=span, adjust=False).mean()

            elif feature.startswith("rsi_") and close is not None:
                period = int(feature.split("_")[1])
                df[feature] = self._compute_rsi(close, period)

            elif feature == "macd" and close is not None:
                ema_12 = close.ewm(span=12, adjust=False).mean()
                ema_26 = close.ewm(span=26, adjust=False).mean()
                df["macd"] = ema_12 - ema_26

            elif feature.startswith("volatility_") and close is not None:
                window = int(feature.split("_")[1])
                df[feature] = close.pct_change().rolling(window=window).std()

            elif feature.startswith("returns_") and close is not None:
                # e.g., returns_1d → 1-day returns
                days = int(feature.replace("returns_", "").replace("d", ""))
                df[feature] = close.pct_change(periods=days)

            elif feature == "day_of_week":
                if hasattr(df.index, "dayofweek"):
                    df["day_of_week"] = df.index.dayofweek
                else:
                    df["day_of_week"] = 0

        # Drop rows with NaN (warmup period)
        valid_mask = df.notna().all(axis=1)
        df = df[valid_mask]

        # Align y with the same indices
        if y is not None:
            y = y.loc[df.index]

        return df, y, metadata

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def inverse_transform(self, X, y, metadata=None, **kwargs) -> tuple:
        return X, y, metadata or {}

