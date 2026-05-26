"""TemporalValidationSplit — sequential split respecting temporal order."""

import pandas as pd


class TemporalValidationSplit:
    """Splits data sequentially (no shuffle) respecting temporal order.
    
    The data must already be sorted chronologically.
    Split is done by proportion: train_size, val_size, test_size.
    
    Follows the ValidationSplit protocol:
    transform(X, y, metadata) → (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
    """

    def __init__(self, train_size: float, val_size: float, test_size: float, **kwargs):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def transform(self, X: pd.DataFrame, y: pd.DataFrame, metadata: dict = None, **kwargs) -> tuple:
        """Split data sequentially by time.
        
        No shuffling — preserves temporal order.
        Returns (X_train, y_train, X_val, y_val, X_test, y_test, metadata).
        """
        metadata = metadata or {}
        n = len(X)

        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)

        X_train = X.iloc[:train_end].copy()
        y_train = y.iloc[:train_end].copy()

        X_val = X.iloc[train_end:val_end].copy()
        y_val = y.iloc[train_end:val_end].copy()

        X_test = X.iloc[val_end:].copy()
        y_test = y.iloc[val_end:].copy()

        return X_train, y_train, X_val, y_val, X_test, y_test, metadata

