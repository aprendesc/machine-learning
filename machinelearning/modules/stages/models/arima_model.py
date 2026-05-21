"""ARIMAModel — ARIMA/SARIMAX model for time series regression using statsmodels."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAModel:
    """ARIMA/SARIMAX model for time series forecasting.

    Univariate model — fits on y_train (target series), ignores X features.
    Predict interprets len(X) as the number of steps to forecast.

    Follows the Model protocol: initialize/fit/predict.
    """

    def __init__(
        self,
        order: list = None,
        seasonal_order: list = None,
        target_column: str = "Close",
        **kwargs,
    ):
        self.order = tuple(order) if order else (5, 1, 2)
        self.seasonal_order = tuple(seasonal_order) if seasonal_order else (0, 0, 0, 0)
        self.target_column = target_column
        self.fitted_model = None
        self.train_series = None

    def initialize(self, X=None, y=None, metadata=None, **kwargs) -> tuple:
        """No-op — model is built in fit()."""
        return X, y, metadata or {}

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test, metadata=None, **kwargs) -> tuple:
        """Fit SARIMAX on y_train series.

        Ignores X (univariate). Uses the target column from y_train.
        Returns ((X_train, y_train, X_val, y_val, X_test, y_test), metadata).
        """
        metadata = metadata or {}

        # Extract the target series
        if isinstance(y_train, pd.DataFrame):
            series = y_train.iloc[:, 0].values
        else:
            series = np.array(y_train).flatten()

        self.train_series = series

        # Fit SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model = model.fit(disp=False, maxiter=200)

        data_tuple = (X_train, y_train, X_val, y_val, X_test, y_test)
        return data_tuple, metadata

    def predict(self, X, y=None, metadata=None, **kwargs) -> tuple:
        """Forecast N steps ahead where N = len(X).

        Returns (None, y_pred: pd.DataFrame, metadata).
        y_pred has one column (target_column) with N forecasted values.
        """
        metadata = metadata or {}

        n_steps = len(X) if X is not None else 1

        # Forecast
        forecast = self.fitted_model.forecast(steps=n_steps)
        forecast_values = np.array(forecast).flatten()

        # Build DataFrame with same index as X if possible
        if X is not None and hasattr(X, "index"):
            y_pred = pd.DataFrame(
                forecast_values, columns=[self.target_column], index=X.index
            )
        else:
            y_pred = pd.DataFrame(forecast_values, columns=[self.target_column])

        return None, y_pred, metadata

