"""FFNNModel — Feed-Forward Neural Network for binary classification using PyTorch."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _FFNNNetwork(nn.Module):
    """Internal PyTorch neural network architecture."""

    def __init__(self, input_size: int, hidden_layers: list, output_size: int = 1):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FFNNModel:
    """Feed-Forward Neural Network for binary classification.
    
    Architecture: Linear → ReLU → ... → Sigmoid.
    Training: Adam optimizer + BCELoss + early stopping on validation loss.
    
    Follows the Model protocol: initialize/fit/predict.
    """

    def __init__(
        self,
        hidden_layers: list = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        prediction_mode: str = None,
        patience: int = 10,
        **kwargs,
    ):
        self.hidden_layers = hidden_layers or [64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prediction_mode = prediction_mode  # "predict_proba" or None
        self.patience = patience
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, X=None, y=None, metadata=None, **kwargs) -> tuple:
        """Build the network architecture. Called before fit."""
        # Architecture is built in fit() when input_size is known
        return X, y, metadata or {}

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test, metadata=None, **kwargs) -> tuple:
        """Train the FFNN with early stopping on validation loss.
        
        Returns ((X_train, y_train, X_val, y_val, X_test, y_test), metadata).
        """
        metadata = metadata or {}

        # Convert to tensors (replace NaN/Inf with 0 for safety)
        import numpy as np
        X_train_arr = np.nan_to_num(X_train.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_arr = np.nan_to_num(X_val.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_t = torch.tensor(X_train_arr, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val_arr, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(self.device)

        # Build model
        input_size = X_train_t.shape[1]
        self.model = _FFNNNetwork(input_size, self.hidden_layers, output_size=1).to(self.device)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        data_tuple = (X_train, y_train, X_val, y_val, X_test, y_test)
        return data_tuple, metadata

    def predict(self, X, y=None, metadata=None, **kwargs) -> tuple:
        """Predict using the trained FFNN.
        
        Returns (None, y_pred: pd.DataFrame, metadata).
        - prediction_mode='predict_proba': returns probabilities (float 0-1)
        - Otherwise: returns class labels (0/1)
        """
        metadata = metadata or {}
        self.model.eval()

        import numpy as np
        X_arr = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_t = torch.tensor(X_arr, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy().flatten()

        if self.prediction_mode == "predict_proba":
            y_pred = pd.DataFrame(probs, columns=["prediction"], index=X.index)
        else:
            classes = (probs >= 0.5).astype(int)
            y_pred = pd.DataFrame(classes, columns=["prediction"], index=X.index)

        return None, y_pred, metadata

