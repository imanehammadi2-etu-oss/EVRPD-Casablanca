"""
===============================================================================
Module : Deep PyTorch LSTM Forecaster
File   : forecasting/models/lstm_forecaster.py
Description : Réseau de neurones récurrent LSTM profond en PyTorch.
===============================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional
from forecasting.base_forecaster import BaseForecaster


class PyTorchLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, pred_len: int, n_targets: int):
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, pred_len * n_targets)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        # Utiliser le dernier état caché
        last_hidden = out[:, -1, :]
        preds = self.fc(last_hidden)
        return preds.view(-1, self.pred_len, self.n_targets)


class LSTMForecaster(BaseForecaster):
    """
    Forecaster basé sur un réseau de neurones récurrent LSTM PyTorch.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="LSTM", config=config)
        self.hidden_size = self.config.get("hidden_size", 128)
        self.num_layers = self.config.get("num_layers", 2)
        self.dropout = self.config.get("dropout", 0.10)
        self.lr = self.config.get("learning_rate", 0.001)
        self.batch_size = self.config.get("batch_size", 32)
        self.max_epochs = self.config.get("max_epochs", 50)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "LSTMForecaster":
        t0 = time.time()
        n_samples, context_len, n_features = X_train.shape
        _, pred_len, n_targets = y_train.shape

        self.model = PyTorchLSTMModel(
            input_dim=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pred_len=pred_len,
            n_targets=n_targets
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.max_epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        self.train_time = time.time() - t0
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("Le modèle LSTM n'a pas été entraîné.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model(tx).cpu().numpy()

        return preds
