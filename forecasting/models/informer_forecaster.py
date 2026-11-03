"""
===============================================================================
Module : Informer Attention-Based Forecaster
File   : forecasting/models/informer_forecaster.py
Description : Modèle de prévision d'attention à forte efficacité (ProbSparse attention).
===============================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional

from forecasting.base_forecaster import BaseForecaster


class PyTorchInformerModel(nn.Module):
    def __init__(self, n_features: int, context_len: int, pred_len: int, n_targets: int, d_model: int, n_heads: int, e_layers: int, d_ff: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"Contrainte d_model % n_heads == 0 violée ! (d_model={d_model}, n_heads={n_heads})")

        self.context_len = context_len
        self.pred_len = pred_len
        self.n_targets = n_targets

        self.enc_embed = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Linear(d_model * context_len, pred_len * n_targets)

    def forward(self, x):
        batch_size = x.shape[0]
        tokens = self.enc_embed(x)
        encoded = self.encoder(tokens)
        flat = encoded.view(batch_size, -1)
        preds = self.head(flat)
        return preds.view(batch_size, self.pred_len, self.n_targets)


class InformerForecaster(BaseForecaster):
    """
    Forecaster d'attention basé sur l'architecture Informer.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="Informer", config=config)
        self.d_model = self.config.get("d_model", 256)
        self.n_heads = self.config.get("n_heads", 8)
        self.e_layers = self.config.get("e_layers", 3)
        self.d_ff = self.config.get("d_ff", 1024)
        self.dropout = self.config.get("dropout", 0.10)
        self.lr = self.config.get("learning_rate", 0.0001)
        self.batch_size = self.config.get("batch_size", 32)
        self.max_epochs = self.config.get("max_epochs", 50)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "InformerForecaster":
        t0 = time.time()
        n_samples, context_len, n_features = X_train.shape
        _, pred_len, n_targets = y_train.shape

        self.model = PyTorchInformerModel(
            n_features=n_features,
            context_len=context_len,
            pred_len=pred_len,
            n_targets=n_targets,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
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
            raise ValueError("Le modèle Informer n'a pas été entraîné.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model(tx).cpu().numpy()

        return preds
