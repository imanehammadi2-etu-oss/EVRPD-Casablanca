"""
===============================================================================
Module : LSTN Forecaster (Long & Short Transformer / LSTM Network)
File   : forecasting/models/lstn_forecaster.py
Description : Architecture hybride combinant récurrence court-terme (LSTM/GRU)
              et attention long-terme (Transformer Multi-Head).
===============================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional

from forecasting.base_forecaster import BaseForecaster


class PyTorchLSTNModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        context_len: int,
        pred_len: int,
        n_targets: int,
        rnn_type: str,
        rnn_hidden: int,
        rnn_layers: int,
        d_model: int,
        n_heads: int,
        n_transformer_layers: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"Contrainte d_model % n_heads == 0 violée ! (d_model={d_model}, n_heads={n_heads})")

        self.context_len = context_len
        self.pred_len = pred_len
        self.n_targets = n_targets

        # Branche Court-Terme (RNN)
        if rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(n_features, rnn_hidden, num_layers=rnn_layers, batch_first=True, dropout=dropout if rnn_layers > 1 else 0.0)
        else:
            self.rnn = nn.LSTM(n_features, rnn_hidden, num_layers=rnn_layers, batch_first=True, dropout=dropout if rnn_layers > 1 else 0.0)

        # Branche Long-Terme (Transformer)
        self.attn_proj = nn.Linear(n_features, d_model)
        trans_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=n_transformer_layers)

        # Fusion
        fusion_dim = rnn_hidden + d_model
        self.head = nn.Linear(fusion_dim, pred_len * n_targets)

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. Court-Terme RNN
        rnn_out, _ = self.rnn(x)
        rnn_last = rnn_out[:, -1, :]

        # 2. Long-Terme Transformer
        t_tokens = self.attn_proj(x)
        trans_out = self.transformer(t_tokens)
        trans_last = trans_out[:, -1, :]

        # 3. Fusion & Tête
        fused = torch.cat([rnn_last, trans_last], dim=-1)
        preds = self.head(fused)
        return preds.view(batch_size, self.pred_len, self.n_targets)


class LSTNForecaster(BaseForecaster):
    """
    Forecaster hybride Long & Short Transformer / LSTM Network.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="LSTN", config=config)
        self.rnn_type = self.config.get("rnn_type", "LSTM")
        self.rnn_hidden = self.config.get("rnn_hidden_size", 128)
        self.rnn_layers = self.config.get("rnn_num_layers", 2)
        self.d_model = self.config.get("d_model", 128)
        self.n_heads = self.config.get("n_heads", 4)
        self.n_trans_layers = self.config.get("n_transformer_layers", 2)
        self.d_ff = self.config.get("d_ff", 512)
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
    ) -> "LSTNForecaster":
        t0 = time.time()
        n_samples, context_len, n_features = X_train.shape
        _, pred_len, n_targets = y_train.shape

        self.model = PyTorchLSTNModel(
            n_features=n_features,
            context_len=context_len,
            pred_len=pred_len,
            n_targets=n_targets,
            rnn_type=self.rnn_type,
            rnn_hidden=self.rnn_hidden,
            rnn_layers=self.rnn_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_transformer_layers=self.n_trans_layers,
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
            raise ValueError("Le modèle LSTN n'a pas été entraîné.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(device)
            preds = self.model(tx).cpu().numpy()

        return preds
