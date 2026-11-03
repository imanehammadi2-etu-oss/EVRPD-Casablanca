"""
===============================================================================
Module : Custom GPT / Patch-Based Transformer Time Series Forecaster
File   : forecasting/models/gpt4ts_forecaster.py
Description : Architecture de prévision de séries temporelles basée sur l'attention
              et le découpage en patchs (type GPT/PatchTST).
===============================================================================
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional

from forecasting.base_forecaster import BaseForecaster
from forecasting.preprocessing import RevIN


class PatchEmbedding(nn.Module):
    def __init__(self, patch_length: int, num_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_length * num_features, d_model)

    def forward(self, patches):
        # patches shape: (batch_size, n_patches, patch_length, num_features)
        batch_size, n_patches, patch_len, num_features = patches.shape
        flat_patches = patches.view(batch_size, n_patches, patch_len * num_features)
        return self.proj(flat_patches) # (batch_size, n_patches, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PyTorchCustomGPTModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        context_len: int,
        pred_len: int,
        n_targets: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()
        # Vérification stricte de la contrainte architecturale
        if d_model % n_heads != 0:
            raise ValueError(f"Contrainte d_model % n_heads == 0 violée ! (d_model={d_model}, n_heads={n_heads})")

        self.context_len = context_len
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # Nombre de patchs
        self.n_patches = math.floor((context_len - patch_len) / patch_stride) + 1

        self.patch_embed = PatchEmbedding(patch_len, n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len * n_targets)

    def forward(self, x):
        # x shape: (batch_size, context_len, n_features)
        batch_size, context_len, n_features = x.shape

        # Patching dynamique
        patches = []
        for i in range(self.n_patches):
            start = i * self.patch_stride
            end = start + self.patch_len
            patches.append(x[:, start:end, :])
        patches_tensor = torch.stack(patches, dim=1) # (batch_size, n_patches, patch_len, n_features)

        # Embedding + Attention
        tokens = self.patch_embed(patches_tensor)
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer_backbone(tokens)

        # Projection finale
        flat_feats = encoded.view(batch_size, -1)
        preds = self.head(flat_feats)
        return preds.view(batch_size, self.pred_len, self.n_targets)


class CustomGPTTimeSeriesForecaster(BaseForecaster):
    """
    Forecaster basés sur une architecture Transformer/GPT à patchs temporels.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="CustomGPT", config=config)
        self.d_model = self.config.get("d_model", 256)
        self.n_heads = self.config.get("n_heads", 8)
        self.n_layers = self.config.get("n_layers", 4)
        self.d_ff = self.config.get("d_ff", 1024)
        self.dropout = self.config.get("dropout", 0.10)
        self.patch_len = self.config.get("patch_length", 12)
        self.patch_stride = self.config.get("patch_stride", 6)
        self.lr = self.config.get("learning_rate", 0.0001)
        self.batch_size = self.config.get("batch_size", 32)
        self.max_epochs = self.config.get("max_epochs", 50)
        self.use_revin = self.config.get("use_revin", True)
        
        # Vérification stricte
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"Contrainte d_model % n_heads == 0 violée ! d_model={self.d_model}, n_heads={self.n_heads}")

        self.model = None
        self.revin = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "CustomGPTTimeSeriesForecaster":
        t0 = time.time()
        n_samples, context_len, n_features = X_train.shape
        _, pred_len, n_targets = y_train.shape

        if self.use_revin:
            self.revin = RevIN(num_features=n_features)
            X_train_proc = self.revin.fit_transform(X_train)
        else:
            X_train_proc = X_train

        self.model = PyTorchCustomGPTModel(
            n_features=n_features,
            context_len=context_len,
            pred_len=pred_len,
            n_targets=n_targets,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dataset = TensorDataset(torch.tensor(X_train_proc, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
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
            raise ValueError("Le modèle CustomGPT n'a pas été entraîné.")

        if self.use_revin and self.revin is not None:
            X_proc = self.revin.transform(X)
        else:
            X_proc = X

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X_proc, dtype=torch.float32).to(device)
            preds = self.model(tx).cpu().numpy()

        if self.use_revin and self.revin is not None:
            preds = self.revin.inverse_transform(preds)

        return preds


# Alias compatible avec la spécification
GPT4TSForecaster = CustomGPTTimeSeriesForecaster
