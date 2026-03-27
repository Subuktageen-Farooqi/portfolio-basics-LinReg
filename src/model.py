"""Model definition for tabular fire risk regression."""

from __future__ import annotations

import torch
from torch import nn


class FireRiskRegressor(nn.Module):
    """Small MLP for tabular fire-risk score regression."""

    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
