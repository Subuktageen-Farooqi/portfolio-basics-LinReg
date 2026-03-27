"""Utility functions for data handling, reproducibility, metrics, and plotting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "wind_speed_kmh",
    "air_quality_index",
    "vegetation_index",
    "distance_to_station_km",
    "response_time_min",
]
TARGET_COLUMN = "fire_risk_score"


class StandardScalerTorch:
    """Simple torch-based standard scaler persisted to JSON."""

    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0, unbiased=False).clamp(min=1e-8)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)

    def save(self, path: Path) -> None:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is empty. Fit before saving.")
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_columns": FEATURE_COLUMNS,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StandardScalerTorch":
        payload = json.loads(path.read_text(encoding="utf-8"))
        scaler = cls()
        scaler.mean = torch.tensor(payload["mean"], dtype=torch.float32)
        scaler.std = torch.tensor(payload["std"], dtype=torch.float32)
        return scaler


def set_seed(seed: int) -> None:
    """Configure deterministic behavior for reproducible training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataframe(data_path: Path) -> pd.DataFrame:
    """Load dataset and ensure all required columns are present."""

    df = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def dataframe_to_tensors(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert selected features/target from DataFrame to float tensors."""

    x = torch.tensor(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(df[TARGET_COLUMN].to_numpy(dtype=np.float32), dtype=torch.float32).view(-1, 1)
    return x, y


def split_indices(n_samples: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create shuffled train/validation/test indices with torch only."""

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    return train_idx, val_idx, test_idx


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² in torch."""

    err = y_pred - y_true
    mae = torch.mean(torch.abs(err))
    rmse = torch.sqrt(torch.mean(err**2))
    ss_res = torch.sum(err**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {"mae": float(mae.item()), "rmse": float(rmse.item()), "r2": float(r2.item())}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def risk_band(score: float) -> str:
    """Simple human-readable risk band for reporting/demo usage."""

    if score < 35:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def add_risk_band_column(scores: Sequence[float]) -> list[str]:
    return [risk_band(float(s)) for s in scores]


def plot_loss_curve(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    plt.plot([line_min, line_max], [line_min, line_max], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Fire Risk")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, alpha=0.8)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
