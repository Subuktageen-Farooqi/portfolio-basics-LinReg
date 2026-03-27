"""Train a PyTorch regressor for fire risk score prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import FireRiskRegressor
from utils import (
    StandardScalerTorch,
    dataframe_to_tensors,
    load_dataframe,
    plot_loss_curve,
    plot_pred_vs_actual,
    plot_residuals,
    regression_metrics,
    save_json,
    set_seed,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fire-risk score regressor.")
    parser.add_argument("--data-path", type=Path, default=Path("data/fire_risk_dataset.csv"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--scaler-path", type=Path, default=Path("outputs/scaler.json"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_dataframe(args.data_path)
    x_all, y_all = dataframe_to_tensors(df)

    train_idx, val_idx, test_idx = split_indices(
        n_samples=len(df), train_ratio=0.7, val_ratio=0.15, seed=args.seed
    )

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]

    scaler = StandardScalerTorch()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    scaler.save(args.scaler_path)

    train_loader = DataLoader(TensorDataset(x_train_scaled, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_scaled, y_val), batch_size=args.batch_size, shuffle=False)

    model = FireRiskRegressor(input_dim=x_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_rmse = float("inf")

    for epoch in range(args.epochs):
        model.train()
        batch_train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())

        mean_train_loss = float(np.mean(batch_train_losses))
        train_losses.append(mean_train_loss)

        model.eval()
        val_batch_losses = []
        val_preds_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_batch_losses.append(criterion(preds, yb).item())
                val_preds_all.append(preds)

        mean_val_loss = float(np.mean(val_batch_losses))
        val_losses.append(mean_val_loss)

        val_preds_tensor = torch.cat(val_preds_all, dim=0)
        val_rmse = regression_metrics(y_val, val_preds_tensor)["rmse"]
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.model_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} "
                f"train_loss={mean_train_loss:.4f} "
                f"val_loss={mean_val_loss:.4f} val_rmse={val_rmse:.4f}"
            )

    best_model = FireRiskRegressor(input_dim=x_train_scaled.shape[1])
    best_model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    best_model.eval()

    with torch.no_grad():
        test_preds = best_model(x_test_scaled)

    metrics = regression_metrics(y_test, test_preds)
    print("Test metrics:", metrics)

    save_json(args.outputs_dir / "metrics.json", metrics)
    save_json(
        args.outputs_dir / "split_info.json",
        {
            "total_rows": len(df),
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "seed": args.seed,
        },
    )

    y_true_np = y_test.squeeze(1).numpy()
    y_pred_np = test_preds.squeeze(1).numpy()
    plot_loss_curve(train_losses, val_losses, args.outputs_dir / "loss_curve.png")
    plot_pred_vs_actual(y_true_np, y_pred_np, args.outputs_dir / "pred_vs_actual.png")
    plot_residuals(y_true_np, y_pred_np, args.outputs_dir / "residuals.png")


if __name__ == "__main__":
    main()
