"""Batch inference for fire-risk score regression model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from model import FireRiskRegressor
from utils import FEATURE_COLUMNS, StandardScalerTorch, add_risk_band_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for fire-risk score regression model.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to CSV with inference rows")
    parser.add_argument("--model-path", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--scaler-path", type=Path, default=Path("outputs/scaler.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/predictions.csv"))
    parser.add_argument("--add-risk-band", action="store_true", help="Add Low/Medium/High band column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    missing = sorted(set(FEATURE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    x = torch.tensor(df[FEATURE_COLUMNS].to_numpy(dtype="float32"), dtype=torch.float32)
    scaler = StandardScalerTorch.load(args.scaler_path)
    x_scaled = scaler.transform(x)

    model = FireRiskRegressor(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = model(x_scaled).squeeze(1).numpy()

    out_df = df.copy()
    out_df["predicted_fire_risk_score"] = preds
    if args.add_risk_band:
        out_df["predicted_fire_risk_band"] = add_risk_band_column(preds)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
