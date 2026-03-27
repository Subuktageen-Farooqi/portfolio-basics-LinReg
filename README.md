# fire-risk-predictor

A lightweight PyTorch portfolio project for tabular regression on wildfire-related operational data.  
The model estimates a continuous `fire_risk_score` that can be used in prioritization and early-warning workflows. The implementation is script-first, reproducible, and intentionally production-aware without unnecessary complexity.

## Problem statement
Emergency response teams need a practical way to prioritize areas where fire risk is rising. This project trains a regression model that predicts fire risk from weather, vegetation, logistics, and air-quality signals.

## Dataset and features
Training data is stored in `data/fire_risk_dataset.csv`.

### Input features
- `temperature_c`
- `humidity_percent`
- `wind_speed_kmh`
- `air_quality_index`
- `vegetation_index`
- `distance_to_station_km`
- `response_time_min`

### Target
- `fire_risk_score`

> Note: If present, `fire_risk_level` is **not** used as a training feature (to avoid leakage). It is used only for optional derived reporting.

## Model approach
- Framework: **PyTorch** end-to-end for splitting, preprocessing, model training, and metrics.
- Architecture: small MLP (2 hidden layers + ReLU + light dropout).
- Optimizer / loss: Adam + MSELoss.
- Data flow:
  1. Deterministic seed setup.
  2. Train/val/test split with torch-based shuffled indices.
  3. Standardization fitted on **train only** and reused at inference.
  4. Best model checkpointed to `best_model.pt` using validation RMSE.
  5. Final MAE / RMSE / R² computed on held-out test set.

## Repo structure
```text
fire-risk-predictor/
├── README.md
├── requirements.txt
├── best_model.pt
├── data/
│   ├── fire_risk_dataset.csv
│   └── sample_inference.csv
├── outputs/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── scaler.json
│   ├── split_info.json
│   ├── loss_curve.png
│   ├── pred_vs_actual.png
│   └── residuals.png
└── src/
    ├── model.py
    ├── train.py
    ├── predict.py
    └── utils.py
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train the model
```bash
python src/train.py
```

Optional training args:
```bash
python src/train.py --epochs 120 --batch-size 64 --learning-rate 1e-3
```

## Run batch prediction on new CSV
```bash
python src/predict.py --input-csv data/sample_inference.csv --add-risk-band
```

Predictions are saved to `outputs/predictions.csv`.

## Evaluation metrics
Current run (held-out test split):
- **MAE:** 4.565
- **RMSE:** 5.706
- **R²:** 0.818

Machine-readable metrics are saved in `outputs/metrics.json`.

## Training and evaluation plots
### Loss curve
![Loss curve](outputs/loss_curve.png)

### Predicted vs actual
![Predicted vs actual](outputs/pred_vs_actual.png)

### Residual distribution
![Residuals](outputs/residuals.png)

## Business relevance
A calibrated fire-risk score can help dispatch teams prioritize patrols, preventive actions, and resource allocation. In an operations setting, this type of model can complement rule-based alerts by turning multiple weak signals into a single sortable risk indicator.

## Limitations
- Current dataset is compact and intended for portfolio demonstration.
- No external geospatial/temporal context yet (seasonality, terrain, historical incidents).
- Further validation is needed before real operational deployment.
