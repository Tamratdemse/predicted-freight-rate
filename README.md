# Freight Rate Prediction

Predicts trucking freight rates from load features using a Gradient Boosting model.

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Run

```bash
# 1. Train the model
python3 src/train.py

# 2. Generate predictions
python3 src/predict.py

# 3. Validate outputs and generate December chart
python3 score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december-chart-inputs.csv \
  --output-dir scorer_results
```

## Files

| File | Purpose |
|---|---|
| `src/data_prep.py` | Data loading, cleaning, feature engineering |
| `src/train.py` | Model training and selection |
| `src/predict.py` | Inference — generates `validation_predictions.csv` and fills December chart |
| `src/evaluate.py` | Accuracy diagnostics (optional) |
| `score.py` | Official validator — checks submission format and generates chart |

## Outputs

| File | Description |
|---|---|
| `validation_predictions.csv` | 12,000 predicted freight rates (main deliverable) |
| `data/december-chart-inputs.csv` | Daily predictions for fixed Dec 2025 lane |
| `scorer_results/candidate_december.png` | December rate chart |

## Model

- **Algorithm:** GradientBoostingRegressor (500 trees, depth 5, lr 0.05)
- **Training data:** `data/train-test.csv` — Jan–Aug 2025 (38,477 rows)
- **Holdout validation:** Sep–Oct 2025 (9,523 rows) — MAE: $119.69, MAPE: 5.22%
- **Key feature:** Distance (correlation 0.91 with rate)
