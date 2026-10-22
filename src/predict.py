"""
predict.py
----------
1. Load trained model + fit_objects saved by train.py
2. Predict posted_rate for all 12,000 loads in validation.csv
   → saves validation_predictions.csv
3. Predict daily rates for december-chart-inputs.csv (required by score.py)
   → fills predicted_rate column in-place
4. Verify the submission
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from data_prep import load_validation, load_template, build_features, ROOT, FEATURE_NAMES

MODEL_PATH       = ROOT / "src" / "model.pkl"
FIT_OBJECTS_PATH = ROOT / "src" / "fit_objects.pkl"
SUBMISSION_PATH  = ROOT / "validation_predictions.csv"
DECEMBER_CSV     = ROOT / "data" / "december-chart-inputs.csv"

# STEP 1 — load model
print("\n" + "="*60)
print("STEP 1 — LOAD MODEL")
print("="*60)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(FIT_OBJECTS_PATH, "rb") as f:
    fit_objects = pickle.load(f)

print(f"  Model      : {type(model).__name__}")
print(f"  Medians    : {fit_objects['medians']}")
print(f"  Equip map  : {fit_objects['equip_map']}")

# STEP 2 — load & preprocess validation data

print("\n" + "="*60)
print("STEP 2 — LOAD & PREPROCESS VALIDATION DATA")
print("="*60)

df_val   = load_validation(verbose=True)
template = load_template()

print(f"\n  Template rows : {len(template):,}")

X_val, feat_names, _ = build_features(df_val, fit_objects=fit_objects)
print(f"  X_val shape  : {X_val.shape}")
print(f"  Features     : {feat_names}")

# STEP 3 — predict

print("\n" + "="*60)
print("STEP 3 — PREDICT")
print("="*60)

preds = model.predict(X_val)

print(f"  Predictions  : {len(preds):,}")
print(f"  Min          : {preds.min():.2f}")
print(f"  Max          : {preds.max():.2f}")
print(f"  Mean         : {preds.mean():.2f}")
print(f"  Median       : {np.median(preds):.2f}")
print(f"  Non-positive : {(preds <= 0).sum()}")

# Safety clip — model should never predict <= 0 but guard anyway
preds = np.clip(preds, a_min=1.0, a_max=None)

# STEP 4 — build submission
print("\n" + "="*60)
print("STEP 4 — BUILD SUBMISSION")
print("="*60)

# Map load_id → prediction
pred_map = dict(zip(df_val["load_id"].values, preds))

submission = template[["load_id"]].copy()
submission["predicted_rate"] = submission["load_id"].map(pred_map)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"  Saved → {SUBMISSION_PATH}")

# STEP 5 — verify
print("\n" + "="*60)
print("STEP 5 — VERIFY SUBMISSION")
print("="*60)

sub = pd.read_csv(SUBMISSION_PATH)

checks = {
    "Row count == 12,000"                   : len(sub) == 12_000,
    "Columns are [load_id, predicted_rate]" : list(sub.columns) == ["load_id", "predicted_rate"],
    "No duplicate load_ids"                 : sub["load_id"].duplicated().sum() == 0,
    "No missing predictions"                : sub["predicted_rate"].isnull().sum() == 0,
    "No infinite predictions"               : not np.isinf(sub["predicted_rate"].values).any(),
    "All predictions > 0"                   : (sub["predicted_rate"] > 0).all(),
    "IDs match template exactly"            : set(template["load_id"]) == set(sub["load_id"]),
    "Row order matches template"            : list(sub["load_id"]) == list(template["load_id"]),
}

all_pass = True
for desc, result in checks.items():
    icon = "✓" if result else "✗"
    if not result:
        all_pass = False
    print(f"  {icon}  {desc}")

print()
if all_pass:
    print("  ✓ All checks passed — submission is ready.")
else:
    print("  ✗ One or more checks FAILED — review above.")

print("\nPrediction distribution:")
print(sub["predicted_rate"].describe().round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — FILL DECEMBER CHART INPUTS  (required by score.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART 2 — FILL DECEMBER CHART INPUTS")
print("="*60)

df_dec = pd.read_csv(DECEMBER_CSV, parse_dates=["date"])
print(f"  Rows : {len(df_dec)}  ({df_dec['date'].min().date()} → {df_dec['date'].max().date()})")

# Build features manually — december file has no market_index / quote_signal,
# so use training medians (same logic as missing-value imputation)
medians   = fit_objects["medians"]
equip_map = fit_objects["equip_map"]

df_dec["weight"]       = df_dec["weight"].fillna(medians["weight"])
df_dec["market_index"] = medians["market_index"]
df_dec["quote_signal"] = medians["quote_signal"]
df_dec["month"]        = df_dec["date"].dt.month
df_dec["day_of_week"]  = df_dec["date"].dt.dayofweek
df_dec["log_distance"] = np.log1p(df_dec["distance"])
df_dec["rate_signal"]  = df_dec["market_index"] * df_dec["quote_signal"]
df_dec["dist_x_signal"]= df_dec["log_distance"] * df_dec["quote_signal"]
df_dec["equipment_code"] = df_dec["equipment"].map(equip_map).fillna(-1).astype(int)

X_dec = df_dec[FEATURE_NAMES].values.astype(np.float64)
dec_preds = np.clip(model.predict(X_dec), a_min=1.0, a_max=None)

print(f"  Predictions  min=${dec_preds.min():.2f}  max=${dec_preds.max():.2f}  mean=${dec_preds.mean():.2f}")

# Write back — preserve exact 7-column format score.py expects
df_out = pd.read_csv(DECEMBER_CSV)
df_out["predicted_rate"] = dec_preds
df_out.to_csv(DECEMBER_CSV, index=False)
print(f"  Saved → {DECEMBER_CSV}")
