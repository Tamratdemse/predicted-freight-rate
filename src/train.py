"""
train.py
--------
1. Load & inspect development data via data_prep
2. Chronological split  (Jan–Aug train / Sep–Oct val)
3. Preprocess — fit only on the training partition
4. Train all candidate models on the training partition
5. Evaluate each on the holdout partition (Sep–Oct)
6. Select the best model (lowest MAE)
7. Save the selected model + its fit_objects to disk
   (same model that was evaluated — no second training phase)
"""

import pickle
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import (
    load_development,
    split_chronological,
    build_features,
    TARGET,
    ROOT,
)

MODEL_PATH       = ROOT / "src" / "model.pkl"
FIT_OBJECTS_PATH = ROOT / "src" / "fit_objects.pkl"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_and_evaluate(name, model, X_tr, y_tr, X_ev, y_ev):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ev)
    mae  = mean_absolute_error(y_ev, pred)
    rms  = rmse(y_ev, pred)
    print(f"  {name:<40}  MAE={mae:>8.2f}   RMSE={rms:>8.2f}")
    return mae, rms, model   # return the FITTED model


# ── STEP 1 — load & inspect ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — LOAD & INSPECT DEVELOPMENT DATA")
print("="*60)

df_dev = load_development(verbose=True)

# ── STEP 2 — chronological split ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — CHRONOLOGICAL SPLIT  (Jan–Aug train / Sep–Oct val)")
print("="*60)

df_tr, df_ev = split_chronological(df_dev, cut_date="2025-08-31")

# ── STEP 3 — preprocess (fit ONLY on training partition) ──────────────────
print("\n" + "="*60)
print("STEP 3 — PREPROCESS & FEATURE ENGINEERING")
print("="*60)

X_tr, feat_names, fit_obj = build_features(df_tr, fit_objects=None)
X_ev, _,          _       = build_features(df_ev, fit_objects=fit_obj)

y_tr = df_tr[TARGET].values
y_ev = df_ev[TARGET].values

print(f"  Features : {feat_names}")
print(f"  X_train  : {X_tr.shape}")
print(f"  X_eval   : {X_ev.shape}")

# ── STEP 4 & 5 — train on split, evaluate on holdout ─────────────────────
print("\n" + "="*60)
print("STEP 4 — TRAIN & EVALUATE ON SPLIT  (Sep–Oct holdout)")
print("="*60)

candidates = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=10)":  Ridge(alpha=10),
    "GradientBoosting(500 trees, depth 5, lr 0.05)": GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        loss="absolute_error",
        random_state=42,
        verbose=0,
    ),
}

results = {}
for name, model in candidates.items():
    mae, rms, fitted_model = train_and_evaluate(
        name, model, X_tr, y_tr, X_ev, y_ev
    )
    results[name] = {"mae": mae, "rmse": rms, "model": fitted_model}

# ── STEP 6 — select best model ────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["mae"])
best      = results[best_name]

print(f"\n  ✓ Selected model : {best_name}")
print(f"    Trained on     : Jan–Aug 2025  ({len(y_tr):,} rows)")
print(f"    Holdout MAE    : {best['mae']:.2f}")
print(f"    Holdout RMSE   : {best['rmse']:.2f}")
print(f"\n  This is the model that will predict validation.csv.")
print(f"  No second training phase — the split-trained model is used directly.")

# ── STEP 7 — save selected model + preprocessing state ───────────────────
print("\n" + "="*60)
print("STEP 5 — SAVE MODEL")
print("="*60)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best["model"], f)
with open(FIT_OBJECTS_PATH, "wb") as f:
    pickle.dump(fit_obj, f)

print(f"  Model saved       → {MODEL_PATH}")
print(f"  Fit objects saved → {FIT_OBJECTS_PATH}")
print(f"  Preprocessing fit on Jan–Aug training partition only.")
print("\nDone — run predict.py to generate submission.")
