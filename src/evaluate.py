"""
evaluate.py
-----------
Deep accuracy diagnostic:
  - Train-set fit  (R², MAE, RMSE, MAPE)
  - Holdout fit    (Sep–Oct)
  - Residual analysis
  - Feature importance
  - Error breakdown by equipment, distance bucket, month
  - Learning curve (how MAE improves with more data)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from data_prep import (
    load_development, split_chronological, build_features, TARGET, ROOT, FEATURE_NAMES
)

MODEL_PATH       = ROOT / "src" / "model.pkl"
FIT_OBJECTS_PATH = ROOT / "src" / "fit_objects.pkl"


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics(label, y_true, y_pred):
    print(f"\n  [{label}]")
    print(f"    R²   : {r2_score(y_true, y_pred):.4f}")
    print(f"    MAE  : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"    RMSE : {rmse(y_true, y_pred):.2f}")
    print(f"    MAPE : {mape(y_true, y_pred):.2f}%")


# ── load data & model ─────────────────────────────────────────────────────────
df_dev = load_development(verbose=False)
df_tr, df_ev = split_chronological(df_dev, cut_date="2025-08-31")

X_tr, _, fit_obj = build_features(df_tr)
X_ev, _, _       = build_features(df_ev, fit_objects=fit_obj)
y_tr = df_tr[TARGET].values
y_ev = df_ev[TARGET].values

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(FIT_OBJECTS_PATH, "rb") as f:
    fit_obj_full = pickle.load(f)

X_full, _, _ = build_features(df_dev, fit_objects=fit_obj_full)
y_full = df_dev[TARGET].values

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1 — TRAIN / HOLDOUT FIT")
print("="*60)

# Refit model on train partition for fair train-set reporting
from sklearn.ensemble import GradientBoostingRegressor
m_split = GradientBoostingRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, loss="absolute_error",
    random_state=42, verbose=0
)
m_split.fit(X_tr, y_tr)

print_metrics("Train partition  (Jan–Aug)", y_tr, m_split.predict(X_tr))
print_metrics("Holdout partition (Sep–Oct)", y_ev, m_split.predict(X_ev))
print_metrics("Full dev set (final model)", y_full, model.predict(X_full))

# Overfitting gap
train_mae = mean_absolute_error(y_tr, m_split.predict(X_tr))
holdout_mae = mean_absolute_error(y_ev, m_split.predict(X_ev))
gap_pct = (holdout_mae - train_mae) / train_mae * 100
print(f"\n  Overfit gap : {gap_pct:.1f}%  (train MAE → holdout MAE)")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2 — RESIDUAL ANALYSIS  (holdout)")
print("="*60)

pred_ev = m_split.predict(X_ev)
resid   = y_ev - pred_ev

print(f"  Mean residual        : {resid.mean():.2f}  (bias — should be ~0)")
print(f"  Std  residual        : {resid.std():.2f}")
print(f"  |Residual| > 500     : {(np.abs(resid) > 500).sum()} / {len(resid)}")
print(f"  |Residual| > 1000    : {(np.abs(resid) > 1000).sum()} / {len(resid)}")
print(f"  |Residual| > 2000    : {(np.abs(resid) > 2000).sum()} / {len(resid)}")

# % of predictions within X% of true value
for pct in [5, 10, 15, 20]:
    within = np.mean(np.abs(resid / y_ev) < pct/100) * 100
    print(f"  Within {pct:2d}% of true    : {within:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3 — FEATURE IMPORTANCE")
print("="*60)

importances = m_split.feature_importances_
order = np.argsort(importances)[::-1]
for rank, idx in enumerate(order, 1):
    bar = "█" * int(importances[idx] * 60)
    print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<18}  {importances[idx]:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4 — ERROR BY DISTANCE BUCKET")
print("="*60)

df_ev2 = df_ev.copy()
df_ev2["pred"]  = pred_ev
df_ev2["resid"] = resid
df_ev2["abs_resid"] = np.abs(resid)

bins   = [0, 300, 600, 1000, 1500, 2000, 9999]
labels = ["<300", "300-600", "600-1k", "1k-1.5k", "1.5k-2k", ">2k"]
df_ev2["dist_bucket"] = pd.cut(df_ev2["distance"], bins=bins, labels=labels)

grp = df_ev2.groupby("dist_bucket", observed=True).agg(
    n=("posted_rate", "count"),
    mae=("abs_resid", "mean"),
    mape_=("resid", lambda x: (np.abs(x / df_ev2.loc[x.index, "posted_rate"])).mean() * 100),
).round(2)
print(grp.to_string())


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5 — ERROR BY EQUIPMENT TYPE")
print("="*60)

grp_eq = df_ev2.groupby("equipment").agg(
    n=("posted_rate", "count"),
    mae=("abs_resid", "mean"),
    mape_=("resid", lambda x: (np.abs(x / df_ev2.loc[x.index, "posted_rate"])).mean() * 100),
).round(2)
print(grp_eq.to_string())


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6 — ERROR BY MONTH (holdout = Sep & Oct)")
print("="*60)

df_ev2["month"] = df_ev2["date"].dt.month
grp_m = df_ev2.groupby("month").agg(
    n=("posted_rate", "count"),
    mae=("abs_resid", "mean"),
    mape_=("resid", lambda x: (np.abs(x / df_ev2.loc[x.index, "posted_rate"])).mean() * 100),
).round(2)
print(grp_m.to_string())


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7 — LEARNING CURVE  (chronological slices)")
print("="*60)

slices = [
    ("Jan–Feb",    "2025-02-28"),
    ("Jan–Apr",    "2025-04-30"),
    ("Jan–Jun",    "2025-06-30"),
    ("Jan–Aug",    "2025-08-31"),
]

print(f"  {'Slice':<12}  {'#Train':>7}  {'Train MAE':>10}  {'Holdout MAE':>12}  {'R² holdout':>11}")
for label, cut in slices:
    df_s, _ = split_chronological(df_dev, cut_date=cut)
    Xs, _, fo = build_features(df_s)
    Xe, _, _  = build_features(df_ev, fit_objects=fo)
    ys = df_s[TARGET].values
    m = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, loss="absolute_error",
        random_state=42, verbose=0
    )
    m.fit(Xs, ys)
    tr_mae  = mean_absolute_error(ys, m.predict(Xs))
    ho_mae  = mean_absolute_error(y_ev, m.predict(Xe))
    ho_r2   = r2_score(y_ev, m.predict(Xe))
    print(f"  {label:<12}  {len(Xs):>7,}  {tr_mae:>10.2f}  {ho_mae:>12.2f}  {ho_r2:>11.4f}")

print("\nDone.")
