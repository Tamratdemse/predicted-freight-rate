"""
data_prep.py
------------
Loads raw CSVs, inspects them, cleans, engineers features, and
returns ready-to-use numpy arrays / DataFrames.

Public API
----------
load_development()   -> pd.DataFrame   (cleaned train-test data)
load_validation()    -> pd.DataFrame   (cleaned validation data)
split_chronological(df, cut_date) -> (df_train, df_val)
build_features(df, fit_objects=None) -> (X, feature_names, fit_objects)
  fit_objects is a dict produced on the training partition and reused on
  any later partition to prevent leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TRAIN_CSV    = DATA / "train-test.csv"
VAL_CSV      = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation-predictions-template.csv"

TARGET = "posted_rate"

# Equipment ordinal order (Dry Van cheapest → Reefer → Flatbed premium)
EQUIP_ORDER = ["Dry Van", "Reefer", "Flatbed"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_and_parse(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _report(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  —  {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"{'='*60}")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    mv = df.isnull().sum()
    print(mv[mv > 0].to_string() if mv.any() else "  none")
    print(f"\nDuplicate rows : {df.duplicated().sum()}")
    if "date" in df.columns:
        print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    if TARGET in df.columns:
        print(f"\nTarget ({TARGET}) distribution:")
        print(df[TARGET].describe().round(2).to_string())
        neg = (df[TARGET] <= 0).sum()
        print(f"  Non-positive  : {neg}")
    if "distance" in df.columns:
        print(f"\nDistance stats:")
        print(df["distance"].describe().round(2).to_string())
        print(f"  Non-positive  : {(df['distance'] <= 0).sum()}")
    if "equipment" in df.columns:
        print(f"\nEquipment counts:")
        print(df["equipment"].value_counts().to_string())
    if TARGET in df.columns:
        num = ["distance", "weight", "market_index", "quote_signal"]
        print(f"\nCorrelation with {TARGET}:")
        print(df[num + [TARGET]].corr()[TARGET].drop(TARGET).round(4).to_string())


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop exact duplicate rows
    - Cap extreme outliers in distance and weight using IQR (only on dev data
      that has a target; validation is left as-is so we never discard a load)
    """
    df = df.drop_duplicates().copy()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_development(verbose: bool = True) -> pd.DataFrame:
    df = _read_and_parse(TRAIN_CSV)
    if verbose:
        _report(df, "Development data (train-test.csv)")
    df = _clean(df)
    return df


def load_validation(verbose: bool = True) -> pd.DataFrame:
    df = _read_and_parse(VAL_CSV)
    if verbose:
        _report(df, "Validation data (validation.csv)")
    return df


def load_template() -> pd.DataFrame:
    return pd.read_csv(TEMPLATE_CSV)


# ─────────────────────────────────────────────────────────────────────────────
# Chronological split
# ─────────────────────────────────────────────────────────────────────────────

def split_chronological(df: pd.DataFrame,
                        cut_date: str = "2025-08-31") -> tuple:
    """
    Split df on date.
      train : date <= cut_date
      val   : date >  cut_date
    Returns (df_train, df_val).
    """
    mask = df["date"] <= pd.Timestamp(cut_date)
    df_train = df[mask].copy()
    df_val   = df[~mask].copy()
    print(f"\nChronological split at {cut_date}:")
    print(f"  Train : {len(df_train):,} rows  "
          f"({df_train['date'].min().date()} → {df_train['date'].max().date()})")
    print(f"  Val   : {len(df_val):,} rows  "
          f"({df_val['date'].min().date()} → {df_val['date'].max().date()})")
    return df_train, df_val


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering + preprocessing
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "distance",
    "log_distance",
    "weight",
    "month",
    "day_of_week",
    "market_index",
    "quote_signal",
    "equipment_code",
    # interaction
    "rate_signal",       # market_index * quote_signal
    "dist_x_signal",     # log_distance * quote_signal
]


def build_features(df: pd.DataFrame,
                   fit_objects: dict | None = None
                   ) -> tuple[np.ndarray, list[str], dict]:
    """
    Parameters
    ----------
    df          : raw DataFrame (may or may not have posted_rate)
    fit_objects : dict returned from a previous call on training data.
                  Pass None on training; pass the returned dict on val/test
                  to prevent leakage.

    Returns
    -------
    X            : np.ndarray  shape (n, len(FEATURE_NAMES))
    feature_names: list[str]
    fit_objects  : dict  (medians, equipment_map)
    """
    df = df.copy()

    # ── 1. Impute missing numerics with TRAINING medians ──────────────────
    num_cols = ["distance", "weight", "market_index", "quote_signal"]
    if fit_objects is None:
        medians = {c: df[c].median() for c in num_cols}
    else:
        medians = fit_objects["medians"]

    for col in num_cols:
        df[col] = df[col].fillna(medians[col])

    # ── 2. Equipment encoding ─────────────────────────────────────────────
    if fit_objects is None:
        seen = df["equipment"].dropna().unique().tolist()
        # put known order first, then any unseen values
        ordered = [e for e in EQUIP_ORDER if e in seen] + \
                  [e for e in seen if e not in EQUIP_ORDER]
        equip_map = {e: i for i, e in enumerate(ordered)}
    else:
        equip_map = fit_objects["equip_map"]

    df["equipment_code"] = df["equipment"].map(equip_map).fillna(-1).astype(int)

    # ── 3. Date features ──────────────────────────────────────────────────
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon … 6=Sun

    # ── 4. Engineered features ────────────────────────────────────────────
    df["log_distance"]  = np.log1p(df["distance"])
    df["rate_signal"]   = df["market_index"] * df["quote_signal"]
    df["dist_x_signal"] = df["log_distance"] * df["quote_signal"]

    fit_out = {"medians": medians, "equip_map": equip_map}

    X = df[FEATURE_NAMES].values.astype(np.float64)
    return X, FEATURE_NAMES, fit_out
