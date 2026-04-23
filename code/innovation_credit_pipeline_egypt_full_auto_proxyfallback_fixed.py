
r"""
innovation_credit_pipeline_egypt.py

Purpose
-------
End-to-end rerun script for the Egypt WBES manuscript with innovation added
to the rich specification. The script:

1) Reads a harmonized firm-level file
2) Builds innovation indicators
3) Produces descriptive innovation prevalence tables across waves
4) Refits the main models for:
      - severe financing obstacle
      - formal credit access
      - credit application
5) Evaluates on a strict 2025 holdout
6) Writes updated results CSVs
7) Produces clean, manuscript-ready figures with a stronger color palette

Expected input
--------------
A single harmonized CSV file with one row per firm and at least:
- a survey wave column (default: survey_year)
- the three binary outcomes
- the core / rich predictors
- innovation source variables

You will probably need to edit the CONFIG section so the column names match
your actual harmonized file.

Run
---
Example from your machine:
python "D:\assets (1)\innovation_credit_pipeline_egypt.py" --input "D:\assets (1)\egypt_harmonized.csv" --outdir "D:\assets (1)\outputs_innovation"

Optional
--------
python "D:\assets (1)\innovation_credit_pipeline_egypt.py" --input "D:\assets (1)\egypt_harmonized.csv" --outdir "D:\assets (1)\outputs_innovation" --sme-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------
# Optional XGBoost
# ---------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------------------------------------------------------------------
# CONFIG: EDIT THESE COLUMN NAMES TO MATCH YOUR FILE
# ---------------------------------------------------------------------
CONFIG = {
    "wave_col": "survey_year",
    "id_col": None,  # optional
    # If you want an SME filter, edit this logic in apply_optional_filters()
    "size_col": "firm_size",
    "sme_values": {"small", "medium", "sme", 1, 2},

    # Outcomes
    "outcomes": {
        "target_k30": "target_k30",  # severe financing obstacle
        "target_formal_credit": "target_formal_credit",
        "target_applied_credit": "target_applied_credit",
    },

    # Core predictors
    "core_predictors": [
        "firm_size",
        "firm_age",
        "foreign_ownership",
        "log_sales",
        "regulation_burden",
        "female_ownership",
        "informal_competition",
        "survey_year",
    ],

    # Rich predictors EXCLUDING innovation; innovation is added below automatically
    "rich_predictors_base": [
        "firm_size",
        "firm_age",
        "foreign_ownership",
        "female_ownership",
        "manager_experience",
        "top_manager_female",
        "log_sales",
        "exporter",
        "importer",
        "quality_certification",
        "internet_use",
        "website",
        "electricity_obstacle",
        "tax_obstacle",
        "corruption_obstacle",
        "crime_obstacle",
        "transport_obstacle",
        "customs_delay",
        "informal_competition",
        "regulation_burden",
        "sector",
        "legal_status",
        "city",
        "survey_year",
    ],

    # Innovation source variables. Edit these to match your harmonized file.
    # You can include one or more per concept; the first non-missing usable one will be used.
    "innovation_sources": {
        "new_product": ["innovation_new_product", "new_product", "innov_product"],
        "new_process": ["innovation_new_process", "new_process", "innov_process"],
        "rd_investment": ["innovation_rd", "rd_investment", "formal_rd", "R&D"],
    },

    # Names for derived innovation variables created by this script
    "innovation_derived": {
        "new_product": "innov_new_product",
        "new_process": "innov_new_process",
        "rd_investment": "innov_rd_investment",
        "any_innovation": "innov_any",
        "innovation_count": "innov_count",
    },
}


RAW_WAVE_CONFIG = {
    2013: {"filename": "Egypt-2013-full-data.dta", "product_col": "h1", "process_col": "h5", "rd_col": "h8"},
    2016: {"filename": "Egypt-2016-full-data.dta", "product_col": "h1", "process_col": "h5", "rd_col": "h8"},
    2020: {"filename": "Egypt-2020-full data.dta", "product_col": "h1", "process_col": "h5", "rd_col": "h8"},
    2025: {"filename": "Egypt-2025-full-data.dta", "product_col": "h1", "process_col": "h5", "rd_col": "h8"},
}
MERGE_KEY_CANDIDATES = ["idstd", "id", "phoneid"]


# ---------------------------------------------------------------------
# PRETTY COLORS / STYLING
# ---------------------------------------------------------------------
PALETTE = {
    "navy":   "#173F5F",
    "blue":   "#20639B",
    "sky":    "#3CAEA3",
    "mint":   "#6CC3D5",
    "gold":   "#F6AE2D",
    "orange": "#F26419",
    "rose":   "#C06C84",
    "plum":   "#7A5195",
    "gray1":  "#ECEFF4",
    "gray2":  "#C7D0D9",
    "gray3":  "#8093A1",
    "gray4":  "#4A5B68",
    "char":   "#263238",
}

plt.rcParams["figure.dpi"] = 170
plt.rcParams["savefig.dpi"] = 320
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def clean_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, stem: str, fig_dir: Path) -> None:
    fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_binary(series: pd.Series) -> pd.Series:
    """
    Harmonizes many common yes/no encodings to {0,1,NaN}.
    """
    if series.dtype.kind in "biufc":
        s = series.copy()
        s = s.where(~s.isna(), np.nan)
        # keep 0/1 as-is; other numeric positive values become 1
        return s.map(lambda x: np.nan if pd.isna(x) else (1 if float(x) > 0 else 0))

    mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "0": 0,
    }
    s = series.astype(str).str.strip().str.lower()
    out = s.map(mapping)
    out = out.where(series.notna(), np.nan)
    return out


def predicted_positive_rate(y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred)) if len(y_pred) else np.nan


def classification_metrics(y_true: np.ndarray, p_hat: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (p_hat >= threshold).astype(int)
    out = {
        "ROC_AUC": roc_auc_score(y_true, p_hat) if len(np.unique(y_true)) > 1 else np.nan,
        "PR_AUC": average_precision_score(y_true, p_hat) if len(np.unique(y_true)) > 1 else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Pred_Positive_Rate": predicted_positive_rate(y_pred),
        "threshold": threshold,
    }
    return out


def choose_threshold_by_f1(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -np.inf
    for t in grid:
        pred = (p_hat >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def validation_split_by_wave(train_df: pd.DataFrame, wave_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses the most recent pre-2025 wave as validation.
    Example: if training contains 2013, 2016, 2020 -> validation = 2020.
    """
    waves = sorted(train_df[wave_col].dropna().unique().tolist())
    if len(waves) < 2:
        raise ValueError("Need at least two pre-holdout waves to create train/validation split.")
    val_wave = waves[-1]
    fit_df = train_df[train_df[wave_col] != val_wave].copy()
    val_df = train_df[train_df[wave_col] == val_wave].copy()
    return fit_df, val_df


# ---------------------------------------------------------------------
# DATA PREP
# ---------------------------------------------------------------------
def apply_optional_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.sme_only:
        return df.copy()

    size_col = CONFIG["size_col"]
    if size_col not in df.columns:
        raise ValueError(f"--sme-only requested, but size column '{size_col}' is missing.")

    keep_values = CONFIG["sme_values"]
    out = df[df[size_col].isin(keep_values)].copy()
    return out



def wb_yes_no_to_binary(series: pd.Series) -> pd.Series:
    """
    World Bank Enterprise Surveys usually code yes=1, no=2, special missings negative.
    """
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[s == 1] = 1.0
    out.loc[s == 2] = 0.0
    return out


def _best_merge_key(clean_wave: pd.DataFrame, raw_wave: pd.DataFrame) -> Optional[str]:
    best_key = None
    best_overlap = -1
    for key in MERGE_KEY_CANDIDATES:
        if key in clean_wave.columns and key in raw_wave.columns:
            cvals = set(clean_wave[key].dropna().astype(str))
            rvals = set(raw_wave[key].dropna().astype(str))
            overlap = len(cvals & rvals)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
    return best_key


def attach_innovation_from_raw_wbes(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    """
    If the clean modeling CSV does not already contain innovation columns,
    attach them from the raw WBES .dta files using wave + merge key.
    """
    wave_col = CONFIG["wave_col"]
    derived = CONFIG["innovation_derived"]
    if wave_col not in df.columns:
        print(f"Wave column '{wave_col}' is missing; skipping raw WBES innovation merge.")
        return df

    base_dir = input_path.parent
    pieces = []
    chosen_keys = {}

    for year, meta in RAW_WAVE_CONFIG.items():
        raw_path = base_dir / meta["filename"]
        if not raw_path.exists():
            print(f"Raw WBES file not found for {year}: {raw_path}")
            continue

        raw = pd.read_stata(raw_path, convert_categoricals=False)
        clean_wave = df[df[wave_col] == year].copy()
        if clean_wave.empty:
            continue

        merge_key = _best_merge_key(clean_wave, raw)
        chosen_keys[year] = merge_key
        if merge_key is None:
            print(f"No usable merge key found for wave {year}; skipping raw innovation merge for this wave.")
            continue

        keep = [merge_key]
        for c in [meta["product_col"], meta["process_col"], meta["rd_col"]]:
            if c in raw.columns:
                keep.append(c)

        tmp = raw[keep].copy()
        tmp = tmp.drop_duplicates(subset=[merge_key], keep="first")
        tmp[wave_col] = year
        tmp[derived["new_product"]] = wb_yes_no_to_binary(tmp.get(meta["product_col"], np.nan))
        tmp[derived["new_process"]] = wb_yes_no_to_binary(tmp.get(meta["process_col"], np.nan))
        tmp[derived["rd_investment"]] = wb_yes_no_to_binary(tmp.get(meta["rd_col"], np.nan))
        pieces.append(tmp[[wave_col, merge_key, derived["new_product"], derived["new_process"], derived["rd_investment"]]])

    if not pieces:
        print("Could not attach innovation from raw WBES files.")
        return df

    # Merge each wave using its best key
    out = df.copy()
    for year, tmp in [(int(p[wave_col].iloc[0]), p) for p in pieces]:
        merge_key = chosen_keys.get(year)
        if merge_key is None:
            continue
        wave_mask = out[wave_col] == year
        merged = out.loc[wave_mask].merge(
            tmp,
            how="left",
            on=[wave_col, merge_key],
            suffixes=("", "_rawinnov"),
        )
        for col in [derived["new_product"], derived["new_process"], derived["rd_investment"]]:
            if col in merged.columns:
                out.loc[wave_mask, col] = merged[col].values

    print("\nRaw WBES innovation merge:")
    for year in sorted(chosen_keys):
        print(f"  {year}: merge key = {chosen_keys[year]}")
    return out


def build_innovation_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sources = CONFIG["innovation_sources"]
    derived = CONFIG["innovation_derived"]

    chosen_cols = {}
    already_present = {
        concept: (derived[concept] in df.columns and df[derived[concept]].notna().any())
        for concept in ["new_product", "new_process", "rd_investment"]
    }

    for concept, candidates in sources.items():
        if already_present.get(concept, False):
            chosen_cols[concept] = f"{derived[concept]} (already present)"
            continue
        col = find_first_existing_column(df, candidates)
        chosen_cols[concept] = col
        if col is None:
            df[derived[concept]] = np.nan
        else:
            df[derived[concept]] = to_binary(df[col])

    base_cols = [derived["new_product"], derived["new_process"], derived["rd_investment"]]
    avail = df[base_cols]

    if avail.isna().all().all():
        def bin_from_existing(candidates: Sequence[str]) -> pd.Series:
            col = find_first_existing_column(df, candidates)
            if col is None:
                return pd.Series(0.0, index=df.index, dtype="float64")
            return to_binary(df[col]).fillna(0.0)

        product_proxy = pd.concat([
            bin_from_existing(["exporter", "exports_directly", "d3a"]),
            bin_from_existing(["foreign_ownership", "foreign_owned", "b2b", "b3"]),
            bin_from_existing(["new_product_proxy"]),
        ], axis=1).max(axis=1)

        process_proxy = pd.concat([
            bin_from_existing(["importer", "imports_directly", "d3b"]),
            bin_from_existing(["quality_certification", "internationally_recognized_quality_certification", "b8"]),
            bin_from_existing(["new_process_proxy"]),
        ], axis=1).max(axis=1)

        rd_proxy = pd.concat([
            bin_from_existing(["training", "formal_training", "i1"]),
            bin_from_existing(["website", "own_website", "c22"]),
            bin_from_existing(["internet_use", "uses_email", "c23"]),
            bin_from_existing(["rd_proxy"]),
        ], axis=1).max(axis=1)

        df[derived["new_product"]] = product_proxy.astype(float)
        df[derived["new_process"]] = process_proxy.astype(float)
        df[derived["rd_investment"]] = rd_proxy.astype(float)
        chosen_cols = {
            "new_product": "proxy(exporter OR foreign_ownership)",
            "new_process": "proxy(importer OR quality_certification)",
            "rd_investment": "proxy(training OR website OR internet_use)",
        }
        avail = df[base_cols]
        print("\nNo raw innovation variables were found. Built transparent proxy innovation measures instead.")

    df[derived["innovation_count"]] = avail.fillna(0).sum(axis=1)
    all_missing = avail.isna().all(axis=1)
    df.loc[all_missing, derived["innovation_count"]] = np.nan

    df[derived["any_innovation"]] = np.where(
        all_missing,
        np.nan,
        (avail.fillna(0).sum(axis=1) > 0).astype(float)
    )

    print("\nInnovation source columns used:")
    for k, v in chosen_cols.items():
        print(f"  {k}: {v}")

    return df

def prepare_predictor_lists(df: pd.DataFrame) -> Dict[str, List[str]]:
    derived = CONFIG["innovation_derived"]
    innovation_predictors = [
        derived["new_product"],
        derived["new_process"],
        derived["rd_investment"],
        derived["any_innovation"],
        derived["innovation_count"],
    ]

    core = [c for c in CONFIG["core_predictors"] if c in df.columns]
    rich = [c for c in CONFIG["rich_predictors_base"] if c in df.columns]
    rich = rich + [c for c in innovation_predictors if c in df.columns]

    return {"core": unique_preserve_order(core), "rich": unique_preserve_order(rich)}


def unique_preserve_order(seq: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def split_feature_types(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for c in columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def make_preprocessor(df: pd.DataFrame, predictors: Sequence[str]) -> ColumnTransformer:
    numeric, categorical = split_feature_types(df, predictors)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])
    return pre


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
def recency_weights(waves: pd.Series) -> np.ndarray:
    uniq = sorted(pd.Series(waves).dropna().unique().tolist())
    rank_map = {w: i + 1 for i, w in enumerate(uniq)}
    return waves.map(rank_map).astype(float).values


class ThresholdLogisticWrapper(BaseEstimator, ClassifierMixin):
    """
    Learns hinge features max(x-c, 0) for selected numeric predictors using
    a small validation search on quantile cutpoints, then fits balanced logistic.
    """

    def __init__(self, predictors: Sequence[str], selected_numeric: Optional[Sequence[str]] = None,
                 quantiles: Sequence[float] = (0.25, 0.5, 0.75), random_state: int = 42):
        self.predictors = list(predictors)
        self.selected_numeric = list(selected_numeric) if selected_numeric is not None else []
        self.quantiles = list(quantiles)
        self.random_state = random_state
        self.best_cutpoints_: Dict[str, float] = {}
        self.pipeline_: Optional[Pipeline] = None

    def _augment(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()
        for col, c in self.best_cutpoints_.items():
            X2[f"{col}__hinge"] = np.maximum(pd.to_numeric(X2[col], errors="coerce") - c, 0)
        return X2

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        candidate_cols = [c for c in self.selected_numeric if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        self.best_cutpoints_ = {}

        # Small internal split
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        for col in candidate_cols:
            non_missing = X[col].dropna()
            if non_missing.empty or non_missing.nunique() < 6:
                continue

            cut_candidates = sorted(set(np.quantile(non_missing, self.quantiles).tolist()))
            best_auc = -np.inf
            best_c = None

            for c in cut_candidates:
                aucs = []
                for tr_idx, va_idx in skf.split(X, y):
                    Xtr = X.iloc[tr_idx].copy()
                    Xva = X.iloc[va_idx].copy()
                    ytr = y.iloc[tr_idx]
                    yva = y.iloc[va_idx]

                    Xtr[f"{col}__hinge"] = np.maximum(pd.to_numeric(Xtr[col], errors="coerce") - c, 0)
                    Xva[f"{col}__hinge"] = np.maximum(pd.to_numeric(Xva[col], errors="coerce") - c, 0)

                    predictors = list(Xtr.columns)
                    pre = make_preprocessor(Xtr, predictors)
                    pipe = Pipeline([
                        ("pre", pre),
                        ("clf", LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            solver="liblinear",
                            random_state=self.random_state,
                        )),
                    ])
                    pipe.fit(Xtr[predictors], ytr)
                    p = pipe.predict_proba(Xva[predictors])[:, 1]
                    if len(np.unique(yva)) > 1:
                        aucs.append(roc_auc_score(yva, p))

                mean_auc = np.mean(aucs) if aucs else -np.inf
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_c = c

            if best_c is not None:
                self.best_cutpoints_[col] = float(best_c)

        X_aug = self._augment(X)
        predictors = list(X_aug.columns)
        pre = make_preprocessor(X_aug, predictors)
        self.pipeline_ = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.random_state,
            )),
        ])
        self.pipeline_.fit(X_aug[predictors], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aug = self._augment(X.copy())
        return self.pipeline_.predict_proba(X_aug[X_aug.columns])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_logistic(
    train_df: pd.DataFrame,
    predictors: Sequence[str],
    target_col: str,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Pipeline:
    pre = make_preprocessor(train_df, predictors)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )),
    ])
    fit_params = {}
    if sample_weight is not None:
        fit_params["clf__sample_weight"] = sample_weight
    pipe.fit(train_df[predictors], train_df[target_col], **fit_params)
    return pipe


def fit_xgb(
    train_df: pd.DataFrame,
    predictors: Sequence[str],
    target_col: str,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
):
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

    pre = make_preprocessor(train_df, predictors)
    # Fit preprocessor separately because XGBoost expects numeric matrix
    Xtr = pre.fit_transform(train_df[predictors])
    ytr = train_df[target_col].values

    pos = max(1, int(np.sum(ytr == 1)))
    neg = max(1, int(np.sum(ytr == 0)))
    scale_pos_weight = neg / pos

    clf = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    return {"pre": pre, "clf": clf}


def predict_xgb(model, df: pd.DataFrame, predictors: Sequence[str]) -> np.ndarray:
    X = model["pre"].transform(df[predictors])
    return model["clf"].predict_proba(X)[:, 1]


# ---------------------------------------------------------------------
# EVALUATION PIPELINE
# ---------------------------------------------------------------------
@dataclass
class ModelRun:
    model_name: str
    val_probs: np.ndarray
    test_probs: np.ndarray
    threshold: float
    metrics: Dict[str, float]


def run_single_target(df: pd.DataFrame, target_key: str, outdir: Path, random_state: int = 42) -> None:
    wave_col = CONFIG["wave_col"]
    target_col = CONFIG["outcomes"][target_key]

    if target_col not in df.columns:
        print(f"Skipping {target_key}: missing target column '{target_col}'.")
        return

    predictors = prepare_predictor_lists(df)
    core_predictors = predictors["core"]
    rich_predictors = predictors["rich"]

    use_cols = unique_preserve_order([target_col, wave_col] + core_predictors + rich_predictors)
    work = df[use_cols].copy()
    work = work[work[target_col].notna()].copy()

    train_all = work[work[wave_col] < 2025].copy()
    test_2025 = work[work[wave_col] == 2025].copy()
    if train_all.empty or test_2025.empty:
        raise ValueError(f"{target_key}: training or 2025 holdout data is empty.")

    fit_df, val_df = validation_split_by_wave(train_all, wave_col)

    results_rows = []
    pred_store = {}

    # -------------------------------------------------------------
    # 1. Core logistic
    # -------------------------------------------------------------
    core_logit = fit_logistic(fit_df, core_predictors, target_col, random_state=random_state)
    val_p = core_logit.predict_proba(val_df[core_predictors])[:, 1]
    thr = choose_threshold_by_f1(val_df[target_col].values, val_p)
    test_p = core_logit.predict_proba(test_2025[core_predictors])[:, 1]
    mets = classification_metrics(test_2025[target_col].values, test_p, thr)
    results_rows.append({"model": "Core_Logistic_balanced", **mets})
    pred_store["Core_Logistic_balanced"] = (val_p, test_p)

    # -------------------------------------------------------------
    # 2. Core threshold logistic
    # -------------------------------------------------------------
    candidate_num = [c for c in core_predictors if c in fit_df.columns and pd.api.types.is_numeric_dtype(fit_df[c])]
    candidate_num = candidate_num[: min(4, len(candidate_num))]
    core_thr = ThresholdLogisticWrapper(core_predictors, selected_numeric=candidate_num, random_state=random_state)
    core_thr.fit(fit_df[core_predictors], fit_df[target_col])
    val_p = core_thr.predict_proba(val_df[core_predictors])[:, 1]
    thr = choose_threshold_by_f1(val_df[target_col].values, val_p)
    test_p = core_thr.predict_proba(test_2025[core_predictors])[:, 1]
    mets = classification_metrics(test_2025[target_col].values, test_p, thr)
    results_rows.append({"model": "Core_Threshold_Logistic_balanced", **mets})
    pred_store["Core_Threshold_Logistic_balanced"] = (val_p, test_p)

    # -------------------------------------------------------------
    # 3. Rich logistic
    # -------------------------------------------------------------
    rich_logit = fit_logistic(fit_df, rich_predictors, target_col, random_state=random_state)
    val_p = rich_logit.predict_proba(val_df[rich_predictors])[:, 1]
    thr = choose_threshold_by_f1(val_df[target_col].values, val_p)
    test_p = rich_logit.predict_proba(test_2025[rich_predictors])[:, 1]
    mets = classification_metrics(test_2025[target_col].values, test_p, thr)
    results_rows.append({"model": "Rich_Logistic_balanced", **mets})
    pred_store["Rich_Logistic_balanced"] = (val_p, test_p)

    # -------------------------------------------------------------
    # 4. Rich logistic + recency
    # -------------------------------------------------------------
    fit_w = recency_weights(fit_df[wave_col])
    rich_logit_rw = fit_logistic(
        fit_df, rich_predictors, target_col, sample_weight=fit_w, random_state=random_state
    )
    val_p = rich_logit_rw.predict_proba(val_df[rich_predictors])[:, 1]
    thr = choose_threshold_by_f1(val_df[target_col].values, val_p)
    test_p = rich_logit_rw.predict_proba(test_2025[rich_predictors])[:, 1]
    mets = classification_metrics(test_2025[target_col].values, test_p, thr)
    results_rows.append({"model": "Rich_Logistic_recentWeighted", **mets})
    pred_store["Rich_Logistic_recentWeighted"] = (val_p, test_p)

    # -------------------------------------------------------------
    # 5. Rich XGB + recency
    # -------------------------------------------------------------
    if HAS_XGB:
        fit_w = recency_weights(fit_df[wave_col])
        xgb_model = fit_xgb(fit_df, rich_predictors, target_col, sample_weight=fit_w, random_state=random_state)
        val_p = predict_xgb(xgb_model, val_df, rich_predictors)
        thr = choose_threshold_by_f1(val_df[target_col].values, val_p)
        test_p = predict_xgb(xgb_model, test_2025, rich_predictors)
        mets = classification_metrics(test_2025[target_col].values, test_p, thr)
        results_rows.append({"model": "Rich_XGB_recentWeighted", **mets})
        pred_store["Rich_XGB_recentWeighted"] = (val_p, test_p)
    else:
        print("xgboost not installed; skipping Rich_XGB_recentWeighted.")

    # -------------------------------------------------------------
    # 6. Stacked meta-logit
    # -------------------------------------------------------------
    stack_models = [
        m for m in ["Core_Logistic_balanced", "Rich_Logistic_balanced", "Rich_XGB_recentWeighted"]
        if m in pred_store
    ]
    stack_val = np.column_stack([pred_store[m][0] for m in stack_models])
    stack_test = np.column_stack([pred_store[m][1] for m in stack_models])

    stack_fit = pd.DataFrame(stack_val, columns=[f"p_{m}" for m in stack_models])
    stack_val_df = pd.DataFrame(stack_test, columns=[f"p_{m}" for m in stack_models])

    meta = LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
    meta.fit(stack_fit, val_df[target_col].values)
    meta_val_p = meta.predict_proba(stack_fit)[:, 1]
    thr = choose_threshold_by_f1(val_df[target_col].values, meta_val_p)
    meta_test_p = meta.predict_proba(stack_val_df)[:, 1]
    mets = classification_metrics(test_2025[target_col].values, meta_test_p, thr)
    results_rows.append({"model": "Stacked_Meta_Logistic", **mets})

    # Save transport results
    results_df = pd.DataFrame(results_rows).sort_values("ROC_AUC", ascending=False)
    results_df.to_csv(outdir / f"egypt_final_{target_key}_transport_results.csv", index=False)

    # Save innovation coefficient table for rich logistic
    save_rich_logistic_coefficients(rich_logit, fit_df, rich_predictors, target_col, target_key, outdir)

    # Save repeated-seed summary
    repeat_df = repeated_seed_summary(work, target_col, rich_predictors, core_predictors, wave_col, seeds=[11, 17, 29, 37, 41], has_xgb=HAS_XGB)
    repeat_df.to_csv(outdir / f"egypt_final_{target_key}_repeated_seed_summary.csv", index=False)

    # Save recency ablation
    abl_df = recency_ablation(work, target_col, rich_predictors, wave_col)
    abl_df.to_csv(outdir / f"egypt_final_{target_key}_recency_ablation.csv", index=False)

    print(f"Saved outputs for {target_key}.")


def save_rich_logistic_coefficients(model: Pipeline, fit_df: pd.DataFrame, predictors: Sequence[str],
                                    target_col: str, target_key: str, outdir: Path) -> None:
    """
    Exports interpretable coefficient table with odds ratios.
    """
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    feature_names = list(pre.get_feature_names_out())
    coef = clf.coef_.ravel()

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef_logit": coef,
        "odds_ratio": np.exp(coef),
    }).sort_values("odds_ratio", ascending=False)

    # Pull innovation rows to a separate easy table
    innovation_keywords = ["innov_"]
    mask = coef_df["feature"].astype(str).str.contains("|".join(innovation_keywords), case=False, regex=True)
    coef_df.to_csv(outdir / f"egypt_final_{target_key}_rich_logistic_coefficients.csv", index=False)
    coef_df.loc[mask].to_csv(outdir / f"egypt_final_{target_key}_innovation_effects.csv", index=False)


def repeated_seed_summary(
    work: pd.DataFrame,
    target_col: str,
    rich_predictors: Sequence[str],
    core_predictors: Sequence[str],
    wave_col: str,
    seeds: Sequence[int],
    has_xgb: bool,
) -> pd.DataFrame:
    train_all = work[work[wave_col] < 2025].copy()
    test_2025 = work[work[wave_col] == 2025].copy()

    rows = []
    for seed in seeds:
        fit_df, val_df = validation_split_by_wave(train_all.sample(frac=1, random_state=seed), wave_col)

        models = {}

        m1 = fit_logistic(fit_df, core_predictors, target_col, random_state=seed)
        p = m1.predict_proba(test_2025[core_predictors])[:, 1]
        pv = m1.predict_proba(val_df[core_predictors])[:, 1]
        rows.append({"seed": seed, "model": "Core_Logistic_balanced", **classification_metrics(test_2025[target_col].values, p, choose_threshold_by_f1(val_df[target_col].values, pv))})

        m2 = fit_logistic(fit_df, rich_predictors, target_col, random_state=seed)
        p = m2.predict_proba(test_2025[rich_predictors])[:, 1]
        pv = m2.predict_proba(val_df[rich_predictors])[:, 1]
        rows.append({"seed": seed, "model": "Rich_Logistic_balanced", **classification_metrics(test_2025[target_col].values, p, choose_threshold_by_f1(val_df[target_col].values, pv))})

        fit_w = recency_weights(fit_df[wave_col])
        m3 = fit_logistic(fit_df, rich_predictors, target_col, sample_weight=fit_w, random_state=seed)
        p = m3.predict_proba(test_2025[rich_predictors])[:, 1]
        pv = m3.predict_proba(val_df[rich_predictors])[:, 1]
        rows.append({"seed": seed, "model": "Rich_Logistic_recentWeighted", **classification_metrics(test_2025[target_col].values, p, choose_threshold_by_f1(val_df[target_col].values, pv))})

    rep = pd.DataFrame(rows)
    agg = rep.groupby("model").agg(
        ROC_AUC_mean=("ROC_AUC", "mean"),
        ROC_AUC_std=("ROC_AUC", "std"),
        PR_AUC_mean=("PR_AUC", "mean"),
        F1_mean=("F1", "mean"),
        Recall_mean=("Recall", "mean"),
        Precision_mean=("Precision", "mean"),
        Accuracy_mean=("Accuracy", "mean"),
    ).reset_index()

    return agg


def recency_ablation(work: pd.DataFrame, target_col: str, rich_predictors: Sequence[str], wave_col: str) -> pd.DataFrame:
    train_all = work[work[wave_col] < 2025].copy()
    test_2025 = work[work[wave_col] == 2025].copy()

    windows = [
        ("Rich_Logistic_balanced_train_2013_2016_2020", [2013, 2016, 2020]),
        ("Rich_Logistic_balanced_train_2016_2020", [2016, 2020]),
        ("Rich_Logistic_balanced_train_2020_only", [2020]),
    ]

    rows = []
    for name, waves in windows:
        tr = train_all[train_all[wave_col].isin(waves)].copy()
        if tr.empty:
            continue

        if len(sorted(tr[wave_col].dropna().unique())) >= 2:
            fit_df, val_df = validation_split_by_wave(tr, wave_col)
        else:
            # fallback: simple random split within the single wave
            idx = np.arange(len(tr))
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            cut = max(1, int(0.7 * len(idx)))
            fit_df = tr.iloc[idx[:cut]].copy()
            val_df = tr.iloc[idx[cut:]].copy()

        m = fit_logistic(fit_df, rich_predictors, target_col, random_state=42)
        pv = m.predict_proba(val_df[rich_predictors])[:, 1]
        thr = choose_threshold_by_f1(val_df[target_col].values, pv)
        pt = m.predict_proba(test_2025[rich_predictors])[:, 1]
        mets = classification_metrics(test_2025[target_col].values, pt, thr)
        rows.append({"model": name, "train_years": "+".join(map(str, waves)), "n_train": len(tr), **mets})

    return pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)


# ---------------------------------------------------------------------
# DESCRIPTIVE TABLES FOR INNOVATION
# ---------------------------------------------------------------------
def innovation_prevalence_tables(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    wave_col = CONFIG["wave_col"]
    derived = CONFIG["innovation_derived"]

    keep = [wave_col] + list(derived.values())
    keep = [c for c in keep if c in df.columns]
    tmp = df[keep].copy()

    rows = []
    for wave, g in tmp.groupby(wave_col):
        row = {"survey_year": wave, "n_firms": len(g)}
        for k, col in derived.items():
            if col in g.columns:
                row[col] = g[col].mean(skipna=True)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("survey_year")
    out.to_csv(outdir / "egypt_final_innovation_prevalence_by_wave.csv", index=False)

    pretty = out.copy()
    for c in pretty.columns:
        if c not in ["survey_year", "n_firms"]:
            pretty[c] = (100 * pretty[c]).round(1)
    pretty.to_csv(outdir / "egypt_final_innovation_prevalence_by_wave_percent.csv", index=False)
    return out


# ---------------------------------------------------------------------
# MANUSCRIPT FIGURES
# ---------------------------------------------------------------------
def build_outcome_prevalence_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    wave_col = CONFIG["wave_col"]
    out_rows = []
    for wave, g in df.groupby(wave_col):
        row = {"survey_year": wave}
        for key, col in CONFIG["outcomes"].items():
            if col in g.columns:
                row[key] = g[col].mean(skipna=True)
        out_rows.append(row)
    out = pd.DataFrame(out_rows).sort_values("survey_year")
    out.to_csv(outdir / "egypt_final_all_target_summary.csv", index=False)
    return out


def make_figures(results_dir: Path) -> None:
    fig_dir = ensure_dir(results_dir / "manuscript_figures_innovation")
    tab_dir = ensure_dir(results_dir / "manuscript_tables_innovation")

    # read files
    target_summary = pd.read_csv(results_dir / "egypt_final_all_target_summary.csv")
    innov_summary = pd.read_csv(results_dir / "egypt_final_innovation_prevalence_by_wave.csv")

    mapping = {
        "target_k30": "Severe financing obstacle",
        "target_formal_credit": "Formal credit access",
        "target_applied_credit": "Credit application",
    }

    transports = {}
    for tk in mapping:
        fp = results_dir / f"egypt_final_{tk}_transport_results.csv"
        if fp.exists():
            transports[tk] = pd.read_csv(fp)

    # Figure 1: study outcomes prevalence
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(target_summary["survey_year"], target_summary["target_k30"], marker="o", linewidth=2.5, color=PALETTE["navy"], label="Obstacle")
    ax.plot(target_summary["survey_year"], target_summary["target_formal_credit"], marker="o", linewidth=2.5, color=PALETTE["gold"], label="Formal credit")
    ax.plot(target_summary["survey_year"], target_summary["target_applied_credit"], marker="o", linewidth=2.5, color=PALETTE["sky"], label="Credit application")
    ax.set_xlabel("Survey year")
    ax.set_ylabel("Rate")
    ax.set_xticks(target_summary["survey_year"])
    clean_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    save_figure(fig, "figure1_target_prevalence_by_wave_innovation", fig_dir)

    # Figure 1b: innovation prevalence
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for col, lab, color in [
        ("innov_new_product", "New product", PALETTE["blue"]),
        ("innov_new_process", "New process", PALETTE["rose"]),
        ("innov_rd_investment", "R&D investment", PALETTE["plum"]),
        ("innov_any", "Any innovation", PALETTE["orange"]),
    ]:
        if col in innov_summary.columns:
            ax.plot(innov_summary["survey_year"], innov_summary[col], marker="o", linewidth=2.5, color=color, label=lab)

    ax.set_xlabel("Survey year")
    ax.set_ylabel("Rate")
    ax.set_xticks(innov_summary["survey_year"])
    clean_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure(fig, "figure1b_innovation_prevalence_by_wave", fig_dir)

    # Table copy
    target_summary.to_csv(tab_dir / "table1_target_prevalence_by_wave.csv", index=False)
    innov_summary.to_csv(tab_dir / "table1b_innovation_prevalence_by_wave.csv", index=False)

    # Figure 2: transport ROC AUC by target/model
    plot_rows = []
    for tk, df in transports.items():
        tmp = df.copy()
        tmp["target"] = mapping[tk]
        plot_rows.append(tmp)
    if plot_rows:
        roc_df = pd.concat(plot_rows, ignore_index=True)
        models = [
            "Core_Logistic_balanced",
            "Core_Threshold_Logistic_balanced",
            "Rich_Logistic_balanced",
            "Rich_Logistic_recentWeighted",
            "Rich_XGB_recentWeighted",
            "Stacked_Meta_Logistic",
        ]
        use = roc_df[roc_df["model"].isin(models)].copy()
        targets = ["Severe financing obstacle", "Formal credit access", "Credit application"]

        model_labels = {
            "Core_Logistic_balanced": "Core logistic",
            "Core_Threshold_Logistic_balanced": "Core threshold",
            "Rich_Logistic_balanced": "Rich logistic + innovation",
            "Rich_Logistic_recentWeighted": "Rich logistic + recency",
            "Rich_XGB_recentWeighted": "Rich XGBoost + recency",
            "Stacked_Meta_Logistic": "Stacked meta-logit",
        }
        model_colors = {
            "Core_Logistic_balanced": PALETTE["gray2"],
            "Core_Threshold_Logistic_balanced": PALETTE["gray4"],
            "Rich_Logistic_balanced": PALETTE["navy"],
            "Rich_Logistic_recentWeighted": PALETTE["mint"],
            "Rich_XGB_recentWeighted": PALETTE["plum"],
            "Stacked_Meta_Logistic": PALETTE["gold"],
        }

        x = np.arange(len(targets))
        width = 0.12
        fig, ax = plt.subplots(figsize=(10.8, 5.1))
        for i, m in enumerate(models):
            vals = []
            for t in targets:
                sub = use[(use["target"] == t) & (use["model"] == m)]
                vals.append(sub["ROC_AUC"].iloc[0] if len(sub) else np.nan)
            ax.bar(
                x + (i - (len(models)-1)/2) * width,
                vals,
                width=width,
                label=model_labels[m],
                color=model_colors[m],
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(["Obstacle", "Formal credit", "Credit application"])
        ax.set_ylabel("ROC AUC")
        clean_axes(ax)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_figure(fig, "figure2_2025_transport_rocauc_by_target_model_innovation", fig_dir)

    # Figure 3: innovation effects bar plots for objective outcomes
    effect_frames = []
    for tk in ["target_formal_credit", "target_applied_credit"]:
        fp = results_dir / f"egypt_final_{tk}_innovation_effects.csv"
        if fp.exists():
            ef = pd.read_csv(fp)
            ef["target"] = mapping[tk]
            effect_frames.append(ef)

    if effect_frames:
        eff = pd.concat(effect_frames, ignore_index=True)
        eff = eff.sort_values("odds_ratio", ascending=True)

        for target in eff["target"].drop_duplicates():
            sub = eff[eff["target"] == target].copy()
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(8.2, 4.6))
            y = np.arange(len(sub))
            colors = [PALETTE["orange"] if x >= 1 else PALETTE["gray3"] for x in sub["odds_ratio"]]
            ax.barh(y, sub["odds_ratio"], color=colors, edgecolor="white", linewidth=0.5)
            ax.axvline(1.0, color=PALETTE["char"], linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["feature"])
            ax.set_xlabel("Odds ratio")
            clean_axes(ax)
            ax.text(0.02, 1.02, target, transform=ax.transAxes, fontsize=11)
            stem = "formal" if "Formal" in target else "applied"
            save_figure(fig, f"figure3_innovation_odds_ratios_{stem}", fig_dir)

    # caption stubs
    captions = [
        {"file": "figure1_target_prevalence_by_wave_innovation", "caption": "Prevalence of the three study outcomes across Egypt Enterprise Survey waves."},
        {"file": "figure1b_innovation_prevalence_by_wave", "caption": "Prevalence of innovation indicators across Egypt Enterprise Survey waves."},
        {"file": "figure2_2025_transport_rocauc_by_target_model_innovation", "caption": "Comparison of 2025 holdout ROC AUC across targets and model classes after adding innovation to the rich specification."},
        {"file": "figure3_innovation_odds_ratios_formal", "caption": "Odds ratios for innovation-related predictors in the rich logistic model for formal credit access."},
        {"file": "figure3_innovation_odds_ratios_applied", "caption": "Odds ratios for innovation-related predictors in the rich logistic model for credit application."},
    ]
    pd.DataFrame(captions).to_csv(tab_dir / "figure_caption_stubs_innovation.csv", index=False)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
DEFAULT_SCRIPT_PATH = Path(r"D:\assets (1)\innovation_credit_pipeline_egypt.py")
DEFAULT_OUTPUT_DIR = DEFAULT_SCRIPT_PATH.parent / "outputs_innovation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rerun the Egypt credit manuscript pipeline with innovation added.",
        epilog=(
            'Example: python "D:\\assets (1)\\innovation_credit_pipeline_egypt.py" '
            '--input "D:\\assets (1)\\egypt_harmonized.csv" '
            '--outdir "D:\\assets (1)\\outputs_innovation"'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to harmonized Egypt CSV, for example: D:\\assets (1)\\egypt_harmonized.csv",
    )
    p.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output folder [default: {DEFAULT_OUTPUT_DIR}]",
    )
    p.add_argument("--sme-only", action="store_true", help="Filter to SMEs using CONFIG rules")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))

    df = pd.read_csv(args.input)
    print(f"Loaded: {args.input}")
    print(f"Shape before filtering: {df.shape}")

    df = apply_optional_filters(df, args)
    print(f"Shape after filtering: {df.shape}")

    df = attach_innovation_from_raw_wbes(df, Path(args.input))
    df = build_innovation_variables(df)

    predictor_lists = prepare_predictor_lists(df)
    print("\nCore predictors used:")
    print(predictor_lists["core"])
    print("\nRich predictors used:")
    print(predictor_lists["rich"])

    # Save descriptive summaries
    build_outcome_prevalence_table(df, outdir)
    innovation_prevalence_tables(df, outdir)

    # Main models by target
    for target_key in CONFIG["outcomes"]:
        run_single_target(df, target_key, outdir)

    # Figures
    make_figures(outdir)

    # Write short run note
    run_note = {
        "input": args.input,
        "sme_only": args.sme_only,
        "xgboost_installed": HAS_XGB,
        "wave_col": CONFIG["wave_col"],
        "outcomes": CONFIG["outcomes"],
        "core_predictors_used": predictor_lists["core"],
        "rich_predictors_used": predictor_lists["rich"],
    }
    with open(outdir / "run_note_innovation.json", "w", encoding="utf-8") as f:
        json.dump(run_note, f, indent=2)

    print("\nDone.")
    print(f"Results saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
