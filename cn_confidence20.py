from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------- LOCAL PATHS -----------------------------------------
MASTER_PATH = "ps3_out/ps3_master_processed.csv"
A52_OOF_PATH = "stage2_arch52/architecture52_oof_survey_predictions.csv"
A52_ALL_PATH = "stage2_arch52/architecture52_all_addresses.csv"
OUTPUT_DIR = "stage2_confidence20"

# -------------------------------- EXPERIMENT SETTINGS ----------------------------------
RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4

# Search over raw conditional quantile shapes. Conformal calibration fixes final coverage.
RAW_QUANTILES = [0.70, 0.80, 0.90]
TARGET_COVERAGES = [0.88, 0.90]
MIN_FINE_GROUP_SIZES = [8, 12]

# Coverage guardrails used when choosing the smallest useful radius configuration.
MIN_OVERALL_COVERAGE = 0.88
MIN_VISITED_COVERAGE = 0.88
MIN_NO_VISIT_COVERAGE = 0.88
MIN_NV_GRID_COVERAGE = 0.85
MAX_PREFERRED_OVERALL_COVERAGE = 0.93

# Prevent numerically silly zero/negative radii before calibration.
RAW_RADIUS_FLOOR_M = 10.0
FINAL_RADIUS_FLOOR_M = 10.0

ROUTER_SOURCES = ["grid", "resid", "extra", "baseline"]



def euclid(x1, y1, x2, y2):
    return np.sqrt((np.asarray(x1, float) - np.asarray(x2, float)) ** 2 +
                   (np.asarray(y1, float) - np.asarray(y2, float)) ** 2)


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for c in names:
        if c in df.columns:
            return c
    return None


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Run Architecture 5.2 first and make sure this notebook is launched from the project root."
        )
    return pd.read_csv(p)




def _simple_imputer(strategy: str):
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)

def onehot_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)
    except TypeError:  # older sklearn
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=2)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)


def finite_conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal quantile: ceil((n+1)*coverage)-th order statistic."""
    x = np.asarray(scores, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    x = np.sort(x)
    k = int(math.ceil((len(x) + 1) * float(coverage)))
    k = min(max(k, 1), len(x))
    return float(x[k - 1])


def coverage_metrics(errors: np.ndarray, radii: np.ndarray) -> Dict[str, float]:
    e = np.asarray(errors, float)
    r = np.asarray(radii, float)
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    if len(e) == 0:
        return {
            "n": 0, "coverage_pct": np.nan, "median_radius_m": np.nan,
            "p90_radius_m": np.nan, "median_error_m": np.nan, "p90_error_m": np.nan,
            "mean_error_m": np.nan, "sharpness_ratio": np.nan,
        }
    return {
        "n": int(len(e)),
        "coverage_pct": float(100 * np.mean(e <= r)),
        "median_radius_m": float(np.median(r)),
        "p90_radius_m": float(np.quantile(r, 0.90)),
        "median_error_m": float(np.median(e)),
        "p90_error_m": float(np.quantile(e, 0.90)),
        "mean_error_m": float(np.mean(e)),
        "sharpness_ratio": float(np.median(r) / max(np.median(e), 1e-9)),
    }


def normalize_boolish(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    st = s.astype(str).str.strip().str.lower()
    mp = {"true": 1, "1": 1, "yes": 1, "y": 1, "false": 0, "0": 0, "no": 0, "n": 0, "nan": 0, "none": 0}
    return st.map(mp).fillna(0).astype(int)


def selected_value(frame: pd.DataFrame, selector_col: str, prefix: str, suffix: str = "") -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    if selector_col not in frame.columns:
        return out
    sel = frame[selector_col].astype(str)
    for src in ROUTER_SOURCES:
        col = f"{prefix}{src}{suffix}"
        if col in frame.columns:
            m = sel.eq(src)
            out.loc[m] = safe_num(frame.loc[m, col])
    return out




def merge_master_features(pred: pd.DataFrame, master: pd.DataFrame, address_id: str) -> pd.DataFrame:
    master_keep = [
        address_id,
        "pt_spread_m", "total_visits", "clean_visits", "flagged_visits",
        "avg_integrity_risk", "agents_involved", "locality_conf", "n_fields_found",
        "script_kn_frac", "script_dev_frac", "is_village_style",
        "locality_method", "town_method", "base_precision", "lm_primary",
        "rel_no_type", "rel_context", "poi_x", "poi_y", "geocoder_x", "geocoder_y",
        "house_no", "cross_no", "main_no", "gali_no", "road_no", "ward_no",
    ]
    master_keep = [c for c in master_keep if c in master.columns]
    # Avoid duplicate columns already emitted by A5.2.
    extras = [address_id] + [c for c in master_keep if c != address_id and c not in pred.columns]
    return pred.merge(master[extras].drop_duplicates(address_id), on=address_id, how="left")


def build_confidence_frame(df: pd.DataFrame, production: bool = False) -> pd.DataFrame:
    f = df.copy()


    if production:
        pin_x = first_existing(f, ["A52_pin_x", "pin_x"])
        pin_y = first_existing(f, ["A52_pin_y", "pin_y"])
        old_r = first_existing(f, ["A52_confidence_radius_m", "confidence_radius_m"])
        method = first_existing(f, ["A52_method", "method"])
    else:
        pin_x = first_existing(f, ["A52_oof_x", "A52_pin_x", "pin_x"])
        pin_y = first_existing(f, ["A52_oof_y", "A52_pin_y", "pin_y"])
        old_r = first_existing(f, ["A52_oof_radius_m", "A52_confidence_radius_m", "confidence_radius_m"])
        method = first_existing(f, ["A52_oof_method", "A52_method", "method"])

    if pin_x is None or pin_y is None:
        raise KeyError("Could not find Architecture 5.2 pin coordinate columns.")

    f["pin_x"] = safe_num(f[pin_x])
    f["pin_y"] = safe_num(f[pin_y])
    f["current_radius_m"] = safe_num(f[old_r]) if old_r else np.nan
    f["pin_method"] = f[method].astype(str) if method else "unknown"

    
    err_col = first_existing(f, ["A52_oof_error_m", "error_m", "actual_error_m"])
    if err_col:
        f["actual_error_m"] = safe_num(f[err_col])

    
    cvc = first_existing(f, ["clean_visit_count", "clean_visits", "pt_k"])
    if cvc is None:
        f["clean_visit_count"] = 0
    else:
        f["clean_visit_count"] = safe_num(f[cvc]).fillna(0)

    f["is_visited"] = (f["clean_visit_count"] > 0).astype(int)
    f["broad_conf_group"] = np.where(f["is_visited"].eq(1), "Visited", "NoVisit")

    evidence = f["evidence_tier"].astype(str) if "evidence_tier" in f.columns else pd.Series("NV_other", index=f.index)
    conf_tier = []
    for idx, row in f.iterrows():
        k = float(row.get("clean_visit_count", 0) or 0)
        if k > 0:
            if k <= 1:
                conf_tier.append("V1")
            elif k <= 2:
                conf_tier.append("V2")
            else:
                conf_tier.append("V3plus")
        else:
            et = str(evidence.loc[idx])
            if et in {"NV_grid", "NV_grid_poi", "NV_poi", "NV_locality", "NV_fallback"}:
                conf_tier.append(et)
            else:
                conf_tier.append("NV_other")
    f["confidence_tier"] = conf_tier

    
    if "router52_ood" in f.columns:
        f["router52_ood_num"] = normalize_boolish(f["router52_ood"])
    else:
        f["router52_ood_num"] = 0

   
    f["top1_predicted_risk_m"] = selected_value(f, "router52_top1", "router52_predicted_risk_", "_m")
    f["top2_predicted_risk_m"] = selected_value(f, "router52_top2", "router52_predicted_risk_", "_m")
    f["top1_adjusted_risk_m"] = selected_value(f, "router52_top1", "router52_adjusted_risk_", "_m")
    f["top2_adjusted_risk_m"] = selected_value(f, "router52_top2", "router52_adjusted_risk_", "_m")

    risk_cols = [f"router52_adjusted_risk_{s}_m" for s in ROUTER_SOURCES if f"router52_adjusted_risk_{s}_m" in f.columns]
    pred_risk_cols = [f"router52_predicted_risk_{s}_m" for s in ROUTER_SOURCES if f"router52_predicted_risk_{s}_m" in f.columns]
    weight_cols = [f"router52_weight_{s}" for s in ROUTER_SOURCES if f"router52_weight_{s}" in f.columns]

    if risk_cols:
        rr = f[risk_cols].apply(pd.to_numeric, errors="coerce")
        f["router_min_adjusted_risk_m"] = rr.min(axis=1)
        f["router_second_adjusted_risk_m"] = rr.apply(lambda r: np.sort(r.dropna().to_numpy(float))[1] if r.notna().sum() >= 2 else np.nan, axis=1)
        f["router_median_adjusted_risk_m"] = rr.median(axis=1)
        f["router_max_adjusted_risk_m"] = rr.max(axis=1)
        f["router_adjusted_risk_spread_m"] = rr.max(axis=1) - rr.min(axis=1)
    if pred_risk_cols:
        rp = f[pred_risk_cols].apply(pd.to_numeric, errors="coerce")
        f["router_min_predicted_risk_m"] = rp.min(axis=1)
        f["router_predicted_risk_spread_m"] = rp.max(axis=1) - rp.min(axis=1)

    f["router_risk_gap_m"] = f["top2_adjusted_risk_m"] - f["top1_adjusted_risk_m"]
    f["router_risk_ratio"] = f["top2_adjusted_risk_m"] / f["top1_adjusted_risk_m"].clip(lower=1.0)

    if weight_cols:
        w = f[weight_cols].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
        row_sum = w.sum(axis=1).replace(0, np.nan)
        wn = w.div(row_sum, axis=0)
        f["router_weight_max"] = wn.max(axis=1)
        f["router_weight_entropy"] = -(wn.where(wn > 0) * np.log(wn.where(wn > 0))).sum(axis=1)
    else:
        f["router_weight_max"] = np.nan
        f["router_weight_entropy"] = np.nan

    # Geometric distances available at prediction time.
    if {"geocoder_x", "geocoder_y"}.issubset(f.columns):
        f["pin_to_baseline_m"] = euclid(f["pin_x"], f["pin_y"], safe_num(f["geocoder_x"]), safe_num(f["geocoder_y"]))
    if {"poi_x", "poi_y"}.issubset(f.columns):
        ok = f[["poi_x", "poi_y"]].notna().all(axis=1)
        f["pin_to_poi_m"] = np.nan
        f.loc[ok, "pin_to_poi_m"] = euclid(
            f.loc[ok, "pin_x"], f.loc[ok, "pin_y"], safe_num(f.loc[ok, "poi_x"]), safe_num(f.loc[ok, "poi_y"])
        )
        if {"geocoder_x", "geocoder_y"}.issubset(f.columns):
            f["baseline_to_poi_m"] = np.nan
            f.loc[ok, "baseline_to_poi_m"] = euclid(
                safe_num(f.loc[ok, "geocoder_x"]), safe_num(f.loc[ok, "geocoder_y"]),
                safe_num(f.loc[ok, "poi_x"]), safe_num(f.loc[ok, "poi_y"])
            )

    return f




NUMERIC_CANDIDATES = [
    "clean_visit_count", "pt_spread_m", "total_visits", "clean_visits", "flagged_visits",
    "avg_integrity_risk", "agents_involved", "locality_conf", "n_fields_found",
    "script_kn_frac", "script_dev_frac", "is_village_style", "current_radius_m",
    "router52_ood_num", "router_grid_extrapolation_score",
    "router_nearest_grid_feature_distance", "router_locality_train_count",
    "top1_predicted_risk_m", "top2_predicted_risk_m", "top1_adjusted_risk_m", "top2_adjusted_risk_m",
    "router_min_adjusted_risk_m", "router_second_adjusted_risk_m", "router_median_adjusted_risk_m",
    "router_max_adjusted_risk_m", "router_adjusted_risk_spread_m",
    "router_min_predicted_risk_m", "router_predicted_risk_spread_m",
    "router_risk_gap_m", "router_risk_ratio", "router_weight_max", "router_weight_entropy",
    "pin_to_baseline_m", "pin_to_poi_m", "baseline_to_poi_m",
    "house_no", "cross_no", "main_no", "gali_no", "road_no", "ward_no",
]

CATEGORICAL_CANDIDATES = [
    "confidence_tier", "evidence_tier", "pin_method", "router52_top1", "router52_top2",
    "router52_ood", "router52_ood_reason", "locality_method", "town_method", "base_precision",
    "lm_primary", "rel_no_type", "rel_context",
]


def usable_features(train: pd.DataFrame, production: pd.DataFrame) -> Tuple[List[str], List[str]]:
    common = set(train.columns) & set(production.columns)
    nums, cats = [], []
    for c in NUMERIC_CANDIDATES:
        if c in common and safe_num(train[c]).notna().sum() >= 3:
            nums.append(c)
    for c in CATEGORICAL_CANDIDATES:
        if c in common and train[c].notna().sum() >= 3 and train[c].astype(str).nunique(dropna=True) >= 2:
            cats.append(c)
    return nums, cats


def coerce_selected_numeric_features(
    train: pd.DataFrame,
    production: pd.DataFrame,
    num_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    """
    Force every selected numeric feature to a true numeric dtype before it reaches
    SimpleImputer/StandardScaler.

    Address-derived fields can legitimately contain tokens such as "206/2".
    Those values must NOT be passed to a median imputer as strings. We coerce
    non-numeric tokens to NaN rather than inventing a numeric interpretation.
    """
    train = train.copy()
    production = production.copy()
    diagnostics = []

    for c in num_cols:
        for label, df in [("OOF", train), ("production", production)]:
            raw = df[c].copy()
            converted = pd.to_numeric(raw, errors="coerce")

            raw_nonmissing = raw.notna()
            newly_missing = raw_nonmissing & converted.isna()

            if newly_missing.any():
                examples = (
                    raw.loc[newly_missing]
                    .astype(str)
                    .drop_duplicates()
                    .head(8)
                    .tolist()
                )
                diagnostics.append({
                    "dataset": label,
                    "feature": c,
                    "n_non_numeric_coerced_to_nan": int(newly_missing.sum()),
                    "examples": examples,
                })

            df[c] = converted.astype(float)

    return train, production, diagnostics


def make_pipeline(num_cols: List[str], cat_cols: List[str], alpha: float, seed: int) -> Pipeline:
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", _simple_imputer("median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", _simple_imputer("most_frequent")),
            ("onehot", onehot_dense()),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    prep = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    reg = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(alpha),
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_depth=3,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("prep", prep), ("model", reg)])


def active_features(sub: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
    nums = [c for c in num_cols if c in sub.columns and safe_num(sub[c]).notna().sum() >= 3]
    cats = [c for c in cat_cols if c in sub.columns and sub[c].notna().sum() >= 3 and sub[c].astype(str).nunique(dropna=True) >= 2]
    return nums, cats


def fit_broad_models(train_df: pd.DataFrame, alpha: float, num_cols: List[str], cat_cols: List[str], seed: int):
    models = {}
    global_nums, global_cats = active_features(train_df, num_cols, cat_cols)
    global_model = make_pipeline(global_nums, global_cats, alpha, seed)
    global_model.fit(train_df[global_nums + global_cats], train_df["actual_error_m"])

    for g in ["Visited", "NoVisit"]:
        sub = train_df[train_df["broad_conf_group"] == g]
        if len(sub) < 15:
            models[g] = (global_model, global_nums, global_cats)
            continue
        nums, cats = active_features(sub, num_cols, cat_cols)
        model = make_pipeline(nums, cats, alpha, seed + (17 if g == "Visited" else 29))
        model.fit(sub[nums + cats], sub["actual_error_m"])
        models[g] = (model, nums, cats)
    return models


def predict_broad_models(models, df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for g in ["Visited", "NoVisit"]:
        idx = df.index[df["broad_conf_group"] == g]
        if len(idx) == 0:
            continue
        model, nums, cats = models[g]
        pred = model.predict(df.loc[idx, nums + cats])
        out.loc[idx] = np.maximum(pred, RAW_RADIUS_FLOOR_M)
    return out


def inner_oof_raw(train_outer: pd.DataFrame, alpha: float, num_cols: List[str], cat_cols: List[str], seed: int) -> pd.Series:
    ystrat = train_outer["broad_conf_group"].astype(str)
    min_count = int(ystrat.value_counts().min())
    n_splits = max(2, min(INNER_FOLDS, min_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw = pd.Series(np.nan, index=train_outer.index, dtype=float)
    arr = np.arange(len(train_outer))
    for j, (tr_pos, va_pos) in enumerate(skf.split(arr, ystrat.to_numpy())):
        tr_idx = train_outer.index[tr_pos]
        va_idx = train_outer.index[va_pos]
        models = fit_broad_models(train_outer.loc[tr_idx], alpha, num_cols, cat_cols, seed + 100 + j)
        raw.loc[va_idx] = predict_broad_models(models, train_outer.loc[va_idx])
    return raw



def make_calibration_map(cal_df: pd.DataFrame, raw_radius: pd.Series, target_coverage: float, min_fine_n: int):
    work = cal_df.copy()
    work["raw_radius"] = raw_radius.reindex(work.index).clip(lower=RAW_RADIUS_FLOOR_M)
    work["ratio"] = work["actual_error_m"] / work["raw_radius"].clip(lower=RAW_RADIUS_FLOOR_M)

    factors = {"fine": {}, "broad": {}, "global": 1.0}
    factor_rows = []

    global_factor = finite_conformal_quantile(work["ratio"].to_numpy(float), target_coverage)
    factors["global"] = global_factor
    factor_rows.append({"level": "global", "group": "ALL", "n": len(work), "factor": global_factor})

    for g, sub in work.groupby("broad_conf_group"):
        if len(sub) >= min_fine_n:
            q = finite_conformal_quantile(sub["ratio"].to_numpy(float), target_coverage)
        else:
            q = global_factor
        factors["broad"][str(g)] = q
        factor_rows.append({"level": "broad", "group": str(g), "n": len(sub), "factor": q})

    for g, sub in work.groupby("confidence_tier"):
        broad = str(sub["broad_conf_group"].iloc[0])
        if len(sub) >= min_fine_n:
            q = finite_conformal_quantile(sub["ratio"].to_numpy(float), target_coverage)
            level = "fine"
        else:
            q = factors["broad"].get(broad, global_factor)
            level = "broad_fallback"
        factors["fine"][str(g)] = q
        factor_rows.append({"level": level, "group": str(g), "n": len(sub), "factor": q})

    return factors, pd.DataFrame(factor_rows)


def apply_calibration(df: pd.DataFrame, raw_radius: pd.Series, factors):
    final = pd.Series(np.nan, index=df.index, dtype=float)
    factor_used = pd.Series(np.nan, index=df.index, dtype=float)
    level_used = pd.Series("global", index=df.index, dtype=object)

    for idx, row in df.iterrows():
        fine = str(row["confidence_tier"])
        broad = str(row["broad_conf_group"])
        if fine in factors["fine"]:
            q = factors["fine"][fine]
            lvl = "fine_or_fallback"
        elif broad in factors["broad"]:
            q = factors["broad"][broad]
            lvl = "broad"
        else:
            q = factors["global"]
            lvl = "global"
        factor_used.loc[idx] = q
        level_used.loc[idx] = lvl
        final.loc[idx] = max(FINAL_RADIUS_FLOOR_M, float(raw_radius.loc[idx]) * float(q))
    return final, factor_used, level_used



def strat_labels(df: pd.DataFrame) -> pd.Series:
    # Use fine tiers when frequent enough for 5 folds, otherwise broad tier.
    counts = df["confidence_tier"].value_counts()
    return df.apply(
        lambda r: r["confidence_tier"] if counts.get(r["confidence_tier"], 0) >= OUTER_FOLDS else r["broad_conf_group"],
        axis=1,
    ).astype(str)


def build_outer_cache(df: pd.DataFrame, alpha: float, num_cols: List[str], cat_cols: List[str]):
    labels = strat_labels(df)
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cache = []
    arr = np.arange(len(df))
    for fold, (tr_pos, te_pos) in enumerate(skf.split(arr, labels.to_numpy())):
        tr_idx = df.index[tr_pos]
        te_idx = df.index[te_pos]
        tr = df.loc[tr_idx]
        te = df.loc[te_idx]

        raw_train_oof = inner_oof_raw(tr, alpha, num_cols, cat_cols, RANDOM_STATE + 1000 * (fold + 1))
        final_models = fit_broad_models(tr, alpha, num_cols, cat_cols, RANDOM_STATE + 5000 + fold)
        raw_test = predict_broad_models(final_models, te)

        cache.append({
            "fold": fold,
            "train_idx": tr_idx,
            "test_idx": te_idx,
            "raw_train_oof": raw_train_oof,
            "raw_test": raw_test,
        })
    return cache


def evaluate_config(df: pd.DataFrame, cache, target_cov: float, min_fine_n: int):
    final = pd.Series(np.nan, index=df.index, dtype=float)
    raw_all = pd.Series(np.nan, index=df.index, dtype=float)
    factor_all = pd.Series(np.nan, index=df.index, dtype=float)
    level_all = pd.Series("", index=df.index, dtype=object)
    fold_all = pd.Series(-1, index=df.index, dtype=int)
    factor_tables = []

    for c in cache:
        tr = df.loc[c["train_idx"]]
        te = df.loc[c["test_idx"]]
        factors, ft = make_calibration_map(tr, c["raw_train_oof"], target_cov, min_fine_n)
        fr, fu, lu = apply_calibration(te, c["raw_test"], factors)
        final.loc[te.index] = fr
        raw_all.loc[te.index] = c["raw_test"]
        factor_all.loc[te.index] = fu
        level_all.loc[te.index] = lu
        fold_all.loc[te.index] = c["fold"]
        ft["outer_fold"] = c["fold"]
        factor_tables.append(ft)

    return final, raw_all, factor_all, level_all, fold_all, pd.concat(factor_tables, ignore_index=True)


def scope_mask(df: pd.DataFrame, scope: str):
    if scope == "ALL":
        return pd.Series(True, index=df.index)
    if scope == "VISITED":
        return df["is_visited"].eq(1)
    if scope == "NO_VISIT":
        return df["is_visited"].eq(0)
    if scope == "NV_GRID":
        return df["confidence_tier"].eq("NV_grid")
    return df["confidence_tier"].eq(scope)


def summarize_radius_model(df: pd.DataFrame, radii: pd.Series, model_name: str):
    rows = []
    scopes = ["ALL", "VISITED", "NO_VISIT", "NV_GRID", "V1", "V2", "V3plus", "NV_grid_poi", "NV_poi", "NV_locality", "NV_fallback"]
    for scope in scopes:
        m = scope_mask(df, scope)
        if not m.any():
            continue
        met = coverage_metrics(df.loc[m, "actual_error_m"], radii.loc[m])
        met.update({"scope": scope, "model": model_name})
        rows.append(met)
    return pd.DataFrame(rows)


def config_selection_row(summary: pd.DataFrame, alpha, target_cov, min_n):
    def get(scope, col):
        x = summary[summary["scope"] == scope]
        return float(x.iloc[0][col]) if len(x) else np.nan

    oc = get("ALL", "coverage_pct") / 100.0
    vc = get("VISITED", "coverage_pct") / 100.0
    nc = get("NO_VISIT", "coverage_pct") / 100.0
    gc = get("NV_GRID", "coverage_pct") / 100.0 if (summary["scope"] == "NV_GRID").any() else np.nan

    mr = get("ALL", "median_radius_m")
    nvr = get("NO_VISIT", "median_radius_m")
    gr = get("NV_GRID", "median_radius_m") if (summary["scope"] == "NV_GRID").any() else nvr

    feasible = (
        oc >= MIN_OVERALL_COVERAGE and
        vc >= MIN_VISITED_COVERAGE and
        nc >= MIN_NO_VISIT_COVERAGE and
        (not np.isfinite(gc) or gc >= MIN_NV_GRID_COVERAGE)
    )
    preferred = feasible and oc <= MAX_PREFERRED_OVERALL_COVERAGE

    # Radius efficiency objective; coverage shortfalls receive huge penalty.
    radius_score = mr + 0.25 * nvr + 0.15 * gr
    shortfall = (
        max(0, MIN_OVERALL_COVERAGE - oc) +
        max(0, MIN_VISITED_COVERAGE - vc) +
        max(0, MIN_NO_VISIT_COVERAGE - nc) +
        (max(0, MIN_NV_GRID_COVERAGE - gc) if np.isfinite(gc) else 0)
    )
    overcoverage_penalty = 1200 * max(0, oc - 0.92)
    score = radius_score + 20000 * shortfall + overcoverage_penalty

    return {
        "raw_quantile": alpha,
        "target_coverage": target_cov,
        "min_fine_group_n": min_n,
        "overall_coverage_pct": 100 * oc,
        "visited_coverage_pct": 100 * vc,
        "no_visit_coverage_pct": 100 * nc,
        "nv_grid_coverage_pct": 100 * gc if np.isfinite(gc) else np.nan,
        "overall_median_radius_m": mr,
        "visited_median_radius_m": get("VISITED", "median_radius_m"),
        "no_visit_median_radius_m": nvr,
        "nv_grid_median_radius_m": gr,
        "feasible": bool(feasible),
        "preferred": bool(preferred),
        "selection_score": float(score),
    }



def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    master = load_csv(MASTER_PATH)
    oof = load_csv(A52_OOF_PATH)
    all_addr = load_csv(A52_ALL_PATH)

    address_id = first_existing(oof, ["address_id", "addr_id", "id"])
    if address_id is None:
        raise KeyError(f"Could not identify address_id in {A52_OOF_PATH}. Columns: {list(oof.columns)}")
    if address_id not in master.columns or address_id not in all_addr.columns:
        raise KeyError(f"Address id '{address_id}' must exist in master and Architecture 5.2 production outputs.")

    oof = merge_master_features(oof, master, address_id)
    all_addr = merge_master_features(all_addr, master, address_id)

    train = build_confidence_frame(oof, production=False)
    prod = build_confidence_frame(all_addr, production=True)

    if "actual_error_m" not in train.columns or train["actual_error_m"].notna().sum() < 50:
        raise KeyError("Architecture 5.2 OOF file must contain A52_oof_error_m (or equivalent actual error).")
    train = train[train["actual_error_m"].notna()].copy()

    num_cols, cat_cols = usable_features(train, prod)
    if not num_cols and not cat_cols:
        raise RuntimeError("No usable confidence features were found in both OOF and production files.")


    train, prod, numeric_coercion_diagnostics = coerce_selected_numeric_features(
        train, prod, num_cols
    )

    print("=" * 100)
    print("CreditNirvana PS3 — Confidence 2.0")
    print("=" * 100)
    print(f"OOF surveyed rows: {len(train)} | Production addresses: {len(prod)}")
    print(f"Numeric confidence features ({len(num_cols)}): {num_cols}")
    print(f"Categorical confidence features ({len(cat_cols)}): {cat_cols}")

    if numeric_coercion_diagnostics:
        print("\nMixed numeric tokens safely coerced to NaN before sklearn:")
        for d in numeric_coercion_diagnostics:
            print(
                f"  - {d['dataset']} / {d['feature']}: "
                f"{d['n_non_numeric_coerced_to_nan']} value(s); "
                f"examples={d['examples']}"
            )
        print(
            "These are treated as missing numeric values rather than being "
            "incorrectly interpreted as numbers."
        )

    print("Pins are frozen: Architecture 5.2 coordinates are never retrained or modified here.\n")

   
    current_summary = summarize_radius_model(train, train["current_radius_m"], "Current_A52_confidence")

    # Search configs. Cache expensive model fits per raw quantile.
    candidate_rows = []
    result_store = {}
    cache_by_alpha = {}

    for alpha in RAW_QUANTILES:
        print(f"Building nested OOF quantile cache for raw quantile={alpha:.2f} ...")
        cache = build_outer_cache(train, alpha, num_cols, cat_cols)
        cache_by_alpha[alpha] = cache
        for target_cov in TARGET_COVERAGES:
            for min_n in MIN_FINE_GROUP_SIZES:
                radii, raw, factor, level, fold, factor_table = evaluate_config(train, cache, target_cov, min_n)
                summary = summarize_radius_model(train, radii, "Confidence_2_0_candidate")
                row = config_selection_row(summary, alpha, target_cov, min_n)
                candidate_rows.append(row)
                result_store[(alpha, target_cov, min_n)] = {
                    "radii": radii, "raw": raw, "factor": factor, "level": level,
                    "fold": fold, "factor_table": factor_table, "summary": summary,
                }

    search = pd.DataFrame(candidate_rows).sort_values(["preferred", "feasible", "selection_score"], ascending=[False, False, True])
    search.to_csv(outdir / "confidence20_candidate_search.csv", index=False)

    # Prefer configs within the desired coverage window, then any feasible config, then best score.
    preferred = search[search["preferred"]]
    feasible = search[search["feasible"]]
    if len(preferred):
        best_row = preferred.sort_values("selection_score").iloc[0]
    elif len(feasible):
        best_row = feasible.sort_values("selection_score").iloc[0]
    else:
        best_row = search.sort_values("selection_score").iloc[0]

    best_key = (float(best_row["raw_quantile"]), float(best_row["target_coverage"]), int(best_row["min_fine_group_n"]))
    best = result_store[best_key]

    # Final OOF comparison.
    best_summary = summarize_radius_model(train, best["radii"], "Confidence_2_0")
    combined_summary = pd.concat([current_summary, best_summary], ignore_index=True)
    combined_summary.to_csv(outdir / "confidence20_oof_summary.csv", index=False)

    oof_out_cols = [address_id, "actual_error_m", "confidence_tier", "broad_conf_group", "evidence_tier", "clean_visit_count", "pin_x", "pin_y", "current_radius_m"]
    for c in ["router52_top1", "router52_top2", "router52_ood", "router52_ood_reason", "top1_adjusted_risk_m", "top2_adjusted_risk_m", "router_risk_ratio", "router_locality_train_count", "router_nearest_grid_feature_distance", "router_grid_extrapolation_score"]:
        if c in train.columns:
            oof_out_cols.append(c)
    oof_out = train[[c for c in oof_out_cols if c in train.columns]].copy()
    oof_out["conf20_raw_radius_m"] = best["raw"]
    oof_out["conf20_calibration_factor"] = best["factor"]
    oof_out["conf20_calibration_level"] = best["level"]
    oof_out["conf20_radius_m"] = best["radii"]
    oof_out["conf20_covered"] = (oof_out["actual_error_m"] <= oof_out["conf20_radius_m"]).astype(int)
    oof_out["current_covered"] = (oof_out["actual_error_m"] <= oof_out["current_radius_m"]).astype(int)
    oof_out["radius_reduction_m"] = oof_out["current_radius_m"] - oof_out["conf20_radius_m"]
    oof_out.to_csv(outdir / "confidence20_oof_predictions.csv", index=False)
    best["factor_table"].to_csv(outdir / "confidence20_oof_calibration_factors.csv", index=False)

    # Train production Confidence 2.0 on all surveyed OOF errors.
    best_alpha, best_target, best_min_n = best_key

    # Cross-fitted raw predictions over all training rows -> calibration ratios for production.
    labels = strat_labels(train)
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + 777)
    cv_raw = pd.Series(np.nan, index=train.index, dtype=float)
    arr = np.arange(len(train))
    for fold, (tr_pos, va_pos) in enumerate(skf.split(arr, labels.to_numpy())):
        tr_idx = train.index[tr_pos]
        va_idx = train.index[va_pos]
        models = fit_broad_models(train.loc[tr_idx], best_alpha, num_cols, cat_cols, RANDOM_STATE + 8000 + fold)
        cv_raw.loc[va_idx] = predict_broad_models(models, train.loc[va_idx])

    prod_factors, prod_factor_table = make_calibration_map(train, cv_raw, best_target, best_min_n)
    prod_factor_table.to_csv(outdir / "confidence20_production_calibration_factors.csv", index=False)

    final_models = fit_broad_models(train, best_alpha, num_cols, cat_cols, RANDOM_STATE + 9999)
    prod_raw = predict_broad_models(final_models, prod)
    prod_radius, prod_factor, prod_level = apply_calibration(prod, prod_raw, prod_factors)

    prod_out_cols = [address_id, "pin_x", "pin_y", "confidence_tier", "broad_conf_group", "evidence_tier", "clean_visit_count", "pin_method", "current_radius_m"]
    for c in ["router52_top1", "router52_top2", "router52_ood", "router52_ood_reason", "top1_adjusted_risk_m", "top2_adjusted_risk_m", "router_risk_ratio", "router_locality_train_count", "router_nearest_grid_feature_distance", "router_grid_extrapolation_score"]:
        if c in prod.columns:
            prod_out_cols.append(c)
    prod_out = prod[[c for c in prod_out_cols if c in prod.columns]].copy()
    prod_out["confidence20_raw_radius_m"] = prod_raw
    prod_out["confidence20_calibration_factor"] = prod_factor
    prod_out["confidence20_calibration_level"] = prod_level
    prod_out["confidence20_radius_m"] = prod_radius
    prod_out.to_csv(outdir / "confidence20_all_addresses.csv", index=False)

    # Group-level OOF diagnostics.
    group_rows = []
    for group_col in ["confidence_tier", "broad_conf_group", "evidence_tier"]:
        if group_col not in train.columns:
            continue
        for g, idx in train.groupby(group_col).groups.items():
            idx = list(idx)
            oldm = coverage_metrics(train.loc[idx, "actual_error_m"], train.loc[idx, "current_radius_m"])
            newm = coverage_metrics(train.loc[idx, "actual_error_m"], best["radii"].loc[idx])
            group_rows.append({
                "group_type": group_col, "group": str(g), "n": len(idx),
                "current_coverage_pct": oldm["coverage_pct"], "conf20_coverage_pct": newm["coverage_pct"],
                "current_median_radius_m": oldm["median_radius_m"], "conf20_median_radius_m": newm["median_radius_m"],
                "median_error_m": newm["median_error_m"],
                "radius_reduction_pct": 100 * (oldm["median_radius_m"] - newm["median_radius_m"]) / max(oldm["median_radius_m"], 1e-9),
            })
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(outdir / "confidence20_group_diagnostics.csv", index=False)

    # Report.
    def row(scope, model):
        x = combined_summary[(combined_summary["scope"] == scope) & (combined_summary["model"] == model)]
        return x.iloc[0].to_dict() if len(x) else {}

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("CreditNirvana PS3 - Confidence 2.0")
    report_lines.append("Architecture 5.2 pins are frozen; only the radius estimator was changed.")
    report_lines.append("=" * 100)
    report_lines.append(f"OOF surveyed rows: {len(train)} | Production addresses: {len(prod)}")
    report_lines.append(f"Selected raw quantile: {best_alpha}")
    report_lines.append(f"Selected conformal target coverage: {best_target}")
    report_lines.append(f"Selected minimum fine-tier calibration size: {best_min_n}")
    report_lines.append("")
    report_lines.append("OOF COMPARISON")
    for scope in ["ALL", "VISITED", "NO_VISIT", "NV_GRID"]:
        old = row(scope, "Current_A52_confidence")
        new = row(scope, "Confidence_2_0")
        if not new:
            continue
        report_lines.append(
            f"{scope:10s} | Current coverage={old.get('coverage_pct', np.nan):6.2f}% radius={old.get('median_radius_m', np.nan):8.2f}m "
            f"| Conf2 coverage={new.get('coverage_pct', np.nan):6.2f}% radius={new.get('median_radius_m', np.nan):8.2f}m "
            f"| actual median error={new.get('median_error_m', np.nan):8.2f}m"
        )
    report_lines.append("")
    report_lines.append("SUCCESS TARGETS")
    report_lines.append("  Coverage target: roughly 88-92% overall / visited / no-visit; NV_grid guardrail >=85% because n=27.")
    report_lines.append("  Radius targets: overall <600m, visited <200m, no-visit <800m, NV_grid <900m when coverage remains honest.")
    report_lines.append("")
    report_lines.append("OUTPUTS")
    for fn in [
        "confidence20_oof_summary.csv", "confidence20_oof_predictions.csv", "confidence20_group_diagnostics.csv",
        "confidence20_candidate_search.csv", "confidence20_oof_calibration_factors.csv",
        "confidence20_production_calibration_factors.csv", "confidence20_all_addresses.csv", "confidence20_model_config.json",
    ]:
        report_lines.append(f"  {fn}")

    (outdir / "confidence20_run_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    config = {
        "architecture": "Confidence_2_0_evidence_conditioned_quantile_plus_hierarchical_conformal",
        "pin_model": "Architecture_5_2_frozen",
        "selected_raw_quantile": best_alpha,
        "selected_target_coverage": best_target,
        "selected_min_fine_group_n": best_min_n,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "coverage_guardrails": {
            "overall_min": MIN_OVERALL_COVERAGE,
            "visited_min": MIN_VISITED_COVERAGE,
            "no_visit_min": MIN_NO_VISIT_COVERAGE,
            "nv_grid_min": MIN_NV_GRID_COVERAGE,
            "preferred_overall_max": MAX_PREFERRED_OVERALL_COVERAGE,
        },
        "production_calibration_factors": prod_factors,
    }
    (outdir / "confidence20_model_config.json").write_text(json.dumps(config, indent=2, default=float), encoding="utf-8")

    print("\nBEST CONFIDENCE 2.0 CONFIG")
    print(best_row.to_string())
    print("\nOOF SUMMARY")
    show = combined_summary[combined_summary["scope"].isin(["ALL", "VISITED", "NO_VISIT", "NV_GRID"])][
        ["scope", "model", "n", "coverage_pct", "median_radius_m", "p90_radius_m", "median_error_m", "p90_error_m", "sharpness_ratio"]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nSaved outputs to: {outdir.resolve()}")



if __name__ == "__main__":
    main()
