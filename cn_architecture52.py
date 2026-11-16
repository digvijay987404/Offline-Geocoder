from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MASTER_PATH = "ps3_out/ps3_master_processed.csv"
SURVEY_PATH = "surveyed_addresses.csv"
OPTIONAL_VISIT_PATH = "ps3_out/visit_features.csv"
OUTPUT_DIR = "stage2_arch5"

SEED = 42
N_FOLDS = 5
CONF_LEVEL = 0.90
MIN_MODEL_ROWS = 25


TAIL_P90_WEIGHT = 1.0
TAIL_MEDIAN_WEIGHT = 0.25
TAIL_MEAN_WEIGHT = 0.05

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

OVERRIDE = {
    "master_address_id": None,
    "survey_address_id": None,
    "truth_x": None,
    "truth_y": None,
    "baseline_x": None,
    "baseline_y": None,
    "pseudo_x": None,
    "pseudo_y": None,
    "poi_x": None,
    "poi_y": None,
    "locality_id": None,
    "visit_x": None,
    "visit_y": None,
    "is_hub2": None,
    "integrity_risk": None,
    "outcome": None,
    "survey_split": None,
}

ROLE_CANDIDATES = {
    "address_id": ["address_id", "addr_id", "addressid", "address_key", "addr_key", "record_id"],
    "address_text": ["address", "address_text", "raw_address", "full_address", "customer_address", "text_raw"],
    "truth_x": ["truth_x", "true_x", "survey_x", "surveyed_x", "ground_truth_x", "gt_x", "actual_x", "x_truth", "x_true", "x"],
    "truth_y": ["truth_y", "true_y", "survey_y", "surveyed_y", "ground_truth_y", "gt_y", "actual_y", "y_truth", "y_true", "y"],
    "baseline_x": ["baseline_x", "commercial_x", "geocoder_x", "commercial_geocoder_x", "base_x", "baseline_pin_x", "geocode_x"],
    "baseline_y": ["baseline_y", "commercial_y", "geocoder_y", "commercial_geocoder_y", "base_y", "baseline_pin_y", "geocode_y"],
    "pseudo_x": ["pseudo_x", "pseudo_truth_x", "pseudo_gt_x", "consensus_x", "clean_median_x", "pseudo_pin_x", "pt_x"],
    "pseudo_y": ["pseudo_y", "pseudo_truth_y", "pseudo_gt_y", "consensus_y", "clean_median_y", "pseudo_pin_y", "pt_y"],
    "poi_x": ["poi_x", "matched_poi_x", "landmark_poi_x", "poi_coord_x", "landmark_x"],
    "poi_y": ["poi_y", "matched_poi_y", "landmark_poi_y", "poi_coord_y", "landmark_y"],
    "locality_id": ["locality_id", "locality", "canonical_locality", "matched_locality", "locality_name"],
    "visit_x": ["visit_x", "checkin_x", "check_in_x", "checkin_coord_x", "gps_x", "visit_easting", "checkin_easting", "agent_x"],
    "visit_y": ["visit_y", "checkin_y", "check_in_y", "checkin_coord_y", "gps_y", "visit_northing", "checkin_northing", "agent_y"],
    "is_hub2": ["is_hub2", "hub2", "is_fake_hub", "fake_hub_flag"],
    "integrity_risk": ["integrity_risk", "visit_integrity_risk", "risk_score", "fraud_risk"],
    "outcome": ["outcome", "visit_outcome", "field_outcome", "result"],
    "survey_split": ["split", "survey_split", "dataset_split", "set", "fold_type", "is_test", "is_holdout", "holdout"],
}

STRUCTURED_NUM_CANDIDATES = [
    "house_no", "house_number", "cross_no", "cross_number", "main_no", "main_number",
    "gali_no", "gali_number", "road_no", "road_number", "ward_no", "ward_number"
]
STRUCTURED_CAT_CANDIDATES = [
    "block", "locality_id", "town_id", "lm_primary", "rel_no_type", "rel_context",
    "poi_id", "locality_method", "town_method", "base_precision"
]
EXTRA_NUM_CANDIDATES = [
    "locality_conf", "lm_conf", "n_fields_found", "script_kn_frac", "script_dev_frac",
    "is_village_style"
]


def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    return out


def find_col(df: pd.DataFrame, role: str, override_key: Optional[str] = None) -> Optional[str]:
    if override_key and OVERRIDE.get(override_key):
        c = norm_col(OVERRIDE[override_key])
        if c not in df.columns:
            raise KeyError(f"Override {override_key}={c!r} not found. Available columns: {list(df.columns)}")
        return c
    for c in ROLE_CANDIDATES.get(role, []):
        if c in df.columns:
            return c
    return None


def require_col(c: Optional[str], role: str, df: pd.DataFrame) -> str:
    if c is None:
        raise KeyError(f"Could not detect {role}. Available columns:\n{list(df.columns)}")
    return c


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) != 0
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def euclid(x1, y1, x2, y2):
    return np.sqrt((np.asarray(x1, float) - np.asarray(x2, float)) ** 2 +
                   (np.asarray(y1, float) - np.asarray(y2, float)) ** 2)


def finite_xy(x, y) -> bool:
    return np.isfinite(float(x)) and np.isfinite(float(y))


def weighted_geometric_median(points: np.ndarray, weights: np.ndarray, max_iter=300, tol=1e-5):
    P = np.asarray(points, float)
    W = np.asarray(weights, float)
    mask = np.all(np.isfinite(P), axis=1) & np.isfinite(W) & (W > 0)
    P, W = P[mask], W[mask]
    if len(P) == 0:
        return np.array([np.nan, np.nan])
    if len(P) == 1:
        return P[0].copy()
    y = np.average(P, axis=0, weights=W)
    for _ in range(max_iter):
        d = np.linalg.norm(P - y, axis=1)
        hit = d < 1e-10
        if hit.any():
            return P[np.argmax(W * hit)]
        ww = W / np.maximum(d, 1e-9)
        y1 = np.sum(P * ww[:, None], axis=0) / ww.sum()
        if np.linalg.norm(y1 - y) < tol:
            return y1
        y = y1
    return y


def onehot_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def usable_num_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            out.append(c)
    return list(dict.fromkeys(out))


def usable_cat_cols(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    out = []
    for c in candidates:
        if c not in df.columns:
            continue
        vals = df[c].replace({"": np.nan})
        if vals.notna().any():
            out.append(c)
    return list(dict.fromkeys(out))


def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric=False):
    transformers = []
    if num_cols:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot_dense()),
            ]),
            cat_cols,
        ))
    if not transformers:
        raise ValueError("No usable model features.")
    return ColumnTransformer(transformers, remainder="drop")


def make_pipeline(kind: str, num_cols: List[str], cat_cols: List[str]):
    if kind == "grid_huber":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
        model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=3000, tol=1e-5)
    elif kind == "resid_hgb":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.045, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=1.0, random_state=SEED
        )
    elif kind == "extra":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        model = ExtraTreesRegressor(
            n_estimators=450, min_samples_leaf=3, max_features=0.85,
            random_state=SEED, n_jobs=-1
        )
    elif kind == "quantile":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        model = GradientBoostingRegressor(
            loss="quantile", alpha=CONF_LEVEL, n_estimators=220,
            learning_rate=0.035, max_depth=2, min_samples_leaf=5,
            random_state=SEED
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("model", model)])


def fit_pair(train: pd.DataFrame, num_cols, cat_cols, tx, ty, kind):
    features = num_cols + cat_cols
    tr = train.loc[train[[tx, ty]].notna().all(axis=1)].copy()
    if len(tr) < MIN_MODEL_ROWS or not features:
        return None, None
    mx = make_pipeline(kind, num_cols, cat_cols)
    my = make_pipeline(kind, num_cols, cat_cols)
    try:
        mx.fit(tr[features], tr[tx])
        my.fit(tr[features], tr[ty])
    except Exception as e:
        # Grid Huber is the only optimization that can occasionally struggle.
        # Fall back to scaled Ridge so the notebook never dies on convergence.
        if kind == "grid_huber":
            pre = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
            mx = Pipeline([("pre", clone(pre)), ("model", Ridge(alpha=10.0))])
            my = Pipeline([("pre", clone(pre)), ("model", Ridge(alpha=10.0))])
            mx.fit(tr[features], tr[tx])
            my.fit(tr[features], tr[ty])
        else:
            raise RuntimeError(f"Failed fitting {kind}: {e}")
    return mx, my


def pair_predict(models, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    out = np.full((len(df), 2), np.nan)
    mx, my = models
    if mx is None or my is None or not features:
        return out
    out[:, 0] = mx.predict(df[features])
    out[:, 1] = my.predict(df[features])
    return out


def conformal_quantile(values: Sequence[float], q=CONF_LEVEL) -> float:
    x = np.sort(np.asarray([v for v in values if np.isfinite(v)], float))
    if len(x) == 0:
        return np.nan
    k = int(math.ceil((len(x) + 1) * q))
    k = min(max(k, 1), len(x))
    return float(x[k - 1])


def summarize(frame: pd.DataFrame, err_col: str, radius_col: Optional[str] = None) -> Dict[str, float]:
    e = pd.to_numeric(frame[err_col], errors="coerce").dropna().to_numpy(float)
    out = {
        "n": int(len(e)),
        "median_error_m": float(np.median(e)) if len(e) else np.nan,
        "p90_error_m": float(np.quantile(e, 0.90)) if len(e) else np.nan,
        "mean_error_m": float(np.mean(e)) if len(e) else np.nan,
        "pct_under_100m": float(np.mean(e <= 100) * 100) if len(e) else np.nan,
        "pct_under_250m": float(np.mean(e <= 250) * 100) if len(e) else np.nan,
        "pct_under_500m": float(np.mean(e <= 500) * 100) if len(e) else np.nan,
    }
    if radius_col and radius_col in frame.columns:
        m = frame[[err_col, radius_col]].notna().all(axis=1)
        if m.any():
            ee = frame.loc[m, err_col].to_numpy(float)
            rr = frame.loc[m, radius_col].to_numpy(float)
            out.update({
                "coverage_pct": float(np.mean(ee <= rr) * 100),
                "median_radius_m": float(np.median(rr)),
                "p90_radius_m": float(np.quantile(rr, 0.90)),
            })
    return out


def build_visit_consensus(addr: pd.DataFrame, master_id: str, px: str, py: str) -> Tuple[pd.DataFrame, str]:
    optional = Path(OPTIONAL_VISIT_PATH)
    if optional.exists():
        try:
            vf = normalize_df_columns(pd.read_csv(optional))
            vmid = find_col(vf, "address_id")
            vx = find_col(vf, "visit_x", "visit_x")
            vy = find_col(vf, "visit_y", "visit_y")
            if vmid and vx and vy:
                vf[vmid] = vf[vmid].astype(str)
                if vmid != master_id:
                    vf = vf.rename(columns={vmid: master_id})
                hub = find_col(vf, "is_hub2", "is_hub2")
                risk = find_col(vf, "integrity_risk", "integrity_risk")
                outcome = find_col(vf, "outcome", "outcome")
                vf[vx] = pd.to_numeric(vf[vx], errors="coerce")
                vf[vy] = pd.to_numeric(vf[vy], errors="coerce")
                v = vf.loc[vf[[vx, vy]].notna().all(axis=1)].copy()
                v["__hub"] = to_bool_series(v[hub]) if hub else False
                if outcome:
                    os_ = v[outcome].astype(str).str.lower().fillna("")
                    v["__untraceable"] = os_.str.contains("not_traceable|not traceable|untraceable", regex=True)
                else:
                    os_ = pd.Series("", index=v.index)
                    v["__untraceable"] = False
                rr = pd.to_numeric(v[risk], errors="coerce").fillna(0.25).clip(0, 1) if risk else pd.Series(0.25, index=v.index)
                w = (1.0 - rr) ** 2
                om = np.ones(len(v), float)
                if outcome:
                    om[os_.str.contains("met_borrower|met borrower", regex=True)] = 1.00
                    om[os_.str.contains("locked_premises|locked premises|locked", regex=True)] = 0.95
                    om[os_.str.contains("shifted|moved", regex=True)] = 0.55
                    om[os_.str.contains("not_traceable|not traceable|untraceable", regex=True)] = 0.05
                v["__w"] = np.maximum(0.01, np.asarray(w) * om)
                v["__clean"] = (~v["__hub"]) & (~v["__untraceable"])

                rows = []
                for aid, g in v.groupby(master_id):
                    cg = g[g["__clean"]]
                    if len(cg):
                        P = cg[[vx, vy]].to_numpy(float)
                        W = cg["__w"].to_numpy(float)
                        geom = weighted_geometric_median(P, W)
                        dist = np.linalg.norm(P - geom, axis=1)
                        spread = float(np.median(dist)) if len(dist) else np.nan
                        medrisk = float(pd.to_numeric(cg[risk], errors="coerce").median()) if risk else 0.25
                    else:
                        geom = np.array([np.nan, np.nan])
                        spread = medrisk = np.nan
                    rows.append({
                        master_id: str(aid),
                        "clean_visit_count": int(len(cg)),
                        "all_visit_count": int(len(g)),
                        "visit_spread_m": spread,
                        "median_integrity_risk": medrisk,
                        "visit_pin_x": geom[0],
                        "visit_pin_y": geom[1],
                    })
                va = pd.DataFrame(rows)
                out = addr.merge(va, how="left", on=master_id)
                out["clean_visit_count"] = pd.to_numeric(out["clean_visit_count"], errors="coerce").fillna(0).astype(int)
                out["all_visit_count"] = pd.to_numeric(out["all_visit_count"], errors="coerce").fillna(0).astype(int)
                return out, f"raw_visit_file:{OPTIONAL_VISIT_PATH}"
        except Exception as e:
            print(f"NOTE: raw visit file could not be used; falling back to pt_x/pt_y. Reason: {e}")

    # Fallback: Stage-1 pseudo-truth is already the clean visit consensus.
    out = addr.copy()
    cc = "clean_visits" if "clean_visits" in out.columns else ("pt_k" if "pt_k" in out.columns else None)
    tc = "total_visits" if "total_visits" in out.columns else cc
    if cc:
        out["clean_visit_count"] = pd.to_numeric(out[cc], errors="coerce").fillna(0).astype(int)
    else:
        out["clean_visit_count"] = out[[px, py]].notna().all(axis=1).astype(int)
    out["all_visit_count"] = pd.to_numeric(out[tc], errors="coerce").fillna(0).astype(int) if tc else out["clean_visit_count"]
    out["visit_spread_m"] = pd.to_numeric(out["pt_spread_m"], errors="coerce") if "pt_spread_m" in out.columns else np.nan
    out["median_integrity_risk"] = pd.to_numeric(out["avg_integrity_risk"], errors="coerce") if "avg_integrity_risk" in out.columns else np.nan
    out["visit_pin_x"] = pd.to_numeric(out[px], errors="coerce")
    out["visit_pin_y"] = pd.to_numeric(out[py], errors="coerce")
    return out, "address_level_pt_consensus"


def build_component_predictions(addr: pd.DataFrame, pseudo_train: pd.DataFrame,
                                px: str, py: str, bx: Optional[str], by: Optional[str],
                                poix: Optional[str], poiy: Optional[str], loc: Optional[str]):
    # Drop all-null features before pipelines; this removes lm_conf warning.
    grid_num = usable_num_cols(pseudo_train, STRUCTURED_NUM_CANDIDATES)
    grid_cat = usable_cat_cols(pseudo_train, STRUCTURED_CAT_CANDIDATES)

    full_num_candidates = STRUCTURED_NUM_CANDIDATES + EXTRA_NUM_CANDIDATES
    if bx: full_num_candidates.append(bx)
    if by: full_num_candidates.append(by)
    if poix: full_num_candidates.append(poix)
    if poiy: full_num_candidates.append(poiy)
    full_num = usable_num_cols(pseudo_train, full_num_candidates)
    full_cat = usable_cat_cols(pseudo_train, STRUCTURED_CAT_CANDIDATES)

    # 1) robust street-grid model
    grid_models = fit_pair(pseudo_train, grid_num, grid_cat, px, py, "grid_huber")
    grid_pred = pair_predict(grid_models, addr, grid_num + grid_cat)

    # 2) residual GBM around commercial geocoder
    resid_pred = np.full((len(addr), 2), np.nan)
    if bx and by and bx in pseudo_train.columns and by in pseudo_train.columns:
        tr = pseudo_train.loc[pseudo_train[[bx, by, px, py]].notna().all(axis=1)].copy()
        tr["__dx"] = tr[px] - tr[bx]
        tr["__dy"] = tr[py] - tr[by]
        rx, ry = fit_pair(tr, full_num, full_cat, "__dx", "__dy", "resid_hgb")
        if rx is not None:
            X = addr[full_num + full_cat]
            dx = rx.predict(X); dy = ry.predict(X)
            base_ok = addr[[bx, by]].notna().all(axis=1).to_numpy()
            resid_pred[base_ok, 0] = addr.loc[base_ok, bx].to_numpy(float) + dx[base_ok]
            resid_pred[base_ok, 1] = addr.loc[base_ok, by].to_numpy(float) + dy[base_ok]

    # 3) A3 direct ExtraTrees
    extra_models = fit_pair(pseudo_train, full_num, full_cat, px, py, "extra")
    extra_pred = pair_predict(extra_models, addr, full_num + full_cat)

    # 4) POI and 5) baseline
    poi_pred = np.full((len(addr), 2), np.nan)
    if poix and poiy:
        poi_pred[:, 0] = pd.to_numeric(addr[poix], errors="coerce")
        poi_pred[:, 1] = pd.to_numeric(addr[poiy], errors="coerce")

    baseline_pred = np.full((len(addr), 2), np.nan)
    if bx and by:
        baseline_pred[:, 0] = pd.to_numeric(addr[bx], errors="coerce")
        baseline_pred[:, 1] = pd.to_numeric(addr[by], errors="coerce")

    # fallback locality centroid from pseudo-truth training addresses
    locality_pred = np.full((len(addr), 2), np.nan)
    if loc and loc in addr.columns and loc in pseudo_train.columns:
        centers = pseudo_train.groupby(loc)[[px, py]].median()
        for i, val in enumerate(addr[loc]):
            if pd.notna(val) and val in centers.index:
                locality_pred[i] = centers.loc[val].to_numpy(float)
    global_center = pseudo_train[[px, py]].median().to_numpy(float)

    components = {
        "grid": grid_pred,
        "resid": resid_pred,
        "extra": extra_pred,
        "poi": poi_pred,
        "baseline": baseline_pred,
    }
    meta = {
        "grid_num": grid_num,
        "grid_cat": grid_cat,
        "full_num": full_num,
        "full_cat": full_cat,
    }
    return components, locality_pred, global_center, meta


def component_disagreement(component_points: Dict[str, np.ndarray], i: int) -> Tuple[float, int]:
    pts = []
    for arr in component_points.values():
        p = arr[i]
        if np.all(np.isfinite(p)):
            pts.append(p)
    if len(pts) < 2:
        return (250.0 if len(pts) == 1 else 600.0), len(pts)
    P = np.vstack(pts)
    d = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            d.append(float(np.linalg.norm(P[a] - P[b])))
    return float(np.median(d)), len(pts)


def inverse_error_weights(cal: pd.DataFrame, source_names: Sequence[str], tx: str, ty: str) -> Dict[str, float]:
    med = {}
    for name in source_names:
        xcol, ycol = f"cand_{name}_x", f"cand_{name}_y"
        m = cal[[xcol, ycol, tx, ty]].notna().all(axis=1)
        if m.sum() >= 3:
            err = euclid(cal.loc[m, xcol], cal.loc[m, ycol], cal.loc[m, tx], cal.loc[m, ty])
            med[name] = float(np.median(err))
    defaults = {"grid": 0.18, "resid": 0.18, "extra": 0.22, "poi": 0.30, "baseline": 0.12}
    if len(med) < 2:
        return defaults
    inv = {k: 1.0 / max(v, 20.0) for k, v in med.items()}
    w = {k: inv.get(k, 0.02) for k in defaults}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def weighted_component_ensemble(frame: pd.DataFrame, weights: Dict[str, float], source_names: Sequence[str]) -> np.ndarray:
    out = np.full((len(frame), 2), np.nan)
    for j, (_, row) in enumerate(frame.iterrows()):
        pts, ws = [], []
        for name in source_names:
            p = np.array([row.get(f"cand_{name}_x", np.nan), row.get(f"cand_{name}_y", np.nan)], float)
            if np.all(np.isfinite(p)) and weights.get(name, 0) > 0:
                pts.append(p); ws.append(weights[name])
        if pts:
            P = np.vstack(pts); W = np.asarray(ws, float); W /= W.sum()
            out[j] = np.sum(P * W[:, None], axis=0)
    return out


def fit_tail_gate(cal_no_visit: pd.DataFrame, base_xy: np.ndarray, tx: str, ty: str):
    """Tune how strongly high-disagreement A4 predictions are pulled toward A3 ExtraTrees."""
    if len(cal_no_visit) < 8:
        return {"alpha_max": 0.5, "q_lo": 0.50, "q_hi": 0.90, "d_lo": 200.0, "d_hi": 800.0}

    d = pd.to_numeric(cal_no_visit["component_disagreement_m"], errors="coerce").fillna(600).to_numpy(float)
    extra = cal_no_visit[["cand_extra_x", "cand_extra_y"]].to_numpy(float)
    truth = cal_no_visit[[tx, ty]].to_numpy(float)
    valid_extra = np.all(np.isfinite(extra), axis=1)

    best = None
    for q_lo in [0.40, 0.50, 0.60]:
        for q_hi in [0.80, 0.90, 0.95]:
            if q_hi <= q_lo:
                continue
            dlo, dhi = np.quantile(d, [q_lo, q_hi])
            if dhi <= dlo + 1e-9:
                dhi = dlo + 1.0
            risk = np.clip((d - dlo) / (dhi - dlo), 0, 1)
            for alpha_max in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                alpha = alpha_max * risk
                pred = base_xy.copy()
                ok = valid_extra & np.all(np.isfinite(pred), axis=1)
                pred[ok] = (1 - alpha[ok, None]) * pred[ok] + alpha[ok, None] * extra[ok]
                e = np.linalg.norm(pred - truth, axis=1)
                e = e[np.isfinite(e)]
                if len(e) < 5:
                    continue
                score = (TAIL_P90_WEIGHT * np.quantile(e, 0.90) +
                         TAIL_MEDIAN_WEIGHT * np.median(e) +
                         TAIL_MEAN_WEIGHT * np.mean(e))
                candidate = (score, alpha_max, q_lo, q_hi, float(dlo), float(dhi),
                             float(np.median(e)), float(np.quantile(e, 0.90)))
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        return {"alpha_max": 0.5, "q_lo": 0.50, "q_hi": 0.90, "d_lo": 200.0, "d_hi": 800.0}
    return {
        "alpha_max": best[1], "q_lo": best[2], "q_hi": best[3],
        "d_lo": best[4], "d_hi": best[5],
        "cal_median_m": best[6], "cal_p90_m": best[7],
    }


def apply_tail_gate(frame: pd.DataFrame, base_xy: np.ndarray, gate: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    d = pd.to_numeric(frame["component_disagreement_m"], errors="coerce").fillna(600).to_numpy(float)
    extra = frame[["cand_extra_x", "cand_extra_y"]].to_numpy(float)
    denom = max(gate["d_hi"] - gate["d_lo"], 1.0)
    risk = np.clip((d - gate["d_lo"]) / denom, 0, 1)
    alpha = gate["alpha_max"] * risk
    out = base_xy.copy()
    ok = np.all(np.isfinite(out), axis=1) & np.all(np.isfinite(extra), axis=1)
    out[ok] = (1 - alpha[ok, None]) * out[ok] + alpha[ok, None] * extra[ok]
    return out, alpha


def fallback_no_visit(frame: pd.DataFrame, xy: np.ndarray, locality_lookup: Dict[str, np.ndarray], global_center: np.ndarray,
                      loc: Optional[str], bx: Optional[str], by: Optional[str]) -> Tuple[np.ndarray, List[str]]:
    out = xy.copy()
    methods = []
    for j, (_, row) in enumerate(frame.iterrows()):
        if np.all(np.isfinite(out[j])):
            methods.append("a4_ensemble+a3_tail_gate")
            continue
        if bx and by and pd.notna(row.get(bx)) and pd.notna(row.get(by)):
            out[j] = [float(row[bx]), float(row[by])]
            methods.append("commercial_fallback")
        elif loc and pd.notna(row.get(loc)) and str(row.get(loc)) in locality_lookup:
            out[j] = locality_lookup[str(row.get(loc))]
            methods.append("locality_fallback")
        else:
            out[j] = global_center
            methods.append("global_centroid_fallback")
    return out, methods


def confidence_feature_lists(df: pd.DataFrame, loc: Optional[str]):
    nums = usable_num_cols(df, [
        "clean_visit_count", "all_visit_count", "visit_spread_m", "median_integrity_risk",
        "component_disagreement_m", "component_source_count", "tail_gate_alpha",
        "locality_conf", "n_fields_found", "script_kn_frac", "script_dev_frac",
        "is_village_style", "dist_pred_to_baseline_m", "dist_pred_to_poi_m"
    ])
    cats = usable_cat_cols(df, ["evidence_tier", loc] if loc else ["evidence_tier"])
    return nums, cats


def fit_confidence_model(train: pd.DataFrame, err_col: str, loc: Optional[str]):
    """A3-style 90th-quantile model with a held-out conformal correction.

    The quantile regressor learns conditional error size. A separate calibration
    slice estimates an additive correction, which is much more reliable than
    calibrating on the same rows used to fit the quantile model.
    """
    num, cat = confidence_feature_lists(train, loc)
    features = num + cat
    tr = train.loc[train[err_col].notna()].copy()
    if len(tr) < 30 or not features:
        return None, num, cat, conformal_quantile(tr[err_col])

    indices = np.arange(len(tr))
    strat = (tr["clean_visit_count"].to_numpy() > 0).astype(int) if "clean_visit_count" in tr.columns else None
    try:
        fit_idx, cal_idx = train_test_split(
            indices, test_size=0.25, random_state=SEED, stratify=strat
        )
    except ValueError:
        fit_idx, cal_idx = train_test_split(indices, test_size=0.25, random_state=SEED)

    qfit = tr.iloc[fit_idx]
    qcal = tr.iloc[cal_idx]
    model = make_pipeline("quantile", num, cat)
    model.fit(qfit[features], qfit[err_col])

    raw_cal = np.maximum(model.predict(qcal[features]), 10.0)
    nonconformity = np.maximum(qcal[err_col].to_numpy(float) - raw_cal, 0.0)
    additive = conformal_quantile(nonconformity)
    if not np.isfinite(additive):
        additive = 0.0
    return model, num, cat, float(additive)


def predict_radius(model_pack, df: pd.DataFrame) -> np.ndarray:
    model, num, cat, correction = model_pack
    if model is None:
        return np.full(len(df), max(float(correction) if np.isfinite(correction) else 500.0, 10.0))
    raw = np.maximum(model.predict(df[num + cat]), 10.0)
    return np.maximum(raw + correction, 10.0)


OUTPUT_DIR = "stage2_arch51"
ROUTER_FOLDS = 5
ROUTER_SOURCES = ["grid", "resid", "extra", "baseline"]
ROUTER_MIN_ROWS = 80


ROUTER_P90_WEIGHT = 1.0
ROUTER_MEDIAN_WEIGHT = 0.25
ROUTER_MEAN_WEIGHT = 0.05
ROUTER_CATASTROPHIC_WEIGHT = 120.0  


def add_candidate_columns(frame: pd.DataFrame, components: Dict[str, np.ndarray]) -> pd.DataFrame:
    out = frame.copy()
    for name, arr in components.items():
        out[f"cand_{name}_x"] = arr[:, 0]
        out[f"cand_{name}_y"] = arr[:, 1]
    return out


def build_support_features(reference: pd.DataFrame, target: pd.DataFrame,
                           grid_num: Sequence[str], loc: Optional[str]) -> pd.DataFrame:
    """Quantify how well each target address is supported by pseudo-truth training data.

    The features are deliberately address-only: no true coordinates are used.
    They measure locality density, street-field coverage, extrapolation beyond the
    known locality range, and nearest distance in normalized street-number space.
    """
    out = pd.DataFrame(index=target.index)
    grid_num = [c for c in grid_num if c in target.columns and c in reference.columns]

    out["router_grid_fields_count"] = 0.0
    for c in grid_num:
        out["router_grid_fields_count"] += pd.to_numeric(target[c], errors="coerce").notna().astype(float)

    if loc and loc in reference.columns and loc in target.columns:
        counts = reference.groupby(loc).size().to_dict()
        out["router_locality_train_count"] = target[loc].map(counts).fillna(0).astype(float)
    else:
        out["router_locality_train_count"] = float(len(reference))

    extrap_count = []
    extrap_score = []
    nearest_dist = []

    if loc and loc in reference.columns:
        groups = {str(k): g for k, g in reference.groupby(loc, dropna=False)}
    else:
        groups = {"__all__": reference}

    for _, row in target.iterrows():
        if loc and loc in target.columns and pd.notna(row.get(loc)):
            ref = groups.get(str(row.get(loc)), reference.iloc[0:0])
        else:
            ref = reference
        if len(ref) == 0:
            ref = reference

        present = []
        scales = {}
        medians = {}
        ex_count = 0
        ex_score = 0.0

        for c in grid_num:
            tv = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            if not np.isfinite(tv):
                continue
            rv = pd.to_numeric(ref[c], errors="coerce").dropna().to_numpy(float)
            if len(rv) < 3:
                continue
            q25, q75 = np.quantile(rv, [0.25, 0.75])
            scale = max(float(q75 - q25), float(np.std(rv)), 1.0)
            lo, hi = float(np.min(rv)), float(np.max(rv))
            if tv < lo:
                ex_count += 1
                ex_score += (lo - tv) / scale
            elif tv > hi:
                ex_count += 1
                ex_score += (tv - hi) / scale
            present.append(c)
            scales[c] = scale
            medians[c] = float(np.median(rv))

        extrap_count.append(float(ex_count))
        extrap_score.append(float(ex_score))

        if not present or len(ref) == 0:
            nearest_dist.append(np.nan)
            continue

        
        cols = present
        R = np.zeros((len(ref), len(cols)), float)
        T = np.zeros(len(cols), float)
        for j, c in enumerate(cols):
            vals = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
            vals = np.where(np.isfinite(vals), vals, medians[c])
            R[:, j] = (vals - medians[c]) / scales[c]
            tv = float(pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0])
            T[j] = (tv - medians[c]) / scales[c]
        d = np.sqrt(np.mean((R - T[None, :]) ** 2, axis=1))
        nearest_dist.append(float(np.min(d)) if len(d) else np.nan)

    out["router_grid_extrapolation_count"] = extrap_count
    out["router_grid_extrapolation_score"] = extrap_score
    out["router_nearest_grid_feature_distance"] = nearest_dist
    return out


def add_router_meta_features(frame: pd.DataFrame, bx: Optional[str], by: Optional[str]) -> pd.DataFrame:
    out = frame.copy()
    pairs = [
        ("grid", "extra"), ("grid", "resid"), ("grid", "baseline"),
        ("extra", "resid"), ("extra", "baseline"), ("resid", "baseline"),
    ]
    dcols = []
    for a, b in pairs:
        c = f"router_d_{a}_{b}_m"
        ax, ay = f"cand_{a}_x", f"cand_{a}_y"
        bxcol, bycol = f"cand_{b}_x", f"cand_{b}_y"
        if all(x in out.columns for x in [ax, ay, bxcol, bycol]):
            ok = out[[ax, ay, bxcol, bycol]].notna().all(axis=1)
            out[c] = np.nan
            if ok.any():
                out.loc[ok, c] = euclid(out.loc[ok, ax], out.loc[ok, ay], out.loc[ok, bxcol], out.loc[ok, bycol])
        else:
            out[c] = np.nan
        dcols.append(c)

    out["router_component_disagreement_m"] = out[dcols].median(axis=1, skipna=True)
    out["router_source_count"] = 0.0
    for s in ROUTER_SOURCES:
        ok = out[[f"cand_{s}_x", f"cand_{s}_y"]].notna().all(axis=1) if f"cand_{s}_x" in out.columns else pd.Series(False, index=out.index)
        out[f"router_has_{s}"] = ok.astype(float)
        out["router_source_count"] += ok.astype(float)

    if bx and by and bx in out.columns and by in out.columns:
        # Raw commercial coordinates help the risk model learn location-specific regimes.
        out["router_baseline_x"] = pd.to_numeric(out[bx], errors="coerce")
        out["router_baseline_y"] = pd.to_numeric(out[by], errors="coerce")
    return out


def router_feature_lists(df: pd.DataFrame, loc: Optional[str]) -> Tuple[List[str], List[str]]:
    num_candidates = [
        "router_grid_fields_count", "router_locality_train_count",
        "router_grid_extrapolation_count", "router_grid_extrapolation_score",
        "router_nearest_grid_feature_distance", "router_component_disagreement_m",
        "router_source_count", "router_d_grid_extra_m", "router_d_grid_resid_m",
        "router_d_grid_baseline_m", "router_d_extra_resid_m",
        "router_d_extra_baseline_m", "router_d_resid_baseline_m",
        "router_has_grid", "router_has_resid", "router_has_extra", "router_has_baseline",
        "router_baseline_x", "router_baseline_y", "locality_conf", "n_fields_found",
        "script_kn_frac", "script_dev_frac", "is_village_style",
    ] + STRUCTURED_NUM_CANDIDATES
    cat_candidates = [
        loc, "locality_method", "town_id", "town_method", "base_precision", "block", "rel_no_type", "rel_context"
    ] if loc else ["locality_method", "town_id", "town_method", "base_precision", "block", "rel_no_type", "rel_context"]
    return usable_num_cols(df, num_candidates), usable_cat_cols(df, [c for c in cat_candidates if c])


def build_pseudo_oof_router_data(addr: pd.DataFrame, pseudo_train: pd.DataFrame,
                                 px: str, py: str, bx: Optional[str], by: Optional[str],
                                 poix: Optional[str], poiy: Optional[str], loc: Optional[str]) -> pd.DataFrame:
    """Create honest simulated-no-visit training examples for the safety controller.

    Every pseudo-truth address is predicted out-of-fold after its visit truth is hidden.
    POI coordinates are blanked on the target side so the router learns the exact NV_grid
    regime we are trying to fix rather than learning from easier POI-anchored cases.
    """
    kf = KFold(n_splits=ROUTER_FOLDS, shuffle=True, random_state=SEED)
    chunks = []
    p = pseudo_train.reset_index(drop=True).copy()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(p), 1):
        ref = p.iloc[tr_idx].copy()
        val_true = p.iloc[va_idx].copy()
        val_sim = val_true.copy()
        if poix and poix in val_sim.columns:
            val_sim[poix] = np.nan
        if poiy and poiy in val_sim.columns:
            val_sim[poiy] = np.nan
        if "poi_id" in val_sim.columns:
            val_sim["poi_id"] = np.nan

        comps, _, _, meta = build_component_predictions(val_sim, ref, px, py, bx, by, poix, poiy, loc)
        f = add_candidate_columns(val_sim, comps)
        supp = build_support_features(ref, val_sim, meta["grid_num"], loc)
        for c in supp.columns:
            f[c] = supp[c].to_numpy()
        f = add_router_meta_features(f, bx, by)
        f["router_fold"] = fold
        f["router_truth_x"] = val_true[px].to_numpy(float)
        f["router_truth_y"] = val_true[py].to_numpy(float)

        for s in ROUTER_SOURCES:
            ok = f[[f"cand_{s}_x", f"cand_{s}_y", "router_truth_x", "router_truth_y"]].notna().all(axis=1)
            f[f"router_error_{s}_m"] = np.nan
            if ok.any():
                f.loc[ok, f"router_error_{s}_m"] = euclid(
                    f.loc[ok, f"cand_{s}_x"], f.loc[ok, f"cand_{s}_y"],
                    f.loc[ok, "router_truth_x"], f.loc[ok, "router_truth_y"]
                )
        # Only examples that actually have some grid/street information are relevant.
        f = f.loc[f["router_grid_fields_count"] > 0].copy()
        chunks.append(f)

    if not chunks:
        raise RuntimeError("Could not construct pseudo-truth OOF router training data.")
    return pd.concat(chunks, ignore_index=True)


def fit_risk_models(train: pd.DataFrame, sources: Sequence[str], num: List[str], cat: List[str]):
    features = num + cat
    models = {}
    medians = {}
    for s in sources:
        ycol = f"router_error_{s}_m"
        tr = train.loc[train[ycol].notna()].copy()
        if len(tr) < 30:
            models[s] = None
            medians[s] = 600.0
            continue
        y = np.log1p(np.maximum(tr[ycol].to_numpy(float), 0.0))
        model = make_pipeline("extra", num, cat)
        model.fit(tr[features], y)
        models[s] = model
        medians[s] = float(np.median(tr[ycol]))
    inv = {s: 1.0 / max(medians.get(s, 600.0), 25.0) for s in sources}
    z = sum(inv.values()) or 1.0
    priors = {s: inv[s] / z for s in sources}
    return models, priors


def predict_risks(models: Dict[str, object], frame: pd.DataFrame,
                  sources: Sequence[str], features: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for s in sources:
        m = models.get(s)
        if m is None:
            out[s] = np.full(len(frame), 800.0)
        else:
            pred = np.expm1(m.predict(frame[features]))
            out[s] = np.clip(pred, 5.0, 5000.0)
    return out


def router_blend(frame: pd.DataFrame, risk_pred: Dict[str, np.ndarray], priors: Dict[str, float],
                 power: float, baseline_floor: float, extrap_penalty: float):
    n = len(frame)
    pred = np.full((n, 2), np.nan)
    weights = {s: np.zeros(n, float) for s in ROUTER_SOURCES}

    extrap = pd.to_numeric(frame.get("router_grid_extrapolation_score", 0.0), errors="coerce").fillna(0).to_numpy(float)

    for i, (_, row) in enumerate(frame.iterrows()):
        raw = {}
        pts = {}
        for s in ROUTER_SOURCES:
            p = np.array([row.get(f"cand_{s}_x", np.nan), row.get(f"cand_{s}_y", np.nan)], float)
            if not np.all(np.isfinite(p)):
                continue
            pts[s] = p
            risk = max(float(risk_pred[s][i]), 10.0)
            score = float(priors.get(s, 0.01)) / (risk ** power)
            if s == "grid":
                score *= math.exp(-extrap_penalty * min(float(extrap[i]), 8.0))
            raw[s] = max(score, 1e-12)
        if not raw:
            continue
        z = sum(raw.values())
        w = {s: raw[s] / z for s in raw}

        # A small commercial-baseline floor protects unsupported extrapolation cases.
        if "baseline" in w and baseline_floor > 0 and w["baseline"] < baseline_floor:
            other = 1.0 - w["baseline"]
            if other > 1e-12:
                scale = (1.0 - baseline_floor) / other
                for s in list(w):
                    if s != "baseline":
                        w[s] *= scale
                w["baseline"] = baseline_floor

        pp = np.zeros(2, float)
        for s, ww in w.items():
            pp += ww * pts[s]
            weights[s][i] = ww
        pred[i] = pp
    return pred, weights


def router_objective(errors: np.ndarray) -> float:
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return np.inf
    catastrophic_rate = float(np.mean(e > 1000.0))
    return (ROUTER_P90_WEIGHT * float(np.quantile(e, 0.90)) +
            ROUTER_MEDIAN_WEIGHT * float(np.median(e)) +
            ROUTER_MEAN_WEIGHT * float(np.mean(e)) +
            ROUTER_CATASTROPHIC_WEIGHT * catastrophic_rate)


def tune_and_fit_router(router_oof: pd.DataFrame, loc: Optional[str]):
    num, cat = router_feature_lists(router_oof, loc)
    features = num + cat
    if len(router_oof) < ROUTER_MIN_ROWS or not features:
        raise RuntimeError(f"Insufficient router training data: {len(router_oof)} rows, {len(features)} features")

    idx = np.arange(len(router_oof))
    fit_idx, val_idx = train_test_split(idx, test_size=0.25, random_state=SEED)
    fit = router_oof.iloc[fit_idx].copy()
    val = router_oof.iloc[val_idx].copy()

    models, priors = fit_risk_models(fit, ROUTER_SOURCES, num, cat)
    risks = predict_risks(models, val, ROUTER_SOURCES, features)
    truth = val[["router_truth_x", "router_truth_y"]].to_numpy(float)

    best = None
    trials = []
    for power in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for floor in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            for penalty in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
                p, _ = router_blend(val, risks, priors, power, floor, penalty)
                e = np.linalg.norm(p - truth, axis=1)
                score = router_objective(e)
                row = {
                    "score": score, "power": power, "baseline_floor": floor,
                    "extrap_penalty": penalty, "median_m": float(np.nanmedian(e)),
                    "p90_m": float(np.nanquantile(e, 0.90)), "mean_m": float(np.nanmean(e)),
                    "catastrophic_rate": float(np.nanmean(e > 1000.0)),
                }
                trials.append(row)
                if best is None or score < best["score"]:
                    best = row

    # Refit risk models on all honest pseudo-OOF examples after tuning.
    final_models, final_priors = fit_risk_models(router_oof, ROUTER_SOURCES, num, cat)
    return {
        "models": final_models,
        "priors": final_priors,
        "num": num,
        "cat": cat,
        "features": features,
        "params": {
            "power": float(best["power"]),
            "baseline_floor": float(best["baseline_floor"]),
            "extrap_penalty": float(best["extrap_penalty"]),
        },
        "tuning_best": best,
        "tuning_trials": pd.DataFrame(trials).sort_values("score").reset_index(drop=True),
    }


def apply_router_pack(pack: Dict, frame: pd.DataFrame):
    risks = predict_risks(pack["models"], frame, ROUTER_SOURCES, pack["features"])
    p, weights = router_blend(
        frame, risks, pack["priors"],
        pack["params"]["power"], pack["params"]["baseline_floor"], pack["params"]["extrap_penalty"]
    )
    return p, weights, risks



A52_OUTPUT_DIR = "stage2_arch52"
A52_ROUTER_CV_FOLDS = 3


A52_P90_WEIGHT = 1.0
A52_CVAR90_WEIGHT = 0.35
A52_MEDIAN_WEIGHT = 0.10
A52_MEAN_WEIGHT = 0.02
A52_CATASTROPHIC_WEIGHT = 500.0


def a52_cvar(errors: np.ndarray, q: float = 0.90) -> float:
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return np.inf
    cut = float(np.quantile(e, q))
    tail = e[e >= cut]
    return float(np.mean(tail)) if len(tail) else cut


def a52_router_objective(errors: np.ndarray) -> float:
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return np.inf
    p90 = float(np.quantile(e, 0.90))
    med = float(np.median(e))
    mean = float(np.mean(e))
    cvar = a52_cvar(e, 0.90)
    catastrophic_rate = float(np.mean(e > 1000.0))
    return (A52_P90_WEIGHT * p90 +
            A52_CVAR90_WEIGHT * cvar +
            A52_MEDIAN_WEIGHT * med +
            A52_MEAN_WEIGHT * mean +
            A52_CATASTROPHIC_WEIGHT * catastrophic_rate)


def a52_risk_cv_predictions(router_oof: pd.DataFrame, loc: Optional[str]):
    """Create honest CV predictions of each expert's error risk.

    Hyperparameters for expert routing are tuned only against these OOF risk
    predictions, never against the surveyed evaluation rows.
    """
    num, cat = router_feature_lists(router_oof, loc)
    features = num + cat
    if len(router_oof) < ROUTER_MIN_ROWS or not features:
        raise RuntimeError(
            f"Insufficient router training data: {len(router_oof)} rows, {len(features)} features"
        )

    kf = KFold(n_splits=A52_ROUTER_CV_FOLDS, shuffle=True, random_state=SEED + 17)
    n = len(router_oof)
    risk_oof = {s: np.full(n, np.nan) for s in ROUTER_SOURCES}

    for tr_idx, va_idx in kf.split(np.arange(n)):
        tr = router_oof.iloc[tr_idx].copy()
        va = router_oof.iloc[va_idx].copy()
        models, _ = fit_risk_models(tr, ROUTER_SOURCES, num, cat)
        rr = predict_risks(models, va, ROUTER_SOURCES, features)
        for s in ROUTER_SOURCES:
            risk_oof[s][va_idx] = rr[s]

    # Global priors are computed only from the pseudo-OOF expert errors.
    medians = {}
    for s in ROUTER_SOURCES:
        vals = pd.to_numeric(router_oof[f"router_error_{s}_m"], errors="coerce").dropna().to_numpy(float)
        medians[s] = float(np.median(vals)) if len(vals) else 600.0
    inv = {s: 1.0 / max(medians[s], 25.0) for s in ROUTER_SOURCES}
    z = sum(inv.values()) or 1.0
    priors = {s: inv[s] / z for s in ROUTER_SOURCES}
    return risk_oof, priors, num, cat, features


def a52_nearest_threshold(frame: pd.DataFrame, q: float) -> float:
    if q > 1.0:
        return np.inf
    vals = pd.to_numeric(
        frame.get("router_nearest_grid_feature_distance", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna().to_numpy(float)
    if len(vals) == 0:
        return np.inf
    return float(np.quantile(vals, q))


def a52_select_route(
    frame: pd.DataFrame,
    risk_pred: Dict[str, np.ndarray],
    priors: Dict[str, float],
    params: Dict[str, object],
):
    """Top-1/top-2 expert router with an explicit OOD mode.

    Unlike Architecture 5.1, this function never blends all four experts.
    It either selects the predicted safest expert or blends only the two safest.
    """
    n = len(frame)
    pred = np.full((n, 2), np.nan)
    weights = {s: np.zeros(n, float) for s in ROUTER_SOURCES}
    adjusted = {s: np.full(n, np.nan) for s in ROUTER_SOURCES}
    chosen1 = np.array([""] * n, dtype=object)
    chosen2 = np.array([""] * n, dtype=object)
    ood_flag = np.zeros(n, dtype=bool)
    ood_reason = np.array([""] * n, dtype=object)

    count = pd.to_numeric(frame.get("router_locality_train_count", 0.0), errors="coerce").fillna(0).to_numpy(float)
    nearest = pd.to_numeric(frame.get("router_nearest_grid_feature_distance", np.nan), errors="coerce").to_numpy(float)
    extrap = pd.to_numeric(frame.get("router_grid_extrapolation_score", 0.0), errors="coerce").fillna(0).to_numpy(float)

    prior_strength = float(params["prior_strength"])
    top1_ratio = float(params["top1_ratio"])
    top2_power = float(params["top2_power"])
    support_penalty = float(params["support_penalty"])
    count_thr = float(params["ood_count_threshold"])
    nearest_thr = float(params["ood_nearest_threshold"])
    extrap_thr = float(params["ood_extrap_threshold"])
    ood_policy = str(params["ood_policy"])

    for i, (_, row) in enumerate(frame.iterrows()):
        pts = {}
        arisk = {}
        for s in ROUTER_SOURCES:
            p = np.array([row.get(f"cand_{s}_x", np.nan), row.get(f"cand_{s}_y", np.nan)], float)
            if not np.all(np.isfinite(p)):
                continue
            pts[s] = p
            r = max(float(risk_pred[s][i]), 5.0)
            prior = max(float(priors.get(s, 0.25)), 1e-4)
            r_adj = r / (prior ** prior_strength)


            low_support_term = 1.0 / math.sqrt(max(count[i], 0.0) + 1.0)
            if s == "grid":
                r_adj *= 1.0 + support_penalty * (min(extrap[i], 8.0) + 2.0 * low_support_term)
            elif s == "resid":
                r_adj *= 1.0 + 0.5 * support_penalty * (min(extrap[i], 8.0) + low_support_term)
            arisk[s] = max(float(r_adj), 5.0)
            adjusted[s][i] = arisk[s]

        if not pts:
            continue

        reasons = []
        if count[i] <= count_thr:
            reasons.append("low_locality_support")
        if np.isfinite(nearest[i]) and nearest[i] >= nearest_thr:
            reasons.append("far_grid_support")
        if extrap[i] >= extrap_thr:
            reasons.append("grid_extrapolation")
        is_ood = len(reasons) > 0
        ood_flag[i] = is_ood
        ood_reason[i] = "+".join(reasons)

        allowed = list(pts.keys())
        if is_ood:
            if ood_policy == "extra_baseline":
                preferred = [s for s in ["extra", "baseline"] if s in pts]
                if preferred:
                    allowed = preferred
            elif ood_policy == "no_grid":
                preferred = [s for s in ["extra", "resid", "baseline"] if s in pts]
                if preferred:
                    allowed = preferred
            elif ood_policy == "extra_only":
                if "extra" in pts:
                    allowed = ["extra"]
                elif "baseline" in pts:
                    allowed = ["baseline"]
            elif ood_policy == "baseline_only":
                if "baseline" in pts:
                    allowed = ["baseline"]
            

        ranked = sorted(allowed, key=lambda s: arisk.get(s, np.inf))
        if not ranked:
            continue
        best = ranked[0]
        chosen1[i] = best

        if len(ranked) == 1:
            weights[best][i] = 1.0
            pred[i] = pts[best]
            continue

        second = ranked[1]
        chosen2[i] = second
        ratio = arisk[second] / max(arisk[best], 1e-9)
        if ratio >= top1_ratio:
            # Strong confidence gap: use the safest predicted expert directly.
            weights[best][i] = 1.0
            pred[i] = pts[best]
        else:
            # Ambiguous case: blend only top two, never all experts.
            sb = 1.0 / (max(arisk[best], 5.0) ** top2_power)
            ss = 1.0 / (max(arisk[second], 5.0) ** top2_power)
            z = sb + ss
            wb = sb / z
            ws = ss / z
            weights[best][i] = wb
            weights[second][i] = ws
            pred[i] = wb * pts[best] + ws * pts[second]

    return pred, weights, adjusted, chosen1, chosen2, ood_flag, ood_reason


def tune_and_fit_router_a52(router_oof: pd.DataFrame, loc: Optional[str]):
    risk_oof, priors, num, cat, features = a52_risk_cv_predictions(router_oof, loc)
    truth = router_oof[["router_truth_x", "router_truth_y"]].to_numpy(float)

    # Candidate OOD distance thresholds are derived only from pseudo-OOF support.
    nearest_qs = [0.90, 0.97, 1.01]  # 1.01 disables nearest-distance OOD detection.
    nearest_thrs = {q: a52_nearest_threshold(router_oof, q) for q in nearest_qs}

    trials = []
    best = None
    for prior_strength in [0.0, 0.5]:
        for top1_ratio in [1.25, 1.60, 2.00]:
            for top2_power in [1.0, 2.0]:
                for support_penalty in [0.0, 0.5]:
                    for count_thr in [0.0, 2.0]:
                        for q in nearest_qs:
                            for extrap_thr in [1.0, np.inf]:
                                for policy in ["all", "extra_baseline", "no_grid", "extra_only"]:
                                    params = {
                                        "prior_strength": prior_strength,
                                        "top1_ratio": top1_ratio,
                                        "top2_power": top2_power,
                                        "support_penalty": support_penalty,
                                        "ood_count_threshold": count_thr,
                                        "ood_nearest_quantile": q,
                                        "ood_nearest_threshold": nearest_thrs[q],
                                        "ood_extrap_threshold": extrap_thr,
                                        "ood_policy": policy,
                                    }
                                    p, _, _, _, _, ood, _ = a52_select_route(
                                        router_oof, risk_oof, priors, params
                                    )
                                    e = np.linalg.norm(p - truth, axis=1)
                                    score = a52_router_objective(e)
                                    row = {
                                        "score": float(score),
                                        "median_m": float(np.nanmedian(e)),
                                        "p90_m": float(np.nanquantile(e, 0.90)),
                                        "cvar90_m": float(a52_cvar(e)),
                                        "mean_m": float(np.nanmean(e)),
                                        "catastrophic_rate": float(np.nanmean(e > 1000.0)),
                                        "ood_rate": float(np.mean(ood)),
                                        **params,
                                    }
                                    
                                    if not np.isfinite(row["ood_extrap_threshold"]):
                                        row["ood_extrap_threshold"] = 1e12
                                    trials.append(row)
                                    if best is None or score < best["score"]:
                                        best = row.copy()

    if best is None:
        raise RuntimeError("Architecture 5.2 router tuning failed.")


    best_params = {
        k: best[k] for k in [
            "prior_strength", "top1_ratio", "top2_power", "support_penalty",
            "ood_count_threshold", "ood_nearest_quantile", "ood_nearest_threshold",
            "ood_extrap_threshold", "ood_policy",
        ]
    }
    if float(best_params["ood_extrap_threshold"]) >= 1e11:
        best_params["ood_extrap_threshold"] = np.inf


    final_models, final_priors = fit_risk_models(router_oof, ROUTER_SOURCES, num, cat)
    return {
        "models": final_models,
        "priors": final_priors,
        "num": num,
        "cat": cat,
        "features": features,
        "params": best_params,
        "tuning_best": best,
        "tuning_trials": pd.DataFrame(trials).sort_values("score").reset_index(drop=True),
    }


def apply_router_pack_a52(pack: Dict, frame: pd.DataFrame):
    risks = predict_risks(pack["models"], frame, ROUTER_SOURCES, pack["features"])
    p, weights, adjusted, top1, top2, ood, reason = a52_select_route(
        frame, risks, pack["priors"], pack["params"]
    )
    return p, weights, risks, adjusted, top1, top2, ood, reason


def make_a52_risk_pipeline(num_cols: List[str], cat_cols: List[str]):
    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    model = ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=4,
        max_features=0.75,
        random_state=SEED + 91,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def fit_risk_models_a52(train: pd.DataFrame, sources: Sequence[str], num: List[str], cat: List[str]):
    features = num + cat
    models = {}
    medians = {}
    for s in sources:
        ycol = f"router_error_{s}_m"
        tr = train.loc[train[ycol].notna()].copy()
        if len(tr) < 30:
            models[s] = None
            medians[s] = 600.0
            continue
        y = np.log1p(np.maximum(tr[ycol].to_numpy(float), 0.0))
        model = make_a52_risk_pipeline(num, cat)
        model.fit(tr[features], y)
        models[s] = model
        medians[s] = float(np.median(tr[ycol]))
    inv = {s: 1.0 / max(medians.get(s, 600.0), 25.0) for s in sources}
    z = sum(inv.values()) or 1.0
    priors = {s: inv[s] / z for s in sources}
    return models, priors


def a52_risk_cv_predictions(router_oof: pd.DataFrame, loc: Optional[str]):
    num, cat = router_feature_lists(router_oof, loc)
    features = num + cat
    if len(router_oof) < ROUTER_MIN_ROWS or not features:
        raise RuntimeError(
            f"Insufficient router training data: {len(router_oof)} rows, {len(features)} features"
        )

    kf = KFold(n_splits=A52_ROUTER_CV_FOLDS, shuffle=True, random_state=SEED + 17)
    n = len(router_oof)
    risk_oof = {s: np.full(n, np.nan) for s in ROUTER_SOURCES}

    for tr_idx, va_idx in kf.split(np.arange(n)):
        tr = router_oof.iloc[tr_idx].copy()
        va = router_oof.iloc[va_idx].copy()
        models, _ = fit_risk_models_a52(tr, ROUTER_SOURCES, num, cat)
        rr = predict_risks(models, va, ROUTER_SOURCES, features)
        for s in ROUTER_SOURCES:
            risk_oof[s][va_idx] = rr[s]

    medians = {}
    for s in ROUTER_SOURCES:
        vals = pd.to_numeric(router_oof[f"router_error_{s}_m"], errors="coerce").dropna().to_numpy(float)
        medians[s] = float(np.median(vals)) if len(vals) else 600.0
    inv = {s: 1.0 / max(medians[s], 25.0) for s in ROUTER_SOURCES}
    z = sum(inv.values()) or 1.0
    priors = {s: inv[s] / z for s in ROUTER_SOURCES}
    return risk_oof, priors, num, cat, features


def tune_and_fit_router_a52(router_oof: pd.DataFrame, loc: Optional[str]):
    risk_oof, priors, num, cat, features = a52_risk_cv_predictions(router_oof, loc)
    truth = router_oof[["router_truth_x", "router_truth_y"]].to_numpy(float)

    nearest_qs = [0.93, 1.01]  # 1.01 disables nearest-distance OOD detection.
    nearest_thrs = {q: a52_nearest_threshold(router_oof, q) for q in nearest_qs}

    trials = []
    best = None
    for prior_strength in [0.0, 0.5]:
        for top1_ratio in [1.40, 1.85]:
            for top2_power in [1.0, 2.0]:
                for support_penalty in [0.0, 0.5]:
                    for count_thr in [0.0, 2.0]:
                        for q in nearest_qs:
                            for extrap_thr in [1.0, np.inf]:
                                for policy in ["all", "extra_baseline", "no_grid"]:
                                    params = {
                                        "prior_strength": prior_strength,
                                        "top1_ratio": top1_ratio,
                                        "top2_power": top2_power,
                                        "support_penalty": support_penalty,
                                        "ood_count_threshold": count_thr,
                                        "ood_nearest_quantile": q,
                                        "ood_nearest_threshold": nearest_thrs[q],
                                        "ood_extrap_threshold": extrap_thr,
                                        "ood_policy": policy,
                                    }
                                    p, _, _, _, _, ood, _ = a52_select_route(
                                        router_oof, risk_oof, priors, params
                                    )
                                    e = np.linalg.norm(p - truth, axis=1)
                                    score = a52_router_objective(e)
                                    row = {
                                        "score": float(score),
                                        "median_m": float(np.nanmedian(e)),
                                        "p90_m": float(np.nanquantile(e, 0.90)),
                                        "cvar90_m": float(a52_cvar(e)),
                                        "mean_m": float(np.nanmean(e)),
                                        "catastrophic_rate": float(np.nanmean(e > 1000.0)),
                                        "ood_rate": float(np.mean(ood)),
                                        **params,
                                    }
                                    if not np.isfinite(row["ood_extrap_threshold"]):
                                        row["ood_extrap_threshold"] = 1e12
                                    trials.append(row)
                                    if best is None or score < best["score"]:
                                        best = row.copy()

    if best is None:
        raise RuntimeError("Architecture 5.2 router tuning failed.")

    best_params = {
        k: best[k] for k in [
            "prior_strength", "top1_ratio", "top2_power", "support_penalty",
            "ood_count_threshold", "ood_nearest_quantile", "ood_nearest_threshold",
            "ood_extrap_threshold", "ood_policy",
        ]
    }
    if float(best_params["ood_extrap_threshold"]) >= 1e11:
        best_params["ood_extrap_threshold"] = np.inf

    final_models, final_priors = fit_risk_models_a52(router_oof, ROUTER_SOURCES, num, cat)
    return {
        "models": final_models,
        "priors": final_priors,
        "num": num,
        "cat": cat,
        "features": features,
        "params": best_params,
        "tuning_best": best,
        "tuning_trials": pd.DataFrame(trials).sort_values("score").reset_index(drop=True),
    }


def main():
    outdir = Path(A52_OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    master = normalize_df_columns(pd.read_csv(MASTER_PATH))
    survey = normalize_df_columns(pd.read_csv(SURVEY_PATH))

    mid = require_col(find_col(master, "address_id", "master_address_id"), "master address id", master)
    sid = require_col(find_col(survey, "address_id", "survey_address_id"), "survey address id", survey)
    tx = require_col(find_col(survey, "truth_x", "truth_x"), "survey truth x", survey)
    ty = require_col(find_col(survey, "truth_y", "truth_y"), "survey truth y", survey)
    px = require_col(find_col(master, "pseudo_x", "pseudo_x"), "pseudo truth x (pt_x)", master)
    py = require_col(find_col(master, "pseudo_y", "pseudo_y"), "pseudo truth y (pt_y)", master)
    bx = find_col(master, "baseline_x", "baseline_x")
    by = find_col(master, "baseline_y", "baseline_y")
    poix = find_col(master, "poi_x", "poi_x")
    poiy = find_col(master, "poi_y", "poi_y")
    loc = find_col(master, "locality_id", "locality_id")

    master[mid] = master[mid].astype(str)
    survey[sid] = survey[sid].astype(str)
    addr = master.groupby(mid, as_index=False).first().copy()

    numeric_cols = [px, py, bx, by, poix, poiy, "pt_k", "pt_spread_m", "clean_visits", "total_visits", "avg_integrity_risk"]
    numeric_cols += STRUCTURED_NUM_CANDIDATES + EXTRA_NUM_CANDIDATES
    for c in [x for x in numeric_cols if x and x in addr.columns]:
        addr[c] = pd.to_numeric(addr[c], errors="coerce")
    survey[tx] = pd.to_numeric(survey[tx], errors="coerce")
    survey[ty] = pd.to_numeric(survey[ty], errors="coerce")

    addr, visit_mode = build_visit_consensus(addr, mid, px, py)
    pseudo_train = addr.loc[addr[[px, py]].notna().all(axis=1)].copy()
    if len(pseudo_train) < MIN_MODEL_ROWS:
        raise RuntimeError(f"Only {len(pseudo_train)} pseudo-truth rows found; need at least {MIN_MODEL_ROWS}.")

    components, locality_pred, global_center, feature_meta = build_component_predictions(
        addr, pseudo_train, px, py, bx, by, poix, poiy, loc
    )
    addr = add_candidate_columns(addr, components)

    dis, cnt = [], []
    for i in range(len(addr)):
        d, c = component_disagreement(components, i)
        dis.append(d); cnt.append(c)
    addr["component_disagreement_m"] = dis
    addr["component_source_count"] = cnt

    locality_lookup = {}
    if loc and loc in addr.columns:
        for i, value in enumerate(addr[loc]):
            if pd.notna(value) and np.all(np.isfinite(locality_pred[i])):
                locality_lookup[str(value)] = locality_pred[i]

    print("Building pseudo-truth OOF NV_grid training examples...")
    router_oof = build_pseudo_oof_router_data(addr, pseudo_train, px, py, bx, by, poix, poiy, loc)
    print("Fitting Architecture 5.2 OOD-aware top-1/top-2 router...")
    router52_pack = tune_and_fit_router_a52(router_oof, loc)
    router52_pack["tuning_trials"].head(30).to_csv(outdir / "architecture52_router_tuning_top30.csv", index=False)

    support_all = build_support_features(pseudo_train, addr, feature_meta["grid_num"], loc)
    for c in support_all.columns:
        addr[c] = support_all[c].to_numpy()
    addr = add_router_meta_features(addr, bx, by)

    r52_xy, r52_weights, r52_risks, r52_adjusted, r52_top1, r52_top2, r52_ood, r52_reason = apply_router_pack_a52(router52_pack, addr)
    addr["router52_pred_x"] = r52_xy[:, 0]
    addr["router52_pred_y"] = r52_xy[:, 1]
    addr["router52_top1"] = r52_top1
    addr["router52_top2"] = r52_top2
    addr["router52_ood"] = r52_ood.astype(int)
    addr["router52_ood_reason"] = r52_reason
    for s in ROUTER_SOURCES:
        addr[f"router52_weight_{s}"] = r52_weights[s]
        addr[f"router52_predicted_risk_{s}_m"] = r52_risks[s]
        addr[f"router52_adjusted_risk_{s}_m"] = r52_adjusted[s]

    survey_keyed = survey.rename(columns={sid: mid}).copy()
    eval_df = survey_keyed.merge(addr, on=mid, how="left", suffixes=("_survey", ""))
    if eval_df["clean_visit_count"].isna().any():
        miss = eval_df.loc[eval_df["clean_visit_count"].isna(), mid].head(10).tolist()
        raise RuntimeError(f"Survey IDs missing from master: {miss}")

    def evidence_tier(r):
        k = int(r["clean_visit_count"])
        if k >= 3: return "V3plus"
        if k == 2: return "V2"
        if k == 1: return "V1"
        has_poi = bool(poix and poiy and pd.notna(r.get(poix)) and pd.notna(r.get(poiy)))
        has_grid = any(pd.notna(r.get(c)) for c in feature_meta["grid_num"])
        if has_grid and has_poi: return "NV_grid_poi"
        if has_grid: return "NV_grid"
        if has_poi: return "NV_poi"
        if loc and pd.notna(r.get(loc)): return "NV_locality"
        return "NV_fallback"

    eval_df["evidence_tier"] = eval_df.apply(evidence_tier, axis=1)
    addr["evidence_tier"] = addr.apply(evidence_tier, axis=1)
    source_names = ["grid", "resid", "extra", "poi", "baseline"]

    def fit_predict_a5_meta(cal_frame: pd.DataFrame, target_frame: pd.DataFrame):
        no_cal = cal_frame.loc[cal_frame["clean_visit_count"] == 0].copy()
        weights = inverse_error_weights(no_cal, source_names, tx, ty)
        base_cal = weighted_component_ensemble(no_cal, weights, source_names)
        gate = fit_tail_gate(no_cal, base_cal, tx, ty)

        no_target = target_frame.loc[target_frame["clean_visit_count"] == 0].copy()
        base_t = weighted_component_ensemble(no_target, weights, source_names)
        gated_t, alpha_t = apply_tail_gate(no_target, base_t, gate)
        gated_t, methods_t = fallback_no_visit(no_target, gated_t, locality_lookup, global_center, loc, bx, by)

        pred = np.full((len(target_frame), 2), np.nan)
        method = np.array([""] * len(target_frame), dtype=object)
        alpha = np.zeros(len(target_frame), float)
        idx_to_pos = {idx: p for p, idx in enumerate(target_frame.index)}

        vis = target_frame["clean_visit_count"] > 0
        pred[vis.to_numpy(), 0] = pd.to_numeric(target_frame.loc[vis, "visit_pin_x"], errors="coerce")
        pred[vis.to_numpy(), 1] = pd.to_numeric(target_frame.loc[vis, "visit_pin_y"], errors="coerce")
        method[vis.to_numpy()] = "visited_weighted_geometric_median"

        for j, idx in enumerate(no_target.index):
            p = idx_to_pos[idx]
            pred[p] = gated_t[j]
            alpha[p] = alpha_t[j]
            method[p] = methods_t[j]
        return pred, method, alpha, weights, gate

    def fit_predict_a52_meta(cal_frame: pd.DataFrame, target_frame: pd.DataFrame):
        pred, method, alpha, weights, gate = fit_predict_a5_meta(cal_frame, target_frame)
        is_nv_grid = (
            (target_frame["clean_visit_count"].to_numpy() == 0) &
            (target_frame["evidence_tier"].astype(str).to_numpy() == "NV_grid")
        )
        rx = pd.to_numeric(target_frame["router52_pred_x"], errors="coerce").to_numpy(float)
        ry = pd.to_numeric(target_frame["router52_pred_y"], errors="coerce").to_numpy(float)
        ok = is_nv_grid & np.isfinite(rx) & np.isfinite(ry)
        pred[ok, 0] = rx[ok]; pred[ok, 1] = ry[ok]
        method[ok] = "NV_grid_A52_OOD_top1_top2_router"
        alpha[ok] = 0.0
        return pred, method, alpha, weights, gate

    strata = np.where(eval_df["clean_visit_count"].to_numpy() > 0, "visited", "no_visit")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    a5_x = np.full(len(eval_df), np.nan); a5_y = np.full(len(eval_df), np.nan)
    a52_x = np.full(len(eval_df), np.nan); a52_y = np.full(len(eval_df), np.nan)
    a52_r = np.full(len(eval_df), np.nan)
    a52_method = np.array([""] * len(eval_df), dtype=object)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(eval_df)), strata), 1):
        cal = eval_df.iloc[tr_idx].copy()
        val = eval_df.iloc[va_idx].copy()

        p5, _, _, _, _ = fit_predict_a5_meta(cal, val)
        p52, m52, al52, _, _ = fit_predict_a52_meta(cal, val)
        a5_x[va_idx] = p5[:, 0]; a5_y[va_idx] = p5[:, 1]
        a52_x[va_idx] = p52[:, 0]; a52_y[va_idx] = p52[:, 1]
        a52_method[va_idx] = m52

        
        pcal, _, alcal, _, _ = fit_predict_a52_meta(cal, cal)
        cal["A52_x"] = pcal[:, 0]; cal["A52_y"] = pcal[:, 1]
        cal["tail_gate_alpha"] = alcal
        cal["A52_error_m"] = euclid(cal["A52_x"], cal["A52_y"], cal[tx], cal[ty])
        cal["dist_pred_to_baseline_m"] = euclid(cal["A52_x"], cal["A52_y"], cal[bx], cal[by]) if bx and by else np.nan
        cal["dist_pred_to_poi_m"] = euclid(cal["A52_x"], cal["A52_y"], cal[poix], cal[poiy]) if poix and poiy else np.nan

        val["A52_x"] = p52[:, 0]; val["A52_y"] = p52[:, 1]
        val["tail_gate_alpha"] = al52
        val["dist_pred_to_baseline_m"] = euclid(val["A52_x"], val["A52_y"], val[bx], val[by]) if bx and by else np.nan
        val["dist_pred_to_poi_m"] = euclid(val["A52_x"], val["A52_y"], val[poix], val[poiy]) if poix and poiy else np.nan
        conf = fit_confidence_model(cal, "A52_error_m", loc)
        a52_r[va_idx] = predict_radius(conf, val)

    eval_df["A5_oof_x"] = a5_x; eval_df["A5_oof_y"] = a5_y
    eval_df["A5_oof_error_m"] = euclid(a5_x, a5_y, eval_df[tx], eval_df[ty])
    eval_df["A52_oof_x"] = a52_x; eval_df["A52_oof_y"] = a52_y
    eval_df["A52_oof_error_m"] = euclid(a52_x, a52_y, eval_df[tx], eval_df[ty])
    eval_df["A52_oof_radius_m"] = a52_r
    eval_df["A52_oof_method"] = a52_method
    if bx and by:
        eval_df["baseline_error_m"] = euclid(eval_df[bx], eval_df[by], eval_df[tx], eval_df[ty])

    
    final_pred, _, final_alpha, _, _ = fit_predict_a52_meta(eval_df, eval_df)
    eval_df["A52_final_x"] = final_pred[:, 0]; eval_df["A52_final_y"] = final_pred[:, 1]
    eval_df["A52_final_error_m"] = euclid(eval_df["A52_final_x"], eval_df["A52_final_y"], eval_df[tx], eval_df[ty])
    eval_df["tail_gate_alpha"] = final_alpha
    eval_df["dist_pred_to_baseline_m"] = euclid(eval_df["A52_final_x"], eval_df["A52_final_y"], eval_df[bx], eval_df[by]) if bx and by else np.nan
    eval_df["dist_pred_to_poi_m"] = euclid(eval_df["A52_final_x"], eval_df["A52_final_y"], eval_df[poix], eval_df[poiy]) if poix and poiy else np.nan
    final_conf = fit_confidence_model(eval_df, "A52_final_error_m", loc)

    all_pred, all_method, all_alpha, _, _ = fit_predict_a52_meta(eval_df, addr)
    all_frame = addr.copy()
    all_frame["A52_pin_x"] = all_pred[:, 0]; all_frame["A52_pin_y"] = all_pred[:, 1]
    all_frame["A52_method"] = all_method; all_frame["tail_gate_alpha"] = all_alpha
    all_frame["dist_pred_to_baseline_m"] = euclid(all_frame["A52_pin_x"], all_frame["A52_pin_y"], all_frame[bx], all_frame[by]) if bx and by else np.nan
    all_frame["dist_pred_to_poi_m"] = euclid(all_frame["A52_pin_x"], all_frame["A52_pin_y"], all_frame[poix], all_frame[poiy]) if poix and poiy else np.nan
    all_frame["A52_confidence_radius_m"] = predict_radius(final_conf, all_frame)

    scopes = {
        "OOF_ALL_SURVEYED": eval_df,
        "OOF_VISITED": eval_df.loc[eval_df["clean_visit_count"] > 0],
        "OOF_NO_VISIT": eval_df.loc[eval_df["clean_visit_count"] == 0],
        "OOF_NV_GRID": eval_df.loc[eval_df["evidence_tier"] == "NV_grid"],
        "OOF_NV_GRID_POI": eval_df.loc[eval_df["evidence_tier"] == "NV_grid_poi"],
        "OOF_NV_POI": eval_df.loc[eval_df["evidence_tier"] == "NV_poi"],
    }
    rows = []
    for scope, fr in scopes.items():
        for model, err, rad in [
            ("Architecture_5_reference", "A5_oof_error_m", None),
            ("Architecture_5_2", "A52_oof_error_m", "A52_oof_radius_m"),
            ("commercial_baseline", "baseline_error_m", None),
        ]:
            if err not in fr.columns:
                continue
            r = summarize(fr, err, rad)
            r.update({"scope": scope, "model": model})
            r["errors_over_1000m"] = int((pd.to_numeric(fr[err], errors="coerce") > 1000).sum())
            rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "architecture52_oof_summary.csv", index=False)


    diag_cols = [
        mid, tx, ty, "evidence_tier", "A5_oof_error_m", "A52_oof_error_m", "baseline_error_m",
        "A5_oof_x", "A5_oof_y", "A52_oof_x", "A52_oof_y",
        "router_grid_fields_count", "router_locality_train_count",
        "router_grid_extrapolation_count", "router_grid_extrapolation_score",
        "router_nearest_grid_feature_distance", "router_component_disagreement_m",
        "router52_top1", "router52_top2", "router52_ood", "router52_ood_reason",
    ]
    for s in ROUTER_SOURCES:
        diag_cols += [
            f"cand_{s}_x", f"cand_{s}_y",
            f"router52_weight_{s}", f"router52_predicted_risk_{s}_m", f"router52_adjusted_risk_{s}_m",
        ]
    nv_diag = eval_df.loc[eval_df["evidence_tier"] == "NV_grid", [c for c in diag_cols if c in eval_df.columns]].copy()
    for s in ROUTER_SOURCES:
        cx, cy = f"cand_{s}_x", f"cand_{s}_y"
        ecol = f"actual_error_{s}_m"
        nv_diag[ecol] = np.nan
        if cx in nv_diag.columns and cy in nv_diag.columns:
            ok = nv_diag[[cx, cy, tx, ty]].notna().all(axis=1)
            if ok.any():
                nv_diag.loc[ok, ecol] = euclid(nv_diag.loc[ok, cx], nv_diag.loc[ok, cy], nv_diag.loc[ok, tx], nv_diag.loc[ok, ty])
    expert_err_cols = [f"actual_error_{s}_m" for s in ROUTER_SOURCES]
    nv_diag["oracle_best_expert"] = nv_diag[expert_err_cols].idxmin(axis=1).str.replace("actual_error_", "", regex=False).str.replace("_m", "", regex=False)
    nv_diag["oracle_best_error_m"] = nv_diag[expert_err_cols].min(axis=1)
    nv_diag["A52_minus_A5_error_m"] = nv_diag["A52_oof_error_m"] - nv_diag["A5_oof_error_m"]
    nv_diag = nv_diag.sort_values("A52_oof_error_m", ascending=False)
    nv_diag.to_csv(outdir / "architecture52_nv_grid_diagnostics.csv", index=False)

    pred_cols = [
        mid, tx, ty, "clean_visit_count", "evidence_tier",
        "A5_oof_x", "A5_oof_y", "A5_oof_error_m",
        "A52_oof_x", "A52_oof_y", "A52_oof_error_m", "A52_oof_radius_m", "A52_oof_method",
    ]
    if bx and by:
        pred_cols += [bx, by, "baseline_error_m"]
    pred_cols += [
        "router52_top1", "router52_top2", "router52_ood", "router52_ood_reason",
        "router_component_disagreement_m", "router_grid_extrapolation_score",
        "router_nearest_grid_feature_distance", "router_locality_train_count",
    ]
    for s in ROUTER_SOURCES:
        pred_cols += [f"router52_weight_{s}", f"router52_predicted_risk_{s}_m", f"router52_adjusted_risk_{s}_m"]
    eval_df[[c for c in pred_cols if c in eval_df.columns]].to_csv(outdir / "architecture52_oof_survey_predictions.csv", index=False)

    all_cols = [
        mid, "A52_pin_x", "A52_pin_y", "A52_confidence_radius_m", "A52_method",
        "clean_visit_count", "evidence_tier", "router52_top1", "router52_top2",
        "router52_ood", "router52_ood_reason", "router_grid_extrapolation_score",
        "router_nearest_grid_feature_distance", "router_locality_train_count",
    ]
    for s in ROUTER_SOURCES:
        all_cols += [f"router52_weight_{s}", f"router52_predicted_risk_{s}_m", f"router52_adjusted_risk_{s}_m"]
    if loc: all_cols.append(loc)
    if bx and by: all_cols += [bx, by]
    if poix and poiy: all_cols += [poix, poiy]
    all_frame[[c for c in all_cols if c in all_frame.columns]].to_csv(outdir / "architecture52_all_addresses.csv", index=False)

    expert_summary = []
    for s in ROUTER_SOURCES:
        e = pd.to_numeric(router_oof[f"router_error_{s}_m"], errors="coerce").dropna().to_numpy(float)
        expert_summary.append({
            "source": s, "n": len(e),
            "median_error_m": float(np.median(e)) if len(e) else np.nan,
            "p90_error_m": float(np.quantile(e, .9)) if len(e) else np.nan,
            "cvar90_m": float(a52_cvar(e)) if len(e) else np.nan,
            "mean_error_m": float(np.mean(e)) if len(e) else np.nan,
            "catastrophic_rate": float(np.mean(e > 1000.0)) if len(e) else np.nan,
        })
    pd.DataFrame(expert_summary).to_csv(outdir / "architecture52_router_pseudo_oof_experts.csv", index=False)

    def get_metric(scope, model, col):
        x = summary[(summary["scope"] == scope) & (summary["model"] == model)]
        return float(x.iloc[0][col]) if len(x) else np.nan

    nv_med = get_metric("OOF_NO_VISIT", "Architecture_5_2", "median_error_m")
    nv_p90 = get_metric("OOF_NO_VISIT", "Architecture_5_2", "p90_error_m")
    vis_med = get_metric("OOF_VISITED", "Architecture_5_2", "median_error_m")
    grid_med = get_metric("OOF_NV_GRID", "Architecture_5_2", "median_error_m")
    grid_p90 = get_metric("OOF_NV_GRID", "Architecture_5_2", "p90_error_m")
    a5_nv_p90 = get_metric("OOF_NO_VISIT", "Architecture_5_reference", "p90_error_m")

    config = {
        "architecture": "Architecture_5_2_OOD_aware_top1_top2_expert_router",
        "unchanged_visited_branch": "A2 weighted geometric median",
        "unchanged_non_NV_grid_branch": "Architecture 5 A4 ensemble + A3 disagreement gate",
        "changed_NV_grid_branch": "OOD-aware predicted-risk top-1/top-2 selection among grid/residual-GBM/ExtraTrees/commercial baseline",
        "unchanged_confidence": "A3 90th-quantile model + calibration",
        "router_training": "pseudo-truth OOF simulated no-visit; target POI hidden; risk models cross-validated; survey truth excluded",
        "router_features_num": router52_pack["num"],
        "router_features_cat": router52_pack["cat"],
        "router_priors": router52_pack["priors"],
        "router_params": router52_pack["params"],
        "router_tuning_best": router52_pack["tuning_best"],
        "visit_mode": visit_mode,
        "success_targets": {"no_visit_median_lte_m": 235.0, "no_visit_p90_lt_m": 700.0, "visited_median_lte_m": 20.0},
        "observed": {"no_visit_median_m": nv_med, "no_visit_p90_m": nv_p90, "visited_median_m": vis_med, "nv_grid_median_m": grid_med, "nv_grid_p90_m": grid_p90},
    }
    (outdir / "architecture52_model_config.json").write_text(json.dumps(config, indent=2, default=float))

    report = []
    report.append("=" * 104)
    report.append("CreditNirvana PS3 - Architecture 5.2 OOD-Aware Top-1/Top-2 NV_grid Expert Router")
    report.append("=" * 104)
    report.append(f"Master addresses: {len(addr):,} | surveyed: {len(eval_df):,} | pseudo-truth: {len(pseudo_train):,}")
    report.append(f"Pseudo-OOF router training examples with grid evidence: {len(router_oof):,}")
    report.append(f"Visit evidence mode: {visit_mode}")
    report.append("")
    report.append("WHAT CHANGED FROM ARCHITECTURE 5.1")
    report.append("  Visited: unchanged -> A2 weighted geometric median")
    report.append("  Non-NV_grid no-visit: unchanged -> Architecture 5")
    report.append("  Confidence: unchanged -> A3 quantile + calibration")
    report.append("  NV_grid: four-way blend removed; use OOD-aware top-1/top-2 expert selection")
    report.append("")
    report.append("ARCHITECTURE 5.2 ROUTER")
    report.append(f"  Learned priors: {router52_pack['priors']}")
    report.append(f"  Tuned params: {router52_pack['params']}")
    report.append(f"  Pseudo-OOF tuning best: {router52_pack['tuning_best']}")
    report.append("")
    report.append("PRIMARY OOF COMPARISON")
    report.append(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    report.append("")
    report.append("TARGET CHECK")
    report.append(f"  No-visit median <= 235m: {'PASS' if np.isfinite(nv_med) and nv_med <= 235 else 'MISS'} ({nv_med:.2f}m)")
    report.append(f"  No-visit p90 < 700m:    {'PASS' if np.isfinite(nv_p90) and nv_p90 < 700 else 'MISS'} ({nv_p90:.2f}m)")
    report.append(f"  Visited median <= 20m:  {'PASS' if np.isfinite(vis_med) and vis_med <= 20 else 'MISS'} ({vis_med:.2f}m)")
    report.append(f"  A5 no-visit p90 reference: {a5_nv_p90:.2f}m")
    report.append(f"  A5.2 NV_grid median/p90:   {grid_med:.2f}m / {grid_p90:.2f}m")
    report.append("")
    report.append("OUTPUTS")
    report.append("  architecture52_oof_summary.csv")
    report.append("  architecture52_oof_survey_predictions.csv")
    report.append("  architecture52_nv_grid_diagnostics.csv")
    report.append("  architecture52_router_tuning_top30.csv")
    report.append("  architecture52_router_pseudo_oof_experts.csv")
    report.append("  architecture52_all_addresses.csv")
    report.append("  architecture52_model_config.json")
    (outdir / "architecture52_run_report.txt").write_text("\n".join(report))

    print("\n".join(report[:19]))
    print("\nPRIMARY OOF COMPARISON")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nTARGET CHECK")
    print(f"No-visit median <=235m: {nv_med:.2f}m")
    print(f"No-visit p90 <700m:     {nv_p90:.2f}m")
    print(f"Visited median <=20m:   {vis_med:.2f}m")
    print(f"NV_grid median / p90:   {grid_med:.2f}m / {grid_p90:.2f}m")
    print(f"\nSaved outputs to: {outdir.resolve()}")



if __name__ == "__main__":
    main()
