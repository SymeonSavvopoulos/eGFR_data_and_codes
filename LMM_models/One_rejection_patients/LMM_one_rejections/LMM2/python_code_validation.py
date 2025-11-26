

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ------------- USER INPUT -------------
MODEL_BUNDLE = Path(r".../LMM_no_rejections/LMM2/model_bundle.json")
EXTERNAL_XLSX = Path(r".../LMM_one_rejections/LMM2/validationcohort.xlsx")      # change me
OUT_DIR       = Path(r".../LMM_one_rejections/LMM2")                 # change me

# Optional overrides (leave as None to use bundle's defaults)
CONDITION_ON_PAST_OVERRIDE = None   # set to True/False or None to use bundle default
EXTERNAL_TIME_COLS_OVERRIDE = {
    # put your external column names here if different.
    # e.g., 42: "gfr_42", 100: "gfr_100", 365: "gfr_365"
    # 42: "gfr_42days",
    # 100: "gfr_100days",
    # 365: "gfr_365",
}

WARN_MISSING_FEATURES = True

# ----------- Helpers to parse fe_names -----------
def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default

def _strip_Q(inner: str) -> str:
    s = inner.strip()
    if s.startswith('Q(') and s.endswith(')'):
        if '"' in s:
            i = s.find('"'); j = s.rfind('"')
        else:
            i = s.find("'"); j = s.rfind("'")
        if i != -1 and j != -1 and j > i:
            return s[i+1:j]
    return s

def _eval_term_component(term: str, time_c: float, feat_row_dict: dict) -> float:
    t = term.strip()
    if t == "Time_c":
        return time_c
    if t.startswith("C("):
        # like C(Q('sex'))[T.M]
        i1 = t.find("("); i2 = t.rfind(")")
        inner = t[i1+1:i2].strip()
        var = _strip_Q(inner)
        val = feat_row_dict.get(var, None)
        if "[T." in t and t.endswith("]"):
            level = t.split("[T.", 1)[1][:-1]
            return 1.0 if (val is not None and str(val) == level) else 0.0
        return 0.0
    if t.startswith("Q("):
        var = _strip_Q(t)
        return _safe_float(feat_row_dict.get(var, 0.0), 0.0)
    return _safe_float(feat_row_dict.get(t, 0.0), 0.0)

def make_X_row(fe_names, time_c, feat_row_dict, extra_time_vars=None):
    feats = {**feat_row_dict, **(extra_time_vars or {})}
    x = np.zeros(len(fe_names), dtype=float)
    for i, name in enumerate(fe_names):
        if name == "Intercept":
            x[i] = 1.0
            continue
        parts = [p.strip() for p in name.split(":")]
        val = 1.0
        for p in parts:
            val *= _eval_term_component(p, time_c, feats)
        x[i] = val
    return x

def resolve_col(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name is None:
            continue
        key = str(name).lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    return None

# ------------- Load bundle & data -------------
bundle = json.loads(MODEL_BUNDLE.read_text(encoding="utf-8"))

fe_names = bundle["fe_names"]
beta = np.array(bundle["fe_params"], dtype=float)
G = np.array(bundle["cov_re"], dtype=float)
sigma2 = float(bundle["sigma2"])

id_col = bundle["id_col"]
time_cols_bundle = {int(k): v for k, v in bundle["time_cols"].items()}  # keys may be strings
times = [int(t) for t in bundle["time_days"]]
assert 365 in times, "Bundle must include 365."
past_times_default = [int(t) for t in bundle.get("past_times", [])]
cond_default = bool(bundle.get("predict_condition_on_past", False))
condition_on_past = cond_default if CONDITION_ON_PAST_OVERRIDE is None else bool(CONDITION_ON_PAST_OVERRIDE)

baseline_feats = bundle.get("feature_cols_baseline", [])
gfr_feat_cols  = bundle.get("feature_cols_gfr", [])
feat_cols_all  = bundle.get("feature_cols_all", baseline_feats + gfr_feat_cols)

apply_gfr_at365_only = bool(bundle.get("apply_gfr_features_at_365_only", True))

# Time centering (must match training: global mean across modeled times)
tyears = {d: d/365.0 for d in times}
tbar = float(np.mean([tyears[d] for d in times]))
Tc = {d: tyears[d] - tbar for d in times}
Z  = {d: np.array([1.0, Tc[d]], dtype=float) for d in times}

# Read external data
df = pd.read_excel(EXTERNAL_XLSX)
df.columns = df.columns.str.strip()

# Honor manual overrides for time columns if provided
time_cols = time_cols_bundle.copy()
for k, v in (EXTERNAL_TIME_COLS_OVERRIDE or {}).items():
    time_cols[int(k)] = v

# Try to resolve time columns robustly if missing
for t in times:
    if t in time_cols and time_cols[t] in df.columns:
        continue
    # Fall-back aliases
    aliases = [time_cols.get(t)]
    if t == 42:
        aliases += ["gfr_42", "egfr_42", "gfr42", "gfr_42days"]
    if t == 100:
        aliases += ["gfr_100", "egfr_100", "gfr100", "gfr_100days"]
    if t == 365:
        aliases += ["gfr_365", "egfr_365", "gfr365", "gfr_365days"]
    found = resolve_col(df, aliases)
    if found:
        time_cols[t] = found

# Build GFR feature columns expected by the model (if present in the bundle)
if "feat_gfr42" in gfr_feat_cols and 42 in time_cols and time_cols[42] in df.columns:
    df["feat_gfr42"] = df[time_cols[42]]
if "feat_gfr100" in gfr_feat_cols and 100 in time_cols and time_cols[100] in df.columns:
    df["feat_gfr100"] = df[time_cols[100]]

# Warn about missing baseline features (optional)
if WARN_MISSING_FEATURES and baseline_feats:
    missing = [c for c in baseline_feats if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} baseline feature(s) missing in external file. "
              f"They will be treated as reference/zero: {missing[:10]}{'...' if len(missing)>10 else ''}")

# ------------- Predict -------------
pred_rows = []
for _, row in df.iterrows():
    # Assemble feature dict for this patient
    feat_row = {c: row[c] for c in feat_cols_all if c in df.columns}

    # Future X at 365 (set is365 = 1 so gfr-feature terms activate if trained that way)
    Xf = make_X_row(fe_names, Tc[365], feat_row, extra_time_vars={"is365": 1})[None, :]

    if not condition_on_past:
        mean_future = float(Xf @ beta)
        se_future = np.nan
    else:
        # Collect observed past outcomes (set is365 = 0 in their X rows)
        y_past, X_past, Z_past = [], [], []
        for t in sorted(past_times_default):
            col = time_cols.get(t, None)
            if not col or col not in df.columns:
                continue
            yv = row.get(col, np.nan)
            if pd.notna(yv):
                y_past.append(float(yv))
                X_past.append(make_X_row(fe_names, Tc[t], feat_row, extra_time_vars={"is365": 0}))
                Z_past.append(Z[t])

        if len(y_past) == 0:
            mean_future = float(Xf @ beta)
            se_future = np.nan
        else:
            y_past = np.asarray(y_past)
            X_past = np.vstack(X_past)
            Z_past = np.vstack(Z_past)

            Var_past = Z_past @ G @ Z_past.T + sigma2 * np.eye(len(y_past))
            Cov_fp   = Z[365] @ G @ Z_past.T
            Var_fut  = float(Z[365] @ G @ Z[365].T + sigma2)

            mean_future = float(Xf @ beta + Cov_fp @ np.linalg.solve(Var_past, (y_past - X_past @ beta)))
            var_future  = float(Var_fut - Cov_fp @ np.linalg.solve(Var_past, Cov_fp.T))
            se_future   = float(np.sqrt(max(var_future, 0.0)))

    pred_rows.append({
        id_col: row.get(id_col, np.nan),
        "gfr_42days": row.get(time_cols.get(42, ""), np.nan),
        "gfr_100days": row.get(time_cols.get(100, ""), np.nan),
        "pred_365_mean": mean_future,
        "pred_365_pi_low":  mean_future - 1.96*se_future if np.isfinite(se_future) else np.nan,
        "pred_365_pi_high": mean_future + 1.96*se_future if np.isfinite(se_future) else np.nan,
        "observed_365": row.get(time_cols.get(365, ""), np.nan),
    })

OUT_DIR.mkdir(parents=True, exist_ok=True)
pred = pd.DataFrame(pred_rows)
pred_path = OUT_DIR / ("external_pred_365_with_gfr_features_" + ("condBLUP" if condition_on_past else "FEonly") + ".xlsx")
pred.to_excel(pred_path, index=False)
print("Saved external predictions to:", pred_path)

# ------------- Optional: metrics & plot -------------
if "observed_365" in pred and pred["observed_365"].notna().any():
    m = pred.dropna(subset=["observed_365"]).copy()
    if len(m) > 0:
        r2   = r2_score(m["observed_365"], m["pred_365_mean"])
        rmse = float(np.sqrt(mean_squared_error(m["observed_365"], m["pred_365_mean"])))
        print(f"External evaluation: R^2 = {r2:.3f}, RMSE = {rmse:.3f}")
        m.to_csv(OUT_DIR / "external_obs_vs_pred.csv", index=False)

        plt.figure()
        plt.scatter(m["observed_365"], m["pred_365_mean"], s=16)
        lo = float(min(m["observed_365"].min(), m["pred_365_mean"].min()))
        hi = float(max(m["observed_365"].max(), m["pred_365_mean"].max()))
        plt.plot([lo, hi], [lo, hi], "--")
        plt.xlabel("Observed 365-day eGFR")
        plt.ylabel("Predicted 365-day eGFR")
        plt.title(f"External set ({'condBLUP' if condition_on_past else 'FE-only'}): R²={r2:.3f}, RMSE={rmse:.2f}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "external_obs_vs_pred.png", dpi=150)
        plt.close()
else:
    print("No observed 365-day column found (or all missing). Metrics skipped.")
