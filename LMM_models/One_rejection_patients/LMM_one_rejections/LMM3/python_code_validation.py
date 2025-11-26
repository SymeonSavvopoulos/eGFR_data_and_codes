
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# ------------- USER INPUT -------------
MODEL_BUNDLE = Path(r".../LMM_no_rejections/LMM3/model_bundle.json")
EXTERNAL_XLSX = Path(r".../LMM_one_rejections/LMM3/validationcohort.xlsx")      # change me
OUT_DIR       = Path(r".../LMM_one_rejections/LMM3")                 # change me


# Optional: force whether to condition on past (None = use bundle default)
CONDITION_ON_PAST_OVERRIDE = None  # set to True/False to override

# Optional: map your validation column names if they differ
EXTERNAL_TIME_COLS_OVERRIDE = {
    # 42: "gfr_42days",
    # 365: "gfr_365",
}

WARN_MISSING_BASELINE = True

# -------- Helpers to parse patsy-like names from fe_names --------
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
    if t.startswith("C("):  # e.g., C(Q('sex'))[T.M]
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
        if not name:
            continue
        key = str(name).lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    return None

# -------- Load bundle & validation data --------
bundle = json.loads(MODEL_BUNDLE.read_text(encoding="utf-8"))

fe_names = bundle["fe_names"]
beta = np.array(bundle["fe_params"], dtype=float)
G = np.array(bundle["cov_re"], dtype=float)
sigma2 = float(bundle["sigma2"])

id_col = bundle["id_col"]
time_cols_bundle = {int(k): v for k, v in bundle["time_cols"].items()}
times = [int(t) for t in bundle["time_days"]]     # [42, 365]
assert 365 in times and 42 in times

baseline_feats = bundle.get("feature_cols_baseline", [])
engineered_feats = bundle.get("feature_cols_engineered", ["feat_gfr42"])
feat_cols_all = bundle.get("feature_cols_all", baseline_feats + engineered_feats)

apply_gfr42_365_only = bool(bundle.get("apply_gfr42_at_365_only", True))
include_time_inter = bool(bundle.get("include_time_interactions", True))
cond_default = bool(bundle.get("predict_condition_on_past", True))
past_times = [int(t) for t in bundle.get("past_times", [42])]
condition_on_past = cond_default if CONDITION_ON_PAST_OVERRIDE is None else bool(CONDITION_ON_PAST_OVERRIDE)

# Time centering (must match training)
tyears = {d: d/365.0 for d in times}
tbar = float(np.mean([tyears[d] for d in times]))
Tc = {d: tyears[d] - tbar for d in times}
Z  = {d: np.array([1.0, Tc[d]], dtype=float) for d in times}

df = pd.read_excel(VALIDATION_XLSX)
df.columns = df.columns.str.strip()

# Allow overrides / auto-detect common aliases
time_cols = time_cols_bundle.copy()
for k, v in (EXTERNAL_TIME_COLS_OVERRIDE or {}).items():
    time_cols[int(k)] = v

for t in times:
    if t in time_cols and time_cols[t] in df.columns:
        continue
    aliases = [time_cols.get(t)]
    if t == 42:  aliases += ["gfr_42", "egfr_42", "gfr42", "gfr_42days"]
    if t == 365: aliases += ["gfr_365", "egfr_365", "gfr365", "gfr_365days"]
    found = resolve_col(df, aliases)
    if found:
        time_cols[t] = found

# Make engineered RHS feature from validation gfr_42days
if 42 in time_cols and time_cols[42] in df.columns:
    df["feat_gfr42"] = df[time_cols[42]]
else:
    df["feat_gfr42"] = 0.0  # fallback (should rarely happen)

# Optional: warn about missing baseline features
if WARN_MISSING_BASELINE and baseline_feats:
    miss = [c for c in baseline_feats if c not in df.columns]
    if miss:
        print(f"Warning: {len(miss)} baseline feature(s) missing in validation file. "
              f"Will treat as 0/ref: {miss[:10]}{'...' if len(miss)>10 else ''}")

# -------- Predict --------
OUT_DIR.mkdir(parents=True, exist_ok=True)
pred_rows = []
for _, row in df.iterrows():
    feat_row = {c: row[c] for c in feat_cols_all if c in df.columns}

    # Future (365). Activate gfr_42 feature only at 365 via is365=1 (matches training design).
    Xf = make_X_row(fe_names, Tc[365], feat_row, extra_time_vars={"is365": 1})[None, :]

    if not condition_on_past:
        mean_365 = float(Xf @ beta)
        se_365 = np.nan
    else:
        # Condition on observed 42d (is365=0)
        y_past, X_past, Z_past = [], [], []
        for t in sorted(past_times):
            col = time_cols.get(t, None)
            if not col or col not in df.columns:
                continue
            yv = row.get(col, np.nan)
            if pd.notna(yv):
                y_past.append(float(yv))
                X_past.append(make_X_row(fe_names, Tc[t], feat_row, extra_time_vars={"is365": 0}))
                Z_past.append(Z[t])

        if len(y_past) == 0:
            mean_365 = float(Xf @ beta)
            se_365 = np.nan
        else:
            y_past = np.asarray(y_past)
            X_past = np.vstack(X_past)
            Z_past = np.vstack(Z_past)
            Var_past = Z_past @ G @ Z_past.T + sigma2 * np.eye(len(y_past))
            Cov_fp   = Z[365] @ G @ Z_past.T
            Var_fut  = float(Z[365] @ G @ Z[365].T + sigma2)

            mean_365 = float(Xf @ beta + Cov_fp @ np.linalg.solve(Var_past, (y_past - X_past @ beta)))
            var_cond = float(Var_fut - Cov_fp @ np.linalg.solve(Var_past, Cov_fp.T))
            se_365   = float(np.sqrt(max(var_cond, 0.0)))

    pred_rows.append({
        id_col: row.get(id_col, np.nan),
        "gfr_42days": row.get(time_cols.get(42, ""), np.nan),
        "pred_365_mean": mean_365,
        "pred_365_pi_low":  mean_365 - 1.96*se_365 if np.isfinite(se_365) else np.nan,
        "pred_365_pi_high": mean_365 + 1.96*se_365 if np.isfinite(se_365) else np.nan,
        "observed_365": row.get(time_cols.get(365, ""), np.nan),
    })

pred = pd.DataFrame(pred_rows)
pred_path = OUT_DIR / f"validation_pred_365_gfr42_{'condBLUP' if condition_on_past else 'FEonly'}.xlsx"
pred.to_excel(pred_path, index=False)
print("Saved validation predictions to:", pred_path)

# -------- Optional: metrics & scatter --------
if pred["observed_365"].notna().any():
    m = pred.dropna(subset=["observed_365"]).copy()
    if len(m) > 0:
        r2   = r2_score(m["observed_365"], m["pred_365_mean"])
        rmse = float(np.sqrt(mean_squared_error(m["observed_365"], m["pred_365_mean"])))
        print(f"Validation: R^2 = {r2:.3f}, RMSE = {rmse:.3f}")
        m.to_csv(OUT_DIR / "validation_obs_vs_pred.csv", index=False)

        plt.figure()
        plt.scatter(m["observed_365"], m["pred_365_mean"], s=16)
        lo = float(min(m["observed_365"].min(), m["pred_365_mean"].min()))
        hi = float(max(m["observed_365"].max(), m["pred_365_mean"].max()))
        plt.plot([lo, hi], [lo, hi], "--")
        plt.xlabel("Observed 365-day eGFR")
        plt.ylabel("Predicted 365-day eGFR")
        plt.title(f"Validation ({'condBLUP' if condition_on_past else 'FE-only'}): R²={r2:.3f}, RMSE={rmse:.2f}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "validation_obs_vs_pred.png", dpi=150)
        plt.close()
else:
    print("No observed 365-day column found (or all missing). Metrics skipped.")

