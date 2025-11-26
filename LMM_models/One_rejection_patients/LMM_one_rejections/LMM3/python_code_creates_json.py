# -*- coding: utf-8 -*-
"""
Predict gfr_365 using baseline features (incl. 'perc') + gfr_42days in a MixedLM.

Data requirements
-----------------
- One row per patient (wide format) with columns:
  - ID column (e.g., 'patient_id')
  - gfr_42days, gfr_365
  - Any baseline features (e.g., 'perc', others)

Model
-----
- Long format over timepoints (42, 365)
- Global-centered time covariate: Time_c
- Random effects: intercept + slope for Time_c  (re_formula="1 + Time_c")
- Fixed effects:
    Time_c + baseline features (+ optional Time_c×baseline-feature interactions)
  + (365-only) gfr_42days as a feature via term is365:feat_gfr42  (avoids leakage at 42d)
- Categorical baseline features handled using C(Q('var')) (no pandas get_dummies)

Prediction
----------
- FE-only: X_365 @ beta
- Conditional (BLUP): X_365 @ beta + Σ_365,past Σ_past,past^{-1}(y_past - X_past beta)
  where y_past uses the observed gfr_42days

Outputs
-------
- model_summary.txt, predictions .xlsx (with 95% PI if conditional)
- cv_metrics.csv for grouped CV by patient (R2 / RMSE)
- residuals.csv, random_effects.csv, and diagnostic plots
"""

import warnings
warnings.filterwarnings("ignore")

# -------------------- Imports --------------------
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error


# -------------------- Small utilities --------------------
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

def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default

def _eval_term_component(term: str, time_c: float, feat_row_dict: dict) -> float:
    t = term.strip()
    if t == "Time_c":
        return time_c
    if t.startswith("C("):
        # e.g., C(Q('sex'))[T.M]
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
    """Build one FE row aligned with fe_params. extra_time_vars merges (e.g., {'is365': 1})."""
    if extra_time_vars is None:
        extra_time_vars = {}
    feats = {**feat_row_dict, **extra_time_vars}
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


# -------------------- Data shaping --------------------
def build_long(dfw_src, id_col, time_cols, feat_cols):
    """
    Wide -> long; add time_years, globally-centered Time_c, and is365 indicator.
    """
    import numpy as _np  # robust against accidental shadowing
    times_sorted = sorted(time_cols.keys())
    rows = []
    for t_day in times_sorted:
        col = time_cols[t_day]
        if col not in dfw_src.columns:
            raise ValueError(f"TIME_COLS maps day {t_day} to '{col}', which is not in the data.")
        tmp = dfw_src[[id_col, col] + feat_cols].copy()
        tmp = tmp.rename(columns={col: "egfr"})
        tmp["time_days"] = t_day
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    long["time_years"] = long["time_days"] / 365.0
    tbar = float(_np.mean([d/365.0 for d in times_sorted]))
    long["Time_c"] = long["time_years"] - tbar
    # 365-only marker using normalized time (numerically robust)
    target_year = max(times_sorted) / 365.0  # 365 -> 1.0
    long["is365"] = _np.isclose(long["time_years"], target_year, rtol=1e-12, atol=1e-12).astype(int)
    return long


# -------------------- Prediction helpers --------------------
def predict_future(dfw_src, feat_cols, fit, id_col, time_cols, past_times, future_time=365,
                   condition_on_past=True):
    """
    Predict y_future for each patient.
      - FE-only: X_future @ beta
      - Conditional: BLUP using observed y_past (recommended)
    """
    fe_names = list(fit.fe_params.index)
    beta = fit.fe_params.values

    times = sorted(time_cols.keys())  # [42, 365]
    tyears = {d: d/365.0 for d in times}
    tbar = float(np.mean([tyears[d] for d in times]))
    Tc = {d: tyears[d] - tbar for d in times}
    Z = {d: np.array([1.0, Tc[d]]) for d in times}

    if condition_on_past:
        G = fit.cov_re.values
        sigma2 = float(fit.scale)

    pred_rows = []
    for _, row in dfw_src.iterrows():
        feats = {c: row[c] for c in feat_cols}

        # Future design at 365 (is365=1)
        Xf = make_X_row(fe_names, Tc[future_time], feats, extra_time_vars={"is365": 1})[None, :]

        if not condition_on_past:
            mean_future = float(Xf @ beta)
            se_future = np.nan
        else:
            # Condition on 42d if available (is365=0 for past)
            y_past, X_past, Z_past = [], [], []
            for t in sorted(past_times):
                if t not in time_cols:
                    continue
                yv = row.get(time_cols[t], np.nan)
                if pd.notna(yv):
                    y_past.append(float(yv))
                    X_past.append(make_X_row(fe_names, Tc[t], feats, extra_time_vars={"is365": 0}))
                    Z_past.append(Z[t])

            if len(y_past) == 0:
                mean_future = float(Xf @ beta)
                se_future = np.nan
            else:
                y_past = np.asarray(y_past)
                X_past = np.vstack(X_past)
                Z_past = np.vstack(Z_past)

                Var_past = Z_past @ G @ Z_past.T + sigma2 * np.eye(len(y_past))
                Cov_future_past = Z[future_time] @ G @ Z_past.T
                Var_future = float(Z[future_time] @ G @ Z[future_time].T + sigma2)

                mean_future = float(Xf @ beta + Cov_future_past @ np.linalg.solve(Var_past, (y_past - X_past @ beta)))
                var_future  = float(Var_future - Cov_future_past @ np.linalg.solve(Var_past, Cov_future_past.T))
                se_future   = float(np.sqrt(max(var_future, 0.0)))

        pred_rows.append({
            id_col: row[id_col],
            "gfr_42days": row.get(time_cols.get(42, ""), np.nan),
            "pred_365_mean": mean_future,
            "pred_365_pi_low": mean_future - 1.96*se_future if np.isfinite(se_future) else np.nan,
            "pred_365_pi_high": mean_future + 1.96*se_future if np.isfinite(se_future) else np.nan,
            "observed_365": row.get(time_cols[future_time], np.nan),
        })
    return pd.DataFrame(pred_rows)


def group_cv(dfw_src, feat_cols, long_fit, formula, id_col, time_cols,
             past_times, future_time=365, n_splits=5, re_formula="1 + Time_c",
             condition_on_past=True):
    # patients used in training
    train_ids_all = long_fit[id_col].unique()
    dfw_clean = (
        dfw_src[dfw_src[id_col].isin(train_ids_all)]
        .drop_duplicates(id_col)
        .reset_index(drop=True)
    )
    n_patients = dfw_clean[id_col].nunique()
    n_splits_eff = min(max(2, n_splits), n_patients) if n_patients >= 2 else 0
    if n_splits_eff < 2:
        return pd.DataFrame(columns=["fold", "R2", "RMSE"])

    times = sorted(time_cols.keys())
    tyears = {d: d/365.0 for d in times}
    tbar = float(np.mean([tyears[d] for d in times]))
    Tc = {d: tyears[d] - tbar for d in times}
    Z = {d: np.array([1.0, Tc[d]]) for d in times}

    gkf = GroupKFold(n_splits=n_splits_eff)
    groups = dfw_clean[id_col].values

    cv_rows = []
    for fold, (tr, te) in enumerate(gkf.split(dfw_clean, groups=groups), start=1):
        train_ids = set(dfw_clean.iloc[tr][id_col].tolist())
        test_ids  = set(dfw_clean.iloc[te][id_col].tolist())

        lc_train = long_fit[long_fit[id_col].isin(train_ids)].copy()
        if lc_train[id_col].nunique() < 2:
            cv_rows.append({"fold": fold, "R2": np.nan, "RMSE": np.nan})
            continue

        try:
            res = smf.mixedlm(formula, lc_train, groups=lc_train[id_col], re_formula=re_formula).fit(
                reml=True, method="lbfgs", maxiter=2000, disp=False
            )
        except Exception:
            cv_rows.append({"fold": fold, "R2": np.nan, "RMSE": np.nan})
            continue

        fe_names = list(res.fe_params.index)
        beta_cv  = res.fe_params.values
        if condition_on_past:
            G_cv  = res.cov_re.values
            s2_cv = float(res.scale)

        preds, obs = [], []
        for pid in test_ids:
            roww = dfw_src[dfw_src[id_col] == pid]
            if roww.empty:
                continue
            roww = roww.iloc[0]
            feat_row = {c: roww[c] for c in feat_cols}

            # Future at 365 (is365=1)
            Xf = make_X_row(fe_names, Tc[future_time], feat_row, extra_time_vars={"is365": 1})[None, :]

            if not condition_on_past:
                yhat = float(Xf @ beta_cv)
            else:
                # Past at 42 (is365=0)
                y_past, X_past, Z_past = [], [], []
                for t in sorted(past_times):
                    if t not in time_cols:
                        continue
                    yv = roww.get(time_cols[t], np.nan)
                    if pd.notna(yv):
                        y_past.append(float(yv))
                        X_past.append(make_X_row(fe_names, Tc[t], feat_row, extra_time_vars={"is365": 0}))
                        Z_past.append(Z[t])
                if len(y_past) == 0:
                    yhat = float(Xf @ beta_cv)
                else:
                    y_past = np.asarray(y_past)
                    X_past = np.vstack(X_past)
                    Z_past = np.vstack(Z_past)
                    Vp = Z_past @ G_cv @ Z_past.T + s2_cv * np.eye(len(y_past))
                    Cfp = Z[future_time] @ G_cv @ Z_past.T
                    yhat = float(Xf @ beta_cv + Cfp @ np.linalg.solve(Vp, (y_past - X_past @ beta_cv)))

            yobs = roww.get(time_cols[future_time], np.nan)
            if pd.notna(yobs):
                preds.append(yhat); obs.append(float(yobs))

        if len(preds) > 0:
            r2   = r2_score(obs, preds)
            rmse = float(np.sqrt(mean_squared_error(obs, preds)))
        else:
            r2, rmse = np.nan, np.nan

        cv_rows.append({"fold": fold, "R2": r2, "RMSE": rmse})

    return pd.DataFrame(cv_rows)


# -------------------- Diagnostics --------------------
def diagnostics(fit, long_fit, id_col, time_cols, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fitted = fit.fittedvalues
    resid  = long_fit["egfr"].values - fitted

    resid_df = long_fit[[id_col, "time_days", "egfr", "Time_c"]].copy()
    resid_df["fitted_cond"] = fitted
    resid_df["resid_cond"]  = resid
    resid_df["visit"] = resid_df["time_days"].map({42:"d42", 365:"d365"})
    resid_df.to_csv(out_dir / "residuals.csv", index=False)

    re = fit.random_effects
    ids = long_fit[id_col].drop_duplicates().tolist()
    u0 = np.array([re[i][0] for i in ids])
    u1 = np.array([re[i][1] for i in ids])
    pd.DataFrame({id_col: ids, "u0_intercept": u0, "u1_slope": u1}).to_csv(out_dir / "random_effects.csv", index=False)

    # Variance components & ICCs
    G  = fit.cov_re.values
    s2 = float(fit.scale)
    var_u0, var_u1, cov_u01 = float(G[0,0]), float(G[1,1]), float(G[0,1])

    def group_var(Tc): return var_u0 + 2*Tc*cov_u01 + (Tc**2)*var_u1
    times  = sorted(time_cols.keys())
    tyears = {d: d/365.0 for d in times}
    tbar   = float(np.mean([tyears[d] for d in times]))
    Tc     = {d: tyears[d] - tbar for d in times}
    icc    = {d: group_var(Tc[d])/(group_var(Tc[d]) + s2) for d in times}

    def save(fig, name):
        fig.tight_layout(); fig.savefig(out_dir / name, dpi=150); plt.close(fig)

    fig = plt.figure(); plt.scatter(fitted, resid, s=12); plt.axhline(0, ls="--")
    plt.xlabel("Fitted"); plt.ylabel("Residual"); plt.title("Residuals vs Fitted"); save(fig, "diag_resid_vs_fitted.png")

    fig = plt.figure(); plt.scatter(fitted, np.sqrt(np.abs(resid)), s=12)
    plt.xlabel("Fitted"); plt.ylabel("Sqrt(|Residual|)"); plt.title("Scale–Location"); save(fig, "diag_scale_location.png")

    fig = sm.qqplot(resid, line="45", fit=True).figure; fig.suptitle("Q–Q residuals"); save(fig, "diag_qq_residuals.png")
    fig = plt.figure(); plt.hist(resid, bins=30); plt.title("Residual histogram"); save(fig, "diag_hist_residuals.png")

    fig = plt.figure(); plt.hist(u0, bins=30); plt.title("Random intercepts u0"); save(fig, "diag_hist_u0.png")
    fig = plt.figure(); plt.hist(u1, bins=30); plt.title("Random slopes u1"); save(fig, "diag_hist_u1.png")

    fig = sm.qqplot(u0, line="45", fit=True).figure; fig.suptitle("Q–Q u0"); save(fig, "diag_qq_u0.png")
    fig = sm.qqplot(u1, line="45", fit=True).figure; fig.suptitle("Q–Q u1"); save(fig, "diag_qq_u1.png")

    fig = plt.figure(); plt.scatter(u0, u1, s=12); plt.xlabel("u0"); plt.ylabel("u1"); plt.title("u0 vs u1"); save(fig, "diag_scatter_u0_u1.png")

    icc_lines = " | ".join([f"{d}d: {icc[d]:.3f}" for d in times])
    (out_dir / "diagnostics_summary.txt").write_text(
        f"Var(u0)={var_u0:.3f}, Var(u1)={var_u1:.3f}, Cov(u0,u1)={cov_u01:.3f}, sigma^2={s2:.3f}\n"
        f"ICC by time: {icc_lines}\n", encoding="utf-8"
    )


# -------------------- Pipeline --------------------
def run_pipeline(
    DATA_XLSX,
    OUT_DIR,
    ID_COL,
    TIME_COLS,                    # {42:'gfr_42days', 365:'gfr_365'}
    PAST_TIMES,                   # [42] if conditioning
    INCLUDE_BASELINE_FEATURES=True,
    INCLUDE_TIME_INTERACTIONS=True,
    RE_FORMULA="1 + Time_c",
    N_FOLDS=5,
    EXCLUDE_365_OUTSIDE=(0, 200),
    FORCE_CATEGORICAL=None,
    APPLY_GFR42_AT_365_ONLY=True,     # avoid leakage
    PREDICT_CONDITION_ON_PAST=True    # BLUP conditioning
):
    # Defensive path coercion
    if isinstance(DATA_XLSX, (list, tuple)): DATA_XLSX = DATA_XLSX[0]
    if isinstance(OUT_DIR,   (list, tuple)): OUT_DIR   = OUT_DIR[0]
    OUT_DIR = Path(OUT_DIR); OUT_DIR.mkdir(parents=True, exist_ok=True)

    if FORCE_CATEGORICAL is None: FORCE_CATEGORICAL = []

    # Load
    dfw = pd.read_excel(DATA_XLSX)

    # Baseline features = all except ID and modeled GFR columns
    gfr_cols = set(TIME_COLS.values())
    base_cols = [ID_COL] + list(gfr_cols)
    feat_cols = [c for c in dfw.columns if c not in base_cols]
    # Exclude any other gfr_* from RHS
    feat_cols = [c for c in feat_cols if not c.lower().startswith("gfr_")]
    leaks = [c for c in feat_cols if c.lower().startswith("gfr_")]
    assert not leaks, f"GFR measurements on RHS not allowed: {leaks}"

    if not INCLUDE_BASELINE_FEATURES:
        feat_cols = []

    # Keep minimal wide
    dfw_keep = dfw[[ID_COL] + list(gfr_cols) + feat_cols].copy()

    # Add engineered gfr_42 feature on RHS (copied under a safe name)
    if 42 in TIME_COLS and TIME_COLS[42] in dfw_keep.columns:
        dfw_keep["feat_gfr42"] = dfw_keep[TIME_COLS[42]]
        if "feat_gfr42" not in feat_cols:
            feat_cols.append("feat_gfr42")
    else:
        # If missing, create a zero column (not typical)
        dfw_keep["feat_gfr42"] = 0.0
        if "feat_gfr42" not in feat_cols:
            feat_cols.append("feat_gfr42")

    # Long table
    long = build_long(dfw_keep, ID_COL, TIME_COLS, feat_cols)

    # Exclude extreme 365 for training only
    min365, max365 = EXCLUDE_365_OUTSIDE
    if min365 is not None and max365 is not None:
        bad_ids = set(dfw.loc[(dfw[TIME_COLS[365]] < min365) | (dfw[TIME_COLS[365]] > max365), ID_COL])
    else:
        bad_ids = set()
    if bad_ids:
        print(f"Excluding {len(bad_ids)} patient(s) from training due to out-of-range 365d eGFR.")
    long_fit = long[~long[ID_COL].isin(bad_ids)].copy()

    # ---- Build FE formula (categoricals via C(Q())) ----
    def q(c): return f"Q('{c}')"
    def is_cat(c): return (c in FORCE_CATEGORICAL) or (dfw[c].dtype == "O") or str(dfw[c].dtype).startswith("category")

    # Separate engineered from baseline
    engineered = {"feat_gfr42"}
    baseline_feats = [c for c in feat_cols if c not in engineered]

    fe_terms = [f"C({q(c)})" if is_cat(c) else q(c) for c in baseline_feats]

    # 365-only application for feat_gfr42 (avoid leakage)
    if APPLY_GFR42_AT_365_ONLY:
        gfr42_term = "Q('is365'):Q('feat_gfr42')"
    else:
        gfr42_term = "Q('feat_gfr42')"  # WARNING: leaks at day 42 row

    rhs_terms = ["Time_c"] + fe_terms + [gfr42_term]
    if INCLUDE_TIME_INTERACTIONS and fe_terms:
        rhs_terms += [f"Time_c:{t}" for t in fe_terms]  # no need to interact Time_c with the 365-only term
    rhs = " + ".join(list(dict.fromkeys(rhs_terms)))
    formula = f"egfr ~ {rhs}"
    print("\nFixed-effects formula:\n", formula, "\n")

    # Fit MixedLM
    print("Fitting mixed model ...")
    fit = smf.mixedlm(formula, long_fit, groups=long_fit[ID_COL], re_formula=RE_FORMULA).fit(
        reml=True, method="lbfgs", maxiter=2000, disp=False
    )
    print("Converged:", fit.converged)
    (OUT_DIR / "model_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")
    # ---- SAVE MODEL BUNDLE for external validation ----
    import json
    
    bundle = {
        # Core model
        "formula": formula,
        "re_formula": RE_FORMULA,
        "fe_names": list(fit.fe_params.index),
        "fe_params": fit.fe_params.tolist(),          # β
        "cov_re": fit.cov_re.values.tolist(),         # G (2x2)
        "sigma2": float(fit.scale),                   # σ²
    
        # Time metadata (must match at apply time)
        "time_days": sorted(TIME_COLS.keys()),        # [42, 365]
        "centering": "global_mean_over_time_days",
    
        # Columns / features used at training
        "id_col": ID_COL,
        "time_cols": TIME_COLS,                       # {"42":"gfr_42days","365":"gfr_365"} (keys may be strings in JSON)
        "feature_cols_engineered": ["feat_gfr42"],    # engineered RHS copy of gfr_42days
        "feature_cols_baseline": [c for c in feat_cols if c != "feat_gfr42"],
        "feature_cols_all": feat_cols[:],             # includes baseline + "feat_gfr42"
        "force_categorical": list(FORCE_CATEGORICAL or []),
    
        # Flags that influence design matrix at apply time
        "include_time_interactions": bool(INCLUDE_TIME_INTERACTIONS),
        "apply_gfr42_at_365_only": bool(APPLY_GFR42_AT_365_ONLY),   # is365:feat_gfr42
        "predict_condition_on_past": bool(PREDICT_CONDITION_ON_PAST),
        "past_times": list(PAST_TIMES),               # usually [42]
    }
    
    (out_path := OUT_DIR / "model_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print("Saved model bundle to:", out_path)

    # Predict
    pred = predict_future(
        dfw_keep, feat_cols, fit, ID_COL, TIME_COLS,
        past_times=PAST_TIMES, future_time=365, condition_on_past=PREDICT_CONDITION_ON_PAST
    )
    out_pred = OUT_DIR / "pred_365_features_gfr42.xlsx"
    pred.to_excel(out_pred, index=False)
    print("Saved predictions to:", out_pred)

    # In-sample metrics
    mask_train = (~pred[ID_COL].isin(bad_ids)) & (~pred["observed_365"].isna())
    if mask_train.any():
        r2_in = r2_score(pred.loc[mask_train, "observed_365"], pred.loc[mask_train, "pred_365_mean"])
        rmse_in = float(np.sqrt(mean_squared_error(pred.loc[mask_train, "observed_365"], pred.loc[mask_train, "pred_365_mean"])))
        print(f"In-sample R^2={r2_in:.3f}, RMSE={rmse_in:.3f}")
        (OUT_DIR / "in_sample_metrics.txt").write_text(f"R2={r2_in:.6f}\nRMSE={rmse_in:.6f}\n", encoding="utf-8")

    # Grouped CV
    if N_FOLDS and N_FOLDS >= 2:
        cv_df = group_cv(
            dfw_keep, feat_cols, long_fit, formula, ID_COL, TIME_COLS,
            past_times=PAST_TIMES, future_time=365, n_splits=N_FOLDS, re_formula=RE_FORMULA,
            condition_on_past=PREDICT_CONDITION_ON_PAST
        )
        cv_path = OUT_DIR / "cv_metrics.csv"
        cv_df.to_csv(cv_path, index=False)
        print("CV metrics saved to:", cv_path)
        if cv_df["R2"].notna().any():
            print(f"CV mean R^2={cv_df['R2'].mean():.3f} (SD {cv_df['R2'].std(ddof=1):.3f}); "
                  f"mean RMSE={cv_df['RMSE'].mean():.3f} (SD {cv_df['RMSE'].std(ddof=1):.3f})")

    # Diagnostics
    diagnostics(fit, long_fit, ID_COL, TIME_COLS, OUT_DIR)
    print("\nAll outputs written to:", OUT_DIR.resolve())


# -------------------- Main --------------------
def main():
    # ---- Adjust to your environment ----
    ID_COL    = "patient_id"
    TIME_COLS = {42: "gfr_42days", 365: "gfr_365"}   # only 42 and 365 in this dataset
    
    DATA_XLSX=r"...No_rejection_patients/LMM_no_rejections/LMM3/Features_gfr42slope_gfr365.xlsx",  # must include 42, 100, 365 columns
    OUT_DIR   = r"...No_rejection_patients/LMM_no_rejections/LMM3"

   # Modeling knobs
    RE_FORMULA = "1 + Time_c"     # random intercept + random slope
    N_FOLDS = 5
    EXCLUDE_365_OUTSIDE = (0, 200)
    FORCE_CATEGORICAL = [
        # e.g., "sex", "do_sex", "do_type2"  # add names of any numeric-coded categoricals if needed
    ]

    # Prediction style
    PREDICT_CONDITION_ON_PAST = True  # True = BLUP (uses observed gfr_42days); False = FE-only
    PAST_TIMES = [42]                 # used only when conditioning

    run_pipeline(
        DATA_XLSX=DATA_XLSX,
        OUT_DIR=OUT_DIR,
        ID_COL=ID_COL,
        TIME_COLS=TIME_COLS,
        PAST_TIMES=PAST_TIMES,
        INCLUDE_BASELINE_FEATURES=True,     # includes 'perc' and other baseline features
        INCLUDE_TIME_INTERACTIONS=True,     # Time_c × baseline features
        RE_FORMULA=RE_FORMULA,
        N_FOLDS=N_FOLDS,
        EXCLUDE_365_OUTSIDE=EXCLUDE_365_OUTSIDE,
        FORCE_CATEGORICAL=FORCE_CATEGORICAL,
        APPLY_GFR42_AT_365_ONLY=True,      # keep gfr_42 feature active only at 365 to avoid leakage
        PREDICT_CONDITION_ON_PAST=PREDICT_CONDITION_ON_PAST
    )

if __name__ == "__main__":
    main()