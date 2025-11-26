# -*- coding: utf-8 -*-
"""
LMM (no features): predict eGFR at 365 days using ONLY the day-42 eGFR.
- Fixed effects: Intercept + Time_c
- Random effects: intercept + slope for Time_c
- No baseline covariates; no one-hot encoding.

Saves: predictions, 5-fold GroupCV metrics, model summary, residuals,
random effects BLUPs, and diagnostic plots.

Author: (you)
"""

import warnings
warnings.filterwarnings("ignore")  # keep console clean

# -------------------- Imports --------------------
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

# -------------------- USER SETTINGS --------------------
DATA_XLSX = r".../LLM_no_rejections/LMM1/only_gfr_42_and_365.xlsx"
OUT_DIR   = Path(r".../LLM_multiple_rejections/LMM1")

ID_COL = "patient_id"
# Model only day 42 and day 365
TIME_COLS = {42: "gfr_42days", 365: "gfr_365days"}

RE_FORMULA = "1 + Time_c"   # random intercept + random slope
N_FOLDS = 5                 # grouped CV folds (by patient)

# Outlier rule for 365-day eGFR in training (kept in predictions)
EXCLUDE_365_OUTSIDE = (0, 200)  # (min, max). Set to (None, None) to disable.

# -------------------- Helpers --------------------
def build_long(dfw_src, id_col, time_cols, feat_cols):
    """Wide -> long; add time_years and within-patient centered Time_c."""
    rows = []
    for t_day in sorted(time_cols.keys()):
        col = time_cols[t_day]
        tmp = dfw_src[[id_col, col] + feat_cols].copy()
        tmp = tmp.rename(columns={col: "egfr"})
        tmp["time_days"] = t_day
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    long["time_years"] = long["time_days"] / 365.0
    long["Time_c"] = long["time_years"] - long.groupby(id_col)["time_years"].transform("mean")
    return long

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
    """
    Evaluate a single component in a product (no ':'):
      - 'Time_c'
      - numeric feature names (possibly quoted: Q('age'))  [unused here]
      - categorical levels (C(Q('sex'))[T.M])              [unused here]
    """
    t = term.strip()
    if t == "Time_c":
        return time_c
    if t.startswith("C("):
        # No features in this script; categorical terms won't appear
        return 0.0
    if t.startswith("Q("):
        # No numeric features either
        return 0.0
    return _safe_float(feat_row_dict.get(t, 0.0), 0.0)

def make_X_row(fe_names, time_c, feat_row_dict):
    """
    Build one fixed-effects row aligned with fit.fe_params.index.
    With no features, this just produces [Intercept, Time_c].
    """
    x = np.zeros(len(fe_names), dtype=float)
    for i, name in enumerate(fe_names):
        if name == "Intercept":
            x[i] = 1.0
            continue
        parts = [p.strip() for p in name.split(":")]
        val = 1.0
        for p in parts:
            val *= _eval_term_component(p, time_c, feat_row_dict)
        x[i] = val
    return x

def predict_365(dfw_src, feat_cols, fit, id_col, time_cols):
    """
    Predict y_365 | (y_42) for each patient.
    Time centering uses only the modeled times (42 and 365).
    """
    fe_names = list(fit.fe_params.index)
    beta = fit.fe_params.values
    G = fit.cov_re.values
    sigma2 = float(fit.scale)

    # Centered times with only 42 & 365
    times = sorted(time_cols.keys())  # [42, 365]
    assert 365 in times and 42 in times, "TIME_COLS must include 42 and 365."
    tyears = {d: d/365.0 for d in times}
    tbar = float(np.mean([tyears[d] for d in times]))
    Tc = {d: tyears[d] - tbar for d in times}
    Z = {d: np.array([1.0, Tc[d]]) for d in times}

    pred_rows = []
    for _, row in dfw_src.iterrows():
        feats = {c: row[c] for c in feat_cols}  # empty in this no-feature setting

        # Condition only on day-42
        y_past, X_past, Z_past = [], [], []
        if pd.notna(row[time_cols[42]]):
            y_past.append(float(row[time_cols[42]]))
            X_past.append(make_X_row(fe_names, Tc[42], feats))
            Z_past.append(Z[42])

        if len(y_past) == 0:
            continue  # nothing to condition on

        y_past = np.asarray(y_past)
        X_past = np.vstack(X_past)
        Z_past = np.vstack(Z_past)
        X365   = make_X_row(fe_names, Tc[365], feats)[None, :]
        Z365v  = Z[365]

        Var_past = Z_past @ G @ Z_past.T + sigma2 * np.eye(len(y_past))
        Cov_future_past = Z365v @ G @ Z_past.T
        Var_future = float(Z365v @ G @ Z365v.T + sigma2)

        mean_future = float(X365 @ beta + Cov_future_past @
                            np.linalg.solve(Var_past, (y_past - X_past @ beta)))
        var_future  = float(Var_future - Cov_future_past @
                            np.linalg.solve(Var_past, Cov_future_past.T))
        se_future = float(np.sqrt(max(var_future, 0.0)))

        pred_rows.append({
            id_col: row[id_col],
            "gfr_42days": row.get(time_cols[42], np.nan),
            "pred_365_mean": mean_future,
            "pred_365_pi_low": mean_future - 1.96*se_future,
            "pred_365_pi_high": mean_future + 1.96*se_future,
            "observed_365": row.get(time_cols[365], np.nan),
        })

    return pd.DataFrame(pred_rows)

def group_cv(dfw_src, feat_cols, long_fit, formula, id_col, time_cols,
             n_splits=5, re_formula="1 + Time_c"):
    """Group CV by patient: train on folds; predict on held-out ids using day-42 only."""
    train_ids_all = long_fit[id_col].unique()
    dfw_clean = dfw_src[dfw_src[id_col].isin(train_ids_all)].drop_duplicates(id_col).reset_index(drop=True)
    groups = dfw_clean[id_col].values

    # Precompute centered times using only 42 & 365
    times = sorted(time_cols.keys())
    tyears = {d: d/365.0 for d in times}
    tbar = float(np.mean([tyears[d] for d in times]))
    Tc = {d: tyears[d] - tbar for d in times}
    Z = {d: np.array([1.0, Tc[d]]) for d in times}

    cv_rows = []
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr, te) in enumerate(gkf.split(dfw_clean, groups=groups), start=1):
        train_ids = set(dfw_clean.iloc[tr][id_col])
        test_ids  = set(dfw_clean.iloc[te][id_col])

        # fit on training subset of long_fit
        lc_train = long_fit[long_fit[id_col].isin(train_ids)].copy()
        res = smf.mixedlm(formula, lc_train, groups=lc_train[id_col], re_formula=re_formula).fit(
            reml=True, method="lbfgs", maxiter=2000, disp=False
        )

        fe_names = list(res.fe_params.index)
        beta_cv = res.fe_params.values
        G_cv = res.cov_re.values
        s2_cv = float(res.scale)

        preds, obs = [], []
        for pid in test_ids:
            roww = dfw_src[dfw_src[id_col]==pid]
            if roww.empty:
                continue
            roww = roww.iloc[0]
            feat_row = {c: roww[c] for c in feat_cols}  # empty

            # Condition only on 42
            y_past, X_past, Z_past = [], [], []
            y42 = roww.get(time_cols[42], np.nan)
            if pd.notna(y42):
                y_past.append(float(y42))
                X_past.append(make_X_row(fe_names, Tc[42], feat_row))
                Z_past.append(Z[42])
            if len(y_past) == 0:
                continue  # cannot predict conditionally

            y_past = np.asarray(y_past)
            X_past = np.vstack(X_past)
            Z_past = np.vstack(Z_past)
            X365 = make_X_row(fe_names, Tc[365], feat_row)[None, :]
            Z365 = Z[365]

            Vp = Z_past @ G_cv @ Z_past.T + s2_cv*np.eye(len(y_past))
            Cfp = Z365 @ G_cv @ Z_past.T
            yhat = float(X365 @ beta_cv + Cfp @ np.linalg.solve(Vp, (y_past - X_past @ beta_cv)))

            yobs = roww.get(time_cols[365], np.nan)
            if pd.notna(yobs):
                preds.append(yhat); obs.append(float(yobs))

        if len(preds) > 0:
            r2 = r2_score(obs, preds)
            rmse = float(np.sqrt(mean_squared_error(obs, preds)))
        else:
            r2, rmse = np.nan, np.nan
        cv_rows.append({"fold": fold, "R2": r2, "RMSE": rmse})

    return pd.DataFrame(cv_rows)

def diagnostics(fit, long_fit, dfw_src, id_col, out_dir):
    """Save residuals, random effects, and diagnostic figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conditional fitted and residuals
    fitted = fit.fittedvalues
    resid = long_fit["egfr"].values - fitted

    resid_df = long_fit[[id_col, "time_days", "egfr", "Time_c"]].copy()
    resid_df["fitted_cond"] = fitted
    resid_df["resid_cond"] = resid
    resid_df["visit"] = resid_df["time_days"].map({42: "d42", 365: "d365"})
    resid_df.to_csv(out_dir / "residuals.csv", index=False)

    # Random effects
    re = fit.random_effects
    ids = long_fit[id_col].drop_duplicates().tolist()
    u0 = np.array([re[i][0] for i in ids])
    u1 = np.array([re[i][1] for i in ids])
    pd.DataFrame({id_col: ids, "u0_intercept": u0, "u1_slope": u1}).to_csv(out_dir / "random_effects.csv", index=False)

    # Variance components and ICC
    G = fit.cov_re.values
    s2 = float(fit.scale)
    var_u0, var_u1, cov_u01 = float(G[0,0]), float(G[1,1]), float(G[0,1])
    def group_var(Tc): return var_u0 + 2*Tc*cov_u01 + (Tc**2)*var_u1
    t42, t365 = 42/365.0, 1.0
    tbar = (t42 + t365)/2.0
    Tc42, Tc365 = t42 - tbar, t365 - tbar
    icc_time0 = var_u0/(var_u0+s2)
    icc_42 = group_var(Tc42)/(group_var(Tc42)+s2)
    icc_365 = group_var(Tc365)/(group_var(Tc365)+s2)

    # Figures
    def save_plot(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)

    fig = plt.figure(); plt.scatter(fitted, resid, s=12); plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted (conditional)"); plt.ylabel("Residual"); plt.title("Residuals vs Fitted")
    save_plot(fig, "diag_resid_vs_fitted.png")

    fig = plt.figure(); plt.scatter(fitted, np.sqrt(np.abs(resid)), s=12)
    plt.xlabel("Fitted (conditional)"); plt.ylabel("Sqrt(|Residual|)"); plt.title("Scale–Location")
    save_plot(fig, "diag_scale_location.png")

    fig_qqr = sm.qqplot(resid, line="45", fit=True).figure
    fig_qqr.suptitle("Q–Q residuals")
    save_plot(fig_qqr, "diag_qq_residuals.png")

    fig = plt.figure(); plt.hist(resid, bins=30); plt.xlabel("Residual"); plt.ylabel("Freq"); plt.title("Residual histogram")
    save_plot(fig, "diag_hist_residuals.png")

    fig = plt.figure(); plt.hist(u0, bins=30); plt.xlabel("u0"); plt.ylabel("Freq"); plt.title("Random intercepts")
    save_plot(fig, "diag_hist_u0.png")

    fig = plt.figure(); plt.hist(u1, bins=30); plt.xlabel("u1"); plt.ylabel("Freq"); plt.title("Random slopes")
    save_plot(fig, "diag_hist_u1.png")

    fig_q0 = sm.qqplot(u0, line="45", fit=True).figure
    fig_q0.suptitle("Q–Q random intercepts")
    save_plot(fig_q0, "diag_qq_u0.png")

    fig_q1 = sm.qqplot(u1, line="45", fit=True).figure
    fig_q1.suptitle("Q–Q random slopes")
    save_plot(fig_q1, "diag_qq_u1.png")

    fig = plt.figure(); plt.scatter(u0, u1, s=12); plt.xlabel("u0"); plt.ylabel("u1"); plt.title("u0 vs u1")
    save_plot(fig, "diag_scatter_u0_u1.png")

    summary_text = f"""
Variance components:
    Var(u0) = {var_u0:.3f}, Var(u1) = {var_u1:.3f}, Cov(u0,u1) = {cov_u01:.3f}, sigma^2 = {s2:.3f}
ICC:
    Time_c=0: {icc_time0:.3f} | 42d: {icc_42:.3f} | 365d: {icc_365:.3f}

Residuals:
    mean = {np.mean(resid):.4f}, sd = {np.std(resid, ddof=1):.3f}, skew = {pd.Series(resid).skew():.3f}, excess kurtosis = {pd.Series(resid).kurt():.3f}
Heteroscedasticity proxies:
    corr(|resid|, fitted) Pearson = {np.corrcoef(np.abs(resid), fitted)[0,1]:.3f}, Spearman = {np.corrcoef(pd.Series(np.abs(resid)).rank(), pd.Series(fitted).rank())[0,1]:.3f}
"""
    (out_dir / "diagnostics_summary.txt").write_text(summary_text, encoding="utf-8")

# -------------------- Main --------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load wide table
    dfw = pd.read_excel(DATA_XLSX)

    # Identify GFR columns; ignore everything else on purpose (no features)
    base_cols = [ID_COL] + list(TIME_COLS.values())
    missing = [c for c in base_cols if c not in dfw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feat_cols = []  # <<<<<< NO FEATURES (intentionally ignore all other columns)

    # Keep only ID + the two GFR columns
    dfw_keep = dfw[[ID_COL] + list(TIME_COLS.values())].copy()

    # Build long table and center time (42 & 365 only)
    long = build_long(dfw_keep, ID_COL, TIME_COLS, feat_cols)

    # Exclude outlier 365 values (for model fitting only)
    min365, max365 = EXCLUDE_365_OUTSIDE
    if (min365 is not None) and (max365 is not None):
        bad_ids = set(dfw.loc[(dfw[TIME_COLS[365]] < min365) | (dfw[TIME_COLS[365]] > max365), ID_COL])
    else:
        bad_ids = set()
    if bad_ids:
        print(f"Excluding {len(bad_ids)} patient(s) from training due to out-of-range 365d eGFR:", sorted(bad_ids))
    long_fit = long[~long[ID_COL].isin(bad_ids)].copy()

    # ---- Fixed-effects formula: only Time_c ----
    formula = "egfr ~ Time_c"
    print("\nFixed-effects formula:\n", formula, "\n")

    # Fit the LMM
    print("Fitting mixed model ...")
    model = smf.mixedlm(formula, long_fit, groups=long_fit[ID_COL], re_formula=RE_FORMULA)
    fit = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
    # ---- SAVE MODEL BUNDLE (version-robust) ----
    import json
    
    bundle = {
        "fe_names": list(fit.fe_params.index),         # ["Intercept", "Time_c"]
        "fe_params": fit.fe_params.tolist(),           # fixed effects β
        "cov_re": fit.cov_re.values.tolist(),          # 2x2 random-effects covariance G
        "sigma2": float(fit.scale),                    # residual variance σ^2
        "time_days": sorted(TIME_COLS.keys()),         # [42, 365]
        "id_col": ID_COL,
        "time_cols": TIME_COLS                         # {"42": "gfr_42days", "365": "gfr_365days"}
    }
    (out_path := OUT_DIR / "model_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print("Saved model bundle to:", out_path)

    print("Converged:", fit.converged)
    (OUT_DIR / "model_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")

    # Predictions for all patients with a 42-day measurement
    pred = predict_365(dfw_keep, feat_cols, fit, ID_COL, TIME_COLS)
    out_pred = OUT_DIR / "pred_365_from_42_NO_FEATURES.xlsx"
    pred.to_excel(out_pred, index=False)
    print("Saved predictions to:", out_pred)

    # In-sample metrics on the training subset (only those we predicted)
    mask_train = (~pred[ID_COL].isin(bad_ids)) & (~pred["observed_365"].isna())
    if mask_train.any():
        r2_in = r2_score(pred.loc[mask_train, "observed_365"], pred.loc[mask_train, "pred_365_mean"])
        rmse_in = float(np.sqrt(mean_squared_error(pred.loc[mask_train, "observed_365"], pred.loc[mask_train, "pred_365_mean"])))
        print(f"In-sample (training)  R^2 = {r2_in:.3f}, RMSE = {rmse_in:.3f}")
    else:
        print("In-sample metrics skipped (no eligible rows).")

    # 5-fold Group CV (by patient), conditioning on 42d only
    cv_df = group_cv(dfw_keep, feat_cols, long_fit, formula, ID_COL, TIME_COLS,
                     n_splits=N_FOLDS, re_formula=RE_FORMULA)
    cv_df.to_csv(OUT_DIR / "cv5_metrics.csv", index=False)
    print("\n5-fold Group CV:")
    print(cv_df)
    if cv_df["R2"].notna().any():
        print(f"Mean R^2 = {cv_df['R2'].mean():.3f} (SD {cv_df['R2'].std(ddof=1):.3f}); "
              f"Mean RMSE = {cv_df['RMSE'].mean():.3f} (SD {cv_df['RMSE'].std(ddof=1):.3f})")

    # Diagnostics
    diagnostics(fit, long_fit, dfw_keep, ID_COL, OUT_DIR)
    print("\nAll outputs written to:", OUT_DIR.resolve())
    
    

if __name__ == "__main__":
    main()
    
    

