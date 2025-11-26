
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

global np


# -------------------- Helpers --------------------
def build_long(dfw_src, id_col, time_cols, feat_cols):
    import numpy as _np  # local alias avoids any accidental shadowing
    times_sorted = sorted(time_cols.keys())
    rows = []
    for t_day in times_sorted:
        col = time_cols[t_day]
        tmp = dfw_src[[id_col, col] + feat_cols].copy()
        tmp = tmp.rename(columns={col: "egfr"})
        tmp["time_days"] = t_day
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    long["time_years"] = long["time_days"] / 365.0
    tbar = float(_np.mean([d / 365.0 for d in times_sorted]))
    long["Time_c"] = long["time_years"] - tbar

    # mark the 365-day row using the normalized time
    FUTURE_DAY = max(times_sorted)  # usually 365
    target_year = FUTURE_DAY / 365.0
    long["is365"] = _np.isclose(long["time_years"], target_year, rtol=1e-12, atol=1e-12).astype(int)
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
    """Turn Q('col') or Q(\"col\") into bare column name."""
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
    - numeric feature names (possibly quoted: Q('age'))
    - categorical levels from patsy: C(Q('sex'))[T.M]
    """
    t = term.strip()
    if t == "Time_c":
        return time_c
    if t.startswith("C("):
        # C(Q('sex'))[T.M]
        i1 = t.find("("); i2 = t.rfind(")")
        inner = t[i1+1:i2].strip()  # e.g., Q('sex')
        var = _strip_Q(inner)       # -> 'sex'
        val = feat_row_dict.get(var, None)
        if "[T." in t and t.endswith("]"):
            level = t.split("[T.", 1)[1][:-1]
            return 1.0 if (val is not None and str(val) == level) else 0.0
        return 0.0  # reference level has no explicit FE column
    if t.startswith("Q("):  # e.g., Q('age')
        var = _strip_Q(t)
        return _safe_float(feat_row_dict.get(var, 0.0), 0.0)
    # plain numeric feature name
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


def predict_future(dfw_src, feat_cols, fit, id_col, time_cols, past_times, future_time=365,
                   condition_on_past=False):
    """
    Predict y_future for each patient.
    If condition_on_past=False: FE-only prediction = X_future @ beta
    If True: conditional BLUP using observed y_past.
    """
    if future_time not in time_cols:
        raise ValueError(f"future_time {future_time} missing from TIME_COLS.")
    fe_names = list(fit.fe_params.index)
    beta = fit.fe_params.values

    times = sorted(time_cols.keys())  # e.g., [42, 100, 365]
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

        # Build design for future=365 (is365 = 1)
        Xf = make_X_row(fe_names, Tc[future_time], feats, extra_time_vars={"is365": 1})[None, :]

        if not condition_on_past:
            mean_future = float(Xf @ beta)
            se_future = np.nan  # FE-only: no conditional variance computed
        else:
            # Collect past outcomes (is365 = 0 for past rows)
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
                # fall back to FE-only if no past observed for this patient
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
                var_future = float(Var_future - Cov_future_past @ np.linalg.solve(Var_past, Cov_future_past.T))
                se_future = float(np.sqrt(max(var_future, 0.0)))

        pred_rows.append({
            id_col: row[id_col],
            "gfr_42days": row.get(time_cols.get(42, ""), np.nan),
            "gfr_100days": row.get(time_cols.get(100, ""), np.nan),
            "pred_365_mean": mean_future,
            "pred_365_pi_low": mean_future - 1.96 * se_future if np.isfinite(se_future) else np.nan,
            "pred_365_pi_high": mean_future + 1.96 * se_future if np.isfinite(se_future) else np.nan,
            "observed_365": row.get(time_cols[future_time], np.nan),
        })
    return pd.DataFrame(pred_rows)


def group_cv(dfw_src, feat_cols, long_fit, formula, id_col, time_cols,
             past_times, future_time=365, n_splits=5, re_formula="1 + Time_c",
             condition_on_past=False):
    """
    Group CV by patient: fit on training folds; predict on held-out patients
    either FE-only or conditioning on selected past times.
    """
    # IDs actually used in training (outliers already excluded in long_fit)
    train_ids_all = long_fit[id_col].unique()
    dfw_clean = (
        dfw_src[dfw_src[id_col].isin(train_ids_all)]
        .drop_duplicates(id_col)
        .reset_index(drop=True)
    )

    # If not enough patients, shrink the #folds
    n_patients = dfw_clean[id_col].nunique()
    n_splits_eff = min(max(2, n_splits), n_patients) if n_patients >= 2 else 0
    if n_splits_eff < 2:
        return pd.DataFrame(columns=["fold", "R2", "RMSE"])

    groups = dfw_clean[id_col].values

    # global centering (consistent with build_long)
    times = sorted(time_cols.keys())
    tyears = {d: d/365.0 for d in times}
    tbar = float(np.mean([tyears[d] for d in times]))
    Tc = {d: tyears[d] - tbar for d in times}
    Z = {d: np.array([1.0, Tc[d]]) for d in times}

    cv_rows = []
    gkf = GroupKFold(n_splits=n_splits_eff)

    for fold, (tr, te) in enumerate(gkf.split(dfw_clean, groups=groups), start=1):
        train_ids = set(dfw_clean.iloc[tr][id_col].tolist())
        test_ids  = set(dfw_clean.iloc[te][id_col].tolist())

        # Fit on training subset of the long table
        lc_train = long_fit[long_fit[id_col].isin(train_ids)].copy()
        if lc_train[id_col].nunique() < 2:
            cv_rows.append({"fold": fold, "R2": np.nan, "RMSE": np.nan})
            continue

        try:
            res = smf.mixedlm(
                formula, lc_train, groups=lc_train[id_col], re_formula=re_formula
            ).fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
        except Exception:
            cv_rows.append({"fold": fold, "R2": np.nan, "RMSE": np.nan})
            continue

        fe_names = list(res.fe_params.index)
        beta_cv = res.fe_params.values

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

            # X at future (365), is365=1
            Xf = make_X_row(fe_names, Tc[future_time], feat_row, extra_time_vars={"is365": 1})[None, :]

            if not condition_on_past:
                yhat = float(Xf @ beta_cv)
            else:
                # Build past conditioning (is365=0)
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


def diagnostics(fit, long_fit, dfw_src, id_col, time_cols, out_dir):
    """Save residuals, random effects, and diagnostic figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conditional fitted and residuals
    fitted = fit.fittedvalues
    resid = long_fit["egfr"].values - fitted
    resid_df = long_fit[[id_col, "time_days", "egfr", "Time_c"]].copy()
    resid_df["fitted_cond"] = fitted
    resid_df["resid_cond"] = resid
    resid_df["visit"] = resid_df["time_days"].map({42: "d42", 100: "d100", 365: "d365"})
    resid_df.to_csv(out_dir / "residuals.csv", index=False)

    # Random effects
    re = fit.random_effects
    ids = long_fit[id_col].drop_duplicates().tolist()
    u0 = np.array([re[i][0] for i in ids])
    u1 = np.array([re[i][1] for i in ids])
    pd.DataFrame({id_col: ids, "u0_intercept": u0, "u1_slope": u1}).to_csv(out_dir / "random_effects.csv", index=False)

    # Variance components and ICC at modeled times
    G = fit.cov_re.values
    s2 = float(fit.scale)
    var_u0, var_u1, cov_u01 = float(G[0,0]), float(G[1,1]), float(G[0,1])

    def group_var(Tc): return var_u0 + 2*Tc*cov_u01 + (Tc**2)*var_u1

    times = sorted(time_cols.keys())
    tyears = {d: d/365.0 for d in times}
    tbar = float(np.mean([tyears[d] for d in times]))
    Tc = {d: tyears[d] - tbar for d in times}
    icc_dict = {d: group_var(Tc[d]) / (group_var(Tc[d]) + s2) for d in times}

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

    fig_q0 = sm.qqplot(u0, line="45", fit=True).figure; fig_q0.suptitle("Q–Q random intercepts")
    save_plot(fig_q0, "diag_qq_u0.png")

    fig_q1 = sm.qqplot(u1, line="45", fit=True).figure; fig_q1.suptitle("Q–Q random slopes")
    save_plot(fig_q1, "diag_qq_u1.png")

    fig = plt.figure(); plt.scatter(u0, u1, s=12); plt.xlabel("u0"); plt.ylabel("u1"); plt.title("u0 vs u1")
    save_plot(fig, "diag_scatter_u0_u1.png")

    # Numeric summary
    resid_mean = float(np.mean(resid))
    resid_sd = float(np.std(resid, ddof=1))
    resid_skew = float(pd.Series(resid).skew())
    resid_kurt = float(pd.Series(resid).kurt())
    pearson_corr = float(np.corrcoef(np.abs(resid), fitted)[0,1])
    spearman_corr = float(np.corrcoef(pd.Series(np.abs(resid)).rank(), pd.Series(fitted).rank())[0,1])

    icc_lines = " | ".join([f"{d}d: {icc_dict[d]:.3f}" for d in times])

    summary_text = f"""
Variance components:
  Var(u0) = {var_u0:.3f}, Var(u1) = {var_u1:.3f}, Cov(u0,u1) = {cov_u01:.3f}, sigma^2 = {s2:.3f}
ICC by modeled time: {icc_lines}

Residuals:
  mean = {resid_mean:.4f}, sd = {resid_sd:.3f}, skew = {resid_skew:.3f}, excess kurtosis = {resid_kurt:.3f}

Heteroscedasticity proxies:
  corr(|resid|, fitted): Pearson = {pearson_corr:.3f}, Spearman = {spearman_corr:.3f}
"""
    (out_dir / "diagnostics_summary.txt").write_text(summary_text.strip() + "\n", encoding="utf-8")


def run_pipeline(
    DATA_XLSX,
    OUT_DIR,
    ID_COL,
    TIME_COLS,                     # dict, e.g., {42:'gfr_42days', 100:'gfr_100days', 365:'gfr_365'}
    PAST_TIMES,                    # list of ints (subset of TIME_COLS keys) used only if conditioning
    INCLUDE_FEATURES=True,
    INCLUDE_TIME_INTERACTIONS=True,
    RE_FORMULA="1 + Time_c",
    N_FOLDS=5,
    EXCLUDE_365_OUTSIDE=(0, 200),
    FORCE_CATEGORICAL=None,
    INCLUDE_GFR_AS_FEATURES=True,          # <--- NEW: put gfr_42/100 on RHS as features
    APPLY_GFR_FEATURES_AT_365_ONLY=True,   # <--- NEW: use those features only at 365 (avoid leakage)
    PREDICT_CONDITION_ON_PAST=False        # <--- NEW: default FE-only predictions
):
    """
    One end-to-end run: fit MixedLM, predict future, CV, diagnostics.
    """
    if FORCE_CATEGORICAL is None:
        FORCE_CATEGORICAL = []

    OUT_DIR = Path(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load wide table
    dfw = pd.read_excel(DATA_XLSX)

    # Identify baseline features (everything except ID and all modeled GFR columns)
    gfr_cols = set(TIME_COLS.values())
    base_cols = [ID_COL] + list(gfr_cols)
    feat_cols = [c for c in dfw.columns if c not in base_cols]
    # Prevent leakage: exclude any other GFR-like columns that might be present
    feat_cols = [c for c in feat_cols if not c.lower().startswith("gfr_")]
    leaks = [c for c in feat_cols if c.lower().startswith("gfr_")]
    assert not leaks, f"GFR measurements on RHS not allowed: {leaks}"

    if not INCLUDE_FEATURES:
        feat_cols = []

    # Keep minimal wide with features (baseline only for now)
    dfw_keep = dfw[[ID_COL] + list(gfr_cols) + feat_cols].copy()

    # --- Add GFR past values as explicit features ---
    gfr_feat_cols = []
    if INCLUDE_GFR_AS_FEATURES:
        if 42 in TIME_COLS and TIME_COLS[42] in dfw_keep.columns:
            dfw_keep["feat_gfr42"] = dfw_keep[TIME_COLS[42]]
            gfr_feat_cols.append("feat_gfr42")
        if 100 in TIME_COLS and TIME_COLS[100] in dfw_keep.columns:
            dfw_keep["feat_gfr100"] = dfw_keep[TIME_COLS[100]]
            gfr_feat_cols.append("feat_gfr100")
        # add to feature list (avoid duplicates)
        for c in gfr_feat_cols:
            if c not in feat_cols:
                feat_cols.append(c)

    # Build long table and global-center time across modeled timepoints
    long = build_long(dfw_keep, ID_COL, TIME_COLS, feat_cols)

    # Exclude outlier 365 values for training only
    min365, max365 = EXCLUDE_365_OUTSIDE
    if min365 is not None and max365 is not None:
        bad_ids = set(dfw.loc[(dfw[TIME_COLS[365]] < min365) | (dfw[TIME_COLS[365]] > max365), ID_COL])
    else:
        bad_ids = set()

    if bad_ids:
        print(f"Excluding {len(bad_ids)} patient(s) from training due to out-of-range 365d eGFR.")

    long_fit = long[~long[ID_COL].isin(bad_ids)].copy()

    # ---- Build fixed-effects formula with robust quoting (no manual OHE) ----
    def q(col): return f"Q('{col}')"
    def is_cat(col):
        return (col in FORCE_CATEGORICAL) or (dfw[col].dtype == "O") or str(dfw[col].dtype).startswith("category")
    def fmt_term(col):  # C(Q('var')) for categoricals; Q('var') for numerics
        # Do NOT wrap feat_gfr* with C()
        if INCLUDE_GFR_AS_FEATURES and col in {"feat_gfr42", "feat_gfr100"}:
            return q(col)
        return f"C({q(col)})" if is_cat(col) else q(col)

    # Split baseline vs gfr-feature terms
    baseline_feats = [c for c in feat_cols if c not in {"feat_gfr42", "feat_gfr100"}]
    fe_terms = [fmt_term(c) for c in baseline_feats]

    # gfr-as-features terms
    gfr_feat_terms = []
    if INCLUDE_GFR_AS_FEATURES:
        has42f = "feat_gfr42" in feat_cols
        has100f = "feat_gfr100" in feat_cols
        if APPLY_GFR_FEATURES_AT_365_ONLY:
            if has42f:  gfr_feat_terms.append("Q('is365'):Q('feat_gfr42')")
            if has100f: gfr_feat_terms.append("Q('is365'):Q('feat_gfr100')")
        else:
            if has42f:  gfr_feat_terms.append("Q('feat_gfr42')")
            if has100f: gfr_feat_terms.append("Q('feat_gfr100')")

    rhs_terms = ["Time_c"] + fe_terms + gfr_feat_terms
    if INCLUDE_TIME_INTERACTIONS and fe_terms:
        rhs_terms += [f"Time_c:{t}" for t in fe_terms]
    rhs = " + ".join(list(dict.fromkeys(rhs_terms)))
    formula = f"egfr ~ {rhs}"
    print("\nFixed-effects formula:\n", formula, "\n")

    # Fit the LMM
    print("Fitting mixed model ...")
    model = smf.mixedlm(formula, long_fit, groups=long_fit[ID_COL], re_formula=RE_FORMULA)
    fit = model.fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
    print("Converged:", fit.converged)
    (OUT_DIR / "model_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")
    
    # ---- SAVE MODEL BUNDLE (for external application) ----
    import json
    
    bundle = {
        # Model pieces
        "formula": formula,
        "re_formula": RE_FORMULA,
        "fe_names": list(fit.fe_params.index),
        "fe_params": fit.fe_params.tolist(),
        "cov_re": fit.cov_re.values.tolist(),
        "sigma2": float(fit.scale),
    
        # Time / centering metadata
        "time_days": sorted(TIME_COLS.keys()),        # e.g., [42, 100, 365]
        "centering": "global_mean_over_time_days",    # informational
    
        # Column mapping and features used during training
        "id_col": ID_COL,
        "time_cols": TIME_COLS,                       # {"42": "gfr_42days", "100": "gfr_100days", "365": "gfr_365"}
        "feature_cols_baseline": baseline_feats,      # baseline (non-GFR) features
        "feature_cols_gfr": gfr_feat_cols,            # e.g., ["feat_gfr42", "feat_gfr100"]
        "feature_cols_all": baseline_feats + gfr_feat_cols,
    
        # Flags that affect how X rows are built at apply-time
        "include_time_interactions": bool(INCLUDE_TIME_INTERACTIONS),
        "include_gfr_as_features": bool(INCLUDE_GFR_AS_FEATURES),
        "apply_gfr_features_at_365_only": bool(APPLY_GFR_FEATURES_AT_365_ONLY),
    
        # Prediction defaults
        "predict_condition_on_past": bool(PREDICT_CONDITION_ON_PAST),
        "past_times": list(PAST_TIMES),
    
        # Optional: how categoricals were forced
        "force_categorical": list(FORCE_CATEGORICAL or []),
    }
    
    (out_path := OUT_DIR / "model_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print("Saved model bundle to:", out_path)


    # Predictions (either FE-only or conditional on y_past)
    pred = predict_future(
        dfw_keep, feat_cols, fit, ID_COL, TIME_COLS, past_times=PAST_TIMES,
        future_time=365, condition_on_past=PREDICT_CONDITION_ON_PAST
    )

    # Name outputs based on settings
    tag_feats = "with_baseline_feats" if INCLUDE_FEATURES else "no_baseline_feats"
    tag_gfr = "gfr_as_features"
    tag_scope = "365only" if APPLY_GFR_FEATURES_AT_365_ONLY else "all_times"
    tag_pred = "condBLUP" if PREDICT_CONDITION_ON_PAST else "FEonly"
    out_pred = OUT_DIR / f"pred_365_{tag_feats}_{tag_gfr}_{tag_scope}_{tag_pred}.xlsx"
    pred.to_excel(out_pred, index=False)
    print("Saved predictions to:", out_pred)

    # In-sample metrics on the training subset (only those we predicted and not excluded)
    mask_train = (~pred[ID_COL].isin(bad_ids)) & (~pred["observed_365"].isna())
    if mask_train.any():
        r2_in = r2_score(pred.loc[mask_train, "observed_365"], pred.loc[mask_train, "pred_365_mean"])
        rmse_in = float(np.sqrt(mean_squared_error(pred.loc[mask_train, "observed_365"], pred.loc[mask_train, "pred_365_mean"])))
        print(f"In-sample (training) R^2 = {r2_in:.3f}, RMSE = {rmse_in:.3f}")
        (OUT_DIR / "in_sample_metrics.txt").write_text(f"R2 = {r2_in:.6f}\nRMSE = {rmse_in:.6f}\n", encoding="utf-8")
    else:
        print("In-sample metrics skipped (no eligible rows).")

    # 5-fold Group CV (by patient)
    cv_df = pd.DataFrame(columns=["fold", "R2", "RMSE"])
    if N_FOLDS and N_FOLDS >= 2:
        cv_df = group_cv(
            dfw_keep, feat_cols, long_fit, formula, ID_COL, TIME_COLS,
            past_times=PAST_TIMES, future_time=365, n_splits=N_FOLDS, re_formula=RE_FORMULA,
            condition_on_past=PREDICT_CONDITION_ON_PAST
        )
        cv_df.to_csv(OUT_DIR / "cv5_metrics.csv", index=False)
        print("\nGroup CV:")
        print(cv_df)
        if cv_df["R2"].notna().any():
            summary = (f"Mean R^2 = {cv_df['R2'].mean():.3f} (SD {cv_df['R2'].std(ddof=1):.3f}); "
                       f"Mean RMSE = {cv_df['RMSE'].mean():.3f} (SD {cv_df['RMSE'].std(ddof=1):.3f})")
            print(summary)
            (OUT_DIR / "cv5_summary.txt").write_text(summary + "\n", encoding="utf-8")

    # Diagnostics
    diagnostics(fit, long_fit, dfw_keep, ID_COL, TIME_COLS, OUT_DIR)
    print("\nAll outputs written to:", OUT_DIR.resolve())


# -------------------- Main --------------------
def main():
    """
    Example run: use gfr_42days + gfr_100days as FEATURES (at 365 only), plus baseline features.
    Predictions are FE-only by default (no BLUP conditioning). Flip flags below if desired.
    """
    # ==== Common settings ====
    ID_COL = "patient_id"
    RE_FORMULA = "1 + Time_c"   # random intercept + random slope
    N_FOLDS = 5
    EXCLUDE_365_OUTSIDE = (0, 200)  # training only
    FORCE_CATEGORICAL = [
        # e.g., "sex", "do_sex", "do_type2"
    ]

    # ==== Paths (adjust to your files) ====
    DATA_XLSX=r"...LMM_no_rejections/LMM2/Features_gfr42100_gfr365.xlsx", 
    OUT_DIR   = r"...LMM_no_rejections/LMM2"

    # ==== Times ====
    TIME_COLS = {42: "gfr_42days", 100: "gfr_100days", 365: "gfr_365"}
    PAST_TIMES = [42, 100]   # only used if PREDICT_CONDITION_ON_PAST=True
    # --- Coerce paths in case a tuple/list was passed by mistake ---
    if isinstance(DATA_XLSX, (list, tuple)):
        DATA_XLSX = DATA_XLSX[0]
    if isinstance(OUT_DIR, (list, tuple)):
        OUT_DIR = OUT_DIR[0]
    DATA_XLSX = str(DATA_XLSX)
    OUT_DIR = Path(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        DATA_XLSX=DATA_XLSX,
        OUT_DIR=OUT_DIR,
        ID_COL=ID_COL,
        TIME_COLS=TIME_COLS,
        PAST_TIMES=PAST_TIMES,
        INCLUDE_FEATURES=True,                    # include baseline features from your sheet
        INCLUDE_TIME_INTERACTIONS=True,           # Time_c × baseline features
        RE_FORMULA=RE_FORMULA,
        N_FOLDS=N_FOLDS,
        EXCLUDE_365_OUTSIDE=EXCLUDE_365_OUTSIDE,
        FORCE_CATEGORICAL=FORCE_CATEGORICAL,
        INCLUDE_GFR_AS_FEATURES=True,             # <-- puts gfr_42/gfr_100 on RHS as features
        APPLY_GFR_FEATURES_AT_365_ONLY=True,      # <-- safe default; avoids leakage at 42/100 rows
        PREDICT_CONDITION_ON_PAST=False           # <-- FE-only predictions; set True to also BLUP-condition
    )


if __name__ == "__main__":
    main()
