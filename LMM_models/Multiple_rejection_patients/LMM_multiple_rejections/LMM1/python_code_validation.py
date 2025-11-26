

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# ---------------- USER INPUT ----------------
MODEL_BUNDLE = Path(r".../LMM_no_rejections/LMM1/model_bundle.json")
EXTERNAL_XLSX = Path(r".../LMM_multiple_rejections/LMM1/mult_rej_patid_gfr42_gfr365_only.xlsx")  # <-- change this
OUT_DIR = Path(r".../LMM_multiple_rejections/LMM1")



# ------------- Load bundle & data ----------
bundle = json.loads(MODEL_BUNDLE.read_text(encoding="utf-8"))

fe_names = bundle["fe_names"]
beta = np.array(bundle["fe_params"], dtype=float)
G = np.array(bundle["cov_re"], dtype=float)
sigma2 = float(bundle["sigma2"])

# times/centering used in training (must be identical at apply time)
times = [int(t) for t in bundle["time_days"]]
assert 42 in times and 365 in times, "Bundle must include 42 and 365."

tyears = {d: d/365.0 for d in times}
tbar = float(np.mean([tyears[d] for d in times]))
Tc = {d: tyears[d] - tbar for d in times}               # centered times
Z  = {d: np.array([1.0, Tc[d]], dtype=float) for d in times}

id_col = bundle["id_col"]
time_cols = {int(k): v for k, v in bundle["time_cols"].items()}  # keys may be strings in JSON

df = pd.read_excel(EXTERNAL_XLSX)




# Validate required columns
required = [id_col, time_cols[42]]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"External file is missing required columns: {missing}")

# -------- Helpers (no features -> X = [1, Time_c]) --------
def make_X_row(fe_names, time_c):
    x = np.zeros(len(fe_names), dtype=float)
    for i, name in enumerate(fe_names):
        if name == "Intercept":
            x[i] = 1.0
        elif name == "Time_c":
            x[i] = time_c
        else:
            # Should not occur in the no-feature model
            x[i] = 0.0
    return x

# ------------- Predict for external -----------
pred_rows = []
for _, row in df.iterrows():
    y42 = row.get(time_cols[42], np.nan)
    if pd.isna(y42):
        continue  # need day-42 to condition

    # Past (day 42)
    y_past = np.array([float(y42)], dtype=float)
    X_past = make_X_row(fe_names, Tc[42])[None, :]      # shape (1, p)
    Z_past = Z[42][None, :]                             # shape (1, 2)

    # Future (day 365)
    X365  = make_X_row(fe_names, Tc[365])[None, :]
    Z365  = Z[365]

    # Gaussian conditioning
    Var_past = Z_past @ G @ Z_past.T + sigma2 * np.eye(1)
    Cov_fp   = Z365 @ G @ Z_past.T         # shape (2,)@(2x2)@(2,1) -> (1,1)
    mean_365 = float(X365 @ beta + Cov_fp @ np.linalg.solve(Var_past, (y_past - X_past @ beta)))
    Var_fut  = float(Z365 @ G @ Z365.T + sigma2)
    Var_cond = float(Var_fut - Cov_fp @ np.linalg.solve(Var_past, Cov_fp.T))
    se_365   = float(np.sqrt(max(Var_cond, 0.0)))

    pred_rows.append({
        id_col: row[id_col],
        "gfr_42days": y42,
        "pred_365_mean": mean_365,
        "pred_365_pi_low":  mean_365 - 1.96*se_365,
        "pred_365_pi_high": mean_365 + 1.96*se_365,
        "observed_365": row.get(time_cols.get(365, ""), np.nan) if (365 in time_cols and time_cols[365] in df.columns) else np.nan,
    })

OUT_DIR.mkdir(parents=True, exist_ok=True)
pred = pd.DataFrame(pred_rows)
pred_path = OUT_DIR / "external_pred_365_from_42_NO_FEATURES.xlsx"
pred.to_excel(pred_path, index=False)
print("Saved external predictions to:", pred_path)

# ------------- Optional: external metrics & plot -------------
if "observed_365" in pred and pred["observed_365"].notna().any():
    m = pred.dropna(subset=["observed_365"]).copy()
    r2   = r2_score(m["observed_365"], m["pred_365_mean"])
    rmse = float(np.sqrt(mean_squared_error(m["observed_365"], m["pred_365_mean"])))
    print(f"External evaluation: R^2 = {r2:.3f}, RMSE = {rmse:.3f}")
    m.to_csv(OUT_DIR / "external_obs_vs_pred.csv", index=False)

    plt.figure()
    plt.scatter(m["observed_365"], m["pred_365_mean"], s=16)
    lims = [min(m["observed_365"].min(), m["pred_365_mean"].min()),
            max(m["observed_365"].max(), m["pred_365_mean"].max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("Observed 365-day eGFR")
    plt.ylabel("Predicted 365-day eGFR")
    plt.title(f"External set: R²={r2:.3f}, RMSE={rmse:.2f}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "external_obs_vs_pred.png", dpi=150)
    plt.close()
else:
    print("No observed 365-day column found (or all missing). Metrics skipped.")
