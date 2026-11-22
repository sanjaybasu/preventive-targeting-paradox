#!/usr/bin/env python3
"""
Empirical Pillar 2 for `preventive-targeting-paradox`:
Real U.S. Randomized Preventive & Population-Health Spending-CATE and Winner's Curse Audit.

STRICT SCOPE RULE (Round-1 6-Lens Panel Consensus):
  Uses ONLY the three real U.S. preventive/population-health spending RCTs on disk:
    1. Illinois Workplace Wellness Study (Jones, Molitor, & Reif, QJE 2019; N = 4,834 randomized;
       N = 3,239 Yr1 / N = 3,307 30-month insured claims sample; 69 Strata_claims).
    2. Camden Hotspotting RCT (Finkelstein, Zhou, Taubman, & Doyle, NEJM 2020; N = 782 consented claims sample).
    3. Oregon Health Insurance Experiment (OHIE; Finkelstein et al., QJE 2012; Taubman et al., Science 2014;
       Baicker et al., NEJM 2013; N = 74,922 lottery / N = 24,646 ED admin / N = 23,741 12m survey /
       N = 12,229 in-person respondents of 20,745 sampled) — explicitly labeled as a
       Medicaid insurance expansion benchmark (not a targeted clinical prevention program).
  Plus AHRQ MEPS 2023 public-use event files for 2023-dollar unit-price sensitivity in OHIE.

  STRICTLY EXCLUDES all acute/critical-care hospital trials (gusto, ist, elaia1, elaia2, crash2)
  and non-U.S. neonatal trials (synbiotic). Zero PHI. Zero leakage from System Three.

Outputs:
  - results/pillar2_rct_audit.json
  - results/pillar2_rct_summary.csv
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from econml.dml import CausalForestDML
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="econml")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# =============================================================================
# GLOBAL CONSTANTS & STRICT SCOPE PATHS
# =============================================================================
SEED = 20261121
N_FOLDS = 5
N_PERM = 500
N_JOBS = min(12, max(1, (os.cpu_count() or 4) - 2))

ROOT = Path(__file__).resolve().parent.parent
ILLINOIS_DIR = Path(os.environ.get("ILLINOIS_DATA_DIR", str(ROOT / "data/raw/illinois/repo/data/csv")))
CAMDEN_DTA = Path(os.environ.get("CAMDEN_DTA_PATH", str(ROOT / "data/raw/camden/analysis_claims.dta")))
OREGON_ZIP = Path(os.environ.get("OREGON_ZIP_PATH", str(ROOT / "data/raw/oregon/oregon_puf.zip")))
MEPS_2023_DIR = ROOT / "data" / "raw"

OUT_DIR = ROOT / "results"
OUT_JSON = OUT_DIR / "pillar2_rct_audit.json"
OUT_CSV = OUT_DIR / "pillar2_rct_summary.csv"

FORBIDDEN_TOKENS = ("gusto", "ist_", "/ist/", "elaia", "crash2", "synbiotic")


def assert_scope_clean() -> None:
    """Enforce zero acute/non-U.S. trial leakage from System Three."""
    for p in (ILLINOIS_DIR, CAMDEN_DTA, OREGON_ZIP, MEPS_2023_DIR):
        s = str(p).lower()
        for tok in FORBIDDEN_TOKENS:
            assert tok not in s, f"SCOPE VIOLATION: forbidden token '{tok}' in path {p}"
        assert p.exists(), f"Missing required input path: {p}"


# =============================================================================
# STATISTICAL & ECONOMETRIC UTILITIES
# =============================================================================
def weighted_cluster_mean_se(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Exact Hajek weighted mean and cluster-robust (HC1 when unique clusters == N) SE.
    Equivalent to WLS(values ~ 1, weights=weights) with cluster-robust variance.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return {
            "n": 0, "n_clusters": 0, "mean": float("nan"), "se": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "t_stat": float("nan"), "p_two_sided": 1.0, "p_one_sided_savings": 1.0,
        }
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return {
            "n": 0, "n_clusters": 0, "mean": float("nan"), "se": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "t_stat": float("nan"), "p_two_sided": 1.0, "p_one_sided_savings": 1.0,
        }
    mu = float(np.sum(w * v) / w_sum)
    r = w * (v - mu)
    if groups is None:
        g_count = n
        ss = float(np.sum(r * r))
    else:
        g_arr = np.asarray(groups)
        _, inv = np.unique(g_arr, return_inverse=True)
        g_count = int(inv.max()) + 1
        r_cl = np.bincount(inv, weights=r, minlength=g_count)
        ss = float(np.sum(r_cl * r_cl))

    if g_count <= 1 or ss <= 0.0:
        se = 0.0
        t_stat = 0.0
        p2 = 1.0
        p1 = 1.0
    else:
        var = (g_count / (g_count - 1.0)) * ss / (w_sum * w_sum)
        se = float(math.sqrt(max(var, 0.0)))
        if se == 0.0:
            t_stat = 0.0
            p2 = 1.0
            p1 = 1.0
        else:
            t_stat = float(mu / se)
            df = max(g_count - 1, 1)
            p2 = float(2.0 * stats.t.sf(abs(t_stat), df=df))
            p1 = float(stats.t.cdf(t_stat, df=df))  # one-sided H1: mu < 0 (net savings)

    z = 1.959963984540054
    return {
        "n": int(n),
        "n_clusters": int(g_count),
        "mean": mu,
        "se": se,
        "ci_low": float(mu - z * se),
        "ci_high": float(mu + z * se),
        "t_stat": t_stat,
        "p_two_sided": p2,
        "p_one_sided_savings": p1,
    }


def wls_cluster_reg(
    y: np.ndarray,
    X_df: pd.DataFrame,
    target_col: str,
    weights: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Exact Stata-equivalent OLS/WLS with HC1 or household-clustered SEs."""
    X_mat = sm.add_constant(X_df.astype(np.float64), has_constant="add")
    y_vec = np.asarray(y, dtype=np.float64)
    if weights is None:
        mod = sm.OLS(y_vec, X_mat)
    else:
        mod = sm.WLS(y_vec, X_mat, weights=np.asarray(weights, dtype=np.float64))

    if groups is None:
        res = mod.fit(cov_type="HC1")
        n_cl = len(y_vec)
    else:
        g_arr = np.asarray(groups)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": g_arr})
        n_cl = int(len(np.unique(g_arr)))

    b = float(res.params[target_col])
    se = float(res.bse[target_col])
    pval = float(res.pvalues[target_col])
    z = 1.959963984540054
    return {
        "n": int(len(y_vec)),
        "n_clusters": n_cl,
        "estimate": b,
        "se": se,
        "ci_low": float(b - z * se),
        "ci_high": float(b + z * se),
        "p_value": pval,
    }


def weighted_quantiles_and_moments(
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute weighted mean, SD, skewness, excess kurtosis, median, P90/P95/P99/Max, zero_frac."""
    v = np.asarray(y, dtype=np.float64)
    w = np.ones(len(v), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w_norm = w / np.sum(w)
    mu = float(np.sum(w_norm * v))
    diff = v - mu
    var = float(np.sum(w_norm * diff ** 2))
    sd = math.sqrt(max(var, 0.0))
    skew = float(np.sum(w_norm * (diff ** 3)) / (sd ** 3)) if sd > 0 else 0.0
    ex_kurt = float(np.sum(w_norm * (diff ** 4)) / (sd ** 4) - 3.0) if sd > 0 else 0.0

    # Weighted quantiles
    order = np.argsort(v)
    v_s = v[order]
    cdf = np.cumsum(w_norm[order])
    def wq(q: float) -> float:
        idx = int(np.searchsorted(cdf, q))
        idx = min(max(idx, 0), len(v_s) - 1)
        return float(v_s[idx])

    return {
        "n": int(len(v)),
        "mean": mu,
        "sd": sd,
        "median": wq(0.50),
        "p90": wq(0.90),
        "p95": wq(0.95),
        "p99": wq(0.99),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
        "skewness": skew,
        "excess_kurtosis": ex_kurt,
        "zero_frac": float(np.sum(w_norm[v == 0.0])),
    }


def compute_theorem3_ncrit(
    sigma_gamma_annual: float,
    p_cov: int,
    n_actual: int,
    deltas: Tuple[float, ...] = (200.0, 500.0, 1000.0),
) -> Dict[str, Any]:
    """
    Exact Theorem 3 sample-size requirements to detect a genuine -Delta/yr subgroup saving:
      - Classical two-sided 80%-power single-subgroup requirement:
            n_80_sub = (z_{0.975} + z_{0.80})^2 * (sigma_Gamma / Delta)^2
        and total trial N for a q=20% targeted subgroup: N_80_q20 = n_80_sub / 0.20
      - Theorem 3 minimax policy-learning bound (Tsybakov margin exponent alpha):
            n_crit(alpha, v) = v * (sigma_Gamma / |Delta|)^((2 + alpha) / (1 + alpha))
        reported for alpha=0 (distribution-free) and alpha=1 (smooth density at zero),
        at v=1 and v=p_cov (covariate dimension).
    """
    z_factor = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) ** 2  # 7.84886...
    out: Dict[str, Any] = {
        "sigma_gamma_annual_usd": float(sigma_gamma_annual),
        "p_covariates": int(p_cov),
        "n_actual": int(n_actual),
        "z_factor_power80_alpha05": float(z_factor),
        "by_target_saving_usd_yr": {},
    }
    for d in deltas:
        ratio = sigma_gamma_annual / float(d)
        n_sub_80 = float(z_factor * (ratio ** 2))
        n_trial_q20_80 = float(n_sub_80 / 0.20)
        n_thm3_a0_v1 = float(ratio ** 2)
        n_thm3_a0_vp = float(p_cov * (ratio ** 2))
        n_thm3_a1_v1 = float(ratio ** 1.5)
        n_thm3_a1_vp = float(p_cov * (ratio ** 1.5))
        out["by_target_saving_usd_yr"][f"minus_{int(d)}"] = {
            "target_saving_usd_yr": -float(d),
            "snr_delta_over_sigma_gamma": float(d / sigma_gamma_annual) if sigma_gamma_annual > 0 else 0.0,
            "n_power80_single_subgroup": round(n_sub_80, 1),
            "N_power80_total_trial_q20": round(n_trial_q20_80, 1),
            "underpowered_multiple_q20": round(n_trial_q20_80 / max(n_actual, 1), 2),
            "n_crit_thm3_alpha0_v1": round(n_thm3_a0_v1, 1),
            "n_crit_thm3_alpha0_vp": round(n_thm3_a0_vp, 1),
            "n_crit_thm3_alpha1_v1": round(n_thm3_a1_v1, 1),
            "n_crit_thm3_alpha1_vp": round(n_thm3_a1_vp, 1),
        }
    return out


# =============================================================================
# MEPS 2023 UNIT COSTS (FOR OHIE 2023-USD SENSITIVITY)
# =============================================================================
def compute_meps_2023_unit_costs() -> Dict[str, float]:
    """Compute exact AHRQ MEPS 2023 survey-weighted mean expenditure per utilization event."""
    specs = {
        "er_visit": ("meps_2023_er.zip", "h248e.dta", "ERXP23X", "PERWT23F"),
        "ip_discharge": ("meps_2023_ip.zip", "h248d.dta", "IPXP23X", "PERWT23F"),
        "office_visit": ("meps_2023_ob.zip", "h248g.dta", "OBXP23X", "PERWT23F"),
        "outpatient_visit": ("meps_2023_op.zip", "h248f.dta", "OPXP23X", "PERWT23F"),
        "rx_fill": ("meps_2023_rx.zip", "h248a.dta", "RXXP23X", "PERWT23F"),
    }
    prices: Dict[str, float] = {}
    for key, (zf, fn, xp_col, wt_col) in specs.items():
        with zipfile.ZipFile(MEPS_2023_DIR / zf) as z:
            df = pd.read_stata(io.BytesIO(z.read(fn)), convert_categoricals=False)
        valid = df[(df[xp_col] >= 0) & (df[wt_col] > 0)]
        prices[key] = float(np.average(valid[xp_col].values, weights=valid[wt_col].values))
    # Finkelstein et al. (QJE 2012) annualized 6-month recall formula uses 3 fills per 6 months per current Rx
    # (2008 MEPS $52/fill * 3 = $156 -> * 2 = $312/yr per current Rx). We apply the identical 3 fills/6mo scaling.
    prices["rx_6mo_per_current_rx"] = 3.0 * prices["rx_fill"]
    return prices


# =============================================================================
# OHIE EXACT STACKED SPENDING REPLICATION (FINKELSTEIN ET AL., QJE 2012 TABLE V)
# =============================================================================
def stacked_ohie_spending_qje(
    df_s12: pd.DataFrame,
    unit_weights_6mo: Dict[str, float],
    first_stage_col: str = "ohp_all_ever_firstn_30sep2009",
) -> Dict[str, Any]:
    """
    Exact Python replication of OHIE `tables_analysis.do` stacked regression + lincom:
      2 * (w_rx * b_rx + w_doc * b_doc + w_er * b_er + w_hosp * b_hosp)
    with equation-specific `draw_survey_12m x numhh_list` fixed effects, normalized `weight_12m`,
    and household_id-clustered cross-equation covariance matrix.
    """
    eq_vars = ["rx_num_mod_12m", "doc_num_mod_12m", "er_num_mod_12m", "hosp_num_mod_12m"]
    mult_vec = np.array([2.0 * unit_weights_6mo[v] for v in eq_vars], dtype=np.float64)

    # Map every household_id in df_s12 to compact index 0..G-1
    hh_ids, hh_inv_all = np.unique(df_s12["household_id"].values, return_inverse=True)
    n_hh = len(hh_ids)
    n_eq = len(eq_vars)

    betas = np.zeros(n_eq, dtype=np.float64)
    c_means = np.zeros(n_eq, dtype=np.float64)
    eq_ns = np.zeros(n_eq, dtype=int)
    scores_hh = np.zeros((n_hh, n_eq), dtype=np.float64)

    # Also compute first-stage score for exact stacked delta-method LATE SE
    fs_sub = df_s12.dropna(subset=[first_stage_col, "weight_12m", "cell"]).copy()
    w_fs = fs_sub["weight_12m"].values / fs_sub["weight_12m"].mean()
    # Frisch-Waugh-Lovell cell demeaning
    cells_fs = fs_sub["cell"].values
    t_fs = fs_sub["treatment"].values.astype(np.float64)
    y_fs = fs_sub[first_stage_col].values.astype(np.float64)
    _, c_inv_fs = np.unique(cells_fs, return_inverse=True)
    w_cell = np.bincount(c_inv_fs, weights=w_fs)
    t_bar = np.bincount(c_inv_fs, weights=w_fs * t_fs) / w_cell
    y_bar = np.bincount(c_inv_fs, weights=w_fs * y_fs) / w_cell
    t_til = t_fs - t_bar[c_inv_fs]
    y_til = y_fs - y_bar[c_inv_fs]
    denom_fs = float(np.sum(w_fs * t_til * t_til))
    beta_fs = float(np.sum(w_fs * t_til * y_til) / denom_fs)
    e_fs = y_til - beta_fs * t_til
    hh_fs_idx = np.searchsorted(hh_ids, fs_sub["household_id"].values)
    score_fs_hh = np.bincount(hh_fs_idx, weights=w_fs * t_til * e_fs / denom_fs, minlength=n_hh)

    for m, var in enumerate(eq_vars):
        sub = df_s12.dropna(subset=[var, "weight_12m", "cell"]).copy()
        eq_ns[m] = len(sub)
        w = sub["weight_12m"].values / sub["weight_12m"].mean()
        t = sub["treatment"].values.astype(np.float64)
        y = sub[var].values.astype(np.float64)

        c_mask = t == 0.0
        c_means[m] = float(np.sum(w[c_mask] * y[c_mask]) / np.sum(w[c_mask]))

        _, c_inv = np.unique(sub["cell"].values, return_inverse=True)
        wc = np.bincount(c_inv, weights=w)
        tb = np.bincount(c_inv, weights=w * t) / wc
        yb = np.bincount(c_inv, weights=w * y) / wc
        tt = t - tb[c_inv]
        yt = y - yb[c_inv]
        denom = float(np.sum(w * tt * tt))
        b = float(np.sum(w * tt * yt) / denom)
        e = yt - b * tt
        betas[m] = b

        hh_idx = np.searchsorted(hh_ids, sub["household_id"].values)
        scores_hh[:, m] = np.bincount(hh_idx, weights=w * tt * e / denom, minlength=n_hh)

    # Stacked cluster-robust covariance matrix across the 4 equations + first stage
    scale = n_hh / (n_hh - 1.0)
    cov_beta = scale * (scores_hh.T @ scores_hh)
    itt_spend = float(mult_vec @ betas)
    se_itt = float(math.sqrt(max(float(mult_vec @ cov_beta @ mult_vec), 0.0)))
    t_itt = itt_spend / se_itt if se_itt > 0 else 0.0
    p_itt = float(2.0 * stats.t.sf(abs(t_itt), df=n_hh - 1))
    c_mean_spend = float(mult_vec @ c_means)

    # Delta-method LATE = itt_spend / beta_fs using joint household influence vector
    s_itt_hh = scores_hh @ mult_vec
    late_spend = float(itt_spend / beta_fs)
    s_late_hh = (s_itt_hh - late_spend * score_fs_hh) / beta_fs
    se_late = float(math.sqrt(scale * float(np.sum(s_late_hh ** 2))))
    p_late = float(2.0 * stats.norm.sf(abs(late_spend / se_late))) if se_late > 0 else 1.0

    z = 1.959963984540054
    return {
        "n_households": int(n_hh),
        "eq_ns": {eq_vars[i]: int(eq_ns[i]) for i in range(n_eq)},
        "control_mean_annual_usd": c_mean_spend,
        "itt_annual_usd": itt_spend,
        "itt_se": se_itt,
        "itt_ci_low": float(itt_spend - z * se_itt),
        "itt_ci_high": float(itt_spend + z * se_itt),
        "itt_p_value": p_itt,
        "first_stage": beta_fs,
        "late_annual_usd": late_spend,
        "late_se": se_late,
        "late_ci_low": float(late_spend - z * se_late),
        "late_ci_high": float(late_spend + z * se_late),
        "late_p_value": p_late,
    }


# =============================================================================
# CROSS-FITTING, DR-LEARNER, CAUSAL FOREST, BLP/GATES & WINNER'S CURSE ENGINE
# =============================================================================
def make_stratified_group_folds(
    strata: np.ndarray,
    treat: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """
    Create deterministic stratified cluster-respecting folds.
    Every cluster (e.g., OHIE household_id) stays intact in a single fold.
    """
    g_arr = np.asarray(groups)
    uniq_g, first_idx, inv = np.unique(g_arr, return_index=True, return_inverse=True)
    g_strata = np.asarray(strata)[first_idx]
    g_treat = np.asarray(treat)[first_idx].astype(int)

    # Combine stratum and treatment; collapse tiny cells (< n_splits) into treatment-only label
    raw_key = pd.Series(g_strata).astype(str) + "_" + pd.Series(g_treat).astype(str)
    vc = raw_key.value_counts()
    small = set(vc[vc < n_splits].index)
    strat_key = np.where(raw_key.isin(small), "SMALL_" + pd.Series(g_treat).astype(str), raw_key)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    g_folds = np.zeros(len(uniq_g), dtype=int)
    for k, (_, te_idx) in enumerate(skf.split(np.zeros(len(uniq_g)), strat_key)):
        g_folds[te_idx] = k
    return g_folds[inv]


def compute_stratum_propensity(
    W_tr: np.ndarray,
    strata_tr: np.ndarray,
    strata_te: np.ndarray,
) -> np.ndarray:
    """Exact design-stratum propensity estimated on training split, clipped to [0.05, 0.95]."""
    global_p = float(np.clip(np.mean(W_tr), 0.05, 0.95))
    df_tr = pd.DataFrame({"s": strata_tr, "w": W_tr})
    p_map = df_tr.groupby("s")["w"].mean().clip(0.05, 0.95).to_dict()
    return np.array([p_map.get(s, global_p) for s in strata_te], dtype=np.float64)


def impute_and_scale(
    X_tr: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit median imputation + standardization strictly on training fold."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_tr_imp = np.where(np.isnan(X_tr), med, X_tr)
    X_te_imp = np.where(np.isnan(X_te), med, X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)
    return X_tr_s, X_te_s


def fit_predict_ensemble(
    X_tr_s: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_te_s: np.ndarray,
    seed: int,
    n_estimators: int = 80,
    max_depth: int = 3,
    num_leaves: int = 8,
    min_child_samples: int = 25,
    reg_lambda: float = 10.0,
) -> np.ndarray:
    """50/50 ensemble of RidgeCV + regularized LightGBM regressor."""
    ridge = RidgeCV(alphas=np.logspace(-1, 5, 13))
    ridge.fit(X_tr_s, y_tr, sample_weight=w_tr)
    pred_r = ridge.predict(X_te_s)

    eff_min_child = max(5, min(min_child_samples, max(5, len(y_tr) // 20)))
    gbm = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=eff_min_child,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        n_jobs=1,
        verbose=-1,
        random_state=seed,
    )
    gbm.fit(X_tr_s, y_tr, sample_weight=w_tr)
    pred_g = gbm.predict(X_te_s)
    return 0.5 * pred_r + 0.5 * pred_g


def compute_blp_and_gates(
    psi: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Any]:
    """
    Chernozhukov, Demirer, Duflo, & Fernandez-Val (Econometrica 2025)
    BLP (beta_1 ATE, beta_2 heterogeneity loading) and GATES (quartile CATEs G1..G4 & G4 - G1),
    evaluated on strictly cross-fitted AIPW scores psi and strictly cross-fitted score S(X).
    Quartile convention:
      G1 = lowest 25% of predicted CATE S(X) (predicted MOST cost-saving)
      G4 = highest 25% of predicted CATE S(X) (predicted LEAST cost-saving / highest net cost)
    """
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(psi, dtype=np.float64)
    s_mean = float(np.sum(w * s) / np.sum(w))
    s_centered = s - s_mean

    # BLP: psi ~ 1 + (S - S_bar) with weights and cluster-robust SEs
    if np.std(s_centered) < 1e-10:
        ate_res = weighted_cluster_mean_se(y, w, groups)
        blp = {
            "beta1_ate": ate_res["mean"],
            "beta1_se": ate_res["se"],
            "beta1_p": ate_res["p_two_sided"],
            "beta2_het": 0.0,
            "beta2_se": float("nan"),
            "beta2_ci_low": float("nan"),
            "beta2_ci_high": float("nan"),
            "beta2_p": 1.0,
            "score_sd": float(np.std(s)),
        }
    else:
        X_blp = pd.DataFrame({"s_centered": s_centered})
        X_mat = sm.add_constant(X_blp, has_constant="add")
        mod = sm.WLS(y, X_mat, weights=w)
        if len(np.unique(groups)) == len(y):
            res = mod.fit(cov_type="HC1")
        else:
            res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
        b1, se1, p1 = float(res.params["const"]), float(res.bse["const"]), float(res.pvalues["const"])
        b2, se2, p2 = float(res.params["s_centered"]), float(res.bse["s_centered"]), float(res.pvalues["s_centered"])
        z = 1.959963984540054
        blp = {
            "beta1_ate": b1,
            "beta1_se": se1,
            "beta1_p": p1,
            "beta2_het": b2,
            "beta2_se": se2,
            "beta2_ci_low": float(b2 - z * se2),
            "beta2_ci_high": float(b2 + z * se2),
            "beta2_p": p2,
            "score_sd": float(np.sqrt(np.sum(w * (s_centered ** 2)) / np.sum(w))),
        }

    # GATES: quartiles of S(X) (deterministic rank tie-breaking with fixed noise-free first-order rank)
    ranks = pd.Series(s).rank(method="first").values
    q_labels = pd.qcut(ranks, 4, labels=[1, 2, 3, 4]).astype(int)

    gates_q: Dict[str, Any] = {}
    for q in (1, 2, 3, 4):
        m = q_labels == q
        st = weighted_cluster_mean_se(y[m], w[m], groups[m])
        gates_q[f"G{q}"] = {
            "n": st["n"],
            "mean_pred_score": float(np.sum(w[m] * s[m]) / np.sum(w[m])),
            "cate": st["mean"],
            "se": st["se"],
            "ci_low": st["ci_low"],
            "ci_high": st["ci_high"],
            "p_two_sided": st["p_two_sided"],
            "p_one_sided_savings": st["p_one_sided_savings"],
        }

    # Exact cluster-robust G4 - G1 contrast via regression on quartile dummies
    D_df = pd.DataFrame({
        "q1": (q_labels == 1).astype(float),
        "q2": (q_labels == 2).astype(float),
        "q3": (q_labels == 3).astype(float),
        "q4": (q_labels == 4).astype(float),
    })
    mod_g = sm.WLS(y, D_df, weights=w)
    if len(np.unique(groups)) == len(y):
        res_g = mod_g.fit(cov_type="HC1")
    else:
        res_g = mod_g.fit(cov_type="cluster", cov_kwds={"groups": groups})
    cov_g = res_g.cov_params()
    diff_41 = float(res_g.params["q4"] - res_g.params["q1"])
    var_41 = float(cov_g.loc["q4", "q4"] + cov_g.loc["q1", "q1"] - 2.0 * cov_g.loc["q4", "q1"])
    se_41 = math.sqrt(max(var_41, 0.0))
    df_g = max(int(len(np.unique(groups))) - 4, 1)
    p_41 = float(2.0 * stats.t.sf(abs(diff_41 / se_41), df=df_g)) if se_41 > 0 else 1.0
    z = 1.959963984540054

    return {
        "blp": blp,
        "gates": gates_q,
        "gates_spread_G4_minus_G1": {
            "estimate": diff_41,
            "se": se_41,
            "ci_low": float(diff_41 - z * se_41),
            "ci_high": float(diff_41 + z * se_41),
            "p_two_sided": p_41,
        },
    }


def evaluate_policy_rule(
    pi_mask: np.ndarray,
    psi_eval: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    c0: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate binary targeting policy pi_mask in {0, 1} on AIPW scores psi_eval at program cost c0.
    Reports BOTH per-eligible-population net cost and conditional per-targeted-enrollee net cost.
    """
    pi = np.asarray(pi_mask, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w_tot = float(np.sum(w))
    q_share = float(np.sum(w * pi) / w_tot) if w_tot > 0 else 0.0
    n_targ = int(np.sum(pi > 0.5))

    # Per-eligible-population net cost: (1/sum w) * sum w_i * pi_i * (psi_i + c0)
    u_pop = pi * (psi_eval + c0)
    pop_res = weighted_cluster_mean_se(u_pop, w, groups)

    # Conditional per-targeted-enrollee net cost: E[psi + c0 | pi = 1]
    if n_targ > 0:
        m = pi > 0.5
        targ_res = weighted_cluster_mean_se(psi_eval[m] + c0, w[m], groups[m])
    else:
        targ_res = {
            "n": 0, "n_clusters": 0, "mean": 0.0, "se": 0.0,
            "ci_low": 0.0, "ci_high": 0.0, "t_stat": 0.0,
            "p_two_sided": 1.0, "p_one_sided_savings": 1.0,
        }

    return {
        "c0": float(c0),
        "n_targeted": n_targ,
        "targeted_share": q_share,
        "net_cost_per_eligible": pop_res["mean"],
        "se_per_eligible": pop_res["se"],
        "ci_low_per_eligible": pop_res["ci_low"],
        "ci_high_per_eligible": pop_res["ci_high"],
        "p_one_sided_savings": pop_res["p_one_sided_savings"],
        "p_two_sided": pop_res["p_two_sided"],
        "net_cost_per_targeted": targ_res["mean"],
        "se_per_targeted": targ_res["se"],
        "ci_low_per_targeted": targ_res["ci_low"],
        "ci_high_per_targeted": targ_res["ci_high"],
    }


def evaluate_trial_cate_and_winners_curse(
    trial_id: str,
    label: str,
    unit: str,
    unit_scale_to_annual: float,
    X_df: pd.DataFrame,
    Y: np.ndarray,
    W: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    groups: np.ndarray,
    baseline_risk_col: Optional[str],
    c0_grid: List[float],
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Run complete Pillar 2 Components 2 & 3 on one RCT spending arm:
      - Zero-leakage 5-fold stratified/household-clustered outer cross-fitting
      - DR-Learner (LightGBM + Ridge ensemble with nested inner cross-fitting)
      - Honest Causal Forest (econml.dml.CausalForestDML)
      - Baseline-Spending / Risk-Score Targeting
      - Naive In-Sample Plug-In ML (same LightGBM + Ridge learner fit without splitting)
      - BLP, GATES, optimal sign rule across c0_grid, top-q curves (10/20/30/50/100%),
        and Empirical Winner's Curse Gap.
    """
    t0 = time.time()
    n, p = X_df.shape
    X_raw = X_df.values.astype(np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)
    s_arr = np.asarray(strata)
    g_arr = np.asarray(groups)

    folds = make_stratified_group_folds(s_arr, W_arr, g_arr, n_splits=N_FOLDS, seed=seed)

    # Pre-encode strata dummies for CausalForestDML nuisance control W
    strata_dummies = pd.get_dummies(pd.Series(s_arr), drop_first=True, dtype=float).values
    use_strata_w = strata_dummies.shape[1] > 0

    # Containers for strictly out-of-fold predictions & evaluation nuisances
    e_oof = np.zeros(n, dtype=np.float64)
    mu0_oof = np.zeros(n, dtype=np.float64)
    mu1_oof = np.zeros(n, dtype=np.float64)
    m_pool_oof = np.zeros(n, dtype=np.float64)
    psi_eval = np.zeros(n, dtype=np.float64)
    theta_dr_oof = np.zeros(n, dtype=np.float64)
    theta_cf_oof = np.zeros(n, dtype=np.float64)

    # Out-of-fold top-q masks (threshold computed strictly on training fold to prevent test leakage!)
    q_grid = [0.10, 0.20, 0.30, 0.50, 1.00]
    topq_dr_oof = {q: np.zeros(n, dtype=bool) for q in q_grid}
    topq_cf_oof = {q: np.zeros(n, dtype=bool) for q in q_grid}
    topq_risk_oof = {q: np.zeros(n, dtype=bool) for q in q_grid}

    # Baseline risk score vector (strictly pre-treatment; if None, use cross-fitted mu0_oof!)
    has_direct_risk = baseline_risk_col is not None and baseline_risk_col in X_df.columns
    direct_risk_raw = X_df[baseline_risk_col].fillna(0.0).values.astype(np.float64) if has_direct_risk else None

    for k in range(N_FOLDS):
        tr = folds != k
        te = folds == k

        X_tr_s, X_te_s = impute_and_scale(X_raw[tr], X_raw[te])
        Y_tr, W_tr, w_tr = Y_arr[tr], W_arr[tr], w_arr[tr]
        s_tr, g_tr = s_arr[tr], g_arr[tr]

        # 1. Outer evaluation nuisances (strictly fit on tr!)
        e_te = compute_stratum_propensity(W_tr, s_tr, s_arr[te])
        e_oof[te] = e_te

        m0_mask = W_tr == 0
        m1_mask = W_tr == 1
        mu0_te = fit_predict_ensemble(X_tr_s[m0_mask], Y_tr[m0_mask], w_tr[m0_mask], X_te_s, seed=seed + 100 + k)
        mu1_te = fit_predict_ensemble(X_tr_s[m1_mask], Y_tr[m1_mask], w_tr[m1_mask], X_te_s, seed=seed + 200 + k)
        m_pool_te = fit_predict_ensemble(X_tr_s, Y_tr, w_tr, X_te_s, seed=seed + 300 + k)

        mu0_oof[te] = mu0_te
        mu1_oof[te] = mu1_te
        m_pool_oof[te] = m_pool_te

        mu_w_te = np.where(W_arr[te] == 1, mu1_te, mu0_te)
        psi_eval[te] = (mu1_te - mu0_te) + ((W_arr[te] - e_te) / (e_te * (1.0 - e_te))) * (Y_arr[te] - mu_w_te)

        # 2. DR-Learner Stage 2 via nested inner 4-fold cross-fitting strictly inside tr (ZERO fold leakage!)
        inner_folds = make_stratified_group_folds(s_tr, W_tr, g_tr, n_splits=4, seed=seed + 1000 + k)
        psi_tr_inner = np.zeros(len(Y_tr), dtype=np.float64)
        for j in range(4):
            itr = inner_folds != j
            ite = inner_folds == j
            e_ite = compute_stratum_propensity(W_tr[itr], s_tr[itr], s_tr[ite])
            im0 = itr & (W_tr == 0)
            im1 = itr & (W_tr == 1)
            mu0_ite = fit_predict_ensemble(X_tr_s[im0], Y_tr[im0], w_tr[im0], X_tr_s[ite], seed=seed + 400 + k * 10 + j, n_estimators=60)
            mu1_ite = fit_predict_ensemble(X_tr_s[im1], Y_tr[im1], w_tr[im1], X_tr_s[ite], seed=seed + 500 + k * 10 + j, n_estimators=60)
            mu_w_ite = np.where(W_tr[ite] == 1, mu1_ite, mu0_ite)
            psi_tr_inner[ite] = (mu1_ite - mu0_ite) + ((W_tr[ite] - e_ite) / (e_ite * (1.0 - e_ite))) * (Y_tr[ite] - mu_w_ite)

        # Fit DR-Learner Stage-2 on (X_tr_s, psi_tr_inner) and predict both tr (for OOF quantile cutoffs) & te
        # Using a single fitted object so train-fold quantile threshold matches test-fold prediction scale!
        ridge_dr = RidgeCV(alphas=np.logspace(-1, 5, 13))
        ridge_dr.fit(X_tr_s, psi_tr_inner, sample_weight=w_tr)
        eff_min_child = max(10, min(25, len(Y_tr) // 20))
        gbm_dr = lgb.LGBMRegressor(
            n_estimators=80, max_depth=3, num_leaves=8, min_child_samples=eff_min_child,
            learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=10.0, n_jobs=1, verbose=-1, random_state=seed + 600 + k,
        )
        gbm_dr.fit(X_tr_s, psi_tr_inner, sample_weight=w_tr)
        th_dr_te = 0.5 * ridge_dr.predict(X_te_s) + 0.5 * gbm_dr.predict(X_te_s)
        th_dr_tr = 0.5 * ridge_dr.predict(X_tr_s) + 0.5 * gbm_dr.predict(X_tr_s)
        theta_dr_oof[te] = th_dr_te

        # 3. Honest Causal Forest (econml.dml.CausalForestDML) fit strictly on tr with cluster-respecting inner CV
        cf_inner_folds = make_stratified_group_folds(s_tr, W_tr, g_tr, n_splits=3, seed=seed + 2000 + k)
        cf_cv_splits = [(np.where(cf_inner_folds != j)[0], np.where(cf_inner_folds == j)[0]) for j in range(3)]
        cf = CausalForestDML(
            model_y=RidgeCV(alphas=np.logspace(-1, 5, 9)),
            model_t=LogisticRegression(C=10.0, max_iter=500),
            discrete_treatment=True,
            n_estimators=200,
            min_samples_leaf=max(15, min(25, len(Y_tr) // 25)),
            max_depth=5,
            honest=True,
            cv=cf_cv_splits,
            n_jobs=min(4, N_JOBS),
            random_state=seed + 700 + k,
        )
        W_ctrl_tr = strata_dummies[tr] if use_strata_w else None
        cf.fit(Y_tr, W_tr, X=X_tr_s, W=W_ctrl_tr, sample_weight=w_tr)
        th_cf_te = cf.effect(X_te_s).ravel()
        th_cf_tr = cf.effect(X_tr_s).ravel()
        theta_cf_oof[te] = th_cf_te

        # 4. Baseline risk score on tr & te (for OHIE-Survey without baseline spending col, use mu0_oof predicted on tr/te)
        if has_direct_risk:
            r_tr = direct_risk_raw[tr]
            r_te = direct_risk_raw[te]
        else:
            # Predict baseline spending mu0 on tr & te using ridge fit on control arm of tr
            r_mod = RidgeCV(alphas=np.logspace(-1, 5, 9))
            r_mod.fit(X_tr_s[m0_mask], Y_tr[m0_mask], sample_weight=w_tr[m0_mask])
            r_tr = r_mod.predict(X_tr_s)
            r_te = r_mod.predict(X_te_s)

        # Compute strictly out-of-fold top-q targeting masks:
        # For DR & CF: target lowest q fraction of predicted spending CATE (most cost-saving)
        # For Risk-Score: target highest q fraction of baseline spending/utilization
        for q in q_grid:
            if q >= 0.999:
                topq_dr_oof[q][te] = True
                topq_cf_oof[q][te] = True
                topq_risk_oof[q][te] = True
            else:
                topq_dr_oof[q][te] = th_dr_te <= np.quantile(th_dr_tr, q)
                topq_cf_oof[q][te] = th_cf_te <= np.quantile(th_cf_tr, q)
                topq_risk_oof[q][te] = r_te >= np.quantile(r_tr, 1.0 - q)

    # Baseline risk score vector for BLP/GATES (negative sign so "lower score = higher baseline spend" matches G1 convention,
    # while also recording direct positive baseline spending quartile ordering cleanly!)
    risk_score_full = direct_risk_raw if has_direct_risk else mu0_oof

    # -------------------------------------------------------------------------
    # NAIVE IN-SAMPLE PLUG-IN ML (Same LightGBM + Ridge DR-Learner fit WITHOUT sample splitting!)
    # -------------------------------------------------------------------------
    X_all_s, _ = impute_and_scale(X_raw, X_raw)
    e_in = compute_stratum_propensity(W_arr, s_arr, s_arr)
    m0_all = W_arr == 0
    m1_all = W_arr == 1
    mu0_in = fit_predict_ensemble(X_all_s[m0_all], Y_arr[m0_all], w_arr[m0_all], X_all_s, seed=seed + 9001)
    mu1_in = fit_predict_ensemble(X_all_s[m1_all], Y_arr[m1_all], w_arr[m1_all], X_all_s, seed=seed + 9002)
    mu_w_in = np.where(W_arr == 1, mu1_in, mu0_in)
    psi_in = (mu1_in - mu0_in) + ((W_arr - e_in) / (e_in * (1.0 - e_in))) * (Y_arr - mu_w_in)

    # Standard unsplit DR-Learner (50/50 Ridge + LightGBM fit & evaluated on the SAME full sample)
    theta_dr_in = fit_predict_ensemble(
        X_all_s, psi_in, w_arr, X_all_s, seed=seed + 9003,
        n_estimators=100, max_depth=4, num_leaves=15, min_child_samples=max(10, min(15, n // 30)), reg_lambda=5.0,
    )

    # Also evaluate Naive In-Sample with the EXACT SAME hyperparameters as cross-fitted DR-Learner
    # so the Winner's Curse Gap is 100% attributable to sample splitting alone!
    theta_dr_in_matched = fit_predict_ensemble(
        X_all_s, psi_in, w_arr, X_all_s, seed=seed + 9004,
        n_estimators=80, max_depth=3, num_leaves=8, min_child_samples=max(10, min(25, n // 20)), reg_lambda=10.0,
    )

    # -------------------------------------------------------------------------
    # COMPONENT 2: Expenditure Distribution, Heavy-Tail Diagnostics & Theorem 3 n_crit
    # -------------------------------------------------------------------------
    dist_native = weighted_quantiles_and_moments(Y_arr, w_arr)
    resid_oof = Y_arr - m_pool_oof
    w_norm = w_arr / np.sum(w_arr)
    sigma_resid = float(math.sqrt(np.sum(w_norm * (resid_oof - np.sum(w_norm * resid_oof)) ** 2)))
    sigma_gamma = float(math.sqrt(np.sum(w_norm * (psi_eval - np.sum(w_norm * psi_eval)) ** 2)))
    r2_oof = float(1.0 - (sigma_resid ** 2) / (dist_native["sd"] ** 2)) if dist_native["sd"] > 0 else 0.0

    thm3_ncrit = compute_theorem3_ncrit(
        sigma_gamma_annual=sigma_gamma * unit_scale_to_annual,
        p_cov=p,
        n_actual=n,
    )

    # -------------------------------------------------------------------------
    # COMPONENT 3A: BLP & GATES (Chernozhukov et al. 2025)
    # -------------------------------------------------------------------------
    blp_gates_dr = compute_blp_and_gates(psi_eval, theta_dr_oof, w_arr, g_arr)
    blp_gates_cf = compute_blp_and_gates(psi_eval, theta_cf_oof, w_arr, g_arr)
    # For Baseline Spending Risk-Score: use -risk_score_full so G1 = highest baseline spending quartile (matches G1 = top-priority)
    blp_gates_risk = compute_blp_and_gates(psi_eval, -risk_score_full, w_arr, g_arr)

    # Sequential Martingale Cross-Fitting (Luedtke & van der Laan, Annals of Statistics 2016)
    # for non-regular optimal-policy inference valid even at the sharp null theta(X) == 0:
    # Train on strictly prior chunks folds < k (k = 1..N_FOLDS-1), evaluate on chunk k.
    theta_dr_seq_chunks: Dict[int, np.ndarray] = {}
    psi_seq_chunks: Dict[int, np.ndarray] = {}
    for k in range(1, N_FOLDS):
        tr = folds < k
        te = folds == k
        X_tr_s, X_te_s = impute_and_scale(X_raw[tr], X_raw[te])
        Y_tr, W_tr, w_tr = Y_arr[tr], W_arr[tr], w_arr[tr]
        s_tr, g_tr = s_arr[tr], g_arr[tr]

        e_te = compute_stratum_propensity(W_tr, s_tr, s_arr[te])
        m0_mask = W_tr == 0
        m1_mask = W_tr == 1
        mu0_te = fit_predict_ensemble(X_tr_s[m0_mask], Y_tr[m0_mask], w_tr[m0_mask], X_te_s, seed=seed + 3100 + k)
        mu1_te = fit_predict_ensemble(X_tr_s[m1_mask], Y_tr[m1_mask], w_tr[m1_mask], X_te_s, seed=seed + 3200 + k)
        mu_w_te = np.where(W_arr[te] == 1, mu1_te, mu0_te)
        psi_seq_chunks[k] = (mu1_te - mu0_te) + ((W_arr[te] - e_te) / (e_te * (1.0 - e_te))) * (Y_arr[te] - mu_w_te)

        # Inner cross-fitted DR pseudo-outcome strictly inside folds < k
        n_in_splits = min(4, max(2, k + 1))
        in_f = make_stratified_group_folds(s_tr, W_tr, g_tr, n_splits=n_in_splits, seed=seed + 3300 + k)
        psi_in_tr = np.zeros(len(Y_tr), dtype=np.float64)
        for j in range(n_in_splits):
            itr = in_f != j
            ite = in_f == j
            e_ite = compute_stratum_propensity(W_tr[itr], s_tr[itr], s_tr[ite])
            im0 = itr & (W_tr == 0)
            im1 = itr & (W_tr == 1)
            m0_ite = fit_predict_ensemble(X_tr_s[im0], Y_tr[im0], w_tr[im0], X_tr_s[ite], seed=seed + 3400 + k * 10 + j, n_estimators=60)
            m1_ite = fit_predict_ensemble(X_tr_s[im1], Y_tr[im1], w_tr[im1], X_tr_s[ite], seed=seed + 3500 + k * 10 + j, n_estimators=60)
            mw_ite = np.where(W_tr[ite] == 1, m1_ite, m0_ite)
            psi_in_tr[ite] = (m1_ite - m0_ite) + ((W_tr[ite] - e_ite) / (e_ite * (1.0 - e_ite))) * (Y_tr[ite] - mw_ite)

        theta_dr_seq_chunks[k] = fit_predict_ensemble(X_tr_s, psi_in_tr, w_tr, X_te_s, seed=seed + 3600 + k)

    # -------------------------------------------------------------------------
    # COMPONENT 3B: Optimal Plug-In Sign Rule pi(X) = 1{theta_hat(X) + c0 < 0} across c0_grid
    # -------------------------------------------------------------------------
    sign_rule_by_c0: List[Dict[str, Any]] = []
    for c0 in c0_grid:
        # Honest OOS DR-Learner (5-fold cross-fitted)
        pi_dr_oos = (theta_dr_oof + c0) < 0.0
        oos_dr = evaluate_policy_rule(pi_dr_oos, psi_eval, w_arr, g_arr, c0=c0)

        # Honest OOS DR-Learner (Luedtke & van der Laan 2016 Sequential Martingale Cross-Fitting)
        vk_list, vark_list, wk_list, qk_list = [], [], [], []
        for k in range(1, N_FOLDS):
            te = folds == k
            pi_k = (theta_dr_seq_chunks[k] + c0) < 0.0
            rk = evaluate_policy_rule(pi_k, psi_seq_chunks[k], w_arr[te], g_arr[te], c0=c0)
            vk_list.append(rk["net_cost_per_eligible"])
            vark_list.append(rk["se_per_eligible"] ** 2)
            wk_list.append(float(np.sum(w_arr[te])))
            qk_list.append(rk["targeted_share"])
        wk_n = np.array(wk_list) / np.sum(wk_list)
        v_mart = float(np.sum(wk_n * np.array(vk_list)))
        se_mart = float(math.sqrt(max(float(np.sum((wk_n ** 2) * np.array(vark_list))), 0.0)))
        t_mart = v_mart / se_mart if se_mart > 0 else 0.0
        z975 = 1.959963984540054
        oos_dr_martingale = {
            "c0": float(c0),
            "targeted_share": float(np.sum(wk_n * np.array(qk_list))),
            "net_cost_per_eligible": v_mart,
            "se_per_eligible": se_mart,
            "ci_low_per_eligible": float(v_mart - z975 * se_mart),
            "ci_high_per_eligible": float(v_mart + z975 * se_mart),
            "p_one_sided_savings": float(stats.norm.cdf(t_mart)) if se_mart > 0 else 1.0,
            "p_two_sided": float(2.0 * stats.norm.sf(abs(t_mart))) if se_mart > 0 else 1.0,
        }

        # Honest OOS Causal Forest
        pi_cf_oos = (theta_cf_oof + c0) < 0.0
        oos_cf = evaluate_policy_rule(pi_cf_oos, psi_eval, w_arr, g_arr, c0=c0)

        # Naive In-Sample (matched hyperparameters -> pure sample-splitting isolation)
        pi_in_m = (theta_dr_in_matched + c0) < 0.0
        app_plugin_m = evaluate_policy_rule(pi_in_m, theta_dr_in_matched, w_arr, g_arr, c0=c0)
        app_emp_m = evaluate_policy_rule(pi_in_m, psi_in, w_arr, g_arr, c0=c0)

        # Naive In-Sample (standard moderate LGBM depth=4, leaves=15 -> realistic practitioner ML)
        pi_in_std = (theta_dr_in + c0) < 0.0
        app_plugin_std = evaluate_policy_rule(pi_in_std, theta_dr_in, w_arr, g_arr, c0=c0)
        app_emp_std = evaluate_policy_rule(pi_in_std, psi_in, w_arr, g_arr, c0=c0)

        sign_rule_by_c0.append({
            "c0": float(c0),
            "honest_oos_dr": oos_dr,
            "honest_oos_dr_martingale": oos_dr_martingale,
            "honest_oos_cf": oos_cf,
            "naive_insample_matched_plugin": app_plugin_m,
            "naive_insample_matched_emp": app_emp_m,
            "naive_insample_standard_plugin": app_plugin_std,
            "naive_insample_standard_emp": app_emp_std,
            "winners_curse_gap_matched_plugin_per_eligible": float(oos_dr["net_cost_per_eligible"] - app_plugin_m["net_cost_per_eligible"]),
            "winners_curse_gap_matched_emp_per_eligible": float(oos_dr["net_cost_per_eligible"] - app_emp_m["net_cost_per_eligible"]),
            "winners_curse_gap_standard_plugin_per_eligible": float(oos_dr["net_cost_per_eligible"] - app_plugin_std["net_cost_per_eligible"]),
            "winners_curse_gap_standard_emp_per_eligible": float(oos_dr["net_cost_per_eligible"] - app_emp_std["net_cost_per_eligible"]),
        })

    # -------------------------------------------------------------------------
    # COMPONENT 3C: Budget-Constrained Top-q Targeting Curves (q in 10, 20, 30, 50, 100%)
    # -------------------------------------------------------------------------
    topq_curves: List[Dict[str, Any]] = []
    for q in q_grid:
        if q >= 0.999:
            pi_in_q = np.ones(n, dtype=bool)
        else:
            pi_in_q = theta_dr_in_matched <= np.quantile(theta_dr_in_matched, q)
        topq_curves.append({
            "q": float(q),
            "dr_oos": evaluate_policy_rule(topq_dr_oof[q], psi_eval, w_arr, g_arr, c0=0.0),
            "cf_oos": evaluate_policy_rule(topq_cf_oof[q], psi_eval, w_arr, g_arr, c0=0.0),
            "baseline_risk_oos": evaluate_policy_rule(topq_risk_oof[q], psi_eval, w_arr, g_arr, c0=0.0),
            "dr_naive_insample_plugin": evaluate_policy_rule(pi_in_q, theta_dr_in_matched, w_arr, g_arr, c0=0.0),
            "dr_naive_insample_emp": evaluate_policy_rule(pi_in_q, psi_in, w_arr, g_arr, c0=0.0),
        })

    ate_aipw = weighted_cluster_mean_se(psi_eval, w_arr, g_arr)
    elapsed = time.time() - t0

    return {
        "trial_id": trial_id,
        "label": label,
        "unit": unit,
        "unit_scale_to_annual": float(unit_scale_to_annual),
        "n": int(n),
        "n_clusters": int(len(np.unique(g_arr))),
        "p_covariates": int(p),
        "covariates_used": list(X_df.columns),
        "ate_aipw": ate_aipw,
        "distribution_and_heavy_tails": {
            **dist_native,
            "residual_spending_sd_oof": sigma_resid,
            "aipw_pseudo_outcome_sd_sigma_gamma": sigma_gamma,
            "r2_baseline_spending_oof": r2_oof,
            "theorem3_ncrit": thm3_ncrit,
        },
        "blp_and_gates": {
            "dr_learner": blp_gates_dr,
            "causal_forest": blp_gates_cf,
            "baseline_risk_score": blp_gates_risk,
        },
        "sign_rule_by_c0": sign_rule_by_c0,
        "topq_targeting_curves": topq_curves,
        "runtime_sec": round(elapsed, 2),
        # Internal arrays returned via helper wrapper below for permutation test? No, keep JSON-clean!
    }


# =============================================================================
# COMPONENT 4: KILL-SWITCH 3B — 500-DRAW SHARP-NULL PERMUTATION PLACEBO TEST
# =============================================================================
def _perm_worker(
    perm_idx: int,
    X_s: np.ndarray,
    Y_resid_oof: np.ndarray,
    Y_resid_in: np.ndarray,
    w_arr: np.ndarray,
    e_design: np.ndarray,
    folds: np.ndarray,
    g_inv: np.ndarray,
    n_clusters: int,
    cl_strata_groups: List[np.ndarray],
    cl_treat_counts: List[int],
    cl_to_rows_indptr: np.ndarray,
    cl_to_rows_indices: np.ndarray,
    base_seed: int,
) -> Dict[str, float]:
    """
    Single sharp-null permutation worker (respecting exact randomization strata & cluster structure).
    Runs IDENTICAL LightGBM + Ridge DR-Learner under:
      (1) Naive In-Sample (fit & evaluated without sample splitting),
      (2) Honest OOS Sequential Martingale Cross-Fitting (Luedtke & van der Laan, Ann. Stat. 2016:
          train on strictly prior folds < k, evaluate on fold k, k=1..4 -> exact calibration at theta==0),
      (3) Diagnostic Pooled 5-Fold Cross-Fitting (documents cross-fold overlap non-regularity at theta==0).
    """
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", category=UserWarning)
    rng = np.random.default_rng(base_seed + perm_idx * 7919)
    n = len(Y_resid_oof)
    W_perm = np.zeros(n, dtype=np.int64)

    # Exact design-respecting cluster-level permutation within each randomization stratum
    for cl_ids, n_treat in zip(cl_strata_groups, cl_treat_counts):
        if n_treat <= 0:
            continue
        chosen = rng.choice(cl_ids, size=n_treat, replace=False)
        for c in chosen:
            start = cl_to_rows_indptr[c]
            end = cl_to_rows_indptr[c + 1]
            W_perm[cl_to_rows_indices[start:end]] = 1

    # Exact design-propensity AIPW scores under sharp null
    ipw = (W_perm - e_design) / (e_design * (1.0 - e_design))
    psi_oof = ipw * Y_resid_oof
    psi_in = ipw * Y_resid_in

    w_sum = float(np.sum(w_arr))
    eff_min_child = max(10, min(25, n // 20))
    eff_min_child_in = max(10, min(15, n // 30))

    # --- Helper for fast weighted cluster mean + SE ---
    def fast_wc_stats(u: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        if mask is not None:
            if not np.any(mask):
                return 0.0, 0.0, 1.0, 1.0
            um = u[mask]
            wm = w_arr[mask]
            gm = g_inv[mask]
            ws = float(np.sum(wm))
            mu = float(np.sum(wm * um) / ws)
            r = wm * (um - mu)
            r_cl = np.bincount(gm, weights=r, minlength=n_clusters)
            active = np.bincount(gm, minlength=n_clusters) > 0
            g_cnt = int(np.sum(active))
            ss = float(np.sum(r_cl[active] ** 2))
        else:
            ws = w_sum
            mu = float(np.sum(w_arr * u) / ws)
            r = w_arr * (u - mu)
            r_cl = np.bincount(g_inv, weights=r, minlength=n_clusters)
            g_cnt = n_clusters
            ss = float(np.sum(r_cl ** 2))
        if g_cnt <= 1 or ss <= 0.0:
            return mu, 0.0, 1.0, 1.0
        se = math.sqrt((g_cnt / (g_cnt - 1.0)) * ss / (ws * ws))
        if se == 0.0:
            return mu, 0.0, 1.0, 1.0
        t_val = mu / se
        df = max(g_cnt - 1, 1)
        p1 = float(stats.t.cdf(t_val, df=df))
        p2 = float(2.0 * stats.t.sf(abs(t_val), df=df))
        return mu, se, p1, p2

    # --- 1. NAIVE IN-SAMPLE PLUG-IN ML (matched hyperparams AND standard practitioner LGBM+Ridge) ---
    ridge_in = Ridge(alpha=100.0)
    ridge_in.fit(X_s, psi_in, sample_weight=w_arr)
    gbm_in = lgb.LGBMRegressor(
        n_estimators=80, max_depth=3, num_leaves=8, min_child_samples=eff_min_child,
        learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=10.0, n_jobs=1, verbose=-1, random_state=base_seed + perm_idx,
    )
    gbm_in.fit(X_s, psi_in, sample_weight=w_arr)
    theta_in = 0.5 * ridge_in.predict(X_s) + 0.5 * gbm_in.predict(X_s)
    pi_in = theta_in < 0.0

    # Standard practitioner depth=4 LGBM+Ridge in-sample
    gbm_std = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, num_leaves=15, min_child_samples=eff_min_child_in,
        learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5.0, n_jobs=1, verbose=-1, random_state=base_seed + perm_idx + 17,
    )
    gbm_std.fit(X_s, psi_in, sample_weight=w_arr)
    theta_in_std = 0.5 * ridge_in.predict(X_s) + 0.5 * gbm_std.predict(X_s)
    pi_in_std = theta_in_std < 0.0

    q_in = float(np.sum(w_arr[pi_in]) / w_sum)
    plugin_in_pop = float(np.sum(w_arr[pi_in] * theta_in[pi_in]) / w_sum) if np.any(pi_in) else 0.0
    plugin_in_targ = float(plugin_in_pop / q_in) if q_in > 0 else 0.0
    emp_in_pop, _, p1_in_pop, _ = fast_wc_stats(pi_in.astype(np.float64) * psi_in)
    emp_in_targ, _, p1_in_targ, _ = fast_wc_stats(psi_in, mask=pi_in)

    q_in_std = float(np.sum(w_arr[pi_in_std]) / w_sum)
    plugin_std_pop = float(np.sum(w_arr[pi_in_std] * theta_in_std[pi_in_std]) / w_sum) if np.any(pi_in_std) else 0.0
    emp_std_pop, _, p1_std_pop, _ = fast_wc_stats(pi_in_std.astype(np.float64) * psi_in)

    # --- 2. HONEST OOS SEQUENTIAL MARTINGALE CROSS-FITTING (Luedtke & van der Laan 2016) ---
    vk_arr = np.zeros(N_FOLDS - 1, dtype=np.float64)
    vark_arr = np.zeros(N_FOLDS - 1, dtype=np.float64)
    wk_arr = np.zeros(N_FOLDS - 1, dtype=np.float64)
    qk_arr = np.zeros(N_FOLDS - 1, dtype=np.float64)
    for k in range(1, N_FOLDS):
        tr = folds < k
        te = folds == k
        rk = Ridge(alpha=100.0)
        rk.fit(X_s[tr], psi_oof[tr], sample_weight=w_arr[tr])
        gk = lgb.LGBMRegressor(
            n_estimators=80, max_depth=3, num_leaves=8, min_child_samples=eff_min_child,
            learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=10.0, n_jobs=1, verbose=-1, random_state=base_seed + perm_idx * 10 + k,
        )
        gk.fit(X_s[tr], psi_oof[tr], sample_weight=w_arr[tr])
        th_k = 0.5 * rk.predict(X_s[te]) + 0.5 * gk.predict(X_s[te])
        pi_k = th_k < 0.0
        uk = np.zeros(n, dtype=np.float64)
        uk[te] = pi_k.astype(np.float64) * psi_oof[te]
        mk, sek, _, _ = fast_wc_stats(uk, mask=te)
        idx = k - 1
        vk_arr[idx] = mk
        vark_arr[idx] = sek * sek
        wk_arr[idx] = float(np.sum(w_arr[te]))
        qk_arr[idx] = float(np.sum(w_arr[te][pi_k]) / wk_arr[idx]) if wk_arr[idx] > 0 else 0.0

    wk_norm = wk_arr / np.sum(wk_arr)
    oos_mart_pop = float(np.sum(wk_norm * vk_arr))
    oos_mart_se = float(math.sqrt(max(float(np.sum((wk_norm ** 2) * vark_arr)), 0.0)))
    q_mart = float(np.sum(wk_norm * qk_arr))
    if oos_mart_se > 0.0:
        t_mart = oos_mart_pop / oos_mart_se
        p1_mart = float(stats.norm.cdf(t_mart))
        p2_mart = float(2.0 * stats.norm.sf(abs(t_mart)))
    else:
        p1_mart = 1.0
        p2_mart = 1.0

    # --- 3. DIAGNOSTIC POOLED 5-FOLD CROSS-FITTED OOS ---
    theta_oos = np.zeros(n, dtype=np.float64)
    for k in range(N_FOLDS):
        tr = folds != k
        te = folds == k
        rk = Ridge(alpha=100.0)
        rk.fit(X_s[tr], psi_oof[tr], sample_weight=w_arr[tr])
        gk = lgb.LGBMRegressor(
            n_estimators=80, max_depth=3, num_leaves=8, min_child_samples=eff_min_child,
            learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=10.0, n_jobs=1, verbose=-1, random_state=base_seed + perm_idx * 20 + k,
        )
        gk.fit(X_s[tr], psi_oof[tr], sample_weight=w_arr[tr])
        theta_oos[te] = 0.5 * rk.predict(X_s[te]) + 0.5 * gk.predict(X_s[te])

    pi_oos = theta_oos < 0.0
    q_oos = float(np.sum(w_arr[pi_oos]) / w_sum)
    oos_pool_pop, oos_pool_se, p1_pool_pop, p2_pool_pop = fast_wc_stats(pi_oos.astype(np.float64) * psi_oof)

    return {
        "q_in": q_in,
        "plugin_in_pop": plugin_in_pop,
        "plugin_in_targ": plugin_in_targ,
        "emp_in_pop": emp_in_pop,
        "emp_in_targ": emp_in_targ,
        "p1_in_pop": p1_in_pop,
        "p1_in_targ": p1_in_targ,
        "q_in_std": q_in_std,
        "plugin_std_pop": plugin_std_pop,
        "emp_std_pop": emp_std_pop,
        "p1_std_pop": p1_std_pop,
        "q_mart": q_mart,
        "oos_mart_pop": oos_mart_pop,
        "oos_mart_se": oos_mart_se,
        "p1_mart": p1_mart,
        "p2_mart": p2_mart,
        "q_oos_pool": q_oos,
        "oos_pool_pop": oos_pool_pop,
        "oos_pool_se": oos_pool_se,
        "p1_pool_pop": p1_pool_pop,
        "p2_pool_pop": p2_pool_pop,
    }


def run_sharp_null_permutation_audit(
    trial_id: str,
    X_df: pd.DataFrame,
    Y: np.ndarray,
    W: np.ndarray,
    weights: np.ndarray,
    strata: np.ndarray,
    groups: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> Dict[str, Any]:
    """
    Kill-Switch 3B: 500-Draw Design-Respecting Sharp-Null Permutation Winner's Curse Placebo Test.
    """
    t0 = time.time()
    n = len(Y)
    X_raw = X_df.values.astype(np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)
    s_arr = np.asarray(strata)
    g_arr = np.asarray(groups)

    folds = make_stratified_group_folds(s_arr, W_arr, g_arr, n_splits=N_FOLDS, seed=seed)

    # Precompute cross-fitted baseline outcome predictor m_hat^{(-k)}(X) (invariant under sharp null!)
    m_oof = np.zeros(n, dtype=np.float64)
    for k in range(N_FOLDS):
        tr = folds != k
        te = folds == k
        X_tr_s, X_te_s = impute_and_scale(X_raw[tr], X_raw[te])
        m_oof[te] = fit_predict_ensemble(X_tr_s, Y_arr[tr], w_arr[tr], X_te_s, seed=seed + 5000 + k)
    Y_resid_oof = Y_arr - m_oof

    # Standardized full X matrix and in-sample baseline residual (for naive in-sample ML)
    X_s, _ = impute_and_scale(X_raw, X_raw)
    m_in = fit_predict_ensemble(X_s, Y_arr, w_arr, X_s, seed=seed + 5999)
    Y_resid_in = Y_arr - m_in

    # Exact design propensity by stratum (invariant under exact within-stratum cluster permutation!)
    e_design = compute_stratum_propensity(W_arr, s_arr, s_arr)

    # Precompute cluster-to-rows CSR structure and cluster-level strata/treatment counts
    uniq_g, first_idx, g_inv = np.unique(g_arr, return_index=True, return_inverse=True)
    n_clusters = len(uniq_g)
    order = np.argsort(g_inv, kind="mergesort")
    counts = np.bincount(g_inv, minlength=n_clusters)
    indptr = np.zeros(n_clusters + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = order.astype(np.int64)

    cl_strata = s_arr[first_idx]
    cl_treat = W_arr[first_idx]
    cl_strata_groups: List[np.ndarray] = []
    cl_treat_counts: List[int] = []
    for st_val in np.unique(cl_strata):
        c_ids = np.where(cl_strata == st_val)[0].astype(np.int64)
        cl_strata_groups.append(c_ids)
        cl_treat_counts.append(int(np.sum(cl_treat[c_ids])))

    res_list = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(_perm_worker)(
            b, X_s, Y_resid_oof, Y_resid_in, w_arr, e_design, folds, g_inv, n_clusters,
            cl_strata_groups, cl_treat_counts, indptr, indices, seed + 100000,
        )
        for b in range(n_perm)
    )
    df_p = pd.DataFrame(res_list)

    fp_in_matched = float(np.mean(df_p["p1_in_pop"] < 0.05))
    fp_in_std = float(np.mean(df_p["p1_std_pop"] < 0.05))
    oos_mart_t1_1s = float(np.mean(df_p["p1_mart"] < 0.05))
    oos_mart_t1_2s = float(np.mean(df_p["p2_mart"] < 0.05))
    oos_pool_t1_1s = float(np.mean(df_p["p1_pool_pop"] < 0.05))
    oos_pool_t1_2s = float(np.mean(df_p["p2_pool_pop"] < 0.05))
    ks3b_pass = bool(oos_mart_t1_1s <= 0.06)

    elapsed = time.time() - t0
    return {
        "trial_id": trial_id,
        "n_permutations": int(n_perm),
        "runtime_sec": round(elapsed, 2),
        "naive_insample_matched": {
            "fp_sign_flip_rate_p05": fp_in_matched,
            "fp_sign_flip_rate_targeted_p05": float(np.mean(df_p["p1_in_targ"] < 0.05)),
            "mean_targeted_share": float(df_p["q_in"].mean()),
            "mean_illusory_savings_plugin_per_eligible": float(df_p["plugin_in_pop"].mean()),
            "mean_illusory_savings_plugin_per_targeted": float(df_p["plugin_in_targ"].mean()),
            "mean_illusory_savings_emp_per_eligible": float(df_p["emp_in_pop"].mean()),
            "mean_illusory_savings_emp_per_targeted": float(df_p["emp_in_targ"].mean()),
            "median_illusory_savings_emp_per_eligible": float(df_p["emp_in_pop"].median()),
            "p95_illusory_savings_emp_per_eligible": float(np.quantile(df_p["emp_in_pop"], 0.05)),
        },
        "naive_insample_standard": {
            "fp_sign_flip_rate_p05": fp_in_std,
            "mean_targeted_share": float(df_p["q_in_std"].mean()),
            "mean_illusory_savings_plugin_per_eligible": float(df_p["plugin_std_pop"].mean()),
            "mean_illusory_savings_emp_per_eligible": float(df_p["emp_std_pop"].mean()),
        },
        "honest_oos_aipw": {
            "inference_method": "Luedtke & van der Laan (Ann. Stat. 2016) Sequential Martingale Cross-Fitting (valid under non-regular sharp null theta == 0)",
            "type1_error_one_sided_savings_p05": oos_mart_t1_1s,
            "type1_error_two_sided_p05": oos_mart_t1_2s,
            "mean_targeted_share": float(df_p["q_mart"].mean()),
            "mean_realized_net_cost_per_eligible": float(df_p["oos_mart_pop"].mean()),
            "emp_sd_realized_net_cost_per_eligible": float(df_p["oos_mart_pop"].std(ddof=1)),
            "rms_analytic_se_per_eligible": float(math.sqrt(float(np.mean(df_p["oos_mart_se"] ** 2)))),
        },
        "honest_oos_pooled_5fold_diagnostic": {
            "note": "Documents cross-fold overlap non-regularity at sharp-null boundary theta(X) == 0 (Luedtke & van der Laan 2016)",
            "type1_error_one_sided_savings_p05": oos_pool_t1_1s,
            "type1_error_two_sided_p05": oos_pool_t1_2s,
            "mean_targeted_share": float(df_p["q_oos_pool"].mean()),
            "mean_realized_net_cost_per_eligible": float(df_p["oos_pool_pop"].mean()),
            "emp_sd_realized_net_cost_per_eligible": float(df_p["oos_pool_pop"].std(ddof=1)),
            "rms_analytic_se_per_eligible": float(math.sqrt(float(np.mean(df_p["oos_pool_se"] ** 2)))),
        },
        "mean_null_winners_curse_gap_matched_plugin": float((df_p["oos_mart_pop"] - df_p["plugin_in_pop"]).mean()),
        "mean_null_winners_curse_gap_matched_emp": float((df_p["oos_mart_pop"] - df_p["emp_in_pop"]).mean()),
        "mean_null_winners_curse_gap_standard_emp": float((df_p["oos_mart_pop"] - df_p["emp_std_pop"]).mean()),
        "ks3b_pass_oos_type1_le_006": ks3b_pass,
    }


# =============================================================================
# ILLINOIS EMPIRICAL CHURN & EMPLOYER TERMINATION MODULE
# =============================================================================
def analyze_illinois_empirical_churn(
    il_claims: pd.DataFrame,
    il_admin: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Directly measure empirical 30-month employer coverage churn (`1 - covg_0816_0119`)
    across baseline spending quartiles in `claims.csv`, AND 12-/30-month employer job
    termination (`terminated_0717`, `terminated_0119`) across age and baseline sick-leave
    quartiles in `firm_admin.csv`.
    NOTE: Illinois public-use CSVs are intentionally unlinked across domains (no cross-file ID).
    """
    out: Dict[str, Any] = {
        "public_use_linkage_note": (
            "Illinois public-use files (claims.csv, firm_admin.csv, biometrics.csv, participation.csv) "
            "are independently ordered without a cross-file individual ID per NBER confidentiality terms. "
            "Coverage churn (1 - covg_0816_0119) is stratified by baseline spending quartiles within claims.csv; "
            "job termination (terminated_0119) is stratified by baseline sick leave and age within firm_admin.csv."
        )
    }

    # 1. Full randomized cohort (N=4,834) coverage retention & termination
    full_covg_30m = float(il_claims["covg_0816_0119"].mean())
    full_term_12m = float(il_admin["terminated_0717"].mean())
    full_term_30m = float(il_admin["terminated_0119"].mean())
    out["full_cohort_N4834"] = {
        "mean_covg_0816_0119": full_covg_30m,
        "mean_uncovered_frac_30m": 1.0 - full_covg_30m,
        "terminated_12m": full_term_12m,
        "terminated_30m": full_term_30m,
        "implied_annual_termination_hazard_h": float(-math.log(1.0 - full_term_30m) / 2.5),
    }

    # 2. Baseline-insured cohort (covg_0715_0716 > 0, N=3,223) AND full-year baseline insured (covg == 1, N=2,884)
    for tag, sub in [
        ("baseline_insured_any", il_claims[il_claims["covg_0715_0716"] > 0].copy()),
        ("baseline_insured_fullyear", il_claims[il_claims["covg_0715_0716"] >= 0.999].copy()),
    ]:
        sub["spend_q"] = pd.qcut(sub["spend_0715_0716"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"])
        sub["churn_frac_30m"] = 1.0 - sub["covg_0816_0119"]
        sub["any_covg_loss_30m"] = (sub["covg_0816_0119"] < 0.999).astype(float)
        sub["full_covg_loss_30m"] = (sub["covg_0816_0119"] == 0.0).astype(float)

        rows = []
        for q_lab, g in sub.groupby("spend_q", observed=False):
            any_loss = float(g["any_covg_loss_30m"].mean())
            rows.append({
                "quartile": str(q_lab),
                "n": int(len(g)),
                "mean_baseline_monthly_spend": float(g["spend_0715_0716"].mean()),
                "mean_baseline_annual_spend": float(g["spend_0715_0716"].mean() * 12.0),
                "mean_covg_12m": float(g["covg_0816_0717"].mean()),
                "mean_covg_30m": float(g["covg_0816_0119"].mean()),
                "mean_churn_frac_30m": float(g["churn_frac_30m"].mean()),
                "any_covg_loss_30m_rate": any_loss,
                "full_covg_loss_30m_rate": float(g["full_covg_loss_30m"].mean()),
                "implied_annual_churn_hazard_h": float(-math.log(max(1.0 - any_loss, 1e-6)) / 2.5),
                "age50_share": float(g["age50"].mean()),
            })

        # Chi-square test across quartiles for any_covg_loss_30m and Kruskal-Wallis for churn_frac_30m
        ct = pd.crosstab(sub["spend_q"], sub["any_covg_loss_30m"])
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(ct)
        kw_groups = [g["churn_frac_30m"].values for _, g in sub.groupby("spend_q", observed=False)]
        kw_stat, kw_p = stats.kruskal(*kw_groups)
        # Difference Q4 - Q1 in any_covg_loss_30m
        q4_loss = rows[3]["any_covg_loss_30m_rate"]
        q1_loss = rows[0]["any_covg_loss_30m_rate"]
        out[tag] = {
            "n": int(len(sub)),
            "overall_any_covg_loss_30m": float(sub["any_covg_loss_30m"].mean()),
            "overall_mean_churn_frac_30m": float(sub["churn_frac_30m"].mean()),
            "overall_implied_annual_hazard_h": float(-math.log(1.0 - sub["any_covg_loss_30m"].mean()) / 2.5),
            "by_baseline_spending_quartile": rows,
            "q4_minus_q1_any_loss_pp": round((q4_loss - q1_loss) * 100.0, 2),
            "chi2_any_loss_p": float(chi2_p),
            "kruskal_churn_frac_p": float(kw_p),
        }

    # 3. Firm admin termination by age group & baseline sick-leave quartile
    fa = il_admin.copy()
    fa["age_grp"] = np.where(fa["age50"] == 1, "50+", np.where(fa["age37_49"] == 1, "37-49", "<37"))
    age_rows = []
    for ag in ["<37", "37-49", "50+"]:
        g = fa[fa["age_grp"] == ag]
        t30 = float(g["terminated_0119"].mean())
        age_rows.append({
            "age_group": ag,
            "n": int(len(g)),
            "terminated_12m": float(g["terminated_0717"].mean()),
            "terminated_30m": t30,
            "implied_annual_termination_hazard_h": float(-math.log(max(1.0 - t30, 1e-6)) / 2.5),
        })
    fa["sick_q"] = pd.qcut(fa["sickleave_0815_0716"].rank(method="first"), 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"])
    sick_rows = []
    for sq, g in fa.groupby("sick_q", observed=False):
        t30 = float(g["terminated_0119"].mean())
        sick_rows.append({
            "quartile": str(sq),
            "n": int(len(g)),
            "mean_baseline_sick_days_yr": float(g["sickleave_0815_0716"].mean()),
            "terminated_12m": float(g["terminated_0717"].mean()),
            "terminated_30m": t30,
            "implied_annual_termination_hazard_h": float(-math.log(max(1.0 - t30, 1e-6)) / 2.5),
        })
    _, chi2_age_p, _, _ = stats.chi2_contingency(pd.crosstab(fa["age_grp"], fa["terminated_0119"]))
    _, chi2_sick_p, _, _ = stats.chi2_contingency(pd.crosstab(fa["sick_q"], fa["terminated_0119"]))
    out["firm_admin_termination"] = {
        "n": int(len(fa)),
        "by_age_group": age_rows,
        "chi2_age_p": float(chi2_age_p),
        "by_baseline_sickleave_quartile": sick_rows,
        "chi2_sickleave_p": float(chi2_sick_p),
    }
    return out


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main() -> None:
    t_start = time.time()
    assert_scope_clean()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("EMPIRICAL PILLAR 2: REAL U.S. SPENDING RCT CATE & WINNER'S CURSE AUDIT")
    print(f"Seed={SEED} | Folds={N_FOLDS} | Permutations={N_PERM} | Workers={N_JOBS}")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # LOAD DATASETS
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading strictly scoped U.S. RCT datasets and MEPS 2023 unit costs...")
    il_claims = pd.read_csv(ILLINOIS_DIR / "claims.csv")
    il_admin = pd.read_csv(ILLINOIS_DIR / "firm_admin.csv")
    cam = pd.read_stata(CAMDEN_DTA)
    cam = cam[cam["Isample_180"] == 1].copy().reset_index(drop=True)

    with zipfile.ZipFile(OREGON_ZIP) as z:
        oh_desc = pd.read_stata(io.BytesIO(z.read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_descriptive_vars.dta")), convert_categoricals=False)
        oh_ed = pd.read_stata(io.BytesIO(z.read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_ed_vars.dta")), convert_categoricals=False)
        oh_s12 = pd.read_stata(io.BytesIO(z.read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_survey12m_vars.dta")), convert_categoricals=False)
        oh_sp = pd.read_stata(io.BytesIO(z.read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_stateprograms_vars.dta")), convert_categoricals=False)
        oh_inp = pd.read_stata(io.BytesIO(z.read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_inperson_vars.dta")), convert_categoricals=False)

    meps23_prices = compute_meps_2023_unit_costs()
    print(f"  Illinois claims N={len(il_claims)}, firm_admin N={len(il_admin)}")
    print(f"  Camden consented 180d sample N={len(cam)}")
    print(f"  OHIE lottery N={len(oh_desc)}, ED admin N={len(oh_ed)}, 12m survey resp N={int((oh_s12['sample_12m_resp']==1).sum())}, in-person resp N={int((oh_inp['sample_resp_inp']==1).sum())}")
    print(f"  MEPS 2023 unit costs: ER=${meps23_prices['er_visit']:.2f}, IP=${meps23_prices['ip_discharge']:.2f}, OB=${meps23_prices['office_visit']:.2f}, RxFill=${meps23_prices['rx_fill']:.2f}")

    # =========================================================================
    # COMPONENT 1: KILL-SWITCH 3A — EXACT PUBLISHED ITT REPLICATION
    # =========================================================================
    print("\n[2/5] Running Kill-Switch 3A: Exact Published ITT Replication...")
    ks3a: Dict[str, Any] = {"illinois_qje_2019": {}, "camden_nejm_2020": {}, "ohie_medicaid_expansion_benchmark": {}}
    ks3a_checks: List[Dict[str, Any]] = []

    # --- 1A. Illinois QJE 2019 ---
    il_yr1_sub = il_claims[(il_claims["covg_0816_0717"] > 0) & il_claims["spend_0816_0717"].notna()].copy()
    il_30m_sub = il_claims[(il_claims["covg_0816_0119"] > 0) & il_claims["spend_0816_0119"].notna()].copy()

    def il_itt_spec(df: pd.DataFrame, y_col: str, w_col: Optional[str], strata_col: Optional[str]) -> Dict[str, Any]:
        wts = df[w_col].values if w_col is not None else None
        mean_val = float(np.average(df[y_col].values, weights=wts)) if wts is not None else float(df[y_col].mean())
        noctrl = wls_cluster_reg(df[y_col].values, df[["treat"]], "treat", weights=wts)
        out_d: Dict[str, Any] = {"n": int(len(df)), "weighted_mean": mean_val, "no_controls": noctrl}
        if strata_col is not None:
            Xs = pd.get_dummies(df[strata_col], prefix="st", drop_first=True, dtype=float)
            Xs.insert(0, "treat", df["treat"].values)
            out_d["strata_fe"] = wls_cluster_reg(df[y_col].values, Xs, "treat", weights=wts)
        return out_d

    il_yr1_rep = il_itt_spec(il_yr1_sub, "spend_0816_0717", "covg_0816_0717", "Strata_claims")
    il_30m_rep = il_itt_spec(il_30m_sub, "spend_0816_0119", "covg_0816_0119", "Strata_claims")
    il_covg_rep = il_itt_spec(il_claims, "covg_0816_0119", None, "Strata_claims")
    il_term12_rep = il_itt_spec(il_admin, "terminated_0717", None, None)
    il_term30_rep = il_itt_spec(il_admin, "terminated_0119", None, None)

    ks3a["illinois_qje_2019"] = {
        "citation": "Jones, Molitor, & Reif (QJE 2019), Table 3 & Appendix Tables A.4a/A.7a/A.7c",
        "yr1_monthly_spend_usd_mo": {**il_yr1_rep, "published_mean": 576.2, "published_itt": 10.8, "published_se": 48.5, "published_n": 3239},
        "yr1_annualized_spend_usd_yr": {
            "weighted_mean": il_yr1_rep["weighted_mean"] * 12.0,
            "no_controls_itt": il_yr1_rep["no_controls"]["estimate"] * 12.0,
            "no_controls_se": il_yr1_rep["no_controls"]["se"] * 12.0,
            "strata_fe_itt": il_yr1_rep["strata_fe"]["estimate"] * 12.0,
            "strata_fe_se": il_yr1_rep["strata_fe"]["se"] * 12.0,
        },
        "m30_monthly_spend_usd_mo": {**il_30m_rep, "published_mean": 650.5, "published_itt": -74.7, "published_se": 58.5, "published_n": 3307},
        "m30_annualized_spend_usd_yr": {
            "weighted_mean": il_30m_rep["weighted_mean"] * 12.0,
            "no_controls_itt": il_30m_rep["no_controls"]["estimate"] * 12.0,
            "no_controls_se": il_30m_rep["no_controls"]["se"] * 12.0,
            "strata_fe_itt": il_30m_rep["strata_fe"]["estimate"] * 12.0,
            "strata_fe_se": il_30m_rep["strata_fe"]["se"] * 12.0,
        },
        "coverage_retention_30m": il_covg_rep,
        "terminated_12m": {**il_term12_rep, "published_mean": 0.113, "published_itt": -0.013, "published_se": 0.010},
        "terminated_30m": {**il_term30_rep, "published_mean": 0.204, "published_itt": 0.002, "published_se": 0.012},
    }
    ks3a_checks.extend([
        {"trial": "Illinois", "metric": "Yr1 monthly spend ITT ($/mo)", "replicated": round(il_yr1_rep["no_controls"]["estimate"], 2), "published": 10.8, "pass": abs(il_yr1_rep["no_controls"]["estimate"] - 10.8) < 0.2},
        {"trial": "Illinois", "metric": "30m monthly spend ITT ($/mo)", "replicated": round(il_30m_rep["no_controls"]["estimate"], 2), "published": -74.7, "pass": abs(il_30m_rep["no_controls"]["estimate"] - (-74.7)) < 0.2},
        {"trial": "Illinois", "metric": "30m termination ITT", "replicated": round(il_term30_rep["no_controls"]["estimate"], 3), "published": 0.002, "pass": abs(il_term30_rep["no_controls"]["estimate"] - 0.002) < 0.002},
    ])

    # --- 1B. Camden NEJM 2020 ---
    cam_prespec = [
        "Imale_100", "Iagebin34", "Iagebin3539", "Iagebin4044", "Iagebin4549",
        "Iagebin5054", "Iagebin5559", "Iagebin6064", "Iagebin6569", "Iagebin7074",
        "Iagebin75", "Iaa_100", "Ihispanic_100",
    ]
    cam_specs = [
        ("Ipost_ie_admit_180_IP_100", ["pre_ie_admit_cnt_180_IP", "preie_admit_cnt_181t365_IP"], 61.70, 0.64, 0.82, 0.2),
        ("post_ie_receipts_180_IP", ["pre_ie_receipts_180_IP", "preie_receipts_181t365_IP"], 17650.0, 480.0, 680.0, 5.0),
        ("post_ie_charges_180_IP", ["pre_ie_charges_180_IP", "preie_charges_181t365_IP"], 114768.0, 1654.0, 3722.0, 10.0),
        ("post_ie_days_cnt_180_IP", ["pre_ie_days_cnt_180_IP", "preie_days_cnt_181t365_IP"], 9.95, -0.59, -0.32, 0.05),
    ]
    ks3a["camden_nejm_2020"]["citation"] = "Finkelstein, Zhou, Taubman, & Doyle (NEJM 2020), Table 4"
    for y_col, lag_cols, pub_c, pub_unadj, pub_adj, tol in cam_specs:
        c_mean = float(cam.loc[cam["Itreatment"] == 0, y_col].mean())
        t_mean = float(cam.loc[cam["Itreatment"] == 1, y_col].mean())
        unadj = wls_cluster_reg(cam[y_col].values, cam[["Itreatment"]], "Itreatment")
        adj = wls_cluster_reg(cam[y_col].values, cam[["Itreatment"] + cam_prespec + lag_cols], "Itreatment")
        ks3a["camden_nejm_2020"][y_col] = {
            "n": int(len(cam)),
            "control_mean": c_mean,
            "treatment_mean": t_mean,
            "unadjusted": unadj,
            "adjusted_prespec_and_lags": adj,
            "published_control_mean": pub_c,
            "published_unadj": pub_unadj,
            "published_adj": pub_adj,
        }
        ks3a_checks.append({
            "trial": "Camden", "metric": f"{y_col} Adj ITT",
            "replicated": round(adj["estimate"], 2), "published": pub_adj,
            "pass": abs(adj["estimate"] - pub_adj) <= tol,
        })

    # --- 1C. OHIE (Explicitly Labeled Medicaid Insurance Expansion Benchmark) ---
    # ED Administrative Sample (Taubman et al., Science 2014; N = 24,646)
    oh_ed_m = oh_ed.merge(oh_desc, on="person_id", how="inner").merge(
        oh_sp[["person_id", "ohp_all_ever_firstn_30sep2009", "snap_ever_prenotify07", "tanf_ever_prenotify07",
               "snap_tot_hh_prenotify07", "tanf_tot_hh_prenotify07"]],
        on="person_id", how="left",
    )
    def ohie_ed_reg(df: pd.DataFrame, y_col: str, pre_cols: List[str]) -> Dict[str, Any]:
        sub = df.dropna(subset=[y_col, "numhh_list"] + pre_cols).copy()
        X = pd.get_dummies(sub["numhh_list"], prefix="nhh", drop_first=True, dtype=float)
        X.insert(0, "treatment", sub["treatment"].values.astype(float))
        for c in pre_cols:
            X[c] = sub[c].values.astype(float)
        res = wls_cluster_reg(sub[y_col].values, X, "treatment", groups=sub["household_id"].values)
        res["control_mean"] = float(sub.loc[sub["treatment"] == 0, y_col].mean())
        return res

    oh_ed_fs = ohie_ed_reg(oh_ed_m, "ohp_all_ever_firstn_30sep2009", [])
    oh_ed_visits_pre = ohie_ed_reg(oh_ed_m, "num_visit_cens_ed", ["num_visit_pre_cens_ed"])
    oh_ed_visits_raw = ohie_ed_reg(oh_ed_m, "num_visit_cens_ed", [])
    oh_ed_any = ohie_ed_reg(oh_ed_m, "any_visit_ed", [])
    oh_ed_charg = ohie_ed_reg(oh_ed_m, "charg_tot_ed", ["charg_tot_pre_ed"])
    oh_ed_edcharg = ohie_ed_reg(oh_ed_m, "ed_charg_tot_ed", ["ed_charg_tot_pre_ed"])
    oh_ed_visits_late = oh_ed_visits_pre["estimate"] / oh_ed_fs["estimate"]

    # 12-Month Mail Survey Sample (Finkelstein et al., QJE 2012; N = 23,741 respondents)
    oh_s12_m = oh_desc.merge(oh_s12, on="person_id", how="inner").merge(
        oh_sp[["person_id", "ohp_all_ever_firstn_30sep2009", "snap_ever_prenotify07", "tanf_ever_prenotify07",
               "snap_tot_hh_prenotify07", "tanf_tot_hh_prenotify07"]],
        on="person_id", how="left",
    ).merge(
        oh_ed[["person_id", "sample_ed", "num_visit_pre_cens_ed", "charg_tot_pre_ed"]],
        on="person_id", how="left",
    )
    oh_s12_r = oh_s12_m[oh_s12_m["sample_12m_resp"] == 1].copy().reset_index(drop=True)
    oh_s12_r["draw_survey_12m"] = oh_s12_r["wave_survey12m"].replace({8: 7})
    oh_s12_r["cell"] = oh_s12_r["draw_survey_12m"].astype(int).astype(str) + "_" + oh_s12_r["numhh_list"].astype(int).astype(str)

    # Exact QJE 2012 preventive care recodes (prepare_data.do)
    oh_s12_r["chl_chk_bin_12m"] = np.where(oh_s12_r["chl_chk_12m"].isin([1, 2]), 1.0, np.where(oh_s12_r["chl_chk_12m"] == 3, 0.0, np.nan))
    oh_s12_r["dia_chk_bin_12m"] = np.where(oh_s12_r["dia_chk_12m"].isin([1, 2]), 1.0, np.where(oh_s12_r["dia_chk_12m"] == 3, 0.0, np.nan))
    oh_s12_r["mam_chk_bin_12m"] = np.where(oh_s12_r["mam_chk_12m"] == 1, 1.0, np.where(oh_s12_r["mam_chk_12m"].isin([2, 3]), 0.0, np.nan))
    oh_s12_r.loc[oh_s12_r["birthyear_list"] > 1968, "mam_chk_bin_12m"] = np.nan
    oh_s12_r["pap_chk_bin_12m"] = np.where(oh_s12_r["pap_chk_12m"] == 1, 1.0, np.where(oh_s12_r["pap_chk_12m"].isin([2, 3]), 0.0, np.nan))

    # Exact QJE 2012 spending proxy & MEPS 2023-priced annualized spending proxy
    qje_unit_6mo = {"rx_num_mod_12m": 156.0, "doc_num_mod_12m": 150.0, "er_num_mod_12m": 435.0, "hosp_num_mod_12m": 7523.0}
    meps23_unit_6mo = {
        "rx_num_mod_12m": meps23_prices["rx_6mo_per_current_rx"],
        "doc_num_mod_12m": meps23_prices["office_visit"],
        "er_num_mod_12m": meps23_prices["er_visit"],
        "hosp_num_mod_12m": meps23_prices["ip_discharge"],
    }
    oh_s12_r["spend_qje_12m"] = 2.0 * (
        qje_unit_6mo["rx_num_mod_12m"] * oh_s12_r["rx_num_mod_12m"]
        + qje_unit_6mo["doc_num_mod_12m"] * oh_s12_r["doc_num_mod_12m"]
        + qje_unit_6mo["er_num_mod_12m"] * oh_s12_r["er_num_mod_12m"]
        + qje_unit_6mo["hosp_num_mod_12m"] * oh_s12_r["hosp_num_mod_12m"]
    )
    oh_s12_r["spend_meps2023_12m"] = 2.0 * (
        meps23_unit_6mo["rx_num_mod_12m"] * oh_s12_r["rx_num_mod_12m"]
        + meps23_unit_6mo["doc_num_mod_12m"] * oh_s12_r["doc_num_mod_12m"]
        + meps23_unit_6mo["er_num_mod_12m"] * oh_s12_r["er_num_mod_12m"]
        + meps23_unit_6mo["hosp_num_mod_12m"] * oh_s12_r["hosp_num_mod_12m"]
    )

    def ohie_s12_reg(df: pd.DataFrame, y_col: str) -> Dict[str, Any]:
        sub = df.dropna(subset=[y_col, "weight_12m", "cell"]).copy()
        X = pd.get_dummies(sub["cell"], prefix="c", drop_first=True, dtype=float)
        X.insert(0, "treatment", sub["treatment"].values.astype(float))
        res = wls_cluster_reg(sub[y_col].values, X, "treatment", weights=sub["weight_12m"].values, groups=sub["household_id"].values)
        csub = sub[sub["treatment"] == 0]
        res["control_mean"] = float(np.average(csub[y_col].values, weights=csub["weight_12m"].values))
        return res

    oh_s12_stacked_qje = stacked_ohie_spending_qje(oh_s12_r, qje_unit_6mo)
    oh_s12_stacked_meps23 = stacked_ohie_spending_qje(oh_s12_r, meps23_unit_6mo)
    oh_s12_outcomes = {}
    for col in [
        "ohp_all_ever_firstn_30sep2009", "rx_num_mod_12m", "doc_num_mod_12m", "er_num_mod_12m",
        "hosp_num_mod_12m", "spend_qje_12m", "spend_meps2023_12m",
        "chl_chk_bin_12m", "dia_chk_bin_12m", "mam_chk_bin_12m", "pap_chk_bin_12m",
    ]:
        oh_s12_outcomes[col] = ohie_s12_reg(oh_s12_r, col)

    # In-person clinical biometrics (Baicker et al., NEJM 2013; N = 12,229 respondents of 20,745 sampled)
    oh_inp_m = oh_inp.merge(oh_desc[["person_id", "household_id", "treatment", "numhh_list"]], on="person_id").merge(
        oh_sp[["person_id", "ohp_all_ever_inperson"]], on="person_id"
    )
    oh_inp_r = oh_inp_m[oh_inp_m["sample_resp_inp"] == 1].copy()
    oh_inp_r["dep_pos_inp"] = np.where(oh_inp_r["phqtot_inp"].notna(), (oh_inp_r["phqtot_inp"] >= 10).astype(float), np.nan)
    oh_inp_outcomes = {}
    for col in ["ohp_all_ever_inperson", "dep_pos_inp", "dia_dx_post_lottery_inp", "catastrophic_exp_inp", "any_oop_spending_inp"]:
        sub = oh_inp_r.dropna(subset=[col, "weight_total_inp", "numhh_list"]).copy()
        X = pd.get_dummies(sub["numhh_list"], prefix="nhh", drop_first=True, dtype=float)
        X.insert(0, "treatment", sub["treatment"].values.astype(float))
        res = wls_cluster_reg(sub[col].values, X, "treatment", weights=sub["weight_total_inp"].values, groups=sub["household_id"].values)
        csub = sub[sub["treatment"] == 0]
        res["control_mean"] = float(np.average(csub[col].values, weights=csub["weight_total_inp"].values))
        oh_inp_outcomes[col] = res

    ks3a["ohie_medicaid_expansion_benchmark"] = {
        "benchmark_classification": "Medicaid insurance expansion benchmark (NOT a targeted clinical prevention program)",
        "citations": "Finkelstein et al. (QJE 2012); Baicker et al. (NEJM 2013); Taubman et al. (Science 2014)",
        "ed_admin_science_2014": {
            "n_total": int(len(oh_ed_m)),
            "n_households": int(oh_ed_m["household_id"].nunique()),
            "first_stage_ohp_ever": oh_ed_fs,
            "num_visit_cens_ed_pre_adj": {**oh_ed_visits_pre, "implied_late": float(oh_ed_visits_late), "published_censored_itt": 0.096, "published_uncensored_late": 0.41},
            "num_visit_cens_ed_unadj": oh_ed_visits_raw,
            "any_visit_ed": {**oh_ed_any, "published_itt": 0.019},
            "charg_tot_ed_pre_adj": oh_ed_charg,
            "ed_charg_tot_ed_pre_adj": oh_ed_edcharg,
        },
        "survey_12m_qje_2012": {
            "n_respondents": int(len(oh_s12_r)),
            "n_households": int(oh_s12_r["household_id"].nunique()),
            "qje_meps_unit_weights_6mo": qje_unit_6mo,
            "meps_2023_unit_weights_6mo": meps23_unit_6mo,
            "meps_2023_raw_event_means": meps23_prices,
            "stacked_annualized_spending_qje2012": {**oh_s12_stacked_qje, "published_control_mean": 3156.0, "published_itt": 225.0, "published_late": 778.0},
            "stacked_annualized_spending_meps2023": oh_s12_stacked_meps23,
            "individual_outcomes": oh_s12_outcomes,
        },
        "inperson_clinical_nejm_2013": {
            "n_sampled": int(len(oh_inp)),
            "n_respondents": int(len(oh_inp_r)),
            "outcomes": oh_inp_outcomes,
            "dep_pos_implied_late_pp": round((oh_inp_outcomes["dep_pos_inp"]["estimate"] / oh_inp_outcomes["ohp_all_ever_inperson"]["estimate"]) * 100.0, 2),
            "published_dep_pos_late_pp": -9.15,
        },
    }
    ks3a_checks.extend([
        {"trial": "OHIE-ED", "metric": "num_visit_cens_ed pre-adj ITT", "replicated": round(oh_ed_visits_pre["estimate"], 4), "published": 0.096, "pass": abs(oh_ed_visits_pre["estimate"] - 0.096) < 0.005},
        {"trial": "OHIE-S12", "metric": "QJE stacked annualized spend ITT ($/yr)", "replicated": round(oh_s12_stacked_qje["itt_annual_usd"], 1), "published": 225.0, "pass": abs(oh_s12_stacked_qje["itt_annual_usd"] - 225.0) < 5.0},
        {"trial": "OHIE-S12", "metric": "chl_chk_bin_12m ITT", "replicated": round(oh_s12_outcomes["chl_chk_bin_12m"]["estimate"], 3), "published": 0.033, "pass": abs(oh_s12_outcomes["chl_chk_bin_12m"]["estimate"] - 0.033) < 0.003},
        {"trial": "OHIE-S12", "metric": "mam_chk_bin_12m ITT", "replicated": round(oh_s12_outcomes["mam_chk_bin_12m"]["estimate"], 3), "published": 0.055, "pass": abs(oh_s12_outcomes["mam_chk_bin_12m"]["estimate"] - 0.055) < 0.003},
        {"trial": "OHIE-INP", "metric": "dep_pos_inp LATE (pp)", "replicated": ks3a["ohie_medicaid_expansion_benchmark"]["inperson_clinical_nejm_2013"]["dep_pos_implied_late_pp"], "published": -9.15, "pass": abs(ks3a["ohie_medicaid_expansion_benchmark"]["inperson_clinical_nejm_2013"]["dep_pos_implied_late_pp"] - (-9.15)) < 0.15},
    ])
    ks3a_pass = all(c["pass"] for c in ks3a_checks)
    ks3a["checks"] = ks3a_checks
    ks3a["ks3a_pass"] = ks3a_pass
    print(f"  KS3A Replication Checks: {sum(c['pass'] for c in ks3a_checks)}/{len(ks3a_checks)} PASSED (KS3A={ks3a_pass})")
    for c in ks3a_checks:
        print(f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['trial']} | {c['metric']}: rep={c['replicated']} vs pub={c['published']}")

    # =========================================================================
    # PREPARE STRICTLY PRE-TREATMENT COVARIATE MATRICES (ZERO POST-TREATMENT LEAKAGE!)
    # =========================================================================
    print("\n[3/5] Running Components 2 & 3: Heavy-Tail Diagnostics, Cross-Fitted CATE (DR/CF/Risk), BLP/GATES & Winner's Curse...")

    # 1. Illinois strictly pre-treatment covariates (claims.csv)
    il_pre_cols = [
        "male", "age50", "age37_49", "white",
        "covg_0715_0716", "covg_1015_0716",
        "spend_0715_0716", "spendHosp_0715_0716", "spendOff_0715_0716", "spendRx_0715_0716", "nonzero_spend_0715_0716",
        "diabetes_1015_0716", "hyperlipidemia_1015_0716", "hypertension_1015_0716",
        "pcp_any_office_1015_0716", "pcp_any_visits_1015_0716", "pcp_total_office_1015_0716", "pcp_total_visits_1015_0716",
        "pos_er_critical_1015_0716", "pos_hospital_1015_0716", "pos_office_outpatient_1015_0716",
    ]
    for c in il_pre_cols:
        assert "0816" not in c and "hra" not in c and "screening" not in c, f"Post-treatment leakage in Illinois: {c}"

    def build_il_X(df: pd.DataFrame) -> pd.DataFrame:
        X = df[il_pre_cols].copy()
        X["baseline_claims_missing"] = df["spend_0715_0716"].isna().astype(float)
        X["log1p_spend_0715_0716"] = np.log1p(df["spend_0715_0716"].fillna(0.0).clip(lower=0.0))
        return X

    X_il_30m = build_il_X(il_30m_sub)
    X_il_yr1 = build_il_X(il_yr1_sub)

    # 2. Camden strictly pre-treatment covariates (analysis_claims.dta)
    cam_pre_cols = [
        "Imale_100", "Iagebin34", "Iagebin3539", "Iagebin4044", "Iagebin4549",
        "Iagebin5054", "Iagebin5559", "Iagebin6064", "Iagebin6569", "Iagebin7074", "Iagebin75",
        "Iaa_100", "Ihispanic_100", "Iwhite_100",
        "pre_ie_admit_cnt_180_IP", "preie_admit_cnt_181t365_IP",
        "pre_ie_days_cnt_180_IP", "preie_days_cnt_181t365_IP",
        "pre_ie_charges_180_IP", "preie_charges_181t365_IP",
        "pre_ie_receipts_180_IP", "preie_receipts_181t365_IP",
        "Imedicaid_100", "Imedicare_100", "IhospitalX_100",
        "Iasthma_100", "Iarthritis_100", "Ichf_100", "Icopd_emph_100", "Icad_100",
        "Idepression_100", "Idiabetes_100", "Ihiv_aids_100", "IHBP_100", "IHLP_100",
        "Iobesity_100", "Iother_mental_health_100", "Isub_abuse_100",
        "Ihas_pcp_100", "Iusual_ed_100", "Isingle_100", "Isocsupport_always_100",
        "Ihousing_own_100", "Ienglish_100", "Ihs_graduate_100", "Iempl_no_100",
        "Iself_health_poor_100", "Imobility_needshelp_100",
    ]
    for c in cam_pre_cols:
        assert "post" not in c.lower() and "discharged" not in c.lower() and "readmit" not in c.lower(), f"Post-treatment leakage in Camden: {c}"
    X_cam = cam[cam_pre_cols].copy()
    X_cam["log1p_pre_receipts_180"] = np.log1p(cam["pre_ie_receipts_180_IP"].fillna(0.0).clip(lower=0.0))

    # 3. OHIE-Survey strictly pre-lottery covariates (N = 17,896 complete QJE spending respondents)
    oh_s12_cc = oh_s12_r.dropna(subset=["spend_qje_12m", "weight_12m", "cell"]).copy().reset_index(drop=True)
    oh_s12_pre_cols = [
        "birthyear_list", "female_list", "english_list", "self_list",
        "first_day_list", "last_day_list", "have_phone_list", "pobox_list",
        "week_list", "zip_msa_list", "numhh_list",
        "snap_ever_prenotify07", "tanf_ever_prenotify07",
        "snap_tot_hh_prenotify07", "tanf_tot_hh_prenotify07",
    ]
    X_oh_s12 = oh_s12_cc[oh_s12_pre_cols].copy()
    X_oh_s12["in_portland_ed_sample"] = (oh_s12_cc["sample_ed"] == 1).astype(float)
    X_oh_s12["num_visit_pre_cens_ed_0fill"] = oh_s12_cc["num_visit_pre_cens_ed"].fillna(0.0)
    X_oh_s12["log1p_charg_pre_ed_0fill"] = np.log1p(oh_s12_cc["charg_tot_pre_ed"].fillna(0.0).clip(lower=0.0))

    # 4. OHIE-ED Administrative strictly pre-lottery covariates (N = 24,621 non-missing charg_tot_ed)
    oh_ed_cc = oh_ed_m.dropna(subset=["charg_tot_ed", "numhh_list"]).copy().reset_index(drop=True)
    oh_ed_pre_cols = [
        "birthyear_list", "female_list", "english_list", "self_list",
        "first_day_list", "last_day_list", "have_phone_list", "pobox_list",
        "week_list", "numhh_list",
        "snap_ever_prenotify07", "tanf_ever_prenotify07",
        "snap_tot_hh_prenotify07", "tanf_tot_hh_prenotify07",
        "any_visit_pre_ed", "num_visit_pre_cens_ed",
        "any_hosp_pre_ed", "num_hosp_pre_cens_ed",
        "any_out_pre_ed", "num_out_pre_cens_ed",
        "any_on_pre_ed", "num_on_pre_cens_ed",
        "any_off_pre_ed", "num_off_pre_cens_ed",
        "num_edcnnp_pre_ed", "num_edcnpa_pre_ed", "num_epct_pre_ed", "num_ne_pre_ed",
        "any_acsc_pre_ed", "any_chron_pre_ed", "any_inj_pre_ed",
        "any_depres_pre_ed", "any_psysub_pre_ed",
        "charg_tot_pre_ed", "ed_charg_tot_pre_ed",
        "num_hiun_pre_cens_ed", "num_loun_pre_cens_ed",
    ]
    for c in oh_ed_pre_cols:
        assert ("_pre_" in c) or ("_list" in c) or ("prenotify07" in c), f"Post-treatment leakage in OHIE-ED: {c}"
    X_oh_ed = oh_ed_cc[oh_ed_pre_cols].copy()
    X_oh_ed["log1p_charg_tot_pre_ed"] = np.log1p(oh_ed_cc["charg_tot_pre_ed"].fillna(0.0).clip(lower=0.0))

    # Run Component 2 & 3 on all spending arms (plus Camden readmission & OHIE MEPS 2023 sensitivity)
    trial_configs = [
        {
            "trial_id": "illinois_30m_spend",
            "label": "Illinois Workplace Wellness (30-month annualized spending, QJE 2019)",
            "unit": "USD/year (annualized = 12 x monthly)",
            "unit_scale_to_annual": 1.0,
            "X_df": X_il_30m,
            "Y": il_30m_sub["spend_0816_0119"].values * 12.0,
            "W": il_30m_sub["treat"].values,
            "weights": il_30m_sub["covg_0816_0119"].values,
            "strata": il_30m_sub["Strata_claims"].values,
            "groups": np.arange(len(il_30m_sub)),
            "baseline_risk_col": "spend_0715_0716",
            "c0_grid": [0.0, 140.0, 250.0, 360.0],
            "run_perm": True,
        },
        {
            "trial_id": "illinois_yr1_spend",
            "label": "Illinois Workplace Wellness (Year-1 annualized spending, QJE 2019)",
            "unit": "USD/year (annualized = 12 x monthly)",
            "unit_scale_to_annual": 1.0,
            "X_df": X_il_yr1,
            "Y": il_yr1_sub["spend_0816_0717"].values * 12.0,
            "W": il_yr1_sub["treat"].values,
            "weights": il_yr1_sub["covg_0816_0717"].values,
            "strata": il_yr1_sub["Strata_claims"].values,
            "groups": np.arange(len(il_yr1_sub)),
            "baseline_risk_col": "spend_0715_0716",
            "c0_grid": [0.0, 140.0, 250.0, 360.0],
            "run_perm": True,
        },
        {
            "trial_id": "camden_180d_receipts",
            "label": "Camden Hotspotting RCT (180-day hospital receipts, NEJM 2020)",
            "unit": "USD/180 days (annualized scale x2 for Theorem 3 n_crit)",
            "unit_scale_to_annual": 2.0,
            "X_df": X_cam,
            "Y": cam["post_ie_receipts_180_IP"].values,
            "W": cam["Itreatment"].values.astype(int),
            "weights": np.ones(len(cam)),
            "strata": np.zeros(len(cam), dtype=int),
            "groups": np.arange(len(cam)),
            "baseline_risk_col": "pre_ie_receipts_180_IP",
            "c0_grid": [0.0, 4000.0, 7000.0, 10000.0],
            "run_perm": True,
        },
        {
            "trial_id": "camden_180d_charges",
            "label": "Camden Hotspotting RCT (180-day hospital charges, NEJM 2020)",
            "unit": "USD/180 days (annualized scale x2 for Theorem 3 n_crit)",
            "unit_scale_to_annual": 2.0,
            "X_df": X_cam,
            "Y": cam["post_ie_charges_180_IP"].values,
            "W": cam["Itreatment"].values.astype(int),
            "weights": np.ones(len(cam)),
            "strata": np.zeros(len(cam), dtype=int),
            "groups": np.arange(len(cam)),
            "baseline_risk_col": "pre_ie_charges_180_IP",
            "c0_grid": [0.0, 4000.0, 10000.0],
            "run_perm": False,
        },
        {
            "trial_id": "ohie_survey_spend_qje2012",
            "label": "OHIE Medicaid Insurance Expansion Benchmark (12m annualized QJE-2012 MEPS spending)",
            "unit": "USD/year (QJE 2012 MEPS weights)",
            "unit_scale_to_annual": 1.0,
            "X_df": X_oh_s12,
            "Y": oh_s12_cc["spend_qje_12m"].values,
            "W": oh_s12_cc["treatment"].values.astype(int),
            "weights": oh_s12_cc["weight_12m"].values,
            "strata": oh_s12_cc["cell"].values,
            "groups": oh_s12_cc["household_id"].values,
            "baseline_risk_col": None,
            "c0_grid": [0.0, 500.0, 1500.0, 3000.0],
            "run_perm": True,
        },
        {
            "trial_id": "ohie_survey_spend_meps2023",
            "label": "OHIE Medicaid Insurance Expansion Benchmark (12m annualized MEPS-2023 spending)",
            "unit": "USD/year (AHRQ MEPS 2023 event prices)",
            "unit_scale_to_annual": 1.0,
            "X_df": X_oh_s12,
            "Y": oh_s12_cc["spend_meps2023_12m"].values,
            "W": oh_s12_cc["treatment"].values.astype(int),
            "weights": oh_s12_cc["weight_12m"].values,
            "strata": oh_s12_cc["cell"].values,
            "groups": oh_s12_cc["household_id"].values,
            "baseline_risk_col": None,
            "c0_grid": [0.0, 500.0, 1500.0, 3000.0],
            "run_perm": False,
        },
        {
            "trial_id": "ohie_ed_admin_charges",
            "label": "OHIE Medicaid Insurance Expansion Benchmark (18m administrative ED+inpatient charges, Science 2014)",
            "unit": "USD/18 months (annualized scale 12/18 for Theorem 3 n_crit)",
            "unit_scale_to_annual": 12.0 / 18.0,
            "X_df": X_oh_ed,
            "Y": oh_ed_cc["charg_tot_ed"].values,
            "W": oh_ed_cc["treatment"].values.astype(int),
            "weights": np.ones(len(oh_ed_cc)),
            "strata": oh_ed_cc["numhh_list"].values,
            "groups": oh_ed_cc["household_id"].values,
            "baseline_risk_col": "charg_tot_pre_ed",
            "c0_grid": [0.0, 500.0, 1500.0, 3000.0],
            "run_perm": True,
        },
    ]

    cate_results: Dict[str, Any] = {}
    for cfg in trial_configs:
        tid = cfg["trial_id"]
        print(f"  -> CATE & Winner's Curse audit: {tid} (N={len(cfg['Y'])}, p={cfg['X_df'].shape[1]})...")
        res = evaluate_trial_cate_and_winners_curse(
            trial_id=tid,
            label=cfg["label"],
            unit=cfg["unit"],
            unit_scale_to_annual=cfg["unit_scale_to_annual"],
            X_df=cfg["X_df"],
            Y=cfg["Y"],
            W=cfg["W"],
            weights=cfg["weights"],
            strata=cfg["strata"],
            groups=cfg["groups"],
            baseline_risk_col=cfg["baseline_risk_col"],
            c0_grid=cfg["c0_grid"],
            seed=SEED,
        )
        cate_results[tid] = res
        d0 = res["sign_rule_by_c0"][0]
        blp = res["blp_and_gates"]["dr_learner"]["blp"]
        print(
            f"     done ({res['runtime_sec']}s) | ATE={res['ate_aipw']['mean']:+.1f} (SE {res['ate_aipw']['se']:.1f}) | "
            f"BLP b2={blp['beta2_het']:+.3f} (p={blp['beta2_p']:.3f}) | "
            f"c0=0: NaivePlugIn={d0['naive_insample_matched_plugin']['net_cost_per_eligible']:+.1f}, "
            f"NaiveEmp={d0['naive_insample_matched_emp']['net_cost_per_eligible']:+.1f} (p={d0['naive_insample_matched_emp']['p_one_sided_savings']:.3g}) vs "
            f"HonestOOS={d0['honest_oos_dr']['net_cost_per_eligible']:+.1f} (SE {d0['honest_oos_dr']['se_per_eligible']:.1f}, p={d0['honest_oos_dr']['p_one_sided_savings']:.3f})"
        )

    # Illinois Empirical Coverage Churn & Employer Termination Module
    print("\n[4/5] Computing Illinois empirical 30-month coverage churn & job termination...")
    il_churn = analyze_illinois_empirical_churn(il_claims, il_admin)
    bi = il_churn["baseline_insured_any"]
    print(
        f"  Baseline-insured (N={bi['n']}): overall 30m any-loss={bi['overall_any_covg_loss_30m']*100:.1f}% "
        f"(h={bi['overall_implied_annual_hazard_h']*100:.1f}%/yr); "
        f"Q4-Q1={bi['q4_minus_q1_any_loss_pp']:+.2f} pp (chi2 p={bi['chi2_any_loss_p']:.4g})"
    )

    # =========================================================================
    # COMPONENT 4: KILL-SWITCH 3B — 500-DRAW SHARP-NULL PERMUTATION PLACEBO TEST
    # =========================================================================
    print(f"\n[5/5] Running Kill-Switch 3B: {N_PERM}-Draw Sharp-Null Permutation Winner's Curse Placebo Test...")
    perm_results: Dict[str, Any] = {}
    for cfg in trial_configs:
        if not cfg["run_perm"]:
            continue
        tid = cfg["trial_id"]
        print(f"  -> Permuting {tid} ({N_PERM} design-respecting draws)...")
        pres = run_sharp_null_permutation_audit(
            trial_id=tid,
            X_df=cfg["X_df"],
            Y=cfg["Y"],
            W=cfg["W"],
            weights=cfg["weights"],
            strata=cfg["strata"],
            groups=cfg["groups"],
            n_perm=N_PERM,
            seed=SEED,
        )
        perm_results[tid] = pres
        ni = pres["naive_insample_matched"]
        ns = pres["naive_insample_standard"]
        ho = pres["honest_oos_aipw"]
        print(
            f"     done ({pres['runtime_sec']}s) | "
            f"Naive FP(matched)={ni['fp_sign_flip_rate_p05']*100:.1f}% (std={ns['fp_sign_flip_rate_p05']*100:.1f}%), "
            f"Mean Illusory Emp={ni['mean_illusory_savings_emp_per_eligible']:+.1f} (PlugIn={ni['mean_illusory_savings_plugin_per_eligible']:+.1f}) | "
            f"Honest OOS Mean={ho['mean_realized_net_cost_per_eligible']:+.1f}, "
            f"OOS Type-I(1s)={ho['type1_error_one_sided_savings_p05']*100:.2f}% (2s={ho['type1_error_two_sided_p05']*100:.2f}%) | "
            f"KS3B={'PASS' if pres['ks3b_pass_oos_type1_le_006'] else 'FAIL'}"
        )

    ks3b_all_pass = all(v["ks3b_pass_oos_type1_le_006"] for v in perm_results.values())
    ks3_overall_pass = bool(ks3a_pass and ks3b_all_pass)

    # =========================================================================
    # SAVE JSON & SUMMARY CSV
    # =========================================================================
    payload = {
        "metadata": {
            "project": "preventive-targeting-paradox",
            "pillar": "Empirical Pillar 2 — Real U.S. Randomized Spending-CATE & Winner's Curse Audit",
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_permutations": N_PERM,
            "scope_verification": {
                "included_rcts": [
                    "Illinois Workplace Wellness Study (Jones, Molitor, & Reif, QJE 2019)",
                    "Camden Hotspotting RCT (Finkelstein, Zhou, Taubman, & Doyle, NEJM 2020)",
                    "Oregon Health Insurance Experiment — Medicaid insurance expansion benchmark (Finkelstein et al., QJE 2012; Baicker et al., NEJM 2013; Taubman et al., Science 2014)",
                ],
                "excluded_acute_and_non_us_trials": ["gusto", "ist", "elaia1", "elaia2", "crash2", "synbiotic"],
                "zero_leakage_verified": True,
                "zero_phi": True,
            },
            "total_runtime_sec": round(time.time() - t_start, 2),
        },
        "kill_switches": {
            "ks3a_published_itt_replication_pass": ks3a_pass,
            "ks3b_500perm_oos_type1_le_006_pass": ks3b_all_pass,
            "ks3_overall_pass": ks3_overall_pass,
        },
        "component1_ks3a_itt_replication": ks3a,
        "component2_and_3_cate_and_winners_curse": cate_results,
        "illinois_empirical_churn_and_termination": il_churn,
        "component4_ks3b_sharp_null_permutations": perm_results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Build flat summary CSV across all evaluated spending trial arms
    summary_rows = []
    for cfg in trial_configs:
        tid = cfg["trial_id"]
        r = cate_results[tid]
        dist = r["distribution_and_heavy_tails"]
        ncrit = dist["theorem3_ncrit"]["by_target_saving_usd_yr"]
        blp_dr = r["blp_and_gates"]["dr_learner"]["blp"]
        gates_dr = r["blp_and_gates"]["dr_learner"]["gates"]
        spread_dr = r["blp_and_gates"]["dr_learner"]["gates_spread_G4_minus_G1"]
        blp_cf = r["blp_and_gates"]["causal_forest"]["blp"]
        s0 = r["sign_rule_by_c0"][0]
        # Realistic nonzero program cost row (index 1 in c0_grid)
        s_cost = r["sign_rule_by_c0"][1]
        p_res = perm_results.get(tid)

        summary_rows.append({
            "trial_id": tid,
            "label": r["label"],
            "is_medicaid_expansion_benchmark": tid.startswith("ohie_"),
            "unit": r["unit"],
            "n": r["n"],
            "n_clusters": r["n_clusters"],
            "p_covariates": r["p_covariates"],
            "mean_spending": round(dist["mean"], 2),
            "sd_spending": round(dist["sd"], 2),
            "skewness": round(dist["skewness"], 2),
            "excess_kurtosis": round(dist["excess_kurtosis"], 2),
            "p95_spending": round(dist["p95"], 2),
            "p99_spending": round(dist["p99"], 2),
            "max_spending": round(dist["max"], 2),
            "residual_sd_oof": round(dist["residual_spending_sd_oof"], 2),
            "sigma_gamma_aipw": round(dist["aipw_pseudo_outcome_sd_sigma_gamma"], 2),
            "r2_baseline_oof": round(dist["r2_baseline_spending_oof"], 4),
            "ncrit_q20_power80_minus200": ncrit["minus_200"]["N_power80_total_trial_q20"],
            "ncrit_q20_power80_minus500": ncrit["minus_500"]["N_power80_total_trial_q20"],
            "ncrit_q20_power80_minus1000": ncrit["minus_1000"]["N_power80_total_trial_q20"],
            "ate_aipw": round(r["ate_aipw"]["mean"], 2),
            "ate_se": round(r["ate_aipw"]["se"], 2),
            "ate_p": round(r["ate_aipw"]["p_two_sided"], 4),
            "dr_blp_beta2": round(blp_dr["beta2_het"], 4),
            "dr_blp_beta2_se": round(blp_dr["beta2_se"], 4),
            "dr_blp_beta2_p": round(blp_dr["beta2_p"], 4),
            "dr_gates_G1_most_saving": round(gates_dr["G1"]["cate"], 2),
            "dr_gates_G1_se": round(gates_dr["G1"]["se"], 2),
            "dr_gates_G4_least_saving": round(gates_dr["G4"]["cate"], 2),
            "dr_gates_G4_se": round(gates_dr["G4"]["se"], 2),
            "dr_gates_G4_minus_G1": round(spread_dr["estimate"], 2),
            "dr_gates_G4_minus_G1_p": round(spread_dr["p_two_sided"], 4),
            "cf_blp_beta2": round(blp_cf["beta2_het"], 4),
            "cf_blp_beta2_p": round(blp_cf["beta2_p"], 4),
            "c0_0_naive_plugin_per_elig": round(s0["naive_insample_matched_plugin"]["net_cost_per_eligible"], 2),
            "c0_0_naive_emp_per_elig": round(s0["naive_insample_matched_emp"]["net_cost_per_eligible"], 2),
            "c0_0_naive_emp_p1s": round(s0["naive_insample_matched_emp"]["p_one_sided_savings"], 5),
            "c0_0_oos_dr_per_elig": round(s0["honest_oos_dr"]["net_cost_per_eligible"], 2),
            "c0_0_oos_dr_se": round(s0["honest_oos_dr"]["se_per_eligible"], 2),
            "c0_0_oos_dr_p1s": round(s0["honest_oos_dr"]["p_one_sided_savings"], 4),
            "c0_0_oos_dr_martingale_per_elig": round(s0["honest_oos_dr_martingale"]["net_cost_per_eligible"], 2),
            "c0_0_oos_dr_martingale_se": round(s0["honest_oos_dr_martingale"]["se_per_eligible"], 2),
            "c0_0_oos_dr_martingale_p1s": round(s0["honest_oos_dr_martingale"]["p_one_sided_savings"], 4),
            "c0_0_oos_cf_per_elig": round(s0["honest_oos_cf"]["net_cost_per_eligible"], 2),
            "c0_0_oos_cf_se": round(s0["honest_oos_cf"]["se_per_eligible"], 2),
            "c0_0_wcg_plugin_per_elig": round(s0["winners_curse_gap_matched_plugin_per_eligible"], 2),
            "c0_0_wcg_emp_per_elig": round(s0["winners_curse_gap_matched_emp_per_eligible"], 2),
            "c0_prog_cost": s_cost["c0"],
            "c0_prog_oos_dr_per_elig": round(s_cost["honest_oos_dr"]["net_cost_per_eligible"], 2),
            "c0_prog_oos_dr_se": round(s_cost["honest_oos_dr"]["se_per_eligible"], 2),
            "c0_prog_oos_dr_targeted_share": round(s_cost["honest_oos_dr"]["targeted_share"], 4),
            "perm_n": p_res["n_permutations"] if p_res else np.nan,
            "perm_naive_matched_fp_rate": round(p_res["naive_insample_matched"]["fp_sign_flip_rate_p05"], 4) if p_res else np.nan,
            "perm_naive_std_fp_rate": round(p_res["naive_insample_standard"]["fp_sign_flip_rate_p05"], 4) if p_res else np.nan,
            "perm_naive_mean_illusory_emp": round(p_res["naive_insample_matched"]["mean_illusory_savings_emp_per_eligible"], 2) if p_res else np.nan,
            "perm_naive_mean_illusory_plugin": round(p_res["naive_insample_matched"]["mean_illusory_savings_plugin_per_eligible"], 2) if p_res else np.nan,
            "perm_oos_mean_net_cost": round(p_res["honest_oos_aipw"]["mean_realized_net_cost_per_eligible"], 2) if p_res else np.nan,
            "perm_oos_type1_1s": round(p_res["honest_oos_aipw"]["type1_error_one_sided_savings_p05"], 4) if p_res else np.nan,
            "perm_oos_type1_2s": round(p_res["honest_oos_aipw"]["type1_error_two_sided_p05"], 4) if p_res else np.nan,
            "perm_ks3b_pass": p_res["ks3b_pass_oos_type1_le_006"] if p_res else np.nan,
        })

    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 90)
    print(f"COMPLETED in {time.time() - t_start:.1f}s")
    print(f"  JSON saved: {OUT_JSON}")
    print(f"  CSV  saved: {OUT_CSV}")
    print(f"  KS3A (Exact ITT Replication): {'PASS' if ks3a_pass else 'FAIL'}")
    print(f"  KS3B (500-Perm OOS Type-I <= 0.06): {'PASS' if ks3b_all_pass else 'FAIL'}")
    print(f"  KS3  OVERALL: {'PASS' if ks3_overall_pass else 'FAIL'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
