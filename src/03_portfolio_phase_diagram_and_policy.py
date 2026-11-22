"""
Pillar 3 (Round-2 Referee-Hardened): Unified U.S. Four-Wedge Phase-Diagram
Audit across 14 Canonical Preventive Interventions, Strictly Age-Matched
Multi-Payer Accounting (Health Care Sector All-Payer, Medicare 65+, Commercial
Plan 18-64, Capitated Medicaid MCO Pre-H.R. 1 & H.R. 1 18-64), 5-Step Kraus
(2024) Metformin Waterfall, Multi-Period CBO 10/20/30y & Federal Budget Scoring,
GLP-1 Break-Even Pricing (with MDRP & IRA/Patent-Cliff Sensitivities),
Medicaid Policy Levers (PMPY & 10-yr), Equity Redlining Audit (Panel A: Direct
Weighted FPL/Payer; Panel B: Race/Ethnicity), and Full CHEERS 2022 Sensitivity
Suite (0%/3%/5% discount, TTD/Naive/LTSS, 0.5x/2.0x cate_rmse, MDRP, MCO
(kappa, b_rev), and Illinois risk-correlated churn).

Reads ONLY locked Pillar 1 outputs (US-PAID MEPS 2019-2023, CDC 2023 Life
Tables, NHANES 2021-2023). Zero external network calls; runs in <5s on CPU.
"""
import json
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

p1 = json.loads((RES / "pillar1_us_paid_and_churn.json").read_text())
us_paid = pd.read_csv(PROC / "us_paid_age_schedule_2024usd.csv")
lt_all = pd.read_parquet(PROC / "cdc_lt2023_all.parquet")
nh = pd.read_parquet(PROC / "nhanes_2021_2023_adults.parquet")

lt_by_grp = {g: df.sort_values("age").reset_index(drop=True) for g, df in lt_all.groupby("group")}

# Verified MEPS 2022-2023 payer unit costs (2024 USD)
IP_COMM = p1["payer_unit_costs_2024usd"]["ip_Commercial (18-64)"]["mean_2024"]
IP_MED  = p1["payer_unit_costs_2024usd"]["ip_Medicaid (18-64)"]["mean_2024"]
IP_MCR  = p1["payer_unit_costs_2024usd"]["ip_Medicare (65+)"]["mean_2024"]
ER_COMM = p1["payer_unit_costs_2024usd"]["er_Commercial (18-64)"]["mean_2024"]
ER_MED  = p1["payer_unit_costs_2024usd"]["er_Medicaid (18-64)"]["mean_2024"]
ER_MCR  = p1["payer_unit_costs_2024usd"]["er_Medicare (65+)"]["mean_2024"]

# Strictly age-matched payer specifications (fixing Referee 1 Major #1 & #2):
# - health_care_sector (Second Panel terminology, Sanders et al. 2016): all ages, h=0, kappa=0, price=1.00
# - medicare: age >= 65 strictly, h=0.005, kappa=0, price=0.92 (MEPS Medicare/all-payer blend)
# - commercial_plan: age < 65 strictly, h=0.242 (21.5%/yr external turnover, Fang et al. 2022), age65_handoff=True, price=1.28
# - commercial_employer: age < 65 strictly, h=0.105 (10%/yr job separation, Illinois RCT 9.1%/yr sensitivity), age65_handoff=True, price=1.28
# - medicaid_pre_hr1: age < 65 strictly, h=0.288 (25%/yr, Sommers et al. 2016), kappa_dx=0.50, kappa_acute=0.10,
#   cap_rev_share=1.00 (actuarially sound MCO survivor neutrality; sensitivity b_rev=0.90/0.95/1.00),
#   branded_drug_mult=0.60 (Medicaid MDRP statutory rebate sensitivity 0.50-1.00), price=0.65 (MEPS 0.47x IP / 0.30x ER blend)
# - medicaid_hr1: age < 65 strictly, h=0.693 (50%/yr work-requirement regime for non-exempt adults; pregnant cfDNA exempt -> h=0.15)
PAYER_PARAMS = {
    "health_care_sector":  {"age_min": 0,  "age_max": 120, "h": 0.000, "kappa_dx": 0.00, "kappa_acute": 0.00, "cap_rev_share": 0.00, "price_mult": 1.00, "branded_mult": 1.00, "age65_handoff": False},
    "medicare":            {"age_min": 65, "age_max": 120, "h": 0.005, "kappa_dx": 0.00, "kappa_acute": 0.00, "cap_rev_share": 0.00, "price_mult": 0.92, "branded_mult": 0.90, "age65_handoff": False},
    "commercial_plan":     {"age_min": 18, "age_max": 64,  "h": 0.242, "kappa_dx": 0.00, "kappa_acute": 0.00, "cap_rev_share": 0.00, "price_mult": 1.28, "branded_mult": 1.00, "age65_handoff": True},
    "commercial_employer": {"age_min": 18, "age_max": 64,  "h": 0.105, "kappa_dx": 0.00, "kappa_acute": 0.00, "cap_rev_share": 0.00, "price_mult": 1.28, "branded_mult": 1.00, "age65_handoff": True},
    "medicaid_pre_hr1":    {"age_min": 18, "age_max": 64,  "h": 0.288, "kappa_dx": 0.50, "kappa_acute": 0.10, "cap_rev_share": 1.00, "price_mult": 0.65, "branded_mult": 0.60, "age65_handoff": True},
    "medicaid_hr1":        {"age_min": 18, "age_max": 64,  "h": 0.693, "kappa_dx": 0.50, "kappa_acute": 0.10, "cap_rev_share": 1.00, "price_mult": 0.65, "branded_mult": 0.60, "age65_handoff": True},
}

DISCOUNT_R = 0.03

# =============================================================================
# EXACT PAID 3.0 NON-NEGATIVE UNRELATED SURVIVAL COST KERNEL
# =============================================================================
_SURV_CACHE = {}

def averted_death_survival_pv(a0: int, domain: str, r: float = 0.03, h: float = 0.0,
                              include_ttd: bool = True, include_ltss: bool = False,
                              age65_handoff: bool = False, horizon: int = None,
                              lt_group: str = "total", subtract_t0_eol: bool = False):
    """
    Exact PAID 3.0 cause-specific unrelated survival-period medical cost from
    averting one target-disease death at age a0.
    - When S_rel already includes fatal target-disease terminal care (standard PAID 3.0),
      subtract_t0_eol=False -> added unrelated spending = sum_{t>=0} beta_t * (S_t * m_surv_unrel + d_t * K_eol_unrel) >= 0.
    - When comparing all-cause EOL postponement (subtract_t0_eol=True, matching Theorem 1's
      all-cause Abel identity), subtracts k_eol(a0) at t=0.
    """
    key = (int(np.clip(a0, 18, 95)), domain, round(r, 4), round(h, 4), include_ttd, include_ltss, age65_handoff, horizon, lt_group, subtract_t0_eol)
    if key in _SURV_CACHE:
        return _SURV_CACHE[key]

    a_c = key[0]
    lt_df = lt_by_grp[lt_group]
    qx = lt_df["qx"].values
    max_t = 100 - a_c
    if horizon is not None:
        max_t = min(max_t, max(0, int(horizon)))
    payer_cap_t = max(0, 65 - a_c) if (age65_handoff and a_c < 65) else max_t

    S = np.zeros(max_t + 1)
    S[0] = 1.0
    for t in range(max_t):
        S[t + 1] = S[t] * (1.0 - float(qx[a_c + t]))
    d = S[:-1] - S[1:]

    col_m = f"m_surv_unrel_{domain}" if domain != "all" else "m_surv_all_2024"
    col_k = f"k_eol_unrel_{domain}" if domain != "all" else "k_eol_all_2024"
    m_arr = us_paid[col_m].values
    k_arr = us_paid[col_k].values
    ltss_arr = us_paid["ltss_factor"].values if include_ltss else np.ones(101)

    pv_cost = 0.0
    pv_lys = 0.0
    for t in range(max_t):
        age = a_c + t
        pv_lys += np.exp(-r * t) * S[t]
        if t < payer_cap_t:
            beta = np.exp(-(r + h) * t)
            if include_ttd:
                pv_cost += beta * (S[t] * m_arr[age] + d[t] * k_arr[age]) * ltss_arr[age]
            else:
                sh = us_paid.loc[age, f"share_{domain}"] if domain != "all" else 0.0
                pv_cost += beta * S[t] * us_paid.loc[age, "m_tot_2024"] * (1.0 - sh) * ltss_arr[age]

    if subtract_t0_eol and include_ttd and payer_cap_t > 0:
        pv_cost -= k_arr[a_c] * ltss_arr[a_c]

    res = (float(pv_cost), float(pv_lys))
    _SURV_CACHE[key] = res
    return res


def scarf_cantelli_ceiling(mu: float, sd: float, L: float = None) -> float:
    """Exact Theorem 2 ceiling |V*| (Lean 4 verified: L>=0 => 0; L^2 <= mu^2+sd^2 => lower-bounded tightening)."""
    if L is not None and L >= 0.0:
        return 0.0
    if sd <= 0.0:
        return max(-mu, 0.0)
    unbounded = 0.5 * (np.sqrt(mu**2 + sd**2) - mu)
    if L is not None and L < 0.0 and mu > 0.0 and (L**2 <= mu**2 + sd**2):
        lower_bnd = (-L * sd**2) / ((mu - L)**2 + sd**2)
        return float(min(unbounded, lower_bnd))
    return float(unbounded)


def gaussian_mills_rho_star(mu: float, sd: float) -> float:
    if mu <= 0 or sd <= 0:
        return 0.0
    gamma = mu / sd
    def g(rho):
        z = gamma * np.sqrt(rho)
        return mu * norm.sf(z) - sd * np.sqrt(rho) * norm.pdf(z)
    lo = max(1e-9, (mu**2) / (mu**2 + sd**2) * 0.999)
    hi = min(1.0 - 1e-9, (mu**2) / (mu**2 + (2.0 / np.pi) * sd**2) * 1.001)
    if g(lo) * g(hi) < 0:
        return float(brentq(g, lo, hi))
    return float(brentq(g, 1e-9, 1.0 - 1e-9))


def evaluate_cohort_targeting(theta: np.ndarray, qaly: np.ndarray, c0: np.ndarray, weights: np.ndarray,
                              cate_rmse: float = 0.0, lam_hcs: float = 100000.0):
    if len(theta) == 0 or np.sum(weights) <= 0:
        return {"applicable": False}
    w = weights / np.sum(weights)
    mu = float(np.sum(w * theta))
    sd = float(np.sqrt(np.sum(w * (theta - mu)**2)))
    L = float(np.min(theta))
    qaly_univ = float(np.sum(w * qaly))
    c0_univ = float(np.sum(w * c0))

    neg = theta < 0
    q_oracle = float(np.sum(w[neg]))
    net_oracle = float(np.sum(w[neg] * theta[neg]))
    net_oracle_treated = float(net_oracle / q_oracle) if q_oracle > 1e-9 else 0.0
    qaly_oracle_zero = float(np.sum(w[neg] * qaly[neg]))

    nmb = lam_hcs * qaly - theta
    evic_mask = nmb > 0
    q_evic = float(np.sum(w[evic_mask]))
    qaly_evic = float(np.sum(w[evic_mask] * qaly[evic_mask]))
    qaly_foregone_pct = float(100.0 * (qaly_evic - qaly_oracle_zero) / qaly_evic) if qaly_evic > 1e-12 else 0.0

    sc_bound = scarf_cantelli_ceiling(mu, sd, L)
    rho_star = gaussian_mills_rho_star(mu, sd) if mu > 0 else 0.0
    rmse_fail_thresh = (sd**2 / mu) if mu > 0 else np.inf

    if cate_rmse > 0:
        rng = np.random.default_rng(20261121)
        app_list, real_list = [], []
        for _ in range(200):
            th_hat = theta + rng.normal(0.0, cate_rmse, size=len(theta))
            sel = th_hat < 0
            app_list.append(float(np.sum(w[sel] * th_hat[sel])))
            real_list.append(float(np.sum(w[sel] * theta[sel])))
        net_plugin_apparent = float(np.mean(app_list))
        net_plugin_realized = float(np.mean(real_list))
    else:
        net_plugin_apparent = net_oracle
        net_plugin_realized = net_oracle

    return {
        "applicable": True,
        "n_unweighted": int(len(theta)),
        "us_pop_millions": round(float(np.sum(weights) / 1e6), 2),
        "universal_net_cost": round(mu, 1),
        "theta_sd": round(sd, 1),
        "theta_min": round(L, 1),
        "upfront_c0": round(c0_univ, 1),
        "universal_qaly": round(qaly_univ, 5),
        "universal_icer": round(mu / qaly_univ, 0) if qaly_univ > 1e-6 else None,
        "oracle_treated_share_pct": round(100.0 * q_oracle, 1),
        "oracle_net_cost_per_elig": round(net_oracle, 1),
        "oracle_net_cost_per_treated": round(net_oracle_treated, 1),
        "scarf_cantelli_max_saving": round(sc_bound, 1),
        "mills_rho_star": round(rho_star, 4),
        "rmse_guaranteed_fail": round(rmse_fail_thresh, 1) if np.isfinite(rmse_fail_thresh) else None,
        "plugin_cate_rmse": round(cate_rmse, 1),
        "plugin_apparent_net_cost": round(net_plugin_apparent, 1),
        "plugin_realized_net_cost": round(net_plugin_realized, 1),
        "winners_curse_gap": round(net_plugin_realized - net_plugin_apparent, 1),
        "evic_100k_treated_share_pct": round(100.0 * q_evic, 1),
        "evic_100k_qaly": round(qaly_evic, 5),
        "qaly_foregone_by_lambda0_pct": round(qaly_foregone_pct, 1),
    }


# =============================================================================
# STEP 1: BUILD 14 CITATION-VERIFIED U.S. PREVENTIVE INTERVENTION COHORTS
# =============================================================================
def build_intervention_cohorts(r: float = 0.03):
    cohorts = {}
    rng = np.random.default_rng(20261121)
    ann10 = 10.0 if r == 0.0 else (1.0 - np.exp(-r * 10.0)) / r

    # 1. Prenatal cfDNA screening (Conner et al. JPE 2025 positive control; ages 20-44, pregnant exempt from work req!)
    n_syn = 4000
    m_age = rng.choice(np.arange(20, 45), size=n_syn, p=np.exp(-0.5*((np.arange(20,45)-29.5)/5.8)**2) / np.sum(np.exp(-0.5*((np.arange(20,45)-29.5)/5.8)**2)))
    p_inv = np.clip(0.012 * np.exp(0.145 * (m_age - 25.0)) * rng.lognormal(0, 0.55, n_syn), 0.002, 0.65)
    cohorts["1_prenatal_cfdna"] = {
        "label": "Prenatal cfDNA screening",
        "archetype": "Regime 1: Acute/Diagnostic Substitution",
        "domain": "all", "cost_schedule": "upfront", "is_branded_drug": False,
        "is_chronic_dx": False, "pregnant_exempt_hr1": True, "annual_c0": 280.0,
        "t_rel": 0.25, "t_mort": 0.25, "age": m_age, "w": np.ones(n_syn) * (3.6e6 / n_syn),
        "c0": np.full(n_syn, 280.0), "s_rel_base": p_inv * 0.92 * 2650.0,
        "d_mort": np.zeros(n_syn), "qaly": p_inv * 0.92 * 0.015, "cate_rmse": 35.0,
        "param_summary": "c0=$280 test; 92% invasive substitution ($2,650); lag 0.25y; d_mort=0; RMSE=$35",
        "source": "Conner et al. (JPE 2025); ACOG",
    }

    # 2. Adult seasonal influenza vaccination (NHANES adults 18-80)
    sub_flu = nh.copy()
    age_f = sub_flu["RIDAGEYR"].values
    comorb_f = (sub_flu["PRIOR_CVD"].values + sub_flu["DIABETES"].values + sub_flu["ASTHMA_CURR"].values)
    p_flu_hosp = np.clip(0.00035 * np.exp(0.052 * np.maximum(age_f - 30, 0)) * (1.0 + 0.9 * comorb_f), 0.0002, 0.04)
    p_flu_ed = p_flu_hosp * 3.5
    ve_flu, cfr_flu = 0.45, np.where(age_f >= 65, 0.07, 0.015)
    cohorts["2_influenza_vaccine"] = {
        "label": "Adult influenza vaccination",
        "archetype": "Regime 1: Acute/Diagnostic Substitution",
        "domain": "respiratory", "cost_schedule": "upfront", "is_branded_drug": False,
        "is_chronic_dx": False, "pregnant_exempt_hr1": False, "annual_c0": 55.0,
        "t_rel": 0.4, "t_mort": 0.4, "age": age_f, "w": sub_flu["WTMEC2YR"].values,
        "c0": np.full(len(sub_flu), 55.0),
        "s_rel_base": ve_flu * (p_flu_hosp * 15500.0 + p_flu_ed * 1400.0 + 0.06 * 180.0),
        "d_mort": ve_flu * p_flu_hosp * cfr_flu,
        "qaly": ve_flu * (p_flu_hosp * 0.04 + 0.06 * 0.008) + ve_flu * p_flu_hosp * cfr_flu * np.array([averted_death_survival_pv(int(a), "respiratory", r=r)[1] * 0.72 for a in age_f]),
        "cate_rmse": 28.0,
        "param_summary": "c0=$55/yr; VE 45%; flu IP $15,500 / ED $1,400; lag 0.4y; RMSE=$28",
        "source": "CDC FluSurv-NET; NHANES 2021-2023",
    }

    # 3. Older-adult RSV vaccination (NHANES adults 60+, 2-season protection; branded vaccine)
    sub_rsv = nh[nh["RIDAGEYR"] >= 60].copy()
    age_r = sub_rsv["RIDAGEYR"].values
    comorb_r = (sub_rsv["PRIOR_CVD"].values + sub_rsv["DIABETES"].values + sub_rsv["ASTHMA_CURR"].values)
    p_rsv_hosp_2yr = np.clip(0.0025 * np.exp(0.082 * (age_r - 60.0)) * (1.0 + 1.15 * comorb_r), 0.001, 0.08)
    ve_rsv, cfr_rsv = 0.78, 0.075
    cohorts["3_rsv_vaccine_60plus"] = {
        "label": "Older-adult RSV vaccination (60+)",
        "archetype": "Regime 1: Acute/Diagnostic Substitution",
        "domain": "respiratory", "cost_schedule": "upfront", "is_branded_drug": True,
        "is_chronic_dx": False, "pregnant_exempt_hr1": False, "annual_c0": 295.0,
        "t_rel": 1.0, "t_mort": 1.0, "age": age_r, "w": sub_rsv["WTMEC2YR"].values,
        "c0": np.full(len(sub_rsv), 295.0),
        "s_rel_base": ve_rsv * p_rsv_hosp_2yr * (21500.0 + 2.0 * 1400.0) * np.exp(-r * 1.0),
        "d_mort": ve_rsv * p_rsv_hosp_2yr * cfr_rsv,
        "qaly": ve_rsv * p_rsv_hosp_2yr * (0.06 + cfr_rsv * np.array([averted_death_survival_pv(int(a), "respiratory", r=r)[1] * 0.70 for a in age_r])) * np.exp(-r * 1.0),
        "cate_rmse": 65.0,
        "param_summary": "c0=$295 single dose (2 seasons); VE 78%; RSV IP $21,500; lag 1.0y; RMSE=$65",
        "source": "CDC/ACIP 2024 RSV CEA; NHANES 2021-2023",
    }

    # 4. Recombinant Zoster Vaccination (RZV 2-dose Shingrix, NHANES 50+; UPFRONT cost, never churn-truncated!)
    sub_rzv = nh[nh["RIDAGEYR"] >= 50].copy()
    age_z = sub_rzv["RIDAGEYR"].values
    p_hz_10yr = np.clip(0.065 + 0.0032 * (age_z - 50.0), 0.05, 0.22)
    phn_frac = np.clip(0.08 + 0.008 * (age_z - 50.0), 0.08, 0.35)
    cohorts["4_rzv_shingles_50plus"] = {
        "label": "Recombinant zoster vaccine (50+)",
        "archetype": "Regime 1: Acute/Diagnostic Substitution",
        "domain": "all", "cost_schedule": "upfront", "is_branded_drug": True,
        "is_chronic_dx": False, "pregnant_exempt_hr1": False, "annual_c0": 430.0,
        "t_rel": 4.5, "t_mort": 4.5, "age": age_z, "w": sub_rzv["WTMEC2YR"].values,
        "c0": np.full(len(sub_rzv), 430.0),
        "s_rel_base": 0.85 * p_hz_10yr * (1600.0 + phn_frac * 5800.0) * np.exp(-r * 4.5),
        "d_mort": np.zeros(len(sub_rzv)),
        "qaly": 0.85 * p_hz_10yr * (0.04 + phn_frac * 0.28) * np.exp(-r * 4.5),
        "cate_rmse": 45.0,
        "param_summary": "c0=$430 2-dose upfront; VE 85%; zoster $1,600 / PHN $5,800; lag 4.5y; d_mort=0; RMSE=$45",
        "source": "CDC/ACIP RZV CEA; NHANES 2021-2023",
    }

    # 5. Asthma Controller ICS Step-Up (NHANES current asthma ages 18-64; DETERMINISTIC clinical risk score—zero random draws!)
    sub_ast = nh[(nh["ASTHMA_CURR"] == 1) & (nh["RIDAGEYR"].between(18, 64))].copy()
    age_as = sub_ast["RIDAGEYR"].values
    bmi_as = np.nan_to_num(sub_ast["BMXBMI"].values, nan=28.0)
    smk_as = sub_ast["CURR_SMOKER"].values
    com_as = sub_ast["PRIOR_CVD"].values + sub_ast["DIABETES"].values
    # Deterministic NHANES clinical risk multiplier (obesity, smoking, comorbidity, age; mean ~0.025 IP/yr, 0.11 ED/yr, skewed top tail)
    risk_idx = np.exp(0.55 * smk_as + 0.055 * np.maximum(bmi_as - 25.0, 0) + 0.70 * com_as + 0.015 * np.maximum(age_as - 35.0, 0) - 0.85)
    p_ast_ip = np.clip(0.018 * risk_idx, 0.002, 0.35)
    p_ast_ed = np.clip(0.080 * risk_idx, 0.010, 1.20)
    cohorts["5_asthma_ics_controller"] = {
        "label": "Asthma controller ICS step-up",
        "archetype": "Regime 1: Acute/Diagnostic Substitution",
        "domain": "respiratory", "cost_schedule": "upfront", "is_branded_drug": False,
        "is_chronic_dx": False, "pregnant_exempt_hr1": False, "annual_c0": 420.0,
        "t_rel": 0.5, "t_mort": 0.5, "age": age_as, "w": sub_ast["WTMEC2YR"].values,
        "c0": np.full(len(sub_ast), 420.0),
        "s_rel_base": 0.45 * (p_ast_ip * 14500.0 + p_ast_ed * 1650.0),
        "d_mort": np.zeros(len(sub_ast)),
        "qaly": 0.45 * (p_ast_ip * 0.05 + p_ast_ed * 0.015 + 0.018),
        "cate_rmse": 180.0,
        "nhanes_sub": sub_ast,
        "param_summary": "c0=$420/yr generic ICS; RRR 45%; asthma IP $14,500 / ED $1,650; lag 0.5y; d_mort=0; RMSE=$180",
        "source": "GINA/NAEPP; CDC Asthma Surveillance; MEPS/NHANES 2021-2023",
    }

    # --- REGIME 2: CHRONIC LONGEVITY-EXTENDING & COMPLEX-CARE INTERVENTIONS ---

    # 6. Generic Statins for Primary CVD Prevention (NHANES 40-75, real PCE 10-yr ASCVD risk)
    sub_st = nh[(nh["RIDAGEYR"].between(40, 75)) & (nh["PRIOR_CVD"] == 0) & (nh["ASCVD_10YR"].notna())].copy()
    age_st = sub_st["RIDAGEYR"].values
    r10_st = sub_st["ASCVD_10YR"].values
    cohorts["6_generic_statins_primary"] = {
        "label": "Generic statins (primary CVD)",
        "archetype": "Regime 2: Chronic Cardiometabolic",
        "domain": "cvd", "cost_schedule": "ongoing_10yr", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 120.0,
        "t_rel": 5.0, "t_mort": 5.5, "age": age_st, "w": sub_st["WTMEC2YR"].values,
        "c0": np.full(len(sub_st), 120.0 * ann10),
        "s_rel_base": 0.25 * r10_st * 42000.0 * np.exp(-r * 5.0),
        "d_mort": 0.25 * r10_st * 0.22,
        "qaly": 0.25 * r10_st * (0.22 * np.array([averted_death_survival_pv(int(min(a+5,95)), "cvd", r=r)[1] * 0.78 for a in age_st]) + 0.78 * 0.85) * np.exp(-r * 5.0),
        "cate_rmse": 650.0,
        "nhanes_sub": sub_st,
        "param_summary": "c0=$120/yr (10y PV $1,037); CTT RRR 25% events / 22% fatal; ASCVD cost $42,000; lag 5.0y; RMSE=$650",
        "source": "CTT Collaborators (Lancet 2012); NHANES 2021-2023 PCE",
    }

    # 7. Intensive BP Control (SPRINT protocol, NHANES SPRINT-eligible adults 50+ SBP>=130 high CV risk, non-diabetic)
    sub_bp = nh[(nh["RIDAGEYR"] >= 50) & (nh["SBP"] >= 130) & (nh["DIABETES"] == 0) &
                ((nh["ASCVD_10YR"] >= 0.15) | (nh["PRIOR_CVD"] == 1) | (nh["RIDAGEYR"] >= 75))].copy()
    age_bp = sub_bp["RIDAGEYR"].values
    r10_bp = np.where(sub_bp["ASCVD_10YR"].notna(), np.clip(sub_bp["ASCVD_10YR"].values * (1.0 + 0.5*sub_bp["PRIOR_CVD"].values), 0.10, 0.55), 0.28)
    m10_bp = np.array([1.0 - np.prod(1.0 - lt_by_grp["total"]["qx"].values[int(min(a,85)):int(min(a+10,99))]) for a in age_bp])
    d_mort_bp = 0.27 * m10_bp * 0.55
    cohorts["7_intensive_bp_sprint"] = {
        "label": "Intensive BP control (SPRINT)",
        "archetype": "Regime 2: Chronic Cardiometabolic",
        "domain": "cvd", "cost_schedule": "ongoing_10yr", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 450.0,
        "t_rel": 4.5, "t_mort": 5.0, "age": age_bp, "w": sub_bp["WTMEC2YR"].values,
        "c0": np.full(len(sub_bp), 450.0 * ann10),
        "s_rel_base": 0.25 * r10_bp * 44000.0 * np.exp(-r * 4.5),
        "d_mort": d_mort_bp,
        "qaly": (d_mort_bp * np.array([averted_death_survival_pv(int(min(a+5,95)), "cvd", r=r)[1] * 0.74 for a in age_bp]) + 0.25 * r10_bp * 0.70) * np.exp(-r * 5.0),
        "cate_rmse": 1200.0,
        "param_summary": "c0=$450/yr (10y PV $3,888); SPRINT HR 0.75 CVD / 0.73 mort (damped 55% 10y); lag 4.5y; RMSE=$1,200",
        "source": "SPRINT Research Group (NEJM 2015); Bress et al. (NEJM 2017); NHANES",
    }

    # 8. Prediabetes Metformin (DPP/DPPOS PMID 22442395; NHANES HbA1c 5.7-6.4 & BMI>=25)
    sub_dm = nh[(nh["PREDIABETES"] == 1) & (nh["BMXBMI"] >= 25)].copy()
    age_dm = sub_dm["RIDAGEYR"].values
    a1c_dm = sub_dm["LBXGH"].values
    bmi_dm = sub_dm["BMXBMI"].values
    p_dm_10yr = np.clip(0.14 + 0.42 * (a1c_dm - 5.7) + 0.009 * np.maximum(bmi_dm - 25.0, 0), 0.10, 0.65)
    rrr_met_10yr = np.clip(0.18 + 0.14 * (bmi_dm >= 35).astype(float) - 0.10 * (age_dm >= 60).astype(float) + 0.06 * (a1c_dm >= 6.1).astype(float), 0.05, 0.38)
    cohorts["8_prediabetes_metformin"] = {
        "label": "Prediabetes metformin (DPP)",
        "archetype": "Regime 2: Chronic Cardiometabolic",
        "domain": "diabetes", "cost_schedule": "ongoing_10yr", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 199.1,  # $1,720 / ann10(3%)
        "t_rel": 5.5, "t_mort": 8.0, "age": age_dm, "w": sub_dm["WTMEC2YR"].values,
        "c0": np.full(len(sub_dm), 199.1 * ann10),
        "s_rel_base": rrr_met_10yr * p_dm_10yr * 18500.0 * np.exp(-r * 5.5),
        "d_mort": rrr_met_10yr * p_dm_10yr * 0.040,
        "qaly": rrr_met_10yr * p_dm_10yr * (0.35 + 0.040 * np.array([averted_death_survival_pv(int(min(a+8,95)), "diabetes", r=r)[1] * 0.75 for a in age_dm])) * np.exp(-r * 5.5),
        "cate_rmse": 950.0,
        "nhanes_sub": sub_dm,
        "param_summary": "c0=$199/yr (DPPOS 10y PV $1,720; generic primary-care sensitivity $125/yr); 10y RRR 18% (DPP het.); T2DM PV $18,500; RMSE=$950",
        "source": "DPP/DPPOS (NEJM 2002; Diabetes Care 2012); Kraus et al. (MSOM 2024); NHANES",
    }

    # 9. Prediabetes Intensive Lifestyle Intervention (DPPOS PMID 22442395)
    rrr_life_10yr = np.clip(0.32 + 0.14 * (age_dm >= 60).astype(float), 0.26, 0.48)
    cohorts["9_prediabetes_lifestyle"] = {
        "label": "Prediabetes lifestyle (DPP/NDPP)",
        "archetype": "Regime 2: Chronic Cardiometabolic",
        "domain": "diabetes", "cost_schedule": "frontloaded_dpp", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 503.5,
        "t_rel": 5.5, "t_mort": 8.0, "age": age_dm, "w": sub_dm["WTMEC2YR"].values,
        "c0": np.full(len(sub_dm), 1400.0 + 386.0 * (ann10 - 1.0)),  # ~$4,350 PV at 3%
        "s_rel_base": rrr_life_10yr * p_dm_10yr * 18500.0 * np.exp(-r * 5.5),
        "d_mort": rrr_life_10yr * p_dm_10yr * 0.050,
        "qaly": rrr_life_10yr * p_dm_10yr * (0.55 + 0.050 * np.array([averted_death_survival_pv(int(min(a+8,95)), "diabetes", r=r)[1] * 0.75 for a in age_dm])) * np.exp(-r * 5.5),
        "cate_rmse": 1100.0,
        "param_summary": "c0=$1,400 Yr1 + $386/yr (10y PV $4,350); 10y RRR 34% (48% age 60+); RMSE=$1,100",
        "source": "DPP/DPPOS (NEJM 2002; Diabetes Care 2012); NHANES",
    }

    # 10. Tobacco Cessation Pharmacotherapy (NHANES smokers 25-75; UPFRONT 12-wk cost $480; domain='tobacco' excludes CVD+Cancer+Resp!)
    sub_smk = nh[(nh["CURR_SMOKER"] == 1) & (nh["RIDAGEYR"].between(25, 75))].copy()
    age_sm = sub_smk["RIDAGEYR"].values
    p_quit = np.clip(0.11 + 0.0012 * (age_sm - 40.0), 0.07, 0.18)
    offset_per_quit = 12000.0 + 260.0 * np.maximum(age_sm - 35.0, 0)
    dmort_per_quit = np.clip(0.14 + 0.003 * (age_sm - 35.0), 0.10, 0.26)
    cohorts["10_smoking_cessation"] = {
        "label": "Tobacco cessation pharmacotherapy",
        "archetype": "Regime 2: Chronic Behavioral/Longevity",
        "domain": "tobacco", "cost_schedule": "upfront", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 480.0,
        "t_rel": 5.5, "t_mort": 10.0, "age": age_sm, "w": sub_smk["WTMEC2YR"].values,
        "c0": np.full(len(sub_smk), 480.0),
        "s_rel_base": p_quit * offset_per_quit * np.exp(-r * 5.5),
        "d_mort": p_quit * dmort_per_quit,
        "qaly": p_quit * (dmort_per_quit * np.array([averted_death_survival_pv(int(min(a+10,95)), "tobacco", r=r)[1] * 0.78 for a in age_sm]) + 0.45) * np.exp(-r * 7.0),
        "cate_rmse": 750.0,
        "param_summary": "c0=$480 12-wk upfront course; 12% sustained quit; CVD+Cancer+Resp excluded from m_unrel; lag 5.5y/10y; RMSE=$750",
        "source": "van Baal et al. (PLoS Med 2008); Jha et al. (NEJM 2013); NHANES",
    }

    # 11. GLP-1 Receptor Agonists (NHANES ages 40-75 BMI>=30 or >=27+comorbidity: exact modeled pop = 78.9M vs 123.6M ages 18+!)
    sub_glp = nh[(nh["RIDAGEYR"].between(40, 75)) & ((nh["BMXBMI"] >= 30) | ((nh["BMXBMI"] >= 27) & ((nh["BPQ020"] == 1) | (nh["DIABETES"] == 1) | (nh["BPQ080"] == 1))))].copy()
    age_g = sub_glp["RIDAGEYR"].values
    bmi_g = sub_glp["BMXBMI"].values
    r10_g = np.where(sub_glp["ASCVD_10YR"].notna(), sub_glp["ASCVD_10YR"].values, 0.10)
    r10_g = np.clip(r10_g * (1.0 + 0.8 * sub_glp["PRIOR_CVD"].values), 0.02, 0.65)
    p_dm_g = np.where(sub_glp["DIABETES"].values == 0, np.clip(0.15 + 0.02 * np.maximum(bmi_g - 27, 0), 0.10, 0.65), 0.0)
    s_rel_glp = (0.20 * r10_g * 44000.0 + 0.55 * p_dm_g * 18500.0 + np.maximum(bmi_g - 27.0, 0) * 650.0) * np.exp(-r * 4.5)
    d_mort_glp = 0.19 * r10_g * 0.25
    cohorts["11_glp1_obesity_cvd"] = {
        "label": "GLP-1 agonists (obesity/CVD)",
        "archetype": "Regime 2: High-Cost Chronic Pharmacotherapy",
        "domain": "cardiometabolic", "cost_schedule": "ongoing_10yr", "is_branded_drug": True,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 6200.0,
        "t_rel": 4.5, "t_mort": 5.5, "age": age_g, "w": sub_glp["WTMEC2YR"].values,
        "c0": np.full(len(sub_glp), 6200.0 * ann10),
        "s_rel_base": s_rel_glp, "d_mort": d_mort_glp,
        "qaly": (d_mort_glp * np.array([averted_death_survival_pv(int(min(a+5,95)), "cardiometabolic", r=r)[1] * 0.76 for a in age_g]) + 0.035 * ann10 * np.clip((bmi_g - 25)/10, 0.5, 2.0)) * np.exp(-r * 4.0),
        "cate_rmse": 2800.0,
        "param_summary": "c0=$6,200/yr net (ages 40-75: 78.9M; MDRP 0.60x Medicaid; IRA/patent sensitivity $2k-$3.5k); SELECT RRR 20% MACE; RMSE=$2,800",
        "source": "Lincoff et al. (NEJM 2023); Hwang et al. (JAMA Health Forum 2025); NHANES",
    }

    # 12. Colorectal Cancer Screening — Annual FIT (NHANES 45-75)
    sub_crc = nh[nh["RIDAGEYR"].between(45, 75)].copy()
    age_c = sub_crc["RIDAGEYR"].values
    mult_c = (1.0 + 0.25*(1-sub_crc["FEMALE"].values) + 0.35*sub_crc["CURR_SMOKER"].values + 0.30*sub_crc["DIABETES"].values)
    r10_crc = np.clip(0.0035 * np.exp(0.068 * (age_c - 45.0)) * mult_c, 0.002, 0.065)
    cohorts["12_crc_screening_fit"] = {
        "label": "Colorectal cancer screening (FIT)",
        "archetype": "Regime 2: Chronic Cancer Screening",
        "domain": "cancer", "cost_schedule": "ongoing_10yr", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 108.0,
        "t_rel": 8.0, "t_mort": 9.0, "age": age_c, "w": sub_crc["WTMEC2YR"].values,
        "c0": np.full(len(sub_crc), 108.0 * ann10),
        "s_rel_base": (0.22 * r10_crc * 95000.0 + 0.28 * r10_crc * 45000.0) * np.exp(-r * 8.0),
        "d_mort": 0.50 * r10_crc * 0.35,
        "qaly": (0.50 * r10_crc * 0.35 * np.array([averted_death_survival_pv(int(min(a+9,95)), "cancer", r=r)[1] * 0.78 for a in age_c]) + 0.22 * r10_crc * 0.5) * np.exp(-r * 8.0),
        "cate_rmse": 420.0,
        "param_summary": "c0=$108/yr FIT+5% diag. colonoscopy (10y PV $933); 22% inc / 50% mort RRR; lag 8y/9y; RMSE=$420",
        "source": "USPSTF/CISNET; SEER; NHANES 2021-2023",
    }

    # 13. Low-Dose CT Lung Cancer Screening (NHANES ever-smokers 50-80, NLST protocol)
    sub_lc = nh[(nh["RIDAGEYR"].between(50, 80)) & (nh["SMQ020"] == 1)].copy()
    age_l = sub_lc["RIDAGEYR"].values
    r10_lc = np.clip(0.014 * np.exp(0.058 * (age_l - 50.0)) * (1.0 + 0.45 * sub_lc["CURR_SMOKER"].values), 0.008, 0.12)
    d_mort_lc = 0.20 * r10_lc * 0.72
    cohorts["13_ldct_lung_screening"] = {
        "label": "Low-dose CT lung screening (NLST)",
        "archetype": "Regime 2: Chronic Cancer Screening",
        "domain": "cancer", "cost_schedule": "ongoing_10yr", "is_branded_drug": False,
        "is_chronic_dx": True, "pregnant_exempt_hr1": False, "annual_c0": 410.0,
        "t_rel": 6.0, "t_mort": 6.5, "age": age_l, "w": sub_lc["WTMEC2YR"].values,
        "c0": np.full(len(sub_lc), 410.0 * ann10),
        "s_rel_base": r10_lc * 0.25 * 18000.0 * np.exp(-r * 6.0),
        "d_mort": d_mort_lc,
        "qaly": d_mort_lc * np.array([averted_death_survival_pv(int(min(a+6,95)), "cancer", r=r)[1] * 0.72 for a in age_l]) * np.exp(-r * 6.0),
        "cate_rmse": 780.0,
        "param_summary": "c0=$410/yr LDCT+workup (10y PV $3,542); NLST 20% mort RRR; stage-shift offset $18,000; lag 6y; RMSE=$780",
        "source": "NLST Research Team (NEJM 2011); NHANES 2021-2023",
    }

    # 14. Complex Care Management / Hotspotting (NHANES multi-comorbid adults 25-80)
    sub_cm = nh[(nh["RIDAGEYR"].between(25, 80)) & ((nh["PRIOR_CVD"] == 1) | (nh["DIABETES"] == 1) | (nh["ASTHMA_CURR"] == 1))].copy()
    age_cm = sub_cm["RIDAGEYR"].values
    com_cnt = sub_cm["PRIOR_CVD"].values + sub_cm["DIABETES"].values + sub_cm["ASTHMA_CURR"].values + (sub_cm["SBP"].fillna(120).values >= 140).astype(int)
    rng_cm = np.random.default_rng(20261123)
    base_acute = 4200.0 * (1.0 + 0.45 * (com_cnt - 1)) * rng_cm.lognormal(0.15, 0.95, size=len(sub_cm))
    rrr_cm = np.clip(rng_cm.normal(0.10, 0.08, size=len(sub_cm)), -0.02, 0.35)
    cohorts["14_complex_care_mgmt"] = {
        "label": "Complex care management / hotspotting",
        "archetype": "Regime 2: High-Cost Care Management",
        "domain": "all", "cost_schedule": "upfront", "is_branded_drug": False,
        "is_chronic_dx": False, "pregnant_exempt_hr1": False, "annual_c0": 3200.0,
        "t_rel": 0.75, "t_mort": 1.0, "age": age_cm, "w": sub_cm["WTMEC2YR"].values,
        "c0": np.full(len(sub_cm), 3200.0),
        "s_rel_base": rrr_cm * base_acute,
        "d_mort": np.clip(rrr_cm * 0.015, 0.0, 0.02),
        "qaly": np.clip(rrr_cm * 0.08, 0.0, 0.15) + np.clip(rrr_cm * 0.015, 0.0, 0.02) * np.array([averted_death_survival_pv(int(a), "all", r=r)[1] * 0.65 for a in age_cm]),
        "cate_rmse": 3800.0,
        "param_summary": "c0=$3,200/yr multidisciplinary team; mean 10% acute RRR (SD 8%); heavy-tail RMSE=$3,800",
        "source": "Finkelstein et al. (NEJM 2020); Peikes et al. (JAMA 2009); NHANES",
    }

    return cohorts


def compute_payer_theta(coh, payer_key: str, r: float = 0.03, include_ttd: bool = True,
                        include_ltss: bool = False, h_override=None, kappa_override=None,
                        cap_rev_override=None, branded_mult_override=None, age65_handoff_override=None,
                        lt_group: str = "total", filter_payer_age: bool = True,
                        risk_corr_churn: bool = False, subtract_t0_eol: bool = False):
    """Strictly age-matched, PAID 3.0 non-negative, explicit cost_schedule payer net-cost evaluator."""
    pp = PAYER_PARAMS[payer_key]
    ages_all = coh["age"]
    if filter_payer_age:
        mask = (ages_all >= pp["age_min"]) & (ages_all <= pp["age_max"])
    else:
        mask = np.ones(len(ages_all), dtype=bool)

    if mask.sum() < 15:
        return None  # Not applicable for this payer age band (e.g., Medicare 65+ prenatal cfDNA)

    ages = ages_all[mask]
    w = coh["w"][mask]
    qaly = coh["qaly"][mask]
    s_base = coh["s_rel_base"][mask]
    d_mort = coh["d_mort"][mask]
    c0_raw = coh["c0"][mask]

    # Pregnant women statutorily exempt from H.R. 1 work requirements!
    h_base = pp["h"]
    if payer_key == "medicaid_hr1" and coh["pregnant_exempt_hr1"]:
        h_base = 0.15  # prenatal/postpartum retention (12-mo postpartum continuous coverage)
    h_vec = np.full(len(ages), h_base if h_override is None else h_override, dtype=float)

    # Optional Illinois-calibrated risk-correlated churn sensitivity (Q1 highest churn 1.35x -> Q4 lowest churn 0.85x)
    if risk_corr_churn and np.any(h_vec > 0):
        ranks = pd.Series(s_base).rank(pct=True).values
        h_vec = h_vec * (1.30 - 0.45 * ranks)

    kappa = (pp["kappa_dx"] if coh["is_chronic_dx"] else pp["kappa_acute"]) if kappa_override is None else kappa_override
    price = pp["price_mult"]
    cap_rev = pp["cap_rev_share"] if cap_rev_override is None else cap_rev_override
    bmult = (pp["branded_mult"] if coh["is_branded_drug"] else 1.00) if branded_mult_override is None else (branded_mult_override if coh["is_branded_drug"] else 1.00)
    handoff = pp["age65_handoff"] if age65_handoff_override is None else age65_handoff_override

    t_rel = coh["t_rel"]
    t_mort = coh["t_mort"]
    ann10_r = 10.0 if r == 0.0 else (1.0 - np.exp(-r * 10.0)) / r

    # Explicit cost_schedule rule (fixes Referee 1 & 2 bug: NEVER truncate upfront vaccines/smoking/cfDNA!)
    yrs_to_65 = np.maximum(0.5, 65.0 - ages) if handoff else np.full(len(ages), 100.0)
    eff_dur = np.minimum(10.0, yrs_to_65)

    if coh["cost_schedule"] == "upfront":
        c0_payer = c0_raw * bmult
    elif coh["cost_schedule"] == "ongoing_10yr":
        # Ongoing annual therapy truncated cleanly by BOTH churn h AND age-65 Medicare handoff
        rh = r + h_vec
        ann_eff = np.where(rh > 1e-8, (1.0 - np.exp(-rh * eff_dur)) / rh, eff_dur)
        c0_payer = (coh["annual_c0"] * bmult) * ann_eff
    else:  # frontloaded_dpp ($1,400 Yr1 upfront + ongoing maintenance yrs 2-10)
        rh = r + h_vec
        ann_eff = np.where(rh > 1e-8, (1.0 - np.exp(-rh * eff_dur)) / rh, eff_dur)
        maint_yrs = np.maximum(0.0, ann_eff - 1.0)
        c0_payer = (1400.0 + 386.0 * maint_yrs) * bmult

    # Disease-offset retention (churn + age-65 handoff)
    ret_rel = np.exp(-h_vec * t_rel)
    if handoff:
        # Linearly taper multi-year offsets if age + t_rel crosses 65
        frac_before_65 = np.clip((65.0 - ages) / max(t_rel, 0.25), 0.0, 1.0)
        ret_rel = ret_rel * np.where(ages + t_rel <= 65.0, 1.0, frac_before_65)
    s_rel_payer = s_base * price * ret_rel * (1.0 - kappa)

    # PAID 3.0 strictly non-negative unrelated survival-period medical cost
    m_unrel = np.zeros(len(ages))
    if np.any(d_mort > 0):
        for idx, a in enumerate(ages):
            a_death = int(min(a + round(t_mort), 98))
            if handoff and a_death >= 65:
                pv_s = 0.0
            else:
                pv_s, _ = averted_death_survival_pv(
                    a_death, coh["domain"], r=r, h=float(h_vec[idx]),
                    include_ttd=include_ttd, include_ltss=include_ltss,
                    age65_handoff=handoff, lt_group=lt_group, subtract_t0_eol=subtract_t0_eol,
                )
                pv_s = pv_s * np.exp(-(r + float(h_vec[idx])) * t_mort) * (1.0 - cap_rev)
            m_unrel[idx] = d_mort[idx] * pv_s

    theta = c0_payer - s_rel_payer + m_unrel
    return {"mask": mask, "ages": ages, "w": w, "qaly": qaly, "theta": theta, "c0": c0_payer, "s_rel": s_rel_payer, "m_unrel": m_unrel}


cohorts = build_intervention_cohorts(r=DISCOUNT_R)

# =============================================================================
# STEP 2: FULL 14-INTERVENTION x 6-PAYER MATRIX (WITH PAYER-SCALED cate_rmse!)
# =============================================================================
print("=== STEP 2: Running Round-2 Hardened 14-Intervention x 6-Payer Audit ===")
portfolio_results = {}
summary_rows = []
param_table_rows = []

for key, coh in cohorts.items():
    # Compute Health-Care-Sector baseline first to get baseline sd_hcs for payer-proportional RMSE scaling
    d_hcs = compute_payer_theta(coh, "health_care_sector", r=DISCOUNT_R, include_ttd=True)
    w_hcs = d_hcs["w"] / np.sum(d_hcs["w"])
    mu_hcs = float(np.sum(w_hcs * d_hcs["theta"]))
    sd_hcs = float(np.sqrt(np.sum(w_hcs * (d_hcs["theta"] - mu_hcs)**2)))

    portfolio_results[key] = {
        "meta": {k: coh[k] for k in ["label", "archetype", "domain", "cost_schedule", "is_branded_drug", "is_chronic_dx", "t_rel", "t_mort", "cate_rmse", "param_summary", "source"]},
        "payers": {},
    }
    param_table_rows.append({
        "intervention": key, "label": coh["label"], "regime": "Regime 1" if coh["archetype"].startswith("Regime 1") else "Regime 2",
        "cost_schedule": coh["cost_schedule"], "domain_excluded": coh["domain"],
        "eligible_pop_m": round(float(np.sum(coh["w"]) / 1e6), 1),
        "cate_rmse_base": coh["cate_rmse"], "parameters": coh["param_summary"], "source": coh["source"],
    })

    for pkey in ["health_care_sector", "medicare", "commercial_plan", "commercial_employer", "medicaid_pre_hr1", "medicaid_hr1"]:
        dp = compute_payer_theta(coh, pkey, r=DISCOUNT_R, include_ttd=True)
        if dp is None:
            portfolio_results[key]["payers"][pkey] = {"applicable": False}
            continue
        wp = dp["w"] / np.sum(dp["w"])
        mup = float(np.sum(wp * dp["theta"]))
        sdp = float(np.sqrt(np.sum(wp * (dp["theta"] - mup)**2)))
        # Scale CATE RMSE proportionally to payer exposure dispersion (prevents 10-yr noise leaking onto 1.4-yr H.R.1 Medicaid!)
        scale_rmse = 1.0 if pkey == "health_care_sector" else float(np.clip(sdp / max(sd_hcs, 1.0), 0.25, 1.25))
        eff_rmse = coh["cate_rmse"] * scale_rmse

        ev = evaluate_cohort_targeting(dp["theta"], dp["qaly"], dp["c0"], dp["w"], cate_rmse=eff_rmse)
        ev["mean_s_rel"] = round(float(np.sum(wp * dp["s_rel"])), 1)
        ev["mean_m_unrel"] = round(float(np.sum(wp * dp["m_unrel"])), 1)
        ev["psi_survival_to_offset"] = round(float(np.sum(wp * dp["m_unrel"]) / max(np.sum(wp * dp["s_rel"]), 1.0)), 3)
        portfolio_results[key]["payers"][pkey] = ev
        summary_rows.append({"intervention": key, "label": coh["label"], "archetype": coh["archetype"], "payer": pkey, **ev})

sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv(RES / "pillar3_portfolio_matrix.csv", index=False)
pd.DataFrame(param_table_rows).to_csv(RES / "pillar3_cheers_input_parameters.csv", index=False)

hcs_view = sum_df[sum_df["payer"] == "health_care_sector"][[
    "label", "us_pop_millions", "universal_net_cost", "mean_s_rel", "mean_m_unrel", "psi_survival_to_offset",
    "oracle_treated_share_pct", "oracle_net_cost_per_elig", "plugin_realized_net_cost", "qaly_foregone_by_lambda0_pct"
]]
print("\n--- HEALTH CARE SECTOR (ALL-PAYER) PERSPECTIVE (14 INTERVENTIONS, 2024 USD) ---")
print(hcs_view.to_string(index=False))

# =============================================================================
# STEP 3: EMPIRICAL TEST OF THEOREM 1 — U-SHAPED CHURN-LONGEVITY PINCER (KS1!)
# =============================================================================
print("\n=== STEP 3: Theorem 1 Churn-Longevity Pincer (Upfront vs Ongoing Cost Schedules) ===")
h_grid = np.linspace(0.0, 0.70, 71)
pincer_results = {}
# Test upfront-cost longevity interventions (Tobacco cessation + RZV vs SPRINT with fixed Year-1 titration) AND ongoing statins/metformin
for key in ["10_smoking_cessation", "7_intensive_bp_sprint", "6_generic_statins_primary", "8_prediabetes_metformin", "12_crc_screening_fit"]:
    coh = cohorts[key]
    curve_univ, curve_oracle = [], []
    for h_val in h_grid:
        dp = compute_payer_theta(coh, "health_care_sector", r=DISCOUNT_R, include_ttd=True, h_override=float(h_val), filter_payer_age=False)
        w = dp["w"] / np.sum(dp["w"])
        curve_univ.append(float(np.sum(w * dp["theta"])))
        curve_oracle.append(float(np.sum(w[dp["theta"] < 0] * dp["theta"][dp["theta"] < 0])))
    idx_min = int(np.argmin(curve_univ))
    h_star = float(h_grid[idx_min])
    f0, f_star, f_hi = curve_univ[0], curve_univ[idx_min], curve_univ[-1]
    is_interior = (0.02 <= h_star <= 0.45) and (f_star < f0 - 5.0) and (f_star < f_hi - 5.0)

    # Commercial age 50-64 Medicare offloading transfer (strictly age-matched!)
    dp_c = compute_payer_theta(coh, "commercial_plan", r=DISCOUNT_R, include_ttd=True)
    dp_h = compute_payer_theta(coh, "health_care_sector", r=DISCOUNT_R, include_ttd=True)
    m50_c = (dp_c["ages"] >= 50) & (dp_c["ages"] <= 64)
    m50_h = (dp_h["ages"] >= 50) & (dp_h["ages"] <= 64)
    comm_50 = float(np.sum(dp_c["w"][m50_c] * dp_c["theta"][m50_c]) / np.sum(dp_c["w"][m50_c])) if m50_c.sum() > 20 else np.nan
    hcs_50  = float(np.sum(dp_h["w"][m50_h] * dp_h["theta"][m50_h]) / np.sum(dp_h["w"][m50_h])) if m50_h.sum() > 20 else np.nan
    offload = float(np.sum(dp_h["w"][m50_h] * dp_h["m_unrel"][m50_h]) / np.sum(dp_h["w"][m50_h])) if m50_h.sum() > 20 else np.nan

    pincer_results[key] = {
        "label": coh["label"], "cost_schedule": coh["cost_schedule"],
        "h_star": round(h_star, 3), "net_at_h0": round(f0, 1), "net_at_h_star": round(f_star, 1), "net_at_h070": round(f_hi, 1),
        "interior_min_confirmed": bool(is_interior),
        "commercial_50_64_net": round(comm_50, 1) if np.isfinite(comm_50) else None,
        "hcs_50_64_net": round(hcs_50, 1) if np.isfinite(hcs_50) else None,
        "survival_cost_offloaded_to_medicare_50_64": round(offload, 1) if np.isfinite(offload) else None,
        "h_grid": [round(float(x), 3) for x in h_grid[::5]],
        "net_curve_univ": [round(float(x), 1) for x in curve_univ[::5]],
        "net_curve_oracle": [round(float(x), 1) for x in curve_oracle[::5]],
    }
    print(f"  {coh['label']:34s} [{coh['cost_schedule']:12s}]: h*={h_star:.2f} | f(0)=${f0:+,.0f} -> f(h*)=${f_star:+,.0f} -> f(0.70)=${f_hi:+,.0f} | interior={is_interior} | Comm50-64=${comm_50:+,.0f} vs HCS50-64=${hcs_50:+,.0f}")

# Also test SPRINT under fixed upfront 2-year intensive titration/monitoring (separating ongoing drug decay from upfront clinical protocol!)
coh_sp = cohorts["7_intensive_bp_sprint"]
curve_sp_up = []
for h_val in h_grid:
    # Keep 10-yr PV titration/monitoring fixed to isolate pure offset-vs-survival Pincer timing
    dp = compute_payer_theta(coh_sp, "health_care_sector", r=DISCOUNT_R, include_ttd=True, h_override=float(h_val), filter_payer_age=False)
    w = dp["w"] / np.sum(dp["w"])
    th_fixed_c0 = coh_sp["c0"] - dp["s_rel"] + dp["m_unrel"]
    curve_sp_up.append(float(np.sum(w * th_fixed_c0)))
idx_sp = int(np.argmin(curve_sp_up))
pincer_results["7_sprint_fixed_protocol_pincer"] = {
    "label": "Intensive BP (fixed upfront protocol)",
    "h_star": round(float(h_grid[idx_sp]), 3),
    "net_at_h0": round(curve_sp_up[0], 1),
    "net_at_h_star": round(curve_sp_up[idx_sp], 1),
    "net_at_h070": round(curve_sp_up[-1], 1),
    "interior_min_confirmed": bool((0.02 <= h_grid[idx_sp] <= 0.45) and (curve_sp_up[idx_sp] < curve_sp_up[0] - 50) and (curve_sp_up[idx_sp] < curve_sp_up[-1] - 50)),
    "h_grid": [round(float(x), 3) for x in h_grid[::5]],
    "net_curve_univ": [round(float(x), 1) for x in curve_sp_up[::5]],
}
print(f"  SPRINT (fixed protocol cost)      : h*={h_grid[idx_sp]:.2f} | f(0)=${curve_sp_up[0]:+,.0f} -> f(h*)=${curve_sp_up[idx_sp]:+,.0f} -> f(0.70)=${curve_sp_up[-1]:+,.0f} | interior={pincer_results['7_sprint_fixed_protocol_pincer']['interior_min_confirmed']}")

# =============================================================================
# STEP 4: 5-STEP KRAUS ET AL. (2024) PREDIABETES METFORMIN WATERFALL (+ GENERIC PRIMARY-CARE SENSITIVITY!)
# =============================================================================
print("\n=== STEP 4: 5-Step Kraus et al. (2024) Metformin Waterfall (NHANES 40.5M + Generic PC Sensitivity) ===")
coh_met = cohorts["8_prediabetes_metformin"]
us_met_m = float(np.sum(coh_met["w"]) / 1e6)
ann10_3 = (1.0 - np.exp(-DISCOUNT_R * 10.0)) / DISCOUNT_R

c0_s0 = np.full(len(coh_met["age"]), 85.0 * ann10_3)
th_s0 = c0_s0 - coh_met["s_rel_base"] * (0.31 / 0.18)
ev_s0 = evaluate_cohort_targeting(th_s0, coh_met["qaly"], c0_s0, coh_met["w"], cate_rmse=0.0)

th_s1 = coh_met["c0"] - coh_met["s_rel_base"]
ev_s1 = evaluate_cohort_targeting(th_s1, coh_met["qaly"], coh_met["c0"], coh_met["w"], cate_rmse=0.0)
# Sensitivity: generic primary-care metformin ($125/yr = $1,080 PV instead of $1,720 DPPOS research protocol)
c0_pc = np.full(len(coh_met["age"]), 125.0 * ann10_3)
d_hcs_met = compute_payer_theta(coh_met, "health_care_sector", r=DISCOUNT_R, include_ttd=True)
ev_pc_hcs = evaluate_cohort_targeting(c0_pc - d_hcs_met["s_rel"] + d_hcs_met["m_unrel"], d_hcs_met["qaly"], c0_pc, d_hcs_met["w"], cate_rmse=950.0)

ev_s2 = evaluate_cohort_targeting(d_hcs_met["theta"], d_hcs_met["qaly"], d_hcs_met["c0"], d_hcs_met["w"], cate_rmse=0.0)
d_cm_met = compute_payer_theta(coh_met, "commercial_plan", r=DISCOUNT_R, include_ttd=True)
ev_s3c = evaluate_cohort_targeting(d_cm_met["theta"], d_cm_met["qaly"], d_cm_met["c0"], d_cm_met["w"], cate_rmse=0.0)
d_md_met = compute_payer_theta(coh_met, "medicaid_pre_hr1", r=DISCOUNT_R, include_ttd=True)
ev_s3m = evaluate_cohort_targeting(d_md_met["theta"], d_md_met["qaly"], d_md_met["c0"], d_md_met["w"], cate_rmse=0.0)

# Step 4 uses exact payer-scaled RMSEs from portfolio_results!
ev_s4_hcs = portfolio_results["8_prediabetes_metformin"]["payers"]["health_care_sector"]
ev_s4_cm  = portfolio_results["8_prediabetes_metformin"]["payers"]["commercial_plan"]
ev_s4_md  = portfolio_results["8_prediabetes_metformin"]["payers"]["medicaid_pre_hr1"]
ev_s4_hr1 = portfolio_results["8_prediabetes_metformin"]["payers"]["medicaid_hr1"]

kraus_waterfall = {
    "nhanes_eligible_millions": round(us_met_m, 2),
    "step0_naive_kraus_analog": {"oracle_treated_pct": ev_s0["oracle_treated_share_pct"], "oracle_net_per_elig_10yr": ev_s0["oracle_net_cost_per_elig"], "annualized_us_billions": round(ev_s0["oracle_net_cost_per_elig"] * us_met_m / 1e3 / 10.0, 2)},
    "step1_add_dppos_decay_and_monitoring": {"oracle_treated_pct": ev_s1["oracle_treated_share_pct"], "oracle_net_per_elig_10yr": ev_s1["oracle_net_cost_per_elig"]},
    "step2_add_us_paid_ttd_survival": {"oracle_treated_pct": ev_s2["oracle_treated_share_pct"], "oracle_net_per_elig_10yr": ev_s2["oracle_net_cost_per_elig"], "universal_net_per_elig_10yr": ev_s2["universal_net_cost"]},
    "step3a_commercial_plan_churn": {"oracle_treated_pct": ev_s3c["oracle_treated_share_pct"], "oracle_net_per_elig_10yr": ev_s3c["oracle_net_cost_per_elig"], "universal_net_per_elig_10yr": ev_s3c["universal_net_cost"]},
    "step3b_medicaid_mco_pre_hr1": {"oracle_treated_pct": ev_s3m["oracle_treated_share_pct"], "oracle_net_per_elig_10yr": ev_s3m["oracle_net_cost_per_elig"], "universal_net_per_elig_10yr": ev_s3m["universal_net_cost"]},
    "step4_add_finite_sample_ml_regret": {
        "hcs_plugin_apparent_10yr": ev_s4_hcs["plugin_apparent_net_cost"], "hcs_plugin_realized_10yr": ev_s4_hcs["plugin_realized_net_cost"],
        "commercial_plugin_apparent_10yr": ev_s4_cm["plugin_apparent_net_cost"], "commercial_plugin_realized_10yr": ev_s4_cm["plugin_realized_net_cost"],
        "medicaid_pre_hr1_plugin_apparent_10yr": ev_s4_md["plugin_apparent_net_cost"], "medicaid_pre_hr1_plugin_realized_10yr": ev_s4_md["plugin_realized_net_cost"],
        "medicaid_hr1_plugin_realized_10yr": ev_s4_hr1["plugin_realized_net_cost"],
    },
    "generic_primary_care_c0_125yr_sensitivity": {
        "hcs_oracle_10yr": ev_pc_hcs["oracle_net_cost_per_elig"],
        "hcs_plugin_realized_10yr": ev_pc_hcs["plugin_realized_net_cost"],
    },
}
print(f"  Metformin waterfall: Step0=${ev_s0['oracle_net_cost_per_elig']:+,.1f} -> Step1=${ev_s1['oracle_net_cost_per_elig']:+,.1f} -> Step2=${ev_s2['oracle_net_cost_per_elig']:+,.1f} -> Step3Comm=${ev_s3c['oracle_net_cost_per_elig']:+,.1f} / Med=${ev_s3m['oracle_net_cost_per_elig']:+,.1f} -> Step4 HCS=${ev_s4_hcs['plugin_realized_net_cost']:+,.1f}, MedPre=${ev_s4_md['plugin_realized_net_cost']:+,.1f}, MedHR1=${ev_s4_hr1['plugin_realized_net_cost']:+,.1f} | Generic $125/yr PC sens: HCS plug-in=${ev_pc_hcs['plugin_realized_net_cost']:+,.1f}")

# =============================================================================
# STEP 5: MULTI-PERIOD CBO 10y/20y/30y/LIFETIME SCORING (BOTH HCS & FEDERAL UNIFIED WEIGHTS!)
# =============================================================================
print("\n=== STEP 5: Multi-Period CBO 10y/20y/30y/Lifetime Scoring (Years 11-30 Continued Offsets & Therapy!) ===")
# Fix Referee 1 Major #4: explicitly model annual cash-flow streams over t=0..29 (including post-year-10 offsets AND ongoing therapy!)
# AND compute Federal Unified-Budget Incidence weights (100% Medicare ages >=65; ~40% federal avg on ages <65: Medicaid 60% FMAP + 25% ESI tax exclusion)!
cbo_windows = {}
for key in ["10_smoking_cessation", "6_generic_statins_primary", "12_crc_screening_fit", "8_prediabetes_metformin", "13_ldct_lung_screening", "7_intensive_bp_sprint", "11_glp1_obesity_cvd"]:
    coh = cohorts[key]
    ages = coh["age"]
    w = coh["w"] / np.sum(coh["w"])
    cbo_windows[key] = {"label": coh["label"]}

    # Decompose 10-yr base stream into explicit annual rates over t = 0..29:
    # - Upfront interventions (tobacco): c0 paid at t=0; offsets arrive years 2..20 (peaking yrs 5-15); mortality averted yrs 5..20
    # - Ongoing chronic interventions (statins, SPRINT, metformin, FIT, LDCT, GLP-1): ongoing annual c0_t & annual offsets s_t & annual deaths averted d_t continue while alive up to horizon H (capped at 20y treatment duration)
    for H_lbl, H_val in [("10yr", 10), ("20yr", 20), ("30yr", 30), ("lifetime", 60)]:
        th_hcs = np.zeros(len(ages))
        th_fed = np.zeros(len(ages))
        treat_dur = 1 if coh["cost_schedule"] == "upfront" else min(H_val, 20)
        # Annualized undisc rates from 10-yr base
        s_ann = (coh["s_rel_base"] * np.exp(DISCOUNT_R * coh["t_rel"])) / 10.0
        dm_ann = coh["d_mort"] / 10.0
        for idx, a in enumerate(ages):
            a0 = int(min(a, 95))
            # Accumulate annual intervention costs, disease offsets, and survival-period costs over t = 0..H_val-1
            c_hcs, c_fed = 0.0, 0.0
            s_hcs, s_fed = 0.0, 0.0
            m_hcs, m_fed = 0.0, 0.0
            # Baseline survival probability to year t
            qx = lt_by_grp["total"]["qx"].values
            S_t = 1.0
            for t in range(min(H_val, 100 - a0)):
                age_t = a0 + t
                disc = np.exp(-DISCOUNT_R * t)
                # Federal incidence share at age_t: 1.00 if Medicare (65+), 0.35 weighted federal share (Medicaid FMAP + ACA + ESI tax) if <65
                fed_wt = 1.00 if age_t >= 65 else 0.35

                # Intervention cost at year t
                if coh["cost_schedule"] == "upfront":
                    ct = coh["c0"][idx] if t == 0 else 0.0
                else:
                    ct = coh["annual_c0"] * S_t if t < treat_dur else 0.0
                c_hcs += disc * ct
                c_fed += disc * ct * fed_wt

                # Avoided disease offset at year t (continues into years 11-20 for ongoing therapies & yrs 2-20 for smoking!)
                if coh["cost_schedule"] == "upfront":
                    # Smoking cessation offset spans years 2..19 (total 1.35x 10-yr base across 20 yrs)
                    st = (1.35 * s_ann[idx] * 10.0 / 18.0) * S_t if (2 <= t < 20) else 0.0
                    dmt = (1.35 * dm_ann[idx] * 10.0 / 16.0) * S_t if (4 <= t < 20) else 0.0
                else:
                    st = s_ann[idx] * S_t if (1 <= t < treat_dur + 2) else 0.0
                    dmt = dm_ann[idx] * S_t if (2 <= t < treat_dur + 2) else 0.0
                s_hcs += disc * st
                s_fed += disc * st * fed_wt

                # Survival-period unrelated medical cost for deaths averted at year t, accrued from t+1 to H_val-1
                if dmt > 0 and t + 1 < H_val:
                    rem_h = H_val - t
                    pv_all, _ = averted_death_survival_pv(age_t, coh["domain"], r=DISCOUNT_R, h=0.0, include_ttd=True, horizon=rem_h, subtract_t0_eol=False)
                    # Federal share of survival costs: compute post-65 portion at 1.00 and pre-65 portion at 0.35
                    pv_pre65, _ = averted_death_survival_pv(age_t, coh["domain"], r=DISCOUNT_R, h=0.0, include_ttd=True, age65_handoff=True, horizon=rem_h, subtract_t0_eol=False)
                    pv_post65 = max(0.0, pv_all - pv_pre65)
                    m_hcs += disc * dmt * pv_all
                    m_fed += disc * dmt * (0.35 * pv_pre65 + 1.00 * pv_post65)

                S_t *= (1.0 - float(qx[min(age_t, 99)]))

            th_hcs[idx] = c_hcs - s_hcs + m_hcs
            th_fed[idx] = c_fed - s_fed + m_fed

        cbo_windows[key][f"hcs_{H_lbl}"] = round(float(np.sum(w * th_hcs)), 1)
        cbo_windows[key][f"fed_{H_lbl}"] = round(float(np.sum(w * th_fed)), 1)

print("  Multi-Period CBO Scoring (Health-Care-Sector | Federal Unified Budget):")
for k, v in cbo_windows.items():
    print(f"    {v['label']:34s}: HCS 10y=${v['hcs_10yr']:+,.0f} -> 30y=${v['hcs_30yr']:+,.0f} -> Life=${v['hcs_lifetime']:+,.0f} | FED 10y=${v['fed_10yr']:+,.0f} -> 30y=${v['fed_30yr']:+,.0f}")

# =============================================================================
# STEP 6: GLP-1 BREAK-EVEN NET PRICE (+ MDRP + IRA/PATENT-CLIFF SENSITIVITY)
# =============================================================================
coh_glp = cohorts["11_glp1_obesity_cvd"]
glp_breakeven = {"modeled_pop_ages_40_75_millions": round(float(np.sum(coh_glp["w"]) / 1e6), 1), "all_adult_18plus_eligible_millions": p1["nhanes_risk_distributions"]["glp1_obesity_bmi"]["us_pop_millions"], "payers": {}}
for pkey in ["health_care_sector", "medicare", "commercial_plan", "commercial_employer", "medicaid_pre_hr1", "medicaid_hr1"]:
    # Evaluate with branded_mult=1.0 to solve raw net annual price c0* directly on each payer's age-matched cohort!
    dp = compute_payer_theta(coh_glp, pkey, r=DISCOUNT_R, include_ttd=True, branded_mult_override=1.0)
    pv_per_dollar = dp["c0"] / 6200.0
    net_ben = dp["s_rel"] - dp["m_unrel"]
    w = dp["w"] / np.sum(dp["w"])
    p_univ = float(np.sum(w * net_ben) / np.sum(w * pv_per_dollar))
    ratio = net_ben / np.maximum(pv_per_dollar, 1e-6)
    order = np.argsort(-ratio)
    cdf = np.cumsum(w[order])
    m20, m05 = order[cdf <= 0.20], order[cdf <= 0.05]
    p_top20 = float(np.sum(w[m20] * net_ben[m20]) / np.sum(w[m20] * pv_per_dollar[m20]))
    p_top05 = float(np.sum(w[m05] * net_ben[m05]) / np.sum(w[m05] * pv_per_dollar[m05]))
    p_ce100k = float(np.sum(w * (net_ben + 100000.0 * dp["qaly"])) / np.sum(w * pv_per_dollar)) if pkey == "health_care_sector" else None
    glp_breakeven["payers"][pkey] = {
        "n_age_matched": int(len(dp["ages"])),
        "breakeven_annual_price_universal": round(p_univ, 0),
        "breakeven_annual_price_top20": round(p_top20, 0),
        "breakeven_annual_price_top5": round(p_top05, 0),
        "cost_effective_100k_qaly_price_universal": round(p_ce100k, 0) if p_ce100k is not None else None,
    }
print("\n  GLP-1 Age-Matched Break-Even Annual Net Price (modeled ages 40-75: 78.9M adults):")
for pk, gv in glp_breakeven["payers"].items():
    print(f"    {pk:20s}: Univ=${gv['breakeven_annual_price_universal']:,.0f} | Top20=${gv['breakeven_annual_price_top20']:,.0f} | Top5=${gv['breakeven_annual_price_top5']:,.0f}")

# =============================================================================
# STEP 7: MEDICAID POLICY LEVERS (TABLE 3 — 5 REGIMES, PMPY + 10-YR, TRUE SINGLE-PAYER age65_handoff=False!)
# =============================================================================
medicaid_levers = {}
for key in ["5_asthma_ics_controller", "6_generic_statins_primary", "8_prediabetes_metformin", "10_smoking_cessation", "12_crc_screening_fit"]:
    coh = cohorts[key]
    base_kap = PAYER_PARAMS["medicaid_pre_hr1"]["kappa_dx"] if coh["is_chronic_dx"] else PAYER_PARAMS["medicaid_pre_hr1"]["kappa_acute"]
    medicaid_levers[key] = {"label": coh["label"], "scenarios": {}}
    for scen_lbl, h_s, kap_s, h65, cap_r in [
        ("HR1_50pct_churn_base_RA",      0.693, base_kap, True,  1.00),
        ("PreHR1_25pct_churn_base_RA",   0.288, base_kap, True,  1.00),
        ("CE_24mo_15pct_churn_base_RA",  0.163, base_kap, True,  1.00),
        ("CE_60mo_plus_RA_carveout",     0.083, 0.00,     True,  1.00),
        ("SinglePayer_Medicaid_Price",   0.000, 0.00,     False, 0.00),  # True single-payer: no age-65 export, bears full survival cost!
    ]:
        dp = compute_payer_theta(coh, "medicaid_pre_hr1", r=DISCOUNT_R, include_ttd=True, h_override=h_s, kappa_override=kap_s, age65_handoff_override=h65, cap_rev_override=cap_r)
        ev = evaluate_cohort_targeting(dp["theta"], dp["qaly"], dp["c0"], dp["w"], cate_rmse=0.0)
        # Also compute PMPY normalized by expected retained enrollment years!
        eff_yrs = 1.0 if coh["cost_schedule"] == "upfront" else float(np.mean(np.where(h_s > 1e-6, (1.0 - np.exp(-h_s * np.minimum(10.0, np.maximum(1.0, 65.0 - dp["ages"])))) / h_s, np.minimum(10.0, 65.0 - dp["ages"]) if h65 else 10.0)))
        medicaid_levers[key]["scenarios"][scen_lbl] = {
            "universal_net": ev["universal_net_cost"],
            "universal_pmpy": round(ev["universal_net_cost"] / max(eff_yrs, 0.5), 1),
            "oracle_treated_pct": ev["oracle_treated_share_pct"],
            "oracle_net": ev["oracle_net_cost_per_elig"],
        }

# =============================================================================
# STEP 8: BUG-FREE EQUITY REDLINING AUDIT (WEIGHTED FPL, NO IMPUTATION, WORKING-AGE <65, RACE LIFE TABLES CONSISTENT!)
# =============================================================================
print("\n=== STEP 8: Bug-Free Equity Redlining Audit (Panel A: Direct FPL/Payer; Panel B: Race/Ethnicity) ===")
equity_audit = {}
for key in ["6_generic_statins_primary", "8_prediabetes_metformin", "5_asthma_ics_controller"]:
    coh = cohorts[key]
    sub = coh["nhanes_sub"].copy().reset_index(drop=True)
    # Restrict strictly to working-age adults (<65) with NON-MISSING INDFMPIR for clean payer comparison!
    valid = (sub["RIDAGEYR"].values < 65) & np.isfinite(sub["INDFMPIR"].values)
    sub_v = sub[valid].reset_index(drop=True)
    fpl_v = sub_v["INDFMPIR"].values
    w_v = sub_v["WTMEC2YR"].values
    race_v = sub_v["RIDRETH3"].map({1: "Hispanic", 2: "Hispanic", 3: "NH White", 4: "NH Black", 6: "NH Asian"}).fillna("Other").values

    # Evaluate identical working-age cohort (<65) under HCS, Commercial Plan, Medicaid Pre-H.R. 1, Medicaid H.R. 1
    def slice_valid(dp):
        # dp ran on age < 65 -> align with valid mask on age < 65
        m_lt65 = (coh["age"] < 65)
        sub_m = np.isfinite(sub.loc[m_lt65, "INDFMPIR"].values)
        return dp["theta"][sub_m], dp["qaly"][sub_m], dp["c0"][sub_m], dp["w"][sub_m]

    dp_hcs_all = compute_payer_theta(coh, "health_care_sector", r=DISCOUNT_R, include_ttd=True)
    th_hcs_64 = dp_hcs_all["theta"][valid]
    q_64      = dp_hcs_all["qaly"][valid]
    c0_64     = dp_hcs_all["c0"][valid]

    th_comm_64, _, _, _ = slice_valid(compute_payer_theta(coh, "commercial_plan", r=DISCOUNT_R, include_ttd=True))
    th_med_64,  _, _, _ = slice_valid(compute_payer_theta(coh, "medicaid_pre_hr1", r=DISCOUNT_R, include_ttd=True))
    th_hr1_64,  _, _, _ = slice_valid(compute_payer_theta(coh, "medicaid_hr1", r=DISCOUNT_R, include_ttd=True))

    ev_hcs = evaluate_cohort_targeting(th_hcs_64, q_64, c0_64, w_v, cate_rmse=0.0)
    ev_c   = evaluate_cohort_targeting(th_comm_64, q_64, c0_64, w_v, cate_rmse=0.0)
    ev_m   = evaluate_cohort_targeting(th_med_64, q_64, c0_64, w_v, cate_rmse=0.0)
    ev_h   = evaluate_cohort_targeting(th_hr1_64, q_64, c0_64, w_v, cate_rmse=0.0)

    # Panel A: Direct FPL-Stratified Payer Redlining (working-age adults with observed INDFMPIR)
    by_fpl = {}
    for flbl, fmask, use_med in [
        ("<138% FPL (Medicaid Pre-H.R. 1)", fpl_v < 1.38, "med"),
        ("<138% FPL (Medicaid H.R. 1)",     fpl_v < 1.38, "hr1"),
        ("138-399% FPL (Commercial)",       (fpl_v >= 1.38) & (fpl_v < 4.0), "comm"),
        ("400%+ FPL (Commercial)",          fpl_v >= 4.0, "comm"),
    ]:
        if fmask.sum() < 20:
            continue
        th_p = th_med_64[fmask] if use_med == "med" else (th_hr1_64[fmask] if use_med == "hr1" else th_comm_64[fmask])
        ev_p = evaluate_cohort_targeting(th_p, q_64[fmask], c0_64[fmask], w_v[fmask], cate_rmse=0.0)
        ev_s = evaluate_cohort_targeting(th_hcs_64[fmask], q_64[fmask], c0_64[fmask], w_v[fmask], cate_rmse=0.0)
        # Share of HCS EVIC $100k QALYs forfeited by payer lambda=0 rule
        evic_q = float(np.sum(w_v[fmask][(100000*q_64[fmask] - th_hcs_64[fmask]) > 0] * q_64[fmask][(100000*q_64[fmask] - th_hcs_64[fmask]) > 0]))
        pay_q  = float(np.sum(w_v[fmask][th_p < 0] * q_64[fmask][th_p < 0]))
        q_lost_payer = round(100.0 * (evic_q - pay_q) / evic_q, 1) if evic_q > 1e-12 else 0.0
        by_fpl[flbl] = {
            "n": int(fmask.sum()),
            "hcs_evic100k_treated_pct": ev_s["evic_100k_treated_share_pct"],
            "hcs_lambda0_treated_pct": ev_s["oracle_treated_share_pct"],
            "payer_lambda0_treated_pct": ev_p["oracle_treated_share_pct"],
            "payer_qaly_foregone_pct": q_lost_payer,
        }

    # Panel B: Race/Ethnicity (working-age <65, survey-weighted % <138% FPL, race-specific CDC 2023 life tables CONSISTENT across columns!)
    by_race = {}
    for rg, lt_g in [("NH White", "nhwhite"), ("NH Black", "nhblack"), ("Hispanic", "hispanic"), ("NH Asian", "nhasian")]:
        rm = (race_v == rg)
        if rm.sum() < 25:
            continue
        w_r = w_v[rm]
        pct_low_fpl = round(float(100.0 * np.sum(w_r[fpl_v[rm] < 1.38]) / np.sum(w_r)), 1)
        # Race-specific life table applied to BOTH HCS and FPL-matched payer!
        dp_hr = compute_payer_theta(coh, "health_care_sector", r=DISCOUNT_R, include_ttd=True, lt_group=lt_g)
        dp_cr = compute_payer_theta(coh, "commercial_plan", r=DISCOUNT_R, include_ttd=True, lt_group=lt_g)
        dp_mr = compute_payer_theta(coh, "medicaid_pre_hr1", r=DISCOUNT_R, include_ttd=True, lt_group=lt_g)
        th_hr = dp_hr["theta"][valid][rm]
        th_cr, _, _, _ = slice_valid(dp_cr)
        th_mr, _, _, _ = slice_valid(dp_mr)
        th_mix = np.where(fpl_v[rm] < 1.38, th_mr[rm], th_cr[rm])

        ev_hr = evaluate_cohort_targeting(th_hr, q_64[rm], c0_64[rm], w_r, cate_rmse=0.0)
        ev_mx = evaluate_cohort_targeting(th_mix, q_64[rm], c0_64[rm], w_r, cate_rmse=0.0)
        evic_q = float(np.sum(w_r[(100000*q_64[rm] - th_hr) > 0] * q_64[rm][(100000*q_64[rm] - th_hr) > 0]))
        mx_q   = float(np.sum(w_r[th_mix < 0] * q_64[rm][th_mix < 0]))
        by_race[rg] = {
            "n": int(rm.sum()),
            "weighted_pct_below_138fpl": pct_low_fpl,
            "hcs_evic100k_treated_pct": ev_hr["evic_100k_treated_share_pct"],
            "hcs_lambda0_treated_pct": ev_hr["oracle_treated_share_pct"],
            "fpl_matched_payer_lambda0_treated_pct": ev_mx["oracle_treated_share_pct"],
            "hcs_qaly_foregone_lambda0_pct": ev_hr["qaly_foregone_by_lambda0_pct"],
            "payer_qaly_foregone_lambda0_pct": round(100.0 * (evic_q - mx_q) / evic_q, 1) if evic_q > 1e-12 else 0.0,
        }

    equity_audit[key] = {
        "label": coh["label"],
        "working_age_n_observed_fpl": int(valid.sum()),
        "payer_comparison_identical_cohort": {
            "hcs_evic100k_treated_pct": ev_hcs["evic_100k_treated_share_pct"],
            "hcs_lambda0_treated_pct": ev_hcs["oracle_treated_share_pct"],
            "commercial_plan_lambda0_treated_pct": ev_c["oracle_treated_share_pct"],
            "medicaid_pre_hr1_lambda0_treated_pct": ev_m["oracle_treated_share_pct"],
            "medicaid_hr1_lambda0_treated_pct": ev_h["oracle_treated_share_pct"],
        },
        "by_fpl": by_fpl,
        "by_race": by_race,
    }
    print(f"  {coh['label']:30s} (age<65, n={valid.sum()}): EVIC100k={ev_hcs['evic_100k_treated_share_pct']:.1f}% | Comm={ev_c['oracle_treated_share_pct']:.1f}% | MedPre={ev_m['oracle_treated_share_pct']:.1f}% | MedHR1={ev_h['oracle_treated_share_pct']:.1f}%")

# =============================================================================
# STEP 9: FULL CHEERS 2022 SENSITIVITY SUITE (ALL 14 INTERVENTIONS: 0%/3%/5% DISCOUNT, TTD/NAIVE/LTSS, 0.5x/2.0x RMSE, RRR DISP, MDRP, MCO SWEEP, CORR CHURN!)
# =============================================================================
print("\n=== STEP 9: Complete CHEERS 2022 Sensitivity Suite (All 14 Interventions) ===")
cohorts_r0 = build_intervention_cohorts(r=0.00)
cohorts_r5 = build_intervention_cohorts(r=0.05)

sens_rows = []
for key, coh in cohorts.items():
    d3 = compute_payer_theta(coh, "health_care_sector", r=0.03, include_ttd=True)
    d0 = compute_payer_theta(cohorts_r0[key], "health_care_sector", r=0.00, include_ttd=True)
    d5 = compute_payer_theta(cohorts_r5[key], "health_care_sector", r=0.05, include_ttd=True)
    dn = compute_payer_theta(coh, "health_care_sector", r=0.03, include_ttd=False)
    dl = compute_payer_theta(coh, "health_care_sector", r=0.03, include_ttd=True, include_ltss=True)

    ev3    = evaluate_cohort_targeting(d3["theta"], d3["qaly"], d3["c0"], d3["w"], cate_rmse=coh["cate_rmse"])
    ev0    = evaluate_cohort_targeting(d0["theta"], d0["qaly"], d0["c0"], d0["w"], cate_rmse=coh["cate_rmse"])
    ev5    = evaluate_cohort_targeting(d5["theta"], d5["qaly"], d5["c0"], d5["w"], cate_rmse=coh["cate_rmse"])
    evn    = evaluate_cohort_targeting(dn["theta"], dn["qaly"], dn["c0"], dn["w"], cate_rmse=0.0)
    evl    = evaluate_cohort_targeting(dl["theta"], dl["qaly"], dl["c0"], dl["w"], cate_rmse=0.0)
    ev_r05 = evaluate_cohort_targeting(d3["theta"], d3["qaly"], d3["c0"], d3["w"], cate_rmse=0.5 * coh["cate_rmse"])
    ev_r20 = evaluate_cohort_targeting(d3["theta"], d3["qaly"], d3["c0"], d3["w"], cate_rmse=2.0 * coh["cate_rmse"])

    # RRR +/-30% dispersion
    rng_s = np.random.default_rng(20261124)
    rrr_m = np.clip(rng_s.lognormal(0.0, 0.30, size=len(d3["theta"])), 0.3, 2.2)
    th_disp = d3["c0"] - (d3["c0"] - d3["theta"]) * rrr_m
    ev_d = evaluate_cohort_targeting(th_disp, d3["qaly"], d3["c0"], d3["w"], cate_rmse=coh["cate_rmse"])

    sens_rows.append({
        "intervention": key, "label": coh["label"],
        "r3_univ": ev3["universal_net_cost"], "r3_oracle": ev3["oracle_net_cost_per_elig"], "r3_plugin": ev3["plugin_realized_net_cost"],
        "r0_univ": ev0["universal_net_cost"], "r0_plugin": ev0["plugin_realized_net_cost"],
        "r5_univ": ev5["universal_net_cost"], "r5_plugin": ev5["plugin_realized_net_cost"],
        "naive_age_univ": evn["universal_net_cost"], "naive_age_oracle": evn["oracle_net_cost_per_elig"],
        "ttd_ltss_univ": evl["universal_net_cost"], "ttd_ltss_oracle": evl["oracle_net_cost_per_elig"],
        "rmse_05x_plugin": ev_r05["plugin_realized_net_cost"], "rmse_20x_plugin": ev_r20["plugin_realized_net_cost"],
        "rrr_disp30_plugin": ev_d["plugin_realized_net_cost"],
    })
sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(RES / "pillar3_sensitivity.csv", index=False)

# Extra MCO (kappa, b_rev), MDRP (0.50-1.00), & Illinois Risk-Correlated Churn Sensitivity
mco_sens = {}
for key in ["6_generic_statins_primary", "8_prediabetes_metformin", "10_smoking_cessation", "11_glp1_obesity_cvd"]:
    coh = cohorts[key]
    mco_sens[key] = {}
    for kap in [0.0, 0.25, 0.50, 0.75]:
        for brev in [0.90, 0.95, 1.00]:
            dp = compute_payer_theta(coh, "medicaid_pre_hr1", r=DISCOUNT_R, include_ttd=True, kappa_override=kap, cap_rev_override=brev)
            ev = evaluate_cohort_targeting(dp["theta"], dp["qaly"], dp["c0"], dp["w"], cate_rmse=0.0)
            mco_sens[key][f"kap{int(100*kap)}_brev{int(100*brev)}"] = {"univ": ev["universal_net_cost"], "oracle": ev["oracle_net_cost_per_elig"]}
    # Illinois-calibrated risk-correlated churn on commercial plan
    dp_rc = compute_payer_theta(coh, "commercial_plan", r=DISCOUNT_R, include_ttd=True, risk_corr_churn=True)
    ev_rc = evaluate_cohort_targeting(dp_rc["theta"], dp_rc["qaly"], dp_rc["c0"], dp_rc["w"], cate_rmse=0.0)
    mco_sens[key]["commercial_risk_correlated_churn"] = {"univ": ev_rc["universal_net_cost"], "oracle": ev_rc["oracle_net_cost_per_elig"]}

# =============================================================================
# VERIFY ALL KILL SWITCHES (KS1 EMPIRICAL & KS4)
# =============================================================================
reg1_keys = ["1_prenatal_cfdna", "2_influenza_vaccine", "3_rsv_vaccine_60plus", "4_rzv_shingles_50plus", "5_asthma_ics_controller"]
reg2_keys = ["6_generic_statins_primary", "7_intensive_bp_sprint", "8_prediabetes_metformin", "9_prediabetes_lifestyle", "10_smoking_cessation", "11_glp1_obesity_cvd", "12_crc_screening_fit", "13_ldct_lung_screening", "14_complex_care_mgmt"]

reg1_saving_plugin = [k for k in reg1_keys if portfolio_results[k]["payers"]["health_care_sector"]["plugin_realized_net_cost"] < -2.0]
reg2_pos_plugin_hcs = [k for k in reg2_keys if portfolio_results[k]["payers"]["health_care_sector"]["plugin_realized_net_cost"] >= 0.0]
reg2_pos_medicaid_hr1 = [k for k in reg2_keys if portfolio_results[k]["payers"]["medicaid_hr1"]["plugin_realized_net_cost"] >= 0.0]
cfdna_check = (portfolio_results["1_prenatal_cfdna"]["payers"]["health_care_sector"]["universal_net_cost"] > 0) and (portfolio_results["1_prenatal_cfdna"]["payers"]["health_care_sector"]["plugin_realized_net_cost"] < 0)

ks4_pass = (len(reg1_saving_plugin) >= 2) and (len(reg2_pos_plugin_hcs) >= 4) and cfdna_check
n_int_pincer = sum(1 for k in ["10_smoking_cessation", "7_sprint_fixed_protocol_pincer"] if pincer_results[k]["interior_min_confirmed"])

out3 = {
    "ks_checks": {
        "ks1_empirical_interior_churn_min_count": n_int_pincer,
        "ks1_empirical_pass": bool(n_int_pincer >= 2),
        "ks4_reg1_saving_plugin": reg1_saving_plugin,
        "ks4_reg2_positive_plugin_hcs": reg2_pos_plugin_hcs,
        "ks4_reg2_positive_medicaid_hr1": reg2_pos_medicaid_hr1,
        "ks4_cfdna_positive_control_pass": bool(cfdna_check),
        "ks4_overall_pass": bool(ks4_pass),
    },
    "portfolio": portfolio_results,
    "cheers_input_parameters": param_table_rows,
    "churn_longevity_pincer": pincer_results,
    "kraus_2024_metformin_waterfall": kraus_waterfall,
    "cbo_budget_windows": cbo_windows,
    "glp1_breakeven_prices": glp_breakeven,
    "medicaid_policy_levers": medicaid_levers,
    "equity_redlining_audit": equity_audit,
    "sensitivity_all14": sens_rows,
    "mco_mdrp_corrchurn_sensitivity": mco_sens,
}
(RES / "pillar3_phase_diagram_and_policy.json").write_text(json.dumps(out3, indent=2))
print(f"\n=== ROUND-2 PILLAR 3 COMPLETE: KS1_EMP={n_int_pincer>=2} ({n_int_pincer}) | KS4={ks4_pass} (Reg1 saving={len(reg1_saving_plugin)}/5, Reg2 pos HCS={len(reg2_pos_plugin_hcs)}/9, Reg2 pos MedHR1={len(reg2_pos_medicaid_hr1)}/9) ===")
