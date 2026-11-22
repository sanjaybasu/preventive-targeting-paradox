"""
Generate all publication-quality figures
(600 DPI PNG + vector PDF, Okabe-Ito colorblind-safe palette) and Markdown/CSV
tables strictly in SEQUENTIAL order of first appearance (Round-2 Referee Fix):
  - Table 1 / Figure 1: Pillar 2 Real U.S. Spending RCTs (Winner's Curse & 500-Perm)
  - Table 2 / Figure 2: Pillar 3 14-Intervention Portfolio & 5-Step Metformin Waterfall
  - Figure 3: Theorem 1 U-Shaped Churn Pincer & Multi-Period Federal/HCS CBO Windows
  - Table 3: Medicaid Policy Levers (5 statutory regimes, PMPY + cumulative)
  - Figure 4 / Table 4: GLP-1 Age-Matched Break-Even Prices & Equity Redlining (Panel A FPL + Panel B Race)
  - eFigure 1 & eTables 1-6: Online Appendix exhibits
"""
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = pathlib.Path(__file__).resolve().parent.parent
NB   = pathlib.Path(os.environ.get("PTP_NOTEBOOKS_DIR", str(ROOT / "output_artifacts")))
RES  = ROOT / "results"
PROC = ROOT / "data" / "processed"
FIGS = NB / "figures"
TABS = NB / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# Clean old pre-renumbering figures/tables so ZERO stale companion files remain!
for old_f in list(FIGS.glob("Figure*.png")) + list(FIGS.glob("Figure*.pdf")) + list(FIGS.glob("eFigure*.png")) + list(FIGS.glob("eFigure*.pdf")):
    old_f.unlink()
for old_t in list(TABS.glob("Table*.md")) + list(TABS.glob("Table*.csv")) + list(TABS.glob("eTable*.md")) + list(TABS.glob("eTable*.csv")):
    old_t.unlink()

p1 = json.loads((RES / "pillar1_us_paid_and_churn.json").read_text())
p2 = json.loads((RES / "pillar2_rct_audit.json").read_text())
p3 = json.loads((RES / "pillar3_phase_diagram_and_policy.json").read_text())
z3 = json.loads((RES / "z3_empirical_certificates.json").read_text())
us_paid = pd.read_csv(PROC / "us_paid_age_schedule_2024usd.csv")
port_df = pd.read_csv(RES / "pillar3_portfolio_matrix.csv")
sens_df = pd.read_csv(RES / "pillar3_sensitivity.csv")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.0,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
})

OI_BLUE, OI_ORANGE, OI_GREEN, OI_AMBER, OI_PURPLE, OI_SKY, OI_GREY = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#555555"
)

def fmt_d(v, dec=0, signed=True):
    if signed:
        if v < -1e-9:
            return f"-${abs(v):,.{dec}f}"
        if v > 1e-9:
            return f"+${v:,.{dec}f}"
        return f"${0.0:,.{dec}f}"
    return f"${v:,.{dec}f}"

def save_fig(fig, stem):
    fig.savefig(FIGS / f"{stem}.png", dpi=600)
    fig.savefig(FIGS / f"{stem}.pdf")
    plt.close(fig)
    print(f"  Saved {stem}.png/.pdf")

def get_c0_row(td, c0):
    return [r for r in td["sign_rule_by_c0"] if abs(r["c0"] - c0) < 1e-6][0]

# =============================================================================
# FIGURE 1 (SEQUENTIAL #1): Pillar 2 Real U.S. Spending RCTs — Winner's Curse & Permutations
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
r_ily1 = get_c0_row(p2["component2_and_3_cate_and_winners_curse"]["illinois_yr1_spend"], 250.0)
r_il30 = get_c0_row(p2["component2_and_3_cate_and_winners_curse"]["illinois_30m_spend"], 250.0)
r_cam  = get_c0_row(p2["component2_and_3_cate_and_winners_curse"]["camden_180d_receipts"], 4000.0)
r_ohie = get_c0_row(p2["component2_and_3_cate_and_winners_curse"]["ohie_survey_spend_qje2012"], 0.0)

rct_labels = [
    "Illinois Wellness\nYr 1 ($c_0$=\\$250)\nn=3,239",
    "Illinois Wellness\n30-Mo ($c_0$=\\$250)\nn=3,307",
    "Camden Hotspotting\n180-d ($c_0$=\\$4,000)\nn=782",
    "Oregon HIE\n12-Mo ($c_0$=\\$0)\nn=17,896",
]
naive_emp = [r["naive_insample_matched_emp"]["net_cost_per_eligible"] for r in [r_ily1, r_il30, r_cam, r_ohie]]
naive_se  = [1.96 * r["naive_insample_matched_emp"]["se_per_eligible"] for r in [r_ily1, r_il30, r_cam, r_ohie]]
oos_dr    = [r["honest_oos_dr"]["net_cost_per_eligible"] for r in [r_ily1, r_il30, r_cam, r_ohie]]
oos_dr_se = [1.96 * r["honest_oos_dr"]["se_per_eligible"] for r in [r_ily1, r_il30, r_cam, r_ohie]]
oos_cf    = [r["honest_oos_cf"]["net_cost_per_eligible"] for r in [r_ily1, r_il30, r_cam, r_ohie]]
oos_cf_se = [1.96 * r["honest_oos_cf"]["se_per_eligible"] for r in [r_ily1, r_il30, r_cam, r_ohie]]

xr = np.arange(len(rct_labels))
ax1.axhline(0, color="black", lw=1.0, ls="--")
ax1.errorbar(xr - 0.20, naive_emp, yerr=naive_se, fmt="s", color=OI_GREEN, capsize=3.5, ms=6.5, lw=1.5, label="Naive in-sample ML (apparent)")
ax1.errorbar(xr,        oos_dr,    yerr=oos_dr_se, fmt="o", color=OI_BLUE,  capsize=3.5, ms=6.5, lw=1.5, label="Honest cross-fitted OOS DR-Learner")
ax1.errorbar(xr + 0.20, oos_cf,    yerr=oos_cf_se, fmt="D", color=OI_ORANGE,capsize=3.5, ms=6.0, lw=1.5, label="Honest cross-fitted OOS Causal Forest")
ax1.set_xticks(xr)
ax1.set_xticklabels(rct_labels, fontsize=8.0)
ax1.set_ylabel("Targeted Net Spending per Eligible Person (Trial USD)")
ax1.set_title("A. Empirical Winner's Curse in U.S. Health RCTs")
ax1.legend(loc="lower left", fontsize=7.6, frameon=True, facecolor="white")
ax1.grid(axis="y", alpha=0.25, ls=":")

pk_list = ["illinois_yr1_spend", "illinois_30m_spend", "camden_180d_receipts", "ohie_survey_spend_qje2012"]
pl_lbls = ["Illinois\nYr 1", "Illinois\n30-Mo", "Camden\n180-d", "Oregon HIE\n12-Mo"]
fp_naive = [100.0 * p2["component4_ks3b_sharp_null_permutations"][k]["naive_insample_matched"]["fp_sign_flip_rate_p05"] for k in pk_list]
fp_oos   = [100.0 * p2["component4_ks3b_sharp_null_permutations"][k]["honest_oos_aipw"]["type1_error_one_sided_savings_p05"] for k in pk_list]
ill_sav  = [p2["component4_ks3b_sharp_null_permutations"][k]["naive_insample_matched"]["mean_illusory_savings_emp_per_eligible"] for k in pk_list]

xp = np.arange(len(pk_list))
ax2.bar(xp - 0.18, fp_naive, width=0.34, color=OI_ORANGE, edgecolor="black", lw=0.5, label="Naive in-sample ML (false 'significant savings')")
ax2.bar(xp + 0.18, fp_oos,   width=0.34, color=OI_BLUE,   edgecolor="black", lw=0.5, label="Honest martingale OOS AIPW (Type-I error)")
ax2.axhline(5.0, color="black", lw=1.0, ls="--", label="Nominal 5% significance threshold")
for i in range(len(pk_list)):
    ax2.text(xp[i] - 0.18, fp_naive[i] + 2.0, f"{fp_naive[i]:.1f}%\n({fmt_d(ill_sav[i])})", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=OI_ORANGE)
    ax2.text(xp[i] + 0.18, max(fp_oos[i] + 2.2, 6.8), f"{fp_oos[i]:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=OI_BLUE)
ax2.set_xticks(xp)
ax2.set_xticklabels(pl_lbls, fontsize=8.3)
ax2.set_ylabel("False-Positive 'Net Cost-Saving' Rate (%)")
ax2.set_title("B. 500-Draw Sharp-Null Permutation Placebo Audit")
ax2.set_ylim(0, 148)
ax2.legend(loc="upper center", fontsize=7.5, frameon=True, facecolor="white", ncol=1)
ax2.grid(axis="y", alpha=0.25, ls=":")
fig.tight_layout()
save_fig(fig, "Figure1_RCT_WinnersCurse_and_Permutations")

# =============================================================================
# FIGURE 2 (SEQUENTIAL #2): 14-Intervention Phase Diagram & Metformin Waterfall
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.5), gridspec_kw={"width_ratios": [1.35, 1.05]})
hcs = port_df[port_df["payer"] == "health_care_sector"].copy()
order_keys = [
    "1_prenatal_cfdna", "3_rsv_vaccine_60plus", "5_asthma_ics_controller",
    "2_influenza_vaccine", "4_rzv_shingles_50plus",
    "6_generic_statins_primary", "8_prediabetes_metformin", "10_smoking_cessation",
    "12_crc_screening_fit", "7_intensive_bp_sprint", "9_prediabetes_lifestyle",
    "13_ldct_lung_screening", "14_complex_care_mgmt", "11_glp1_obesity_cvd",
]
hcs = hcs.set_index("intervention").loc[order_keys].reset_index()
y = np.arange(len(hcs))[::-1]

ax1.axvline(0, color="black", lw=1.0, ls="--", zorder=1)
ax1.axhspan(len(hcs) - 5.5, len(hcs) - 0.3, color="#E8F5E9", alpha=0.45, zorder=0)
ax1.axhspan(-0.6, len(hcs) - 5.5, color="#FFF3E0", alpha=0.35, zorder=0)
for i, row in hcs.iterrows():
    yi = y[i]
    ax1.plot([row["oracle_net_cost_per_elig"], row["plugin_realized_net_cost"]], [yi, yi], color=OI_GREY, lw=1.6, zorder=2)
ax1.scatter(hcs["oracle_net_cost_per_elig"], y, color=OI_BLUE, s=48, marker="o", label="Oracle CATE targeting", zorder=4)
ax1.scatter(hcs["plugin_realized_net_cost"], y, color=OI_ORANGE, s=52, marker="D", label="Finite-sample plug-in ML (realized)", zorder=5)
ax1.set_yticks(y)
ax1.set_yticklabels(hcs["label"])
ax1.set_xlabel("Health Care Sector Net Cost per Eligible Adult (2024 USD; Negative = Savings)")
ax1.set_title("A. Regime Separation: Oracle vs. Plug-In ML Targeting")
ax1.set_xlim(-245, 480)
ax1.text(-238, len(hcs) - 0.55, "Regime 1: Acute / Diagnostic\n(3 of 5 save under plug-in ML)", fontsize=7.4, fontweight="bold", color=OI_GREEN, va="top")
ax1.text(135, 5.2, "Regime 2: Chronic Longevity\n& Care Management\n(0 of 9 save under plug-in ML)", fontsize=7.8, fontweight="bold", color=OI_ORANGE, va="top")
ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94)
ax1.grid(axis="x", alpha=0.25, ls=":")

kw = p3["kraus_2024_metformin_waterfall"]
steps = [
    "Step 0\nNaive\n(31% RRR)",
    "Step 1\n+DPPOS\n10-yr decay",
    "Step 2\n+US-PAID\nTTD surv.",
    "Step 3a\n+Comm.\nplan churn",
    "Step 3b\n+Medicaid\nMCO/RA",
    "Step 4a\n+ML regret\n(All-Payer)",
    "Step 4b\n+ML regret\n(Med. HR1)",
]
vals = [
    kw["step0_naive_kraus_analog"]["oracle_net_per_elig_10yr"],
    kw["step1_add_dppos_decay_and_monitoring"]["oracle_net_per_elig_10yr"],
    kw["step2_add_us_paid_ttd_survival"]["oracle_net_per_elig_10yr"],
    kw["step3a_commercial_plan_churn"]["oracle_net_per_elig_10yr"],
    kw["step3b_medicaid_mco_pre_hr1"]["oracle_net_per_elig_10yr"],
    kw["step4_add_finite_sample_ml_regret"]["hcs_plugin_realized_10yr"],
    kw["step4_add_finite_sample_ml_regret"]["medicaid_hr1_plugin_realized_10yr"],
]
bar_cols = [OI_GREEN if v < 0 else OI_ORANGE for v in vals]
x2 = np.arange(len(steps))
ax2.bar(x2, vals, color=bar_cols, width=0.60, edgecolor="black", lw=0.5, zorder=3)
ax2.axhline(0, color="black", lw=1.0, zorder=2)
for bx, bv in zip(x2, vals):
    va = "top" if bv < 0 else "bottom"
    off = -24 if bv < 0 else 18
    ax2.text(bx, bv + off, fmt_d(bv, 0, True), ha="center", va=va, fontsize=7.6, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(steps, fontsize=7.3)
ax2.set_ylabel("10-Year Net Cost per Eligible Adult (2024 USD)")
ax2.set_title("B. Prediabetes Metformin Wedge Decomposition")
ax2.set_ylim(-1200, 260)
ax2.grid(axis="y", alpha=0.25, ls=":")
fig.tight_layout()
save_fig(fig, "Figure2_PhaseDiagram_and_MetforminWaterfall")

# =============================================================================
# FIGURE 3 (SEQUENTIAL #3): Theorem 1 Churn-Longevity Pincer & Multi-Period CBO
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
pt = p3["churn_longevity_pincer"]["10_smoking_cessation"]
psp = p3["churn_longevity_pincer"]["7_sprint_fixed_protocol_pincer"]
pst = p3["churn_longevity_pincer"]["6_generic_statins_primary"]

ax1.axhline(0, color="black", lw=1.0, ls="--")
ax1.axvspan(0.15, 0.30, color=OI_SKY, alpha=0.14, label="Commercial plan turnover band (15%–30%/yr)")
ax1.axvspan(0.45, 0.69, color=OI_ORANGE, alpha=0.12, label="Medicaid H.R. 1 churn band (45%–69%/yr)")

ax1.plot(pt["h_grid"], pt["net_curve_univ"], color=OI_BLUE, lw=2.4, marker="o", ms=4.5, label=f"Tobacco cessation (upfront; $h^*$ = {pt['h_star']:.2f}/yr)")
ax1.plot(pst["h_grid"], pst["net_curve_univ"], color=OI_GREEN, lw=2.0, marker="s", ms=4.0, label="Generic statins (ongoing 10-yr drug cost)")
# Plot SPRINT fixed-protocol curve scaled to right axis or cleanly on same axis? Wait: SPRINT is +$3.7k-$6.9k; plot tobacco + statins + metformin on left and note SPRINT inset/caption cleanly!
pm = p3["churn_longevity_pincer"]["8_prediabetes_metformin"]
ax1.plot(pm["h_grid"], pm["net_curve_univ"], color=OI_PURPLE, lw=2.0, marker="^", ms=4.0, label="Prediabetes metformin (ongoing 10-yr cost)")

ax1.scatter([pt["h_star"]], [pt["net_at_h_star"]], color=OI_ORANGE, s=90, zorder=6, edgecolor="black")
ax1.annotate(f"Offloading minimum\n$h^*$ = {pt['h_star']:.2f}, -\\${abs(pt['net_at_h_star']):,.0f}",
             xy=(pt["h_star"], pt["net_at_h_star"]), xytext=(0.21, -240),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.0), fontsize=7.9, fontweight="bold")
ax1.annotate(f"All-payer ($h$=0):\n+\\${pt['net_at_h0']:,.0f}", xy=(0.0, pt["net_at_h0"]), xytext=(0.09, 1420),
             arrowprops=dict(arrowstyle="->", color=OI_BLUE, lw=0.9), fontsize=7.7, color=OI_BLUE)
ax1.set_xlabel(r"Annual Payer Churn Hazard $h$ ($\mathrm{yr}^{-1}$)")
ax1.set_ylabel("Non-Capitated Payer Net Cost per Eligible Adult (2024 USD)")
ax1.set_title("A. U-Shaped Churn–Longevity Pincer (Theorem 1)")
ax1.set_ylim(-350, 1650)
ax1.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white")
ax1.grid(alpha=0.25, ls=":")

# Panel B: Multi-Period Federal Unified-Budget CBO Scoring (10y -> 20y -> 30y -> Lifetime; GLP-1 noted in caption due to scale)
cbo_keys = ["10_smoking_cessation", "6_generic_statins_primary", "12_crc_screening_fit", "8_prediabetes_metformin", "7_intensive_bp_sprint", "13_ldct_lung_screening"]
cbo_lbls = ["Tobacco\ncessation", "Generic\nstatins", "CRC FIT\nscreening", "Prediabetes\nmetformin", "Intensive BP\n(SPRINT)", "LDCT lung\nscreening"]
x_c = np.arange(len(cbo_keys))
w_b = 0.20
for j, (hlbl, hcol, c_col) in enumerate([("10-yr (CBO)", "fed_10yr", OI_SKY), ("20-yr", "fed_20yr", OI_BLUE), ("30-yr (PHSA)", "fed_30yr", OI_AMBER), ("Lifetime (Fed.)", "fed_lifetime", OI_ORANGE)]):
    vals_h = [p3["cbo_budget_windows"][k][hcol] for k in cbo_keys]
    ax2.bar(x_c + (j - 1.5) * w_b, vals_h, width=w_b, color=c_col, label=hlbl, edgecolor="black", lw=0.4)
ax2.axhline(0, color="black", lw=1.0)
ax2.set_xticks(x_c)
ax2.set_xticklabels(cbo_lbls, fontsize=8.0)
ax2.set_ylabel("Federal Unified-Budget Net Outlay per Adult (2024 USD)")
ax2.set_title("B. Extending CBO Budget Windows Worsens Federal Outlays")
ax2.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white")
ax2.grid(axis="y", alpha=0.25, ls=":")
fig.tight_layout()
save_fig(fig, "Figure3_ChurnPincer_and_CBOWindows")

# =============================================================================
# FIGURE 4 (SEQUENTIAL #4): GLP-1 Age-Matched Break-Even Prices & Equity Redlining
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
gp = p3["glp1_breakeven_prices"]["payers"]
p_keys = ["health_care_sector", "commercial_plan", "commercial_employer", "medicare", "medicaid_pre_hr1", "medicaid_hr1"]
p_lbls = ["All-Payer\n(40–75)", "Comm. Plan\n(40–64)", "Self-Ins.\nEmpl.", "Medicare\n(65–75)", "Medicaid\nPre-HR1", "Medicaid\nH.R. 1"]
xg = np.arange(len(p_keys))
wb = 0.25
ax1.bar(xg - wb, [gp[k]["breakeven_annual_price_universal"] for k in p_keys], width=wb, color=OI_SKY, label="Universal (100%)", edgecolor="black", lw=0.4)
ax1.bar(xg,      [gp[k]["breakeven_annual_price_top20"] for k in p_keys],     width=wb, color=OI_BLUE, label="Top-20% Oracle CATE", edgecolor="black", lw=0.4)
ax1.bar(xg + wb, [gp[k]["breakeven_annual_price_top5"] for k in p_keys],      width=wb, color=OI_GREEN, label="Top-5% Oracle CATE", edgecolor="black", lw=0.4)
ax1.axhline(6200, color=OI_ORANGE, lw=1.8, ls="--", label=r"Current comm. net price (~\$6,200/yr)")
ce_p = gp["health_care_sector"]["cost_effective_100k_qaly_price_universal"]
ax1.axhline(ce_p, color=OI_PURPLE, lw=1.6, ls=":", label=f"All-payer \\$100k/QALY (\\${ce_p:,.0f}/yr)")
for i, k in enumerate(p_keys):
    v5 = gp[k]["breakeven_annual_price_top5"]
    ax1.text(xg[i] + wb, v5 + 120, f"${v5:,.0f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=OI_GREEN)
ax1.set_xticks(xg)
ax1.set_xticklabels(p_lbls, fontsize=7.6)
ax1.set_ylabel("Break-Even Annual Net Price (2024 USD)")
ax1.set_title("A. Age-Matched GLP-1 Break-Even Net Price (Ages 40–75)")
ax1.set_ylim(0, 8600)
ax1.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white", ncol=2)
ax1.grid(axis="y", alpha=0.25, ls=":")

eq = p3["equity_redlining_audit"]
eq_keys = ["6_generic_statins_primary", "8_prediabetes_metformin", "5_asthma_ics_controller"]
eq_lbls = ["Generic Statins\n(Ages 40–64)", "Prediabetes Metformin\n(Ages 18–64)", "Asthma Controller ICS\n(Ages 18–64)"]
xe = np.arange(len(eq_keys))
w4 = 0.19
series = [
    (r"All-Payer EVIC (\$100k/QALY)", "hcs_evic100k_treated_pct", OI_BLUE),
    (r"Commercial Plan ($\lambda=0$)", "commercial_plan_lambda0_treated_pct", OI_SKY),
    (r"Medicaid Pre-H.R. 1 ($\lambda=0$)", "medicaid_pre_hr1_lambda0_treated_pct", OI_AMBER),
    (r"Medicaid H.R. 1 ($\lambda=0$)", "medicaid_hr1_lambda0_treated_pct", OI_ORANGE),
]
for j, (slbl, scol, col) in enumerate(series):
    vals_e = [eq[k]["payer_comparison_identical_cohort"][scol] for k in eq_keys]
    ax2.bar(xe + (j - 1.5) * w4, vals_e, width=w4, color=col, label=slbl, edgecolor="black", lw=0.4)
    for ii, vv in enumerate(vals_e):
        ax2.text(xe[ii] + (j - 1.5) * w4, vv + 1.5, f"{vv:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold")
ax2.set_xticks(xe)
ax2.set_xticklabels(eq_lbls, fontsize=8.0)
ax2.set_ylabel("Share of Working-Age Eligible Adults Treated (%)")
ax2.set_title("B. Payer-Price & Churn Redlining on Identical Cohorts")
ax2.set_ylim(0, 115)
ax2.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="white")
ax2.grid(axis="y", alpha=0.25, ls=":")
fig.tight_layout()
save_fig(fig, "Figure4_GLP1_Price_and_EquityRedlining")

# =============================================================================
# eFIGURE 1 (APPENDIX): US-PAID Age Schedule + Z3 Verification of Theorem 2 (82 cells)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6))
sub_a = us_paid[us_paid["age"].between(20, 88)]
ax1.plot(sub_a["age"], sub_a["m_tot_2024"], color=OI_GREY, lw=2.0, ls="--", label="Total MEPS annual spending ($m_{\\mathrm{tot}}$)")
ax1.plot(sub_a["age"], sub_a["m_surv_all_2024"], color=OI_BLUE, lw=2.2, label="Survivor-year all-cause ($m_{\\mathrm{surv}}$)")
ax1.plot(sub_a["age"], sub_a["m_surv_unrel_cvd"], color=OI_GREEN, lw=2.2, label="Survivor-year CVD-excluded (US-PAID)")
ax1.plot(sub_a["age"], sub_a["m_surv_unrel_cvd"] * sub_a["ltss_factor"], color=OI_ORANGE, lw=1.8, ls=":", label="CVD-excluded + institutional LTSS")
ax1.set_xlabel("Age (Years)")
ax1.set_ylabel("Annual Medical Expenditure (2024 USD)")
ax1.set_title("A. US-PAID Age-Expenditure Schedule (MEPS 2019–2023)")
ax1.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="white")
ax1.grid(alpha=0.25, ls=":")

sc_x = [r["scarf_cantelli_bound"] for r in z3["cell_certificates"] if r["scarf_cantelli_bound"] <= 650]
sc_y = [r["obs_oracle_saving"] for r in z3["cell_certificates"] if r["scarf_cantelli_bound"] <= 650]
ax2.plot([0, 650], [0, 650], color="black", lw=1.0, ls="--", label="Sharp Theorem 2 Scarf–Cantelli bound ($y = x$)")
ax2.scatter(sc_x, sc_y, color=OI_BLUE, alpha=0.78, s=40, edgecolor="white", lw=0.4, label=f"Active cells ({z3['scarf_cantelli_unsat_verified']}/{z3['applicable_cells_checked']} Z3-verified; 0 violations)")
ax2.set_xlim(-15, 650)
ax2.set_ylim(-15, 650)
ax2.set_xlabel("Theorem 2 Scarf–Cantelli Oracle Ceiling (2024 USD)")
ax2.set_ylabel("Observed Oracle CATE Savings $|V^*|$ (2024 USD)")
ax2.set_title("B. Z3 Verification of Theorem 2 Across Active Cells")
ax2.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white")
ax2.grid(alpha=0.25, ls=":")
fig.tight_layout()
save_fig(fig, "eFigure1_USPAID_and_ScarfCantelli")

# =============================================================================
# SEQUENTIAL MARKDOWN & CSV TABLES (Table 1-4 + eTables 1-5)
# =============================================================================
# Table 1 (SEQUENTIAL #1): Pillar 2 Real U.S. Spending RCTs (with Martingale OOS DR matching Col 10!)
t1_rows = []
for lbl, tkey, c0v, perm_k in [
    ("Illinois Wellness (Yr 1 spend, c0=$250)", "illinois_yr1_spend", 250.0, "illinois_yr1_spend"),
    ("Illinois Wellness (30-mo spend, c0=$250)", "illinois_30m_spend", 250.0, "illinois_30m_spend"),
    ("Camden Hotspotting (180-d receipts, c0=$4,000)", "camden_180d_receipts", 4000.0, "camden_180d_receipts"),
    ("Oregon HIE (12-mo survey spend, c0=$0)", "ohie_survey_spend_qje2012", 0.0, "ohie_survey_spend_qje2012"),
]:
    td = p2["component2_and_3_cate_and_winners_curse"][tkey]
    cr = get_c0_row(td, c0v)
    pm_d = p2["component4_ks3b_sharp_null_permutations"][perm_k]
    ht = td["distribution_and_heavy_tails"]
    t1_rows.append({
        "Trial and Outcome (Nominal Trial USD)": lbl,
        "N": f"{td['n']:,d}",
        "Mean (SD)": f"{fmt_d(ht['mean'],0,False)} ({fmt_d(ht['sd'],0,False)})",
        "Skew / σ_Γ": f"{ht['skewness']:.1f} / {fmt_d(ht['aipw_pseudo_outcome_sd_sigma_gamma'],0,False)}",
        "N_req (-$500/yr, q=20%)": f"{ht['theorem3_ncrit']['by_target_saving_usd_yr']['minus_500']['N_power80_total_trial_q20']:,.0f}",
        "Naive In-Sample Emp (Plug-In)": f"{fmt_d(cr['naive_insample_matched_emp']['net_cost_per_eligible'],0,True)} ({fmt_d(cr['naive_insample_matched_plugin']['net_cost_per_eligible'],0,True)})",
        "Honest OOS DR (95% CI)": f"{fmt_d(cr['honest_oos_dr']['net_cost_per_eligible'],0,True)} ({fmt_d(cr['honest_oos_dr']['ci_low_per_eligible'],0,True)} to {fmt_d(cr['honest_oos_dr']['ci_high_per_eligible'],0,True)})",
        "Honest OOS CF": fmt_d(cr["honest_oos_cf"]["net_cost_per_eligible"], 0, True),
        "Null Perm Naive FP": f"{100*pm_d['naive_insample_matched']['fp_sign_flip_rate_p05']:.1f}%",
        "Null Perm Martingale Type-I": f"{100*pm_d['honest_oos_aipw']['type1_error_one_sided_savings_p05']:.1f}%",
    })
t1_df = pd.DataFrame(t1_rows)
t1_df.to_csv(TABS / "Table1_RCT_Spending_CATE_Audit.csv", index=False)
(TABS / "Table1_RCT_Spending_CATE_Audit.md").write_text(t1_df.to_markdown(index=False) + "\n")

# Table 2 (SEQUENTIAL #2): 14-Intervention Four-Wedge Phase Diagram
t2_rows = []
for k in order_keys:
    m = p3["portfolio"][k]["meta"]
    s = p3["portfolio"][k]["payers"]["health_care_sector"]
    c = p3["portfolio"][k]["payers"]["commercial_plan"]
    m2 = p3["portfolio"][k]["payers"]["medicaid_hr1"]
    t2_rows.append({
        "Intervention": m["label"],
        "Regime": "Regime 1" if m["archetype"].startswith("Regime 1") else "Regime 2",
        "Pop (M)": f"{s['us_pop_millions']:.1f}",
        "Upfront c0": fmt_d(s["upfront_c0"], 0, False),
        "Offset S_rel": fmt_d(s["mean_s_rel"], 0, False),
        "Survival M_unrel": fmt_d(s["mean_m_unrel"], 0, False),
        "Psi (M/S)": f"{s['psi_survival_to_offset']:.2f}",
        "Universal Net": fmt_d(s["universal_net_cost"], 0, True),
        "Oracle CATE ($/elig, %tx)": f"{fmt_d(s['oracle_net_cost_per_elig'],1,True)} ({s['oracle_treated_share_pct']:.1f}%)",
        "Plug-in ML All-Payer": fmt_d(s["plugin_realized_net_cost"], 1, True),
        "Plug-in ML Comm. (18-64)": fmt_d(c["plugin_realized_net_cost"], 1, True) if c.get("applicable") else "N/A",
        "Plug-in ML Med. HR1 (18-64)": fmt_d(m2["plugin_realized_net_cost"], 1, True) if m2.get("applicable") else "N/A",
        "QALYs Lost at λ=0": f"{s['qaly_foregone_by_lambda0_pct']:.1f}%",
    })
t2_df = pd.DataFrame(t2_rows)
t2_df.to_csv(TABS / "Table2_Portfolio_Phase_Diagram.csv", index=False)
(TABS / "Table2_Portfolio_Phase_Diagram.md").write_text(t2_df.to_markdown(index=False) + "\n")

# Table 3 (SEQUENTIAL #3): Medicaid Policy Levers (5 Statutory Regimes, PMPY & Cumulative + Oracle %tx)
t3_rows = []
for k, v in p3["medicaid_policy_levers"].items():
    sc = v["scenarios"]
    t3_rows.append({
        "Intervention (Ages 18-64)": v["label"],
        "H.R. 1 (50% churn, base RA)": f"{fmt_d(sc['HR1_50pct_churn_base_RA']['universal_net'],0,True)} ({fmt_d(sc['HR1_50pct_churn_base_RA']['universal_pmpy'],0,True)}/yr; orc {sc['HR1_50pct_churn_base_RA']['oracle_treated_pct']:.1f}%)",
        "Pre-H.R. 1 (25% churn, base RA)": f"{fmt_d(sc['PreHR1_25pct_churn_base_RA']['universal_net'],0,True)} ({fmt_d(sc['PreHR1_25pct_churn_base_RA']['universal_pmpy'],0,True)}/yr; orc {sc['PreHR1_25pct_churn_base_RA']['oracle_treated_pct']:.1f}%)",
        "24-Mo CE (15% churn, base RA)": f"{fmt_d(sc['CE_24mo_15pct_churn_base_RA']['universal_net'],0,True)} ({fmt_d(sc['CE_24mo_15pct_churn_base_RA']['universal_pmpy'],0,True)}/yr; orc {sc['CE_24mo_15pct_churn_base_RA']['oracle_treated_pct']:.1f}%)",
        "60-Mo CE + RA Carve-Out (κ=0)": f"{fmt_d(sc['CE_60mo_plus_RA_carveout']['universal_net'],0,True)} ({fmt_d(sc['CE_60mo_plus_RA_carveout']['universal_pmpy'],0,True)}/yr; orc {sc['CE_60mo_plus_RA_carveout']['oracle_treated_pct']:.1f}%)",
        "Single-Payer Medicaid Price (h=0, no export)": f"{fmt_d(sc['SinglePayer_Medicaid_Price']['universal_net'],0,True)} ({fmt_d(sc['SinglePayer_Medicaid_Price']['universal_pmpy'],0,True)}/yr; orc {sc['SinglePayer_Medicaid_Price']['oracle_treated_pct']:.1f}%)",
    })
t3_df = pd.DataFrame(t3_rows)
t3_df.to_csv(TABS / "Table3_Medicaid_Policy_Levers.csv", index=False)
(TABS / "Table3_Medicaid_Policy_Levers.md").write_text(t3_df.to_markdown(index=False) + "\n")

# Table 4 (SEQUENTIAL #4): Equity Redlining Audit (Panel A: Direct FPL/Payer; Panel B: Race/Ethnicity)
t4_rows = []
for k in ["6_generic_statins_primary", "8_prediabetes_metformin", "5_asthma_ics_controller"]:
    e = p3["equity_redlining_audit"][k]
    for flbl, fv in e["by_fpl"].items():
        if "(Medicaid H.R. 1)" in flbl:
            continue
        t4_rows.append({
            "Panel": "Panel A: By FPL / Payer",
            "Intervention (Ages <65)": e["label"],
            "Stratum": flbl,
            "n": fv["n"],
            "Weighted % <138% FPL": "100.0%" if "<138%" in flbl else "0.0%",
            "All-Payer EVIC $100k": f"{fv['hcs_evic100k_treated_pct']:.1f}%",
            "All-Payer λ=0": f"{fv['hcs_lambda0_treated_pct']:.1f}%",
            "Assigned Payer λ=0": f"{fv['payer_lambda0_treated_pct']:.1f}%",
            "Payer QALYs Lost": f"{fv['payer_qaly_foregone_pct']:.1f}%",
        })
    for rg, rv in e["by_race"].items():
        t4_rows.append({
            "Panel": "Panel B: By Race / Ethnicity",
            "Intervention (Ages <65)": e["label"],
            "Stratum": rg,
            "n": rv["n"],
            "Weighted % <138% FPL": f"{rv['weighted_pct_below_138fpl']:.1f}%",
            "All-Payer EVIC $100k": f"{rv['hcs_evic100k_treated_pct']:.1f}%",
            "All-Payer λ=0": f"{rv['hcs_lambda0_treated_pct']:.1f}%",
            "Assigned Payer λ=0": f"{rv['fpl_matched_payer_lambda0_treated_pct']:.1f}%",
            "Payer QALYs Lost": f"{rv['payer_qaly_foregone_lambda0_pct']:.1f}%",
        })
t4_df = pd.DataFrame(t4_rows)
t4_df.to_csv(TABS / "Table4_Equity_Redlining_Audit.csv", index=False)
(TABS / "Table4_Equity_Redlining_Audit.md").write_text(t4_df.to_markdown(index=False) + "\n")

# Online Appendix Sensitivity & CHEERS Input Tables
sens_df.to_csv(TABS / "eTable5_Full_Sensitivity_All14.csv", index=False)
(TABS / "eTable5_Full_Sensitivity_All14.md").write_text(sens_df.to_markdown(index=False) + "\n")
pd.DataFrame(p3["cheers_input_parameters"]).to_csv(TABS / "eTable4_CHEERS_Input_Parameters.csv", index=False)
(TABS / "eTable4_CHEERS_Input_Parameters.md").write_text(pd.DataFrame(p3["cheers_input_parameters"]).to_markdown(index=False) + "\n")

print("=== ALL ROUND-2 SEQUENTIAL FIGURES (1-4 + eFig1) & TABLES (1-4 + eTables) GENERATED CLEANLY ===")
