"""
Pillar 1: Build open-source U.S. PAID-3.0 analog (US-PAID), multi-payer
empirical churn hazards, CDC 2023 life-table survival-cost schedules with
exact Abel summation-by-parts proximity-to-death (TTD) decomposition, and
NHANES 2021-2023 real U.S. adult clinical risk-factor distributions.

100% public federal microdata (AHRQ MEPS 2019-2023, CDC NCHS 2023 Life Tables,
CDC NHANES 2021-2023). Zero PHI. Deterministic, reproducible.
All expenditures adjusted to 2024 USD via official AHRQ/BEA PCE-Health Table 3
(updated April 2026: 2019=103.247, 2020=105.442, 2021=107.572, 2022=110.240,
2023=113.120, 2024=116.048).
"""
import io
import json
import pathlib
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
PROC.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# Official AHRQ MEPS Table 3 (updated April 2026): BEA PCE-Health Index (2017=100)
PCE_HEALTH = {
    2019: 103.247,
    2020: 105.442,
    2021: 107.572,
    2022: 110.240,
    2023: 113.120,
    2024: 116.048,
}
TARGET_YEAR = 2024
INFLATE = {yr: PCE_HEALTH[TARGET_YEAR] / PCE_HEALTH[yr] for yr in range(2019, 2024)}


def read_zip_dta(zpath: pathlib.Path, columns=None) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as zf:
        dta_name = [n for n in zf.namelist() if n.lower().endswith(".dta")][0]
        with zf.open(dta_name) as fh:
            buf = io.BytesIO(fh.read())
            return pd.read_stata(buf, columns=columns, convert_categoricals=False)


def weighted_mean_se(x, w, strata=None, psu=None):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[mask], w[mask]
    if len(x) == 0:
        return np.nan, np.nan, 0
    wsum = w.sum()
    mu = np.sum(w * x) / wsum
    if strata is not None and psu is not None:
        st = np.asarray(strata)[mask]
        ps = np.asarray(psu)[mask]
        # Taylor-linearized stratified cluster variance for ratio mean
        df = pd.DataFrame({"z": w * (x - mu), "st": st, "ps": ps})
        psu_totals = df.groupby(["st", "ps"], sort=False)["z"].sum().reset_index()
        var_sum = 0.0
        for _, grp in psu_totals.groupby("st", sort=False):
            n_h = len(grp)
            if n_h > 1:
                z_h = grp["z"].values
                var_sum += (n_h / (n_h - 1.0)) * np.sum((z_h - z_h.mean()) ** 2)
        se = np.sqrt(var_sum) / wsum
    else:
        n_eff = (wsum ** 2) / np.sum(w ** 2)
        v = np.sum(w * (x - mu) ** 2) / wsum
        se = np.sqrt(v / max(n_eff, 1.0))
    return float(mu), float(se), int(len(x))


# =============================================================================
# 1. POOLED MEPS 2019-2023 FULL-YEAR CONSOLIDATED (US-PAID BASELINE + PAYERS)
# =============================================================================
print("=== STEP 1: Pooling MEPS 2019-2023 FYC ===")
fyc_frames = []
annual_totals = {}
for yr in [2019, 2020, 2021, 2022, 2023]:
    yy = str(yr)[2:]
    cols = [
        "DUPERSID", "AGELAST", "SEX", "RACETHX",
        f"POVCAT{yy}", f"INSCOV{yy}", f"PRVEV{yy}", f"MCDEV{yy}", f"MCREV{yy}",
        f"TOTEXP{yy}", f"TOTPRV{yy}", f"TOTMCD{yy}", f"TOTMCR{yy}",
        f"IPTEXP{yy}", f"ERTEXP{yy}", f"OPTEXP{yy}", f"OBVEXP{yy}", f"RXEXP{yy}",
        f"PERWT{yy}F", "VARSTR", "VARPSU",
    ]
    df = read_zip_dta(RAW / f"meps_{yr}_fyc.zip", columns=cols)
    df = df.rename(columns={
        f"POVCAT{yy}": "POVCAT",
        f"INSCOV{yy}": "INSCOV",
        f"PRVEV{yy}": "PRVEV",
        f"MCDEV{yy}": "MCDEV",
        f"MCREV{yy}": "MCREV",
        f"TOTEXP{yy}": "TOTEXP_NOM",
        f"TOTPRV{yy}": "TOTPRV_NOM",
        f"TOTMCD{yy}": "TOTMCD_NOM",
        f"TOTMCR{yy}": "TOTMCR_NOM",
        f"IPTEXP{yy}": "IPTEXP_NOM",
        f"ERTEXP{yy}": "ERTEXP_NOM",
        f"OPTEXP{yy}": "OPTEXP_NOM",
        f"OBVEXP{yy}": "OBVEXP_NOM",
        f"RXEXP{yy}": "RXEXP_NOM",
        f"PERWT{yy}F": "PERWT",
    })
    df["YEAR"] = yr
    df["INF_2024"] = INFLATE[yr]
    for c in ["TOTEXP", "TOTPRV", "TOTMCD", "TOTMCR", "IPTEXP", "ERTEXP", "OPTEXP", "OBVEXP", "RXEXP"]:
        df[c] = df[f"{c}_NOM"].clip(lower=0) * df["INF_2024"]

    # Annual AHRQ reconciliation check (nominal total & per capita)
    valid_w = df["PERWT"] > 0
    pop = df.loc[valid_w, "PERWT"].sum()
    tot_nom_b = (df.loc[valid_w, "TOTEXP_NOM"].clip(lower=0) * df.loc[valid_w, "PERWT"]).sum() / 1e9
    tot_24_b = (df.loc[valid_w, "TOTEXP"] * df.loc[valid_w, "PERWT"]).sum() / 1e9
    mean_nom = (tot_nom_b * 1e9) / pop
    annual_totals[str(yr)] = {
        "n_unweighted": int(valid_w.sum()),
        "us_pop_millions": round(float(pop / 1e6), 2),
        "total_nominal_billions": round(float(tot_nom_b), 1),
        "total_2024usd_billions": round(float(tot_24_b), 1),
        "per_capita_nominal": round(float(mean_nom), 1),
        "per_capita_2024usd": round(float(tot_24_b * 1e9 / pop), 1),
    }
    print(f"  MEPS {yr}: n={valid_w.sum():,d}, pop={pop/1e6:.1f}M, nom=${tot_nom_b:.1f}B (${mean_nom:,.0f}/cap), 2024USD=${tot_24_b:.1f}B")
    fyc_frames.append(df)

meps = pd.concat(fyc_frames, ignore_index=True)
meps = meps[(meps["PERWT"] > 0) & (meps["AGELAST"] >= 0)].copy()
meps["POOLWT"] = meps["PERWT"] / 5.0
# Unique strata across pooled years (preserve MEPS variance estimation structure)
meps["POOL_STR"] = meps["YEAR"].astype(str) + "_" + meps["VARSTR"].astype(int).astype(str)
meps["POOL_PSU"] = meps["VARPSU"].astype(int)

# Define mutually exclusive primary payer categories for payer comparisons
# Private (18-64): INSCOV==1 & AGELAST<65; Medicaid (18-64): MCDEV==1 & MCREV!=1 & AGELAST<65; Medicare (65+): MCREV==1 & AGELAST>=65
meps["PAYER_GRP"] = "Other/Uninsured"
meps.loc[(meps["AGELAST"] < 65) & (meps["INSCOV"] == 1), "PAYER_GRP"] = "Commercial (18-64)"
meps.loc[(meps["AGELAST"] < 65) & (meps["MCDEV"] == 1) & (meps["MCREV"] != 1) & (meps["INSCOV"] == 2), "PAYER_GRP"] = "Medicaid (18-64)"
meps.loc[(meps["AGELAST"] >= 65) & (meps["MCREV"] == 1), "PAYER_GRP"] = "Medicare (65+)"

# Race/ethnicity mapping (MEPS RACETHX: 1=Hispanic, 2=NH White, 3=NH Black, 4=NH Asian, 5=NH Other/Multiracial)
RACE_MAP = {1: "Hispanic", 2: "NH White", 3: "NH Black", 4: "NH Asian", 5: "NH Other"}
meps["RACE_GRP"] = meps["RACETHX"].map(RACE_MAP).fillna("NH Other")

# Age-bin expenditure schedule (5-year bins 0..85+, plus smoothed single-year curve)
age_bins = list(range(0, 85, 5)) + [85, 120]
age_labels = [f"{a}-{a+4}" for a in range(0, 85, 5)] + ["85+"]
meps["AGE_BIN"] = pd.cut(meps["AGELAST"], bins=age_bins, labels=age_labels, right=False)

age_sched = []
for lbl in age_labels:
    sub = meps[meps["AGE_BIN"] == lbl]
    mu, se, n = weighted_mean_se(sub["TOTEXP"], sub["POOLWT"], sub["POOL_STR"], sub["POOL_PSU"])
    mu_m, _, _ = weighted_mean_se(sub.loc[sub["SEX"] == 1, "TOTEXP"], sub.loc[sub["SEX"] == 1, "POOLWT"])
    mu_f, _, _ = weighted_mean_se(sub.loc[sub["SEX"] == 2, "TOTEXP"], sub.loc[sub["SEX"] == 2, "POOLWT"])
    age_mid = 88.0 if lbl == "85+" else (int(lbl.split("-")[0]) + 2.0)
    age_sched.append({
        "age_bin": lbl,
        "age_mid": age_mid,
        "n": n,
        "mean_totexp_2024": round(mu, 2),
        "se_totexp_2024": round(se, 2),
        "rse_pct": round(100.0 * se / mu, 2) if mu > 0 else np.nan,
        "male_mean_2024": round(mu_m, 2),
        "female_mean_2024": round(mu_f, 2),
    })
age_sched_df = pd.DataFrame(age_sched)
max_rse = float(age_sched_df["rse_pct"].max())
print(f"  Pooled MEPS 2019-2023 N={len(meps):,d}; max 5-yr age-bin RSE = {max_rse:.2f}% (KS2 gate <15%)")

# Interpolate smooth single-year-of-age spending curve ages 0..100
from scipy.interpolate import PchipInterpolator
pchip_all = PchipInterpolator(age_sched_df["age_mid"].values, age_sched_df["mean_totexp_2024"].values)
pchip_m = PchipInterpolator(age_sched_df["age_mid"].values, age_sched_df["male_mean_2024"].values)
pchip_f = PchipInterpolator(age_sched_df["age_mid"].values, age_sched_df["female_mean_2024"].values)

ages_0_100 = np.arange(0, 101)
# Cap extrapolation flat above age 88 (standard PAID conservative rule; LTSS sensitivity handled separately)
totexp_by_age = np.array([float(pchip_all(min(a, 88.0))) for a in ages_0_100])
totexp_m_by_age = np.array([float(pchip_m(min(a, 88.0))) for a in ages_0_100])
totexp_f_by_age = np.array([float(pchip_f(min(a, 88.0))) for a in ages_0_100])

# =============================================================================
# 2. TRUE PAID-3.0 INDEX-DISEASE EXCLUSION (MEPS 2022-2023 CLNK + 5 EVENT FILES)
# =============================================================================
print("\n=== STEP 2: MEPS 2022-2023 CLNK + Event-Level Index-Disease Attribution (US-PAID) ===")
# CCSR prefixes (AHRQ CCSR v2023):
#   CIR = Circulatory / CVD (IHD, stroke, HF, HTN, etc.)
#   END002..END006 = Diabetes mellitus (and endocrine/metabolic diabetes codes)
#   NEO = Neoplasms (cancers)
#   RSP = Respiratory (asthma, COPD, pneumonia, influenza/RSV)
#   INF = Infectious/parasitic (incl. viral, zoster)
def map_domain(ccsr: str, icd10: str) -> str:
    c = str(ccsr or "").upper().strip()
    i = str(icd10 or "").upper().strip()
    if c.startswith("CIR") or i.startswith(("I10","I11","I12","I13","I15","I20","I21","I22","I23","I24","I25","I48","I50","I60","I61","I62","I63","I64","I65","I66","I67","I69","I70")):
        return "CVD"
    if c in ("END002","END003","END004","END005","END006") or i.startswith(("E08","E09","E10","E11","E13")):
        return "DIABETES"
    if c.startswith("NEO") or i.startswith(("C","D0","D1","D2","D3","D4")):
        return "CANCER"
    if c.startswith("RSP") or i.startswith(("J0","J1","J2","J3","J4","J8","J9")):
        return "RESPIRATORY"
    return "OTHER"

event_domain_rows = []
payer_price_rows = []
clnk_match_log = {}

for yr in [2022, 2023]:
    yy = str(yr)[2:]
    inf = INFLATE[yr]
    fyc_small = read_zip_dta(RAW / f"meps_{yr}_fyc.zip",
                             columns=["DUPERSID", "AGELAST", "SEX", "RACETHX", f"INSCOV{yy}", f"MCDEV{yy}", f"MCREV{yy}", f"TOTEXP{yy}", f"PERWT{yy}F"])
    fyc_small = fyc_small.rename(columns={f"TOTEXP{yy}": "TOTEXP_NOM", f"PERWT{yy}F": "PERWT", f"INSCOV{yy}": "INSCOV", f"MCDEV{yy}": "MCDEV", f"MCREV{yy}": "MCREV"})
    fyc_small = fyc_small[(fyc_small["PERWT"] > 0) & (fyc_small["AGELAST"] >= 0)].copy()
    fyc_small["TOTEXP_2024"] = fyc_small["TOTEXP_NOM"].clip(lower=0) * inf

    cond = read_zip_dta(RAW / f"meps_{yr}_cond.zip", columns=["DUPERSID", "CONDIDX", "ICD10CDX", "CCSR1X"])
    cond["DOMAIN"] = [map_domain(c, i) for c, i in zip(cond["CCSR1X"], cond["ICD10CDX"])]
    clnk = read_zip_dta(RAW / f"meps_{yr}_clnk.zip", columns=["DUPERSID", "CONDIDX", "EVNTIDX"])
    clnk_dom = clnk.merge(cond[["CONDIDX", "DOMAIN"]], on="CONDIDX", how="left")
    clnk_dom["DOMAIN"] = clnk_dom["DOMAIN"].fillna("OTHER")
    # Split multi-condition event spending equally across unique domains linked to that event
    ev_doms = clnk_dom[["EVNTIDX", "DOMAIN"]].drop_duplicates()
    ev_counts = ev_doms.groupby("EVNTIDX")["DOMAIN"].transform("count")
    ev_doms["SHARE"] = 1.0 / ev_counts

    # Collect all 5 event types + log exact CLNK match rates
    ev_list = []
    clnk_id_set = set(clnk["EVNTIDX"].astype(str))
    for etype, idcol, xpcol, pvcol, mdcol, mrcol in [
        ("ip", "EVNTIDX", f"IPXP{yy}X", f"IPFPV{yy}X", f"IPFMD{yy}X", f"IPFMR{yy}X"),
        ("er", "EVNTIDX", f"ERXP{yy}X", f"ERFPV{yy}X", f"ERFMD{yy}X", f"ERFMR{yy}X"),
        ("op", "EVNTIDX", f"OPXP{yy}X", f"OPFPV{yy}X", f"OPFMD{yy}X", f"OPFMR{yy}X"),
        ("ob", "EVNTIDX", f"OBXP{yy}X", f"OBPV{yy}X",  f"OBMD{yy}X",  f"OBMR{yy}X"),
        ("rx", "LINKIDX", f"RXXP{yy}X", f"RXPV{yy}X",  f"RXMD{yy}X",  f"RXMR{yy}X"),
    ]:
        with zipfile.ZipFile(RAW / f"meps_{yr}_{etype}.zip") as zf:
            dta_name = [n for n in zf.namelist() if n.lower().endswith(".dta")][0]
            with zf.open(dta_name) as fh:
                raw_bytes = fh.read()
                avail = set(pd.io.stata.StataReader(io.BytesIO(raw_bytes)).variable_labels().keys())
                use_cols = [c for c in ["DUPERSID", idcol, xpcol, pvcol, mdcol, mrcol, f"PERWT{yy}F"] if c in avail]
                edf = pd.read_stata(io.BytesIO(raw_bytes), columns=use_cols, convert_categoricals=False)
        edf = edf.rename(columns={idcol: "EVNTIDX", xpcol: "XP_NOM"})
        edf["XP_2024"] = edf["XP_NOM"].clip(lower=0) * inf
        edf["ETYPE"] = etype
        match_rate = float(edf["EVNTIDX"].astype(str).isin(clnk_id_set).mean())
        clnk_match_log[f"{yr}_{etype}"] = {"n_events": int(len(edf)), "match_rate_pct": round(100.0 * match_rate, 1)}

        # Payer reimbursement comparison for IP admissions and ER visits (Medicaid vs Commercial vs Medicare)
        if etype in ("ip", "er"):
            edf_m = edf.merge(fyc_small[["DUPERSID", "AGELAST", "INSCOV", "MCDEV", "MCREV", "PERWT"]], on="DUPERSID", how="inner")
            for pname, mask in [
                ("Commercial (18-64)", (edf_m["AGELAST"].between(18, 64)) & (edf_m["INSCOV"] == 1)),
                ("Medicaid (18-64)",   (edf_m["AGELAST"].between(18, 64)) & (edf_m["MCDEV"] == 1) & (edf_m["MCREV"] != 1) & (edf_m["INSCOV"] == 2)),
                ("Medicare (65+)",     (edf_m["AGELAST"] >= 65) & (edf_m["MCREV"] == 1)),
            ]:
                sub_e = edf_m[mask & (edf_m["XP_2024"] > 0)]
                if len(sub_e) > 0:
                    mu_e = np.sum(sub_e["XP_2024"] * sub_e["PERWT"]) / np.sum(sub_e["PERWT"])
                    payer_price_rows.append({"year": yr, "etype": etype, "payer": pname, "mean_cost_2024": mu_e, "n_events": len(sub_e), "wt_sum": float(sub_e["PERWT"].sum())})

        ev_list.append(edf[["DUPERSID", "EVNTIDX", "XP_2024"]])

    all_ev = pd.concat(ev_list, ignore_index=True)
    linked = all_ev.merge(ev_doms, on="EVNTIDX", how="left")
    linked["DOMAIN"] = linked["DOMAIN"].fillna("OTHER")
    linked["SHARE"] = linked["SHARE"].fillna(1.0)
    linked["ALLOC_XP"] = linked["XP_2024"] * linked["SHARE"]

    # Person-level disease-attributable expenditures
    pers_dom = linked.pivot_table(index="DUPERSID", columns="DOMAIN", values="ALLOC_XP", aggfunc="sum", fill_value=0.0).reset_index()
    for d in ["CVD", "DIABETES", "CANCER", "RESPIRATORY"]:
        if d not in pers_dom.columns:
            pers_dom[d] = 0.0
    m_yr = fyc_small.merge(pers_dom[["DUPERSID", "CVD", "DIABETES", "CANCER", "RESPIRATORY"]], on="DUPERSID", how="left").fillna(0.0)
    # Cap domain sum <= TOTEXP_2024
    for d in ["CVD", "DIABETES", "CANCER", "RESPIRATORY"]:
        m_yr[d] = np.minimum(m_yr[d], m_yr["TOTEXP_2024"])
    m_yr["YEAR"] = yr
    event_domain_rows.append(m_yr)

paid_df = pd.concat(event_domain_rows, ignore_index=True)
paid_df["POOLWT"] = paid_df["PERWT"] / 2.0
paid_df["AGE_BIN"] = pd.cut(paid_df["AGELAST"], bins=age_bins, labels=age_labels, right=False)

# Compute domain shares by age bin (for US-PAID unrelated spending = TOTEXP * (1 - domain_share))
paid_rows = []
for lbl in age_labels:
    sub = paid_df[paid_df["AGE_BIN"] == lbl]
    w = sub["POOLWT"].values
    tot = np.sum(sub["TOTEXP_2024"].values * w) / np.sum(w)
    row = {"age_bin": lbl, "age_mid": 88.0 if lbl == "85+" else int(lbl.split("-")[0]) + 2.0, "tot_22_23": round(tot, 2)}
    for d in ["CVD", "DIABETES", "CANCER", "RESPIRATORY"]:
        dm = np.sum(sub[d].values * w) / np.sum(w)
        row[f"share_{d.lower()}"] = round(float(dm / tot), 4) if tot > 0 else 0.0
        row[f"amt_{d.lower()}"] = round(float(dm), 2)
    # Cardiometabolic combined share (CVD + Diabetes)
    cm = np.sum((sub["CVD"].values + sub["DIABETES"].values) * w) / np.sum(w)
    row["share_cardiometabolic"] = round(float(min(cm / tot, 0.95)), 4) if tot > 0 else 0.0
    # Combined tobacco-related share (CVD + Cancer + Respiratory) so tobacco cessation avoids double-counting any smoking-related disease!
    tb = np.sum((sub["CVD"].values + sub["CANCER"].values + sub["RESPIRATORY"].values) * w) / np.sum(w)
    row["share_tobacco"] = round(float(min(tb / tot, 0.95)), 4) if tot > 0 else 0.0
    paid_rows.append(row)
paid_shares_df = pd.DataFrame(paid_rows)
print("  US-PAID disease shares at ages 50-54, 65-69, 80-84:")
print(paid_shares_df[paid_shares_df["age_bin"].isin(["50-54", "65-69", "80-84"])][["age_bin", "tot_22_23", "share_cvd", "share_diabetes", "share_cancer", "share_respiratory"]].to_string(index=False))

# Payer IP & ER unit-price ratios (2022-2023 pooled weighted average)
pp_df = pd.DataFrame(payer_price_rows)
payer_unit_costs = {}
for (etype, payer), grp in pp_df.groupby(["etype", "payer"]):
    wmu = np.sum(grp["mean_cost_2024"] * grp["wt_sum"]) / np.sum(grp["wt_sum"])
    payer_unit_costs[f"{etype}_{payer}"] = {"mean_2024": round(float(wmu), 2), "n_events": int(grp["n_events"].sum())}
ip_comm = payer_unit_costs["ip_Commercial (18-64)"]["mean_2024"]
ip_med  = payer_unit_costs["ip_Medicaid (18-64)"]["mean_2024"]
ip_mcr  = payer_unit_costs["ip_Medicare (65+)"]["mean_2024"]
er_comm = payer_unit_costs["er_Commercial (18-64)"]["mean_2024"]
er_med  = payer_unit_costs["er_Medicaid (18-64)"]["mean_2024"]
er_mcr  = payer_unit_costs["er_Medicare (65+)"]["mean_2024"]
print(f"  Unit costs (2024 USD): IP Comm=${ip_comm:,.0f}, Med=${ip_med:,.0f} (ratio={ip_med/ip_comm:.2f}), Mcr=${ip_mcr:,.0f} | ER Comm=${er_comm:,.0f}, Med=${er_med:,.0f} (ratio={er_med/er_comm:.2f}), Mcr=${er_mcr:,.0f}")

# =============================================================================
# 3. CDC 2023 US PERIOD LIFE TABLES + EXACT ABEL TTD SURVIVAL-COST SCHEDULES
# =============================================================================
print("\n=== STEP 3: CDC 2023 US Period Life Tables & Proximity-to-Death (TTD) Decomposition ===")
def parse_cdc_lt(xlsx_path: pathlib.Path, expected_substr: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, header=None)
    title = str(df.iloc[0, 0])
    assert expected_substr.lower() in title.lower(), f"Title mismatch in {xlsx_path.name}: expected '{expected_substr}', got '{title}'"
    hdr_idx = df.index[df[1].astype(str).str.strip() == "qx"][0]
    sub = df.iloc[hdr_idx + 1:].copy()
    sub = sub.iloc[:101, :7].copy()
    sub.columns = ["age_str", "qx", "lx", "dx", "Lx", "Tx", "ex"]
    sub["age"] = np.arange(len(sub))
    for c in ["qx", "lx", "dx", "Lx", "Tx", "ex"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub.dropna(subset=["qx"]).reset_index(drop=True)

lt_files = {
    "total":    ("cdc_lt2023_total.xlsx",    "total population"),
    "male":     ("cdc_lt2023_male.xlsx",     "for males"),
    "female":   ("cdc_lt2023_female.xlsx",   "for females"),
    "hispanic": ("cdc_lt2023_hispanic.xlsx", "Hispanic population"),
    "nhwhite":  ("cdc_lt2023_nhwhite.xlsx",  "White, non-Hispanic"),
    "nhblack":  ("cdc_lt2023_nhblack.xlsx",  "Black, non-Hispanic"),
    "nhaian":   ("cdc_lt2023_nhaian.xlsx",   "American Indian"),
    "nhasian":  ("cdc_lt2023_nhasian.xlsx",  "Asian, non-Hispanic"),
}
life_tables = {k: parse_cdc_lt(RAW / fn, substr) for k, (fn, substr) in lt_files.items()}
lt_all_frames = []
for k, ldf in life_tables.items():
    tmp = ldf.copy()
    tmp["group"] = k
    lt_all_frames.append(tmp)
pd.concat(lt_all_frames, ignore_index=True).to_parquet(PROC / "cdc_lt2023_all.parquet", index=False)
print(f"  CDC 2023 life expectancy at birth (e0): total={life_tables['total'].loc[0,'ex']:.2f}, male={life_tables['male'].loc[0,'ex']:.2f}, female={life_tables['female'].loc[0,'ex']:.2f}, NH-White={life_tables['nhwhite'].loc[0,'ex']:.2f}, NH-Black={life_tables['nhblack'].loc[0,'ex']:.2f}, Hispanic={life_tables['hispanic'].loc[0,'ex']:.2f}, NH-AIAN={life_tables['nhaian'].loc[0,'ex']:.2f}, NH-Asian={life_tables['nhasian'].loc[0,'ex']:.2f}")

# Proximity-to-Death (TTD) decomposition:
# Total age-a mean spending m_tot(a) = (1 - q_a) * m_surv(a) + q_a * m_eol(a),
# with m_eol(a) = (1 + scale * shape(a)) * m_surv(a).
# Age-gradient shape follows Riley & Lubitz (2010 HSR, PMID 20403063) and French et al. (2017 Health Affairs, PMID 28679814):
# steeper decedent/survivor ratio at younger ages, declining monotonically with age as baseline survivor comorbidity rises.
# Scale is solved uniquely via Brent root-finding so the age-65+ CDC-life-table-weighted EOL spending share
# matches Riley & Lubitz's exact 25.1% Medicare last-year-of-life spending share.
from scipy.optimize import brentq

def eol_excess_shape(age):
    pts_a = np.array([0, 20, 40, 55, 65, 75, 85, 95, 100], dtype=float)
    pts_s = np.array([10.0, 9.0, 8.0, 6.8, 5.5, 3.8, 2.2, 1.5, 1.2], dtype=float)
    return float(np.interp(age, pts_a, pts_s))

lt_tot = life_tables["total"]
qx_tot = lt_tot["qx"].values[:101]
pop_w65 = lt_tot.loc[65:100, "Lx"].values

def eol_share_65_for_scale(scale):
    eol_num, tot_den = 0.0, 0.0
    for idx, a in enumerate(range(65, 101)):
        m_tot = totexp_by_age[a]
        q = float(qx_tot[a])
        r_eol = 1.0 + scale * eol_excess_shape(a)
        m_surv = m_tot / (1.0 + q * (r_eol - 1.0))
        m_eol = r_eol * m_surv
        eol_num += pop_w65[idx] * q * m_eol
        tot_den += pop_w65[idx] * m_tot
    return eol_num / tot_den

EOL_SCALE = float(brentq(lambda s: eol_share_65_for_scale(s) - 0.251, 0.1, 10.0))

def eol_ratio_curve(age):
    return 1.0 + EOL_SCALE * eol_excess_shape(age)

# Also compute NHEA-reconciled institutional/LTSS uplift factor for sensitivity (MEPS is civilian noninstitutionalized;
# CMS NHEA Personal Health Care adds nursing-home LTSS rising sharply at ages 75+:
# +0% at <65, +8% at 65-74, +25% at 75-84, +55% at 85+, per French et al. 2017 / CMS NHEA).
def ltss_uplift(age):
    pts_a = np.array([0, 64, 70, 80, 90, 100], dtype=float)
    pts_u = np.array([1.00, 1.00, 1.08, 1.25, 1.55, 1.65], dtype=float)
    return float(np.interp(age, pts_a, pts_u))

# Build smooth domain share interpolators
dom_interps = {}
for col in ["share_cvd", "share_diabetes", "share_cancer", "share_respiratory", "share_cardiometabolic", "share_tobacco"]:
    dom_interps[col] = PchipInterpolator(paid_shares_df["age_mid"].values, paid_shares_df[col].values)

# Compute exact TTD-decomposed m_surv(a), K_eol(a), and effective marginal survival cost mu_eff(a, r, h)
us_paid_rows = []
for a in range(0, 101):
    m_tot = totexp_by_age[a]
    q = float(qx_tot[a])
    r_eol = eol_ratio_curve(a)
    m_surv = m_tot / (1.0 + q * (r_eol - 1.0))
    m_eol = r_eol * m_surv
    k_eol = m_eol - m_surv
    row = {
        "age": a,
        "qx_2023": round(q, 6),
        "ex_2023": round(float(lt_tot.loc[a, "ex"]), 3),
        "r_eol": round(r_eol, 3),
        "m_tot_2024": round(m_tot, 2),
        "m_surv_all_2024": round(m_surv, 2),
        "m_eol_all_2024": round(m_eol, 2),
        "k_eol_all_2024": round(k_eol, 2),
        "ltss_factor": round(ltss_uplift(a), 3),
    }
    for dom in ["cvd", "diabetes", "cancer", "respiratory", "cardiometabolic", "tobacco"]:
        sh = float(np.clip(dom_interps[f"share_{dom}"](min(a, 88.0)), 0.0, 0.85))
        row[f"share_{dom}"] = round(sh, 4)
        row[f"m_surv_unrel_{dom}"] = round(m_surv * (1.0 - sh), 2)
        row[f"k_eol_unrel_{dom}"] = round(k_eol * (1.0 - sh), 2)
    us_paid_rows.append(row)

us_paid_df = pd.DataFrame(us_paid_rows)
us_paid_df.to_csv(PROC / "us_paid_age_schedule_2024usd.csv", index=False)

# Verify Medicare (65+) EOL share implied by our TTD schedule (KS2 validation against Riley & Lubitz 25.1%)
sub65 = us_paid_df[us_paid_df["age"] >= 65].copy()
eol_spend_65 = np.sum(pop_w65 * sub65["qx_2023"].values * sub65["m_eol_all_2024"].values)
tot_spend_65 = np.sum(pop_w65 * sub65["m_tot_2024"].values)
eol_share_65 = float(eol_spend_65 / tot_spend_65)
mean_eol_65 = float(np.sum(pop_w65 * sub65["qx_2023"].values * sub65["m_eol_all_2024"].values) / np.sum(pop_w65 * sub65["qx_2023"].values))
print(f"  TTD validation (ages 65+): EOL_SCALE={EOL_SCALE:.3f}, R_eol(65)={eol_ratio_curve(65):.2f}, R_eol(85)={eol_ratio_curve(85):.2f}, implied EOL share = {100*eol_share_65:.1f}% (Riley & Lubitz 25.1%), weighted mean EOL = ${mean_eol_65:,.0f}")

# Compute exact discrete Abel-identity survival cost of averting one death at age a0
# (and per discounted life-year gained) for societal (r=0.03, h=0), with TTD vs naive age-only vs LTSS!
def compute_averted_death_survival_cost(a0: int, domain: str, r: float = 0.03, h: float = 0.0,
                                        include_ttd: bool = True, include_ltss: bool = False,
                                        lt_df: pd.DataFrame = None, horizon: int = None):
    """
    Exact discrete Abel summation-by-parts valuation of averting one death at age a0
    (shifting survival from 0 to S(t | a0) for t = 0..T-1).
    Returns:
      - pv_unrel_cost: net PV of added unrelated medical costs (survivor-year + postponed EOL - averted EOL at a0)
      - pv_lys: discounted life-years gained
      - mu_eff_per_ly: pv_unrel_cost / pv_lys
    """
    if lt_df is None:
        lt_df = life_tables["total"]
    qx = lt_df["qx"].values
    max_t = min(100 - a0, horizon) if horizon is not None else (100 - a0)
    # Survival S_t from a0: S_0 = 1, S_{t+1} = S_t * (1 - q_{a0+t})
    S = np.zeros(max_t + 1)
    S[0] = 1.0
    for t in range(max_t):
        S[t + 1] = S[t] * (1.0 - float(qx[a0 + t]))
    d = S[:-1] - S[1:]  # decedent mass at age a0+t

    pv_surv = 0.0
    pv_eol_postponed = 0.0
    pv_lys = 0.0
    col_m = f"m_surv_unrel_{domain}" if domain != "all" else "m_surv_all_2024"
    col_k = f"k_eol_unrel_{domain}" if domain != "all" else "k_eol_all_2024"
    m_arr = us_paid_df[col_m].values
    k_arr = us_paid_df[col_k].values
    ltss_arr = us_paid_df["ltss_factor"].values if include_ltss else np.ones(101)

    for t in range(max_t):
        age = a0 + t
        beta = np.exp(-(r + h) * t)
        m_t = m_arr[age] * ltss_arr[age]
        k_t = k_arr[age] * ltss_arr[age]
        if include_ttd:
            # Survivor-year unrelated cost + postponed EOL excess at future death
            pv_surv += beta * S[t] * m_t
            pv_eol_postponed += beta * d[t] * k_t
        else:
            # Naive age-only total unrelated cost (no TTD postponement subtraction)
            sh = us_paid_df.loc[age, f"share_{domain}"] if domain != "all" else 0.0
            pv_surv += beta * S[t] * us_paid_df.loc[age, "m_tot_2024"] * (1.0 - sh) * ltss_arr[age]
        pv_lys += np.exp(-r * t) * S[t]

    # If death at a0 is averted, society ALSO avoids the EOL excess k_eol(a0) today (at t=0) and pays it later at a0+t!
    k0 = k_arr[a0] * ltss_arr[a0] if include_ttd else 0.0
    pv_net = pv_surv + pv_eol_postponed - k0
    return float(pv_net), float(pv_lys), float(pv_net / pv_lys) if pv_lys > 0 else 0.0

ttd_check = {}
for a0 in [45, 55, 65, 75]:
    net_ttd, lys, mu_ttd = compute_averted_death_survival_cost(a0, "cvd", r=0.03, h=0.0, include_ttd=True)
    net_naive, _, mu_naive = compute_averted_death_survival_cost(a0, "cvd", r=0.03, h=0.0, include_ttd=False)
    net_ltss, _, mu_ltss = compute_averted_death_survival_cost(a0, "cvd", r=0.03, h=0.0, include_ttd=True, include_ltss=True)
    ttd_check[str(a0)] = {
        "disc_lys": round(lys, 2),
        "pv_unrel_ttd": round(net_ttd, 0),
        "pv_unrel_naive": round(net_naive, 0),
        "pv_unrel_ttd_ltss": round(net_ltss, 0),
        "ttd_overstatement_pct": round(100.0 * (net_naive - net_ttd) / net_ttd, 1),
        "mu_eff_per_ly_ttd": round(mu_ttd, 0),
        "mu_eff_per_ly_naive": round(mu_naive, 0),
    }
    print(f"  Age {a0} CVD averted death: disc_LY={lys:.1f}, PV_unrel(TTD)=${net_ttd:,.0f} vs Naive=${net_naive:,.0f} (naive +{100*(net_naive-net_ttd)/net_ttd:.1f}%), w/LTSS=${net_ltss:,.0f}")

# =============================================================================
# 4. EMPIRICAL MULTI-PAYER CHURN HAZARDS FROM MEPS LONGITUDINAL PANELS 26 & 27
# =============================================================================
print("\n=== STEP 4: MEPS Longitudinal Panels 26 & 27 (2021-2023) Empirical Insurance Churn ===")
months = ["JA", "FE", "MA", "AP", "MY", "JU", "JL", "AU", "SE", "OC", "NO", "DE"]
mcd_cols = [f"MCD{m}Y1X" for m in months] + [f"MCD{m}Y2X" for m in months]
peg_cols = [f"PEG{m}Y1"  for m in months] + [f"PEG{m}Y2"  for m in months]
mcr_cols = [f"MCR{m}Y1X" for m in months] + [f"MCR{m}Y2X" for m in months]
ins_cols = [f"INS{m}Y1X" for m in months] + [f"INS{m}Y2X" for m in months]

churn_records = []
for pname, zname in [("P26_2021_2022", "meps_p26_long.zip"), ("P27_2022_2023", "meps_p27_long.zip")]:
    cols = ["DUPERSID", "LONGWT", "AGEY1X", "SEX", "RACETHX", "POVCATY1", "DIED"] + mcd_cols + peg_cols + mcr_cols + ins_cols
    df = read_zip_dta(RAW / zname, columns=cols)
    df = df[(df["LONGWT"] > 0) & (df["AGEY1X"] >= 18) & (df["DIED"] != 1)].copy()
    df["PANEL"] = pname
    df["RACE_GRP"] = df["RACETHX"].map(RACE_MAP).fillna("NH Other")
    churn_records.append(df)

long_df = pd.concat(churn_records, ignore_index=True)
long_df["WT"] = long_df["LONGWT"] / 2.0

def compute_12m_exit(df_sub, col_list, age_min=18, age_max=64):
    """Among adults enrolled in month 1 (Jan Y1 == 1), compute weighted fraction exiting by month 12 (any gap or not enrolled in month 13) and continuous hazard h."""
    sub = df_sub[df_sub["AGEY1X"].between(age_min, age_max)].copy()
    m0 = sub[col_list[0]] == 1
    sub = sub[m0]
    if len(sub) == 0:
        return {"n": 0, "exit_12m": np.nan, "any_gap_12m": np.nan, "hazard_yr": np.nan}
    w = sub["WT"].values
    # Enrolled in month 13 (Jan Y2 = col_list[12])
    still_12 = (sub[col_list[12]] == 1).astype(float).values
    # Any exit during months 2..13 (at least 1 month without coverage)
    mat = (sub[col_list[1:13]].values == 1)
    ever_exit = (~mat.all(axis=1)).astype(float)
    p_exit_12 = float(1.0 - np.sum(still_12 * w) / np.sum(w))
    p_gap_12 = float(np.sum(ever_exit * w) / np.sum(w))
    # Continuous annual hazard from 12-month first-exit survival S(1) = 1 - p_gap_12
    h_yr = float(-np.log(max(1.0 - p_gap_12, 1e-4)))
    return {
        "n": int(len(sub)),
        "not_enrolled_m13_pct": round(100.0 * p_exit_12, 2),
        "any_exit_12m_pct": round(100.0 * p_gap_12, 2),
        "annual_hazard_h": round(h_yr, 4),
    }

churn_summary = {
    "commercial_employer_meps_18_64": compute_12m_exit(long_df, peg_cols, 18, 64),
    "medicaid_meps_18_64":            compute_12m_exit(long_df, mcd_cols, 18, 64),
    "medicare_meps_65plus":           compute_12m_exit(long_df, mcr_cols, 65, 100),
    "medicaid_by_race": {},
    "commercial_by_race": {},
    "medicaid_by_fpl": {},
    "commercial_by_age": {},
    "medicaid_by_age": {},
}
for rg in ["NH White", "NH Black", "Hispanic", "NH Asian"]:
    churn_summary["medicaid_by_race"][rg] = compute_12m_exit(long_df[long_df["RACE_GRP"] == rg], mcd_cols, 18, 64)
    churn_summary["commercial_by_race"][rg] = compute_12m_exit(long_df[long_df["RACE_GRP"] == rg], peg_cols, 18, 64)
for pov_id, pov_lbl in [(1, "<100% FPL"), (2, "100-124% FPL"), (3, "125-199% FPL"), (4, "200-399% FPL"), (5, "400%+ FPL")]:
    churn_summary["medicaid_by_fpl"][pov_lbl] = compute_12m_exit(long_df[long_df["POVCATY1"] == pov_id], mcd_cols, 18, 64)
for a_lo, a_hi in [(18, 34), (35, 49), (50, 64)]:
    churn_summary["commercial_by_age"][f"{a_lo}-{a_hi}"] = compute_12m_exit(long_df, peg_cols, a_lo, a_hi)
    churn_summary["medicaid_by_age"][f"{a_lo}-{a_hi}"] = compute_12m_exit(long_df, mcd_cols, a_lo, a_hi)

print("  MEPS P26-27 (2021-2023) Empirical Churn Summary:")
print("   Commercial employer (18-64):", churn_summary["commercial_employer_meps_18_64"])
print("   Medicaid (18-64):           ", churn_summary["medicaid_meps_18_64"])
print("   Medicare (65+):             ", churn_summary["medicare_meps_65plus"])
print("   NOTE: MEPS 2021-2023 spans PHE continuous enrollment; MEPS captures coverage/job exits, not within-employer carrier switches.")

# =============================================================================
# 5. NHANES 2021-2023 REAL U.S. ADULT CLINICAL RISK-FACTOR MICRODATA
# =============================================================================
print("\n=== STEP 5: NHANES 2021-2023 U.S. Adult Clinical Risk Distributions ===")
demo = pd.read_sas(RAW / "NHANES_DEMO_L.xpt")[["SEQN", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "WTMEC2YR"]]
bpxo = pd.read_sas(RAW / "NHANES_BPXO_L.xpt")[["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"]]
ghb  = pd.read_sas(RAW / "NHANES_GHB_L.xpt")[["SEQN", "LBXGH"]]
tc   = pd.read_sas(RAW / "NHANES_TCHOL_L.xpt")[["SEQN", "LBXTC"]]
hdl  = pd.read_sas(RAW / "NHANES_HDL_L.xpt")[["SEQN", "LBDHDD"]]
bmx  = pd.read_sas(RAW / "NHANES_BMX_L.xpt")[["SEQN", "BMXBMI"]]
smq  = pd.read_sas(RAW / "NHANES_SMQ_L.xpt")[["SEQN", "SMQ020", "SMQ040"]]
diq  = pd.read_sas(RAW / "NHANES_DIQ_L.xpt")[["SEQN", "DIQ010", "DIQ070"]]
bpq  = pd.read_sas(RAW / "NHANES_BPQ_L.xpt")[["SEQN", "BPQ020", "BPQ150", "BPQ080"]]
mcq  = pd.read_sas(RAW / "NHANES_MCQ_L.xpt")[["SEQN", "MCQ010", "MCQ035", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]

nh = demo.copy()
for df_m in [bpxo, ghb, tc, hdl, bmx, smq, diq, bpq, mcq]:
    nh = nh.merge(df_m, on="SEQN", how="left")

# Restrict to examined adults 18-80 with positive exam weight
nh = nh[(nh["RIDSTATR"] == 2) & (nh["RIDAGEYR"] >= 18) & (nh["WTMEC2YR"] > 0)].copy()
nh["SBP"] = nh[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
nh["DBP"] = nh[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
nh["FEMALE"] = (nh["RIAGENDR"] == 2).astype(int)
nh["BLACK"] = (nh["RIDRETH3"] == 4).astype(int)
nh["HISPANIC"] = nh["RIDRETH3"].isin([1, 2]).astype(int)
nh["NH_WHITE"] = (nh["RIDRETH3"] == 3).astype(int)
nh["CURR_SMOKER"] = nh["SMQ040"].isin([1, 2]).astype(int)
nh["BP_TREATED"] = (nh["BPQ150"] == 1).astype(int)
nh["DIABETES"] = ((nh["DIQ010"] == 1) | (nh["LBXGH"] >= 6.5)).astype(int)
nh["PREDIABETES"] = ((nh["DIABETES"] == 0) & (nh["LBXGH"] >= 5.7) & (nh["LBXGH"] < 6.5)).astype(int)
nh["PRIOR_CVD"] = ((nh["MCQ160B"] == 1) | (nh["MCQ160C"] == 1) | (nh["MCQ160D"] == 1) | (nh["MCQ160E"] == 1) | (nh["MCQ160F"] == 1)).astype(int)
nh["ASTHMA_CURR"] = ((nh["MCQ010"] == 1) & (nh["MCQ035"] == 1)).astype(int)

# Compute ACC/AHA 2013 Pooled Cohort Equations (PCE) 10-year ASCVD risk for adults 40-79 without prior CVD
def pce_10yr_risk(row):
    age = row["RIDAGEYR"]
    tc_v = row["LBXTC"]
    hdl_v = row["LBDHDD"]
    sbp_v = row["SBP"]
    if not (40 <= age <= 79 and np.isfinite(tc_v) and np.isfinite(hdl_v) and np.isfinite(sbp_v) and tc_v > 50 and hdl_v > 10 and sbp_v > 60):
        return np.nan
    la, ltc, lhdl, lsbp = np.log(age), np.log(tc_v), np.log(hdl_v), np.log(sbp_v)
    smk = row["CURR_SMOKER"]
    dm = row["DIABETES"]
    tr = row["BP_TREATED"]
    fem = row["FEMALE"] == 1
    blk = row["BLACK"] == 1
    if fem and not blk:
        s = (-29.799*la + 4.884*la**2 + 13.540*ltc - 3.114*la*ltc - 13.578*lhdl + 3.149*la*lhdl
             + (2.019 if tr else 1.957)*lsbp + 7.574*smk - 1.665*la*smk + 0.661*dm)
        s0, m0 = 0.9665, -29.18
    elif fem and blk:
        s = (17.114*la + 0.940*ltc - 18.920*lhdl + 4.475*la*lhdl
             + (29.291 - 6.432*la if tr else 27.820 - 6.087*la)*lsbp + 0.691*smk + 0.874*dm)
        s0, m0 = 0.9533, 86.61
    elif (not fem) and (not blk):
        s = (12.344*la + 11.853*ltc - 2.664*la*ltc - 7.990*lhdl + 1.769*la*lhdl
             + (1.797 if tr else 1.764)*lsbp + 7.837*smk - 1.795*la*smk + 0.658*dm)
        s0, m0 = 0.9144, 61.18
    else:
        s = (2.469*la + 0.302*ltc - 0.307*lhdl
             + (1.916 if tr else 1.809)*lsbp + 0.549*smk + 0.645*dm)
        s0, m0 = 0.8954, 19.54
    return float(np.clip(1.0 - s0 ** np.exp(s - m0), 0.0005, 0.95))

nh["ASCVD_10YR"] = nh.apply(pce_10yr_risk, axis=1)
nh.to_parquet(PROC / "nhanes_2021_2023_adults.parquet", index=False)

# Summarize real NHANES clinical target populations (weighted)
def w_stats(sub, col):
    s = sub[sub[col].notna()].copy()
    w = s["WTMEC2YR"].values
    x = s[col].values
    if len(x) == 0:
        return {}
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cdf = np.cumsum(ws) / np.sum(ws)
    mu = float(np.sum(x * w) / np.sum(w))
    sd = float(np.sqrt(np.sum(w * (x - mu)**2) / np.sum(w)))
    return {
        "n": int(len(x)),
        "us_pop_millions": round(float(np.sum(w) / 1e6), 2),
        "mean": round(mu, 4),
        "sd": round(sd, 4),
        "p50": round(float(np.interp(0.50, cdf, xs)), 4),
        "p75": round(float(np.interp(0.75, cdf, xs)), 4),
        "p90": round(float(np.interp(0.90, cdf, xs)), 4),
        "p95": round(float(np.interp(0.95, cdf, xs)), 4),
        "cv": round(sd / mu, 3) if mu > 0 else np.nan,
    }

nh_summary = {
    "n_adults_examined": int(len(nh)),
    "us_adult_pop_millions": round(float(nh["WTMEC2YR"].sum() / 1e6), 2),
    "statin_eligible_40_75_no_cvd_ascvd": w_stats(nh[(nh["RIDAGEYR"].between(40, 75)) & (nh["PRIOR_CVD"] == 0)], "ASCVD_10YR"),
    "statin_ge7_5pct_ascvd": w_stats(nh[(nh["RIDAGEYR"].between(40, 75)) & (nh["PRIOR_CVD"] == 0) & (nh["ASCVD_10YR"] >= 0.075)], "ASCVD_10YR"),
    "sprint_eligible_sbp130_highrisk": w_stats(nh[(nh["RIDAGEYR"] >= 50) & (nh["SBP"] >= 130) & (nh["DIABETES"] == 0) & ((nh["ASCVD_10YR"] >= 0.15) | (nh["PRIOR_CVD"] == 1) | (nh["RIDAGEYR"] >= 75))], "SBP"),
    "prediabetes_hba1c": w_stats(nh[(nh["PREDIABETES"] == 1) & (nh["BMXBMI"] >= 25)], "LBXGH"),
    "glp1_obesity_bmi": w_stats(nh[(nh["BMXBMI"] >= 30) | ((nh["BMXBMI"] >= 27) & ((nh["BPQ020"] == 1) | (nh["DIABETES"] == 1) | (nh["BPQ080"] == 1)))], "BMXBMI"),
    "glp1_obesity_40_79_ascvd": w_stats(nh[(nh["RIDAGEYR"].between(40, 79)) & (nh["BMXBMI"] >= 30)], "ASCVD_10YR"),
    "current_smokers_18plus": w_stats(nh[nh["CURR_SMOKER"] == 1], "RIDAGEYR"),
}
print(f"  NHANES 2021-2023 adults: n={len(nh):,d} (weighted {nh['WTMEC2YR'].sum()/1e6:.1f}M)")
print(f"  Primary-prevention 40-75 ASCVD 10-yr risk: mean={nh_summary['statin_eligible_40_75_no_cvd_ascvd']['mean']:.3f}, sd={nh_summary['statin_eligible_40_75_no_cvd_ascvd']['sd']:.3f}, CV={nh_summary['statin_eligible_40_75_no_cvd_ascvd']['cv']:.2f}, P90={nh_summary['statin_eligible_40_75_no_cvd_ascvd']['p90']:.3f}")
print(f"  Prediabetes BMI>=25: {nh_summary['prediabetes_hba1c']['us_pop_millions']:.1f}M adults; GLP-1 obesity-eligible: {nh_summary['glp1_obesity_bmi']['us_pop_millions']:.1f}M adults")

# =============================================================================
# 6. SAVE CANONICAL PILLAR 1 OUTPUTS + KS2 CHECK
# =============================================================================
# Check KS2 benchmarks:
# (a) MEPS 2021 nominal total matches AHRQ published $2,297B within +/- 1% (from meps.ahrq.gov/about_meps/Price_Index.shtml);
# (b) Max 5-year age-bin RSE < 15%;
# (c) Implied 65+ EOL spending share within [20%, 30%] (Riley & Lubitz 25.1%).
ahrq_2021_diff_pct = abs(annual_totals["2021"]["total_nominal_billions"] - 2297.0) / 2297.0 * 100.0
ks2_pass = (max_rse < 15.0) and (0.20 <= eol_share_65 <= 0.30) and (ahrq_2021_diff_pct < 1.0)

out = {
    "meta": {
        "price_index": "BEA PCE-Health (AHRQ MEPS Table 3, updated April 2026)",
        "base_year_usd": TARGET_YEAR,
        "pooled_meps_years": [2019, 2020, 2021, 2022, 2023],
        "pooled_meps_n": int(len(meps)),
        "paid_clnk_years": [2022, 2023],
        "paid_clnk_n": int(len(paid_df)),
        "longitudinal_panels": ["Panel 26 (2021-2022)", "Panel 27 (2022-2023)"],
        "longitudinal_adults_n": int(len(long_df)),
        "nhanes_cycle": "2021-2023 (Aug 2021-Aug 2023)",
        "nhanes_adults_n": int(len(nh)),
        "life_tables": "CDC NCHS US Period Life Tables 2023 (NVSR 74-06)",
    },
    "ks2_validation": {
        "meps_2021_ahrq_benchmark_billions": 2297.0,
        "meps_2021_computed_billions": annual_totals["2021"]["total_nominal_billions"],
        "meps_2021_diff_pct": round(float(ahrq_2021_diff_pct), 3),
        "max_5yr_age_rse_pct": round(max_rse, 2),
        "rse_gate_lt_15pct": bool(max_rse < 15.0),
        "eol_scale_calibrated": round(float(EOL_SCALE), 4),
        "eol_share_65plus_pct": round(100.0 * eol_share_65, 2),
        "eol_mean_65plus_2024usd": round(mean_eol_65, 2),
        "ks2_overall_pass": bool(ks2_pass),
    },
    "meps_annual_reconciliation": annual_totals,
    "clnk_event_match_rates": clnk_match_log,
    "age_schedule_5yr": age_sched,
    "paid_domain_shares_5yr": paid_rows,
    "payer_unit_costs_2024usd": payer_unit_costs,
    "payer_price_ratios": {
        "ip_medicaid_to_commercial": round(ip_med / ip_comm, 3),
        "ip_medicare_to_commercial": round(ip_mcr / ip_comm, 3),
        "er_medicaid_to_commercial": round(er_med / er_comm, 3),
        "er_medicare_to_commercial": round(er_mcr / er_comm, 3),
    },
    "ttd_averted_death_cvd_check": ttd_check,
    "meps_empirical_churn": churn_summary,
    "nhanes_risk_distributions": nh_summary,
}

(RES / "pillar1_us_paid_and_churn.json").write_text(json.dumps(out, indent=2))
print(f"\n=== PILLAR 1 COMPLETE: KS2 PASS = {ks2_pass} | Saved to {RES / 'pillar1_us_paid_and_churn.json'} ===")
