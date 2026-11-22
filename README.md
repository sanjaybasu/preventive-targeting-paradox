# Can Precision Targeting Make Prevention Save Money? Theory and U.S. Evidence

Code, formal machine-checked proofs (Lean 4.20 + Mathlib4), Z3 4.16.0 empirical verification scripts, and offline unit tests accompanying:

> Basu S, Venkataramani AS. *Can Precision Targeting Make Prevention Save Money? Theory and U.S. Evidence.* Under review, 2026.

## Repository Contents (Code Only)

No raw data, microdata, derived result files, or protected health information are stored in this repository. All empirical inputs are publicly downloadable from federal agencies and public randomized trial replication archives.

- `lean/`: Lean 4.20.0 + Mathlib4 library (`59` formally verified lemmas and theorems across four modules, zero `sorry`, standard axioms only):
  - `PreventiveTargetingParadox/Theorem1_AccountingAndChurn.lean` — discrete Abel proximity-to-death identity, three-perspective payer accounting, zero-cost impossibility, and the U-shaped Churn–Longevity Pincer.
  - `PreventiveTargetingParadox/Theorem2_ScarfCantelliFrechet.lean` — sharp unconstrained Scarf–Cantelli oracle ceiling, exact lower-bounded tightening ($L^2 \le \mu^2 + \sigma^2$), and submodular coupling.
  - `PreventiveTargetingParadox/Theorem3_WinnersCurseMills.lean` — distribution-free Winner's Curse Sandwich and Gaussian Mills-ratio sign-flip threshold.
  - `PreventiveTargetingParadox/Theorem4_NeymanPearsonAndAntinomy.lean` — budget-neutral cross-subsidization duality and longevity–equity targeting inversion.
  - `VerifyAll.lean` — axiom audit verifying standard axioms only across all 59 statements.
- `src/`: Reproducible Python pipeline:
  - `01_build_us_paid_and_churn.py` — constructs the open-source U.S. unrelated expenditure schedule (`US-PAID`) from AHRQ MEPS 2019–2023, CDC NCHS 2023 Period Life Tables, and CDC NHANES 2021–2023.
  - `02_rct_spending_cate_audit.py` — replicates published intention-to-treat benchmarks and runs cross-fitted DR-Learner, honest Causal Forest, and 500-draw sequential martingale sharp-null permutation audits on Illinois Workplace Wellness, Camden Hotspotting, and the Oregon Health Insurance Experiment.
  - `03_portfolio_phase_diagram_and_policy.py` — evaluates 14 U.S. preventive interventions across age-matched payer perspectives, 5-step prediabetes metformin decomposition, multi-period CBO 10/20/30-year federal budget windows, GLP-1 break-even pricing, Medicaid statutory regimes, equity redlining audit, and full sensitivity sweeps.
  - `05_verify_canonical.py` — programmatically re-derives and verifies canonical numbers from locked outputs.
  - `06_generate_figures_and_tables.py` — generates publication-grade figures (600 DPI PNG + vector PDF) and CSV/Markdown tables directly from locked JSON/CSV artifacts.
  - `07_build_and_verify_references.py` — verifies every bibliography DOI and title against the CrossRef REST API.
- `z3_audit/04_z3_certify.py`: Z3 nonlinear real arithmetic (`QF_NRA`) empirical audit verifying Theorem 2's unconstrained and lower-bounded Scarf–Cantelli ceilings and evaluating Theorem 3's Gaussian-benchmark diagnostic across all active intervention-by-payer cells.
- `tests/test_core_theorems_and_bounds.py`: Offline `pytest` unit test suite verifying core mathematical identities, Scarf–Cantelli extremals, Mills-ratio bounds, churn minima, and budget-neutral duality.

## Public Data Sources

1. **AHRQ MEPS (2019–2023):** Full-Year Consolidated (`h216`, `h224`, `h233`, `h243`, `h251`), Medical Conditions (`h241`, `h249`), Event files (`IP`, `ER`, `OP`, `OB`, `RX`), Condition-Event Link (`h239if1`, `h248if1`), and Longitudinal Panels 26–27 (`h244`, `h252`) from [meps.ahrq.gov](https://meps.ahrq.gov/).
2. **CDC NCHS 2023 U.S. Period Life Tables (NVSR 74-06):** Eight demographic tables from [cdc.gov/nchs](https://www.cdc.gov/nchs/products/life_tables.htm).
3. **CDC NHANES (2021–2023):** Examination, demographic, laboratory, and questionnaire XPT modules from [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/).
4. **Public U.S. RCT Replication Archives:**
   - Illinois Workplace Wellness Study (`ILLINOIS_DATA_DIR`, Jones, Molitor, & Reif, 2019, NBER/GitHub archive).
   - Camden Hotspotting RCT (`CAMDEN_DTA_PATH`, Finkelstein et al., 2020, J-PAL / Harvard Dataverse).
   - Oregon Health Insurance Experiment (`OREGON_ZIP_PATH`, Finkelstein et al., 2012; Baicker et al., 2013; Taubman et al., 2014, NBER PUF).

## Running Verification & Pipeline

```bash
# 1. Run offline unit tests (no data required)
pip install -r requirements.txt
pytest tests/

# 2. Verify all 59 Lean 4 theorems (requires Lean 4.20.0 / elan)
cd lean
lake exe cache get
lake build
cd ..

# 3. Run full empirical pipeline (after placing public inputs in data/raw/ or setting env vars)
python src/01_build_us_paid_and_churn.py
python src/02_rct_spending_cate_audit.py
python src/03_portfolio_phase_diagram_and_policy.py
python z3_audit/04_z3_certify.py
python src/05_verify_canonical.py
python src/06_generate_figures_and_tables.py
```

## License

MIT License (see `LICENSE`).
