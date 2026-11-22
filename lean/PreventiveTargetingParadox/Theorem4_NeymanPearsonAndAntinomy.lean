import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

/-!
# Theorem 4: Generalized Neyman–Pearson Duality, Budget-Neutral Cross-Subsidization, and the Longevity–Targeting Antinomy

This module formalizes:
1. **Generalized Neyman–Pearson & Budget-Neutral Cross-Subsidization Duality**:
   Over finite patient types with nonnegative weights `w_i ≥ 0`:
   - (i) Unconstrained cost minimization is solved by `π*_cost(i) = 1{θ_i < 0}`;
   - (ii) Upfront-budget-constrained cost minimization is solved by the ROI threshold
     `π*_ROI(i) = 1{-θ_i / c_{0,i} ≥ η}`;
   - (iii) Budget-neutral health maximization (`max ∑ w_i π_i τ_{H,i}` s.t. `∑ w_i π_i θ_i ≤ 0`)
     is solved by `π*_BN(i) = 1{θ_i ≤ λ_BN τ_{H,i}}`, which strictly dominates `π*_cost` in health
     whenever negative-cost surplus can cross-subsidize positive-cost high-QALY patients.
2. **Longevity–Targeting Antinomy (Rank Inversion Theorem)**:
   For any societal WTP `λ > 0` and two patient profiles with `τ_{H,1} > τ_{H,2} > 0`,
   `θ_1 > 0 > θ_2`, and `λ τ_{H,1} - θ_1 > λ τ_{H,2} - θ_2 > 0`, societal EVIC strictly prioritizes
   type 1 over type 2 whereas a `λ = 0` self-financing mandate strictly rejects type 1 and treats
   only type 2; accompanied by a parameterized longevity-coupling mechanism and an explicit witness.
-/

namespace PreventiveTargetingParadox.Theorem4

open scoped BigOperators

/-! ## Part 4A: Generalized Neyman–Pearson & Budget-Neutral Cross-Subsidization Duality -/

/-- A fractional targeting policy `π` is feasible on `s` if `0 ≤ π(i) ≤ 1` for all `i ∈ s`. -/
def isFeasiblePolicy {ι : Type*} (s : Finset ι) (π : ι → ℝ) : Prop :=
  ∀ i ∈ s, 0 ≤ π i ∧ π i ≤ 1

/-- Unconstrained cost-minimizing threshold rule `π*_cost(i) = 1{θ_i < 0}`. -/
noncomputable def piCost {ι : Type*} (θ : ι → ℝ) (i : ι) : ℝ :=
  if θ i < 0 then 1 else 0

/-- Upfront-budget ROI threshold rule `π*_ROI(i) = 1{-θ_i / c_{0,i} ≥ η}`. -/
noncomputable def piROI {ι : Type*} (θ c0 : ι → ℝ) (η : ℝ) (i : ι) : ℝ :=
  if η ≤ -θ i / c0 i then 1 else 0

/-- Budget-neutral health-maximizing threshold rule `π*_BN(i) = 1{θ_i ≤ λ_BN τ_{H,i}}`. -/
noncomputable def piBN {ι : Type*} (θ τH : ι → ℝ) (lam : ℝ) (i : ι) : ℝ :=
  if θ i ≤ lam * τH i then 1 else 0

lemma piCost_feasible {ι : Type*} (s : Finset ι) (θ : ι → ℝ) :
    isFeasiblePolicy s (piCost θ) := by
  intro i _
  unfold piCost
  split_ifs <;> constructor <;> norm_num

lemma piROI_feasible {ι : Type*} (s : Finset ι) (θ c0 : ι → ℝ) (η : ℝ) :
    isFeasiblePolicy s (piROI θ c0 η) := by
  intro i _
  unfold piROI
  split_ifs <;> constructor <;> norm_num

lemma piBN_feasible {ι : Type*} (s : Finset ι) (θ τH : ι → ℝ) (lam : ℝ) :
    isFeasiblePolicy s (piBN θ τH lam) := by
  intro i _
  unfold piBN
  split_ifs <;> constructor <;> norm_num

/-- Total net cost under `π*_cost` equals `∑ w_i min(θ_i, 0)`. -/
theorem unconstrained_cost_min_eq_sum_min {ι : Type*} (s : Finset ι) (w θ : ι → ℝ) :
    ∑ i ∈ s, w i * piCost θ i * θ i = ∑ i ∈ s, w i * min (θ i) 0 := by
  refine Finset.sum_congr rfl fun i _ => ?_
  unfold piCost
  split_ifs with h
  · rw [min_eq_left (le_of_lt h)]; ring
  · rw [min_eq_right (not_lt.1 h)]; ring

/-- **(i) Unconstrained Cost-Minimization Optimality**:
Among all feasible policies `π ∈ [0, 1]^s`, `π*_cost(i) = 1{θ_i < 0}` minimizes total net cost
`∑ w_i π_i θ_i`. -/
theorem unconstrained_cost_min_optimal {ι : Type*} (s : Finset ι) (w θ π : ι → ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hπ : isFeasiblePolicy s π) :
    ∑ i ∈ s, w i * piCost θ i * θ i ≤ ∑ i ∈ s, w i * π i * θ i := by
  refine Finset.sum_le_sum fun i hi => ?_
  rcases hπ i hi with ⟨h0, h1⟩
  have hwi := hw i hi
  rw [mul_assoc, mul_assoc]
  refine mul_le_mul_of_nonneg_left ?_ hwi
  unfold piCost
  split_ifs with h
  · nlinarith
  · have hge : 0 ≤ θ i := not_lt.1 h
    nlinarith

/-- **(ii) Upfront-Budget ROI Threshold Optimality (Generalized Neyman–Pearson)**:
For `c_{0,i} > 0` and Lagrange multiplier `η ≥ 0`, `π*_ROI(i) = 1{-θ_i / c_{0,i} ≥ η}` minimizes
total net cost `∑ w_i π_i θ_i` over all feasible policies using no more upfront budget than
`π*_ROI` (and, with complementary slackness `η * (B - ∑ w_i π*_ROI(i) c_{0,i}) = 0`, over all
feasible policies with `∑ w_i π_i c_{0,i} ≤ B`). -/
theorem upfront_budget_roi_optimal {ι : Type*} (s : Finset ι) (w θ c0 π : ι → ℝ) {η B : ℝ}
    (hw : ∀ i ∈ s, 0 ≤ w i) (hc0 : ∀ i ∈ s, 0 < c0 i) (hη : 0 ≤ η)
    (hπ : isFeasiblePolicy s π)
    (hbudget : ∑ i ∈ s, w i * π i * c0 i ≤ B)
    (hcs : η * (B - ∑ i ∈ s, w i * piROI θ c0 η i * c0 i) = 0) :
    ∑ i ∈ s, w i * piROI θ c0 η i * θ i ≤ ∑ i ∈ s, w i * π i * θ i := by
  have hlag : ∑ i ∈ s, w i * piROI θ c0 η i * (θ i + η * c0 i) ≤
      ∑ i ∈ s, w i * π i * (θ i + η * c0 i) := by
    refine Finset.sum_le_sum fun i hi => ?_
    rcases hπ i hi with ⟨h0, h1⟩
    have hwi := hw i hi
    have hci := hc0 i hi
    rw [mul_assoc, mul_assoc]
    refine mul_le_mul_of_nonneg_left ?_ hwi
    unfold piROI
    split_ifs with h
    · have hle0 : θ i + η * c0 i ≤ 0 := by
        have hmul : η * c0 i ≤ -θ i := (le_div_iff₀ hci).1 h
        linarith
      nlinarith
    · have hgt0 : 0 < θ i + η * c0 i := by
        have hlt : -θ i / c0 i < η := not_le.1 h
        have hmul : -θ i < η * c0 i := (div_lt_iff₀ hci).1 hlt
        linarith
      nlinarith
  have hsplit_roi : ∑ i ∈ s, w i * piROI θ c0 η i * (θ i + η * c0 i) =
      (∑ i ∈ s, w i * piROI θ c0 η i * θ i) + η * (∑ i ∈ s, w i * piROI θ c0 η i * c0 i) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_; ring
  have hsplit_pi : ∑ i ∈ s, w i * π i * (θ i + η * c0 i) =
      (∑ i ∈ s, w i * π i * θ i) + η * (∑ i ∈ s, w i * π i * c0 i) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_; ring
  rw [hsplit_roi, hsplit_pi] at hlag
  have hηB : η * (∑ i ∈ s, w i * π i * c0 i) ≤ η * (∑ i ∈ s, w i * piROI θ c0 η i * c0 i) := by
    nlinarith [mul_le_mul_of_nonneg_left hbudget hη, hcs]
  linarith

/-- **(iii) Budget-Neutral Health Maximization Optimality**:
If `λ_BN > 0` and the threshold policy `π*_BN(i) = 1{θ_i ≤ λ_BN τ_{H,i}}` exhausts the
budget-neutrality constraint (`∑ w_i π*_BN(i) θ_i = 0`), then `π*_BN` maximizes total health
`∑ w_i π_i τ_{H,i}` among ALL feasible budget-neutral policies (`∑ w_i π_i θ_i ≤ 0`). -/
theorem budget_neutral_health_optimal {ι : Type*} (s : Finset ι) (w θ τH π : ι → ℝ) {lam : ℝ}
    (hw : ∀ i ∈ s, 0 ≤ w i) (hlam : 0 < lam) (hπ : isFeasiblePolicy s π)
    (hπ_bn : ∑ i ∈ s, w i * π i * θ i ≤ 0)
    (hbn_exact : ∑ i ∈ s, w i * piBN θ τH lam i * θ i = 0) :
    ∑ i ∈ s, w i * π i * τH i ≤ ∑ i ∈ s, w i * piBN θ τH lam i * τH i := by
  have hlag : ∑ i ∈ s, w i * π i * (lam * τH i - θ i) ≤
      ∑ i ∈ s, w i * piBN θ τH lam i * (lam * τH i - θ i) := by
    refine Finset.sum_le_sum fun i hi => ?_
    rcases hπ i hi with ⟨h0, h1⟩
    have hwi := hw i hi
    rw [mul_assoc, mul_assoc]
    refine mul_le_mul_of_nonneg_left ?_ hwi
    unfold piBN
    split_ifs with h
    · nlinarith
    · have hlt : lam * τH i < θ i := not_le.1 h
      nlinarith
  have hsplit_pi : ∑ i ∈ s, w i * π i * (lam * τH i - θ i) =
      lam * (∑ i ∈ s, w i * π i * τH i) - (∑ i ∈ s, w i * π i * θ i) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_; ring
  have hsplit_bn : ∑ i ∈ s, w i * piBN θ τH lam i * (lam * τH i - θ i) =
      lam * (∑ i ∈ s, w i * piBN θ τH lam i * τH i) - (∑ i ∈ s, w i * piBN θ τH lam i * θ i) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_; ring
  rw [hsplit_pi, hsplit_bn, hbn_exact, sub_zero] at hlag
  nlinarith

/-- **Strict Health Dominance of `π*_BN` over `π*_cost` (Threshold Form)**:
Whenever `λ_BN > 0`, health effects are nonnegative (`τ_{H,i} ≥ 0`), `π*_BN` remains budget-neutral,
and at least one positively weighted patient with nonnegative net cost (`0 ≤ θ_{i₁} ≤ λ_BN τ_{H,i₁}`)
and positive health gain (`τ_{H,i₁} > 0`) is cross-subsidized, `π*_BN` achieves strictly greater
population health than `π*_cost`. -/
theorem budget_neutral_threshold_strictly_dominates_cost_min {ι : Type*}
    (s : Finset ι) (w θ τH : ι → ℝ) {lam : ℝ}
    (hw : ∀ i ∈ s, 0 ≤ w i) (hlam : 0 < lam)
    (hτ : ∀ i ∈ s, 0 ≤ τH i)
    (hcross : ∃ i1 ∈ s, 0 < w i1 ∧ 0 ≤ θ i1 ∧ θ i1 ≤ lam * τH i1 ∧ 0 < τH i1) :
    ∑ i ∈ s, w i * piCost θ i * τH i < ∑ i ∈ s, w i * piBN θ τH lam i * τH i := by
  rcases hcross with ⟨i1, hi1, hw1, hθ1_ge, hθ1_le, hτ1_pos⟩
  have hle : ∀ i ∈ s, w i * piCost θ i * τH i ≤ w i * piBN θ τH lam i * τH i := by
    intro i hi
    have hwi := hw i hi
    have hτi := hτ i hi
    unfold piCost piBN
    split_ifs with h1 h2
    · exact le_rfl
    · exfalso; exact h2 (by nlinarith)
    · nlinarith
    · exact le_rfl
  refine Finset.sum_lt_sum hle ⟨i1, hi1, ?_⟩
  unfold piCost piBN
  rw [if_neg (not_lt.2 hθ1_ge), if_pos hθ1_le]
  nlinarith

/-- **Constructive Cross-Subsidization Dominance Theorem**:
Whenever unconstrained cost minimization generates a strict negative-cost surplus
(`∑ w_i π*_cost(i) θ_i < 0`) and there exists a positively weighted patient `i₁` with positive
net cost (`θ_{i₁} > 0`) and positive clinical benefit (`τ_{H,i₁} > 0`), there exists an explicit
feasible budget-neutral policy `π_cross` (`∑ w_i π_cross(i) θ_i ≤ 0`) delivering strictly greater
population health than `π*_cost`. -/
theorem cross_subsidization_strictly_dominates_cost_min {ι : Type*} [DecidableEq ι]
    (s : Finset ι) (w θ τH : ι → ℝ)
    (_hw : ∀ i ∈ s, 0 ≤ w i)
    (hsurplus : ∑ i ∈ s, w i * piCost θ i * θ i < 0)
    (hpos_patient : ∃ i1 ∈ s, 0 < w i1 ∧ 0 < θ i1 ∧ 0 < τH i1) :
    ∃ π_cross : ι → ℝ,
      isFeasiblePolicy s π_cross ∧
      ∑ i ∈ s, w i * π_cross i * θ i ≤ 0 ∧
      ∑ i ∈ s, w i * piCost θ i * τH i < ∑ i ∈ s, w i * π_cross i * τH i := by
  rcases hpos_patient with ⟨i1, hi1, hw1, hθ1, hτ1⟩
  set S := -(∑ i ∈ s, w i * piCost θ i * θ i)
  have hS_pos : 0 < S := neg_pos.2 hsurplus
  set α := min 1 (S / (w i1 * θ i1))
  have hα_pos : 0 < α := lt_min zero_lt_one (div_pos hS_pos (mul_pos hw1 hθ1))
  have hα_le1 : α ≤ 1 := min_le_left _ _
  have hα_cost : α * (w i1 * θ i1) ≤ S := by
    have hle : α ≤ S / (w i1 * θ i1) := min_le_right _ _
    exact (le_div_iff₀ (mul_pos hw1 hθ1)).1 hle
  let π_cross : ι → ℝ := fun i => if i = i1 then α else piCost θ i
  have hcost_i1 : piCost θ i1 = 0 := by
    unfold piCost
    exact if_neg (not_lt.2 (le_of_lt hθ1))
  refine ⟨π_cross, ?_, ?_, ?_⟩
  · intro i hi
    simp only [π_cross]
    split_ifs
    · exact ⟨le_of_lt hα_pos, hα_le1⟩
    · exact piCost_feasible s θ i hi
  · have hdiff : ∑ i ∈ s, w i * π_cross i * θ i =
        (∑ i ∈ s, w i * piCost θ i * θ i) + α * (w i1 * θ i1) := by
      have hterm : ∀ i ∈ s, w i * π_cross i * θ i =
          w i * piCost θ i * θ i + (if i = i1 then α * (w i1 * θ i1) else 0) := by
        intro i _
        simp only [π_cross]
        split_ifs with h
        · subst h; rw [hcost_i1]; ring
        · ring
      rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, Finset.sum_ite_eq' s i1, if_pos hi1]
    rw [hdiff]
    simp only [S] at hα_cost
    linarith
  · have hdiff : ∑ i ∈ s, w i * π_cross i * τH i =
        (∑ i ∈ s, w i * piCost θ i * τH i) + α * (w i1 * τH i1) := by
      have hterm : ∀ i ∈ s, w i * π_cross i * τH i =
          w i * piCost θ i * τH i + (if i = i1 then α * (w i1 * τH i1) else 0) := by
        intro i _
        simp only [π_cross]
        split_ifs with h
        · subst h; rw [hcost_i1]; ring
        · ring
      rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, Finset.sum_ite_eq' s i1, if_pos hi1]
    rw [hdiff]
    have hgain : 0 < α * (w i1 * τH i1) := mul_pos hα_pos (mul_pos hw1 hτ1)
    linarith

/-! ## Part 4B: Longevity–Targeting Antinomy (Rank Inversion Theorem) -/

/-- Societal Expected Value of Individualized Care (Net Monetary Benefit) at willingness-to-pay
threshold `λ`: `EVIC(λ; τ_H, θ) = λ * τ_H - θ`. -/
def evic (lam τH θ : ℝ) : ℝ := lam * τH - θ

/-- **Longevity–Targeting Antinomy (Rank Inversion Theorem)**:
For any societal willingness-to-pay `λ > 0` and two patient profiles with positive clinical QALY
gains `τ_{H,1} > τ_{H,2} > 0`, if survival-period cost inflation drives `θ₁ > 0 > θ₂` while
`λ τ_{H,1} - θ₁ > λ τ_{H,2} - θ₂ > 0`, then:
1. Societal EVIC strictly prioritizes type 1 over type 2, with both strictly cost-effective;
2. Under a `λ = 0` self-financing mandate, the ranking strictly inverts (`EVIC(0)` is positive
   for type 2 and negative for type 1);
3. The unconstrained cost-minimizing rule `π*_cost` strictly rejects type 1 (`= 0`) and treats
   only type 2 (`= 1`). -/
theorem longevity_targeting_antinomy {lam τH1 τH2 θ1 θ2 : ℝ}
    (_hlam : 0 < lam) (_hτ2 : 0 < τH2) (_hτ12 : τH2 < τH1)
    (hθ1 : 0 < θ1) (hθ2 : θ2 < 0)
    (hevic2 : 0 < lam * τH2 - θ2)
    (hevic12 : lam * τH2 - θ2 < lam * τH1 - θ1) :
    let θ_pair : Fin 2 → ℝ := ![θ1, θ2]
    evic lam τH2 θ2 < evic lam τH1 θ1 ∧
    0 < evic lam τH2 θ2 ∧
    evic 0 τH1 θ1 < 0 ∧
    0 < evic 0 τH2 θ2 ∧
    piCost θ_pair 0 = 0 ∧
    piCost θ_pair 1 = 1 := by
  intro θ_pair
  refine ⟨hevic12, hevic2, ?_, ?_, ?_, ?_⟩
  · unfold evic; linarith
  · unfold evic; linarith
  · unfold piCost
    simp only [θ_pair, Matrix.cons_val_zero]
    exact if_neg (not_lt.2 (le_of_lt hθ1))
  · unfold piCost
    simp only [θ_pair, Matrix.cons_val_one, Matrix.head_cons]
    exact if_pos hθ2

/-- **Parameterized Longevity-Coupling Mechanism**:
For a preventive intervention with upfront cost `c₀`, non-fatal disease offset `S₀ > c₀`,
baseline non-fatal QALY gain `τ₀ ≥ 0`, averted-death life-years `L_QALY > 0`, mortality relative
risk reduction `r > 0`, and unrelated annual survivor medical cost `m > 0`, when society values
health above annual survivor spending (`λ > m`), ANY pair of patients straddling the critical
baseline mortality risk `p_crit = (S₀ - c₀) / (m * L_QALY * r)` (`0 < p₂ < p_crit < p₁`)
triggers the exact Longevity–Targeting Antinomy: the higher-mortality patient has strictly larger
QALY gain and societal EVIC, yet is strictly rejected by a self-financing mandate. -/
theorem longevity_coupling_rank_inversion_general
    {lam m c0 S0 τ0 L_qaly r p1 p2 : ℝ}
    (hm : 0 < m) (hlam_gt_m : m < lam)
    (_hc0_lt_S0 : c0 < S0) (hτ0 : 0 ≤ τ0)
    (hL : 0 < L_qaly) (hr : 0 < r)
    (hp2_pos : 0 < p2)
    (hp2_lt : p2 < (S0 - c0) / (m * L_qaly * r))
    (hp1_gt : (S0 - c0) / (m * L_qaly * r) < p1) :
    let τH := fun p => τ0 + L_qaly * r * p
    let θ := fun p => c0 - S0 + m * L_qaly * r * p
    let θ_pair : Fin 2 → ℝ := ![θ p1, θ p2]
    0 < τH p2 ∧ τH p2 < τH p1 ∧
    θ p2 < 0 ∧ 0 < θ p1 ∧
    0 < evic lam (τH p2) (θ p2) ∧
    evic lam (τH p2) (θ p2) < evic lam (τH p1) (θ p1) ∧
    evic 0 (τH p1) (θ p1) < 0 ∧ 0 < evic 0 (τH p2) (θ p2) ∧
    piCost θ_pair 0 = 0 ∧ piCost θ_pair 1 = 1 := by
  intro τH θ θ_pair
  have hmLr : 0 < m * L_qaly * r := mul_pos (mul_pos hm hL) hr
  have hLr : 0 < L_qaly * r := mul_pos hL hr
  have hp2_lt_p1 : p2 < p1 := lt_trans hp2_lt hp1_gt
  have hτ2 : 0 < τH p2 := by
    simp only [τH]
    have : 0 < L_qaly * r * p2 := mul_pos hLr hp2_pos
    linarith
  have hτ12 : τH p2 < τH p1 := by
    simp only [τH]
    nlinarith [mul_lt_mul_of_pos_left hp2_lt_p1 hLr]
  have hθ2 : θ p2 < 0 := by
    simp only [θ]
    have hmul : m * L_qaly * r * p2 < S0 - c0 := by
      rwa [lt_div_iff₀ hmLr, mul_comm] at hp2_lt
    linarith
  have hθ1 : 0 < θ p1 := by
    simp only [θ]
    have hmul : S0 - c0 < m * L_qaly * r * p1 := by
      rwa [div_lt_iff₀ hmLr, mul_comm] at hp1_gt
    linarith
  have hlam : 0 < lam := lt_trans hm hlam_gt_m
  have hevic2 : 0 < lam * τH p2 - θ p2 := by nlinarith
  have hevic12 : lam * τH p2 - θ p2 < lam * τH p1 - θ p1 := by
    simp only [τH, θ]
    have hdiff : (lam * (τ0 + L_qaly * r * p1) - (c0 - S0 + m * L_qaly * r * p1)) -
        (lam * (τ0 + L_qaly * r * p2) - (c0 - S0 + m * L_qaly * r * p2)) =
        (lam - m) * (L_qaly * r) * (p1 - p2) := by ring
    have hpos : 0 < (lam - m) * (L_qaly * r) * (p1 - p2) :=
      mul_pos (mul_pos (by linarith) hLr) (by linarith)
    linarith
  have hant := longevity_targeting_antinomy hlam hτ2 hτ12 hθ1 hθ2 hevic2 hevic12
  exact ⟨hτ2, hτ12, hθ2, hθ1, hant.2.1, hant.1, hant.2.2.1, hant.2.2.2.1,
    hant.2.2.2.2.1, hant.2.2.2.2.2⟩

/-- **Constructive Concrete Witness of the Longevity–Targeting Antinomy**:
With realistic clinical parameters (`λ = 100,000`, `m = 15,000`, `c₀ = 500`, `S₀ = 2,000`,
`τ₀ = 0.05`, `L_QALY = 5`, `r = 0.25`, `p₂ = 0.05 < p_crit = 0.08 < p₁ = 0.15`), higher baseline
mortality risk triggers the exact rank inversion between societal EVIC and self-financing. -/
theorem longevity_coupling_rank_inversion_witness :
    ∃ (lam m c0 S0 τ0 L_qaly r p1 p2 : ℝ),
      0 < m ∧ m < lam ∧ 0 < c0 ∧ c0 < S0 ∧ 0 < τ0 ∧ 0 < L_qaly ∧ 0 < r ∧
      0 < p2 ∧ p2 < p1 ∧ p1 < 1 ∧
      let τH := fun p => τ0 + L_qaly * r * p
      let θ := fun p => c0 - S0 + m * L_qaly * r * p
      let θ_pair : Fin 2 → ℝ := ![θ p1, θ p2]
      0 < τH p2 ∧ τH p2 < τH p1 ∧
      θ p2 < 0 ∧ 0 < θ p1 ∧
      0 < evic lam (τH p2) (θ p2) ∧
      evic lam (τH p2) (θ p2) < evic lam (τH p1) (θ p1) ∧
      evic 0 (τH p1) (θ p1) < 0 ∧ 0 < evic 0 (τH p2) (θ p2) ∧
      piCost θ_pair 0 = 0 ∧ piCost θ_pair 1 = 1 := by
  refine ⟨100000, 15000, 500, 2000, 1 / 20, 5, 1 / 4, 3 / 20, 1 / 20,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, ?_⟩
  exact longevity_coupling_rank_inversion_general
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end PreventiveTargetingParadox.Theorem4
