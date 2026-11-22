import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# Theorem 1: Proximity-to-Death Accounting, Multi-Payer Impossibility, and the Churn–Longevity Pincer

This module formalizes:
1. **Discrete Abel Summation-by-Parts Proximity-to-Death (TTD) Identity**: exact algebraic
   rewriting of survivor-year unrelated costs plus end-of-life (EOL) spike cost shifts in terms
   of survival gains `ΔS_t = S*_t - S_t` and effective marginal survival cost
   `μ_eff(t) = β_t m_t - (β_{t-1} K_{t-1} - β_t K_t)`.
2. **Three-Perspective Multi-Payer Accounting & Zero-Cost Impossibility**: Societal (`θ_soc`),
   Non-Capitated Employer/ASO (`θ_emp`), and Capitated MCO (`θ_MCO`, with capitation revenue `B`
   and risk-adjustment clawback `κ_RA * S_rel`), together with exact zero-cost impossibility bounds.
3. **Churn–Longevity Pincer (Non-Monotonicity Theorem)**: exact calculus analysis of
   `f(h) = c₀ - A e^{-h t₁} + M e^{-h t₂}`, establishing the unique interior minimizer `h*`,
   strict monotonicity on each side, the closed-form pincer impossibility bound, and a
   constructive Private-to-Medicare offloading paradox witness.
-/

namespace PreventiveTargetingParadox.Theorem1

open scoped BigOperators
open Real Set

/-! ## Part 1A: Discrete Abel Summation-by-Parts Proximity-to-Death (TTD) Identity -/

/-- Decedent mass in period `t`, defined as `S(t) - S(t + 1)`. -/
def decedents (S : ℕ → ℝ) (t : ℕ) : ℝ := S t - S (t + 1)

/-- Effective marginal survival cost at period `t`:
`μ_eff(t) = β(t) * m(t) - (β(t - 1) * K(t - 1) - β(t) * K(t))`,
capturing survivor-year unrelated spending minus the discounted EOL postponement saving. -/
def muEff (β m K : ℕ → ℝ) (t : ℕ) : ℝ :=
  β t * m t - (β (t - 1) * K (t - 1) - β t * K t)

/-- Pure discrete Abel summation by parts on `Finset.range T`. -/
lemma abel_summation_by_parts (a Δ : ℕ → ℝ) (T : ℕ) :
    ∑ t ∈ Finset.range T, a t * (Δ t - Δ (t + 1)) =
      if T = 0 then 0
      else a 0 * Δ 0 - a (T - 1) * Δ T +
        ∑ t ∈ Finset.Ico 1 T, (a t - a (t - 1)) * Δ t := by
  induction T with
  | zero => simp
  | succ n ih =>
    cases n with
    | zero =>
      simp
      ring
    | succ k =>
      rw [Finset.sum_range_succ, ih]
      simp only [Nat.add_eq_zero, one_ne_zero, and_false, ↓reduceIte, Nat.add_sub_cancel]
      have hle : 1 ≤ k + 1 := Nat.succ_le_succ (Nat.zero_le k)
      rw [Finset.sum_Ico_succ_top hle]
      simp only [Nat.add_sub_cancel]
      ring

/-- **Discrete Abel Summation-by-Parts Proximity-to-Death (TTD) Identity** over `Ico 1 T`.
Under boundary conditions `S(0) = S*(0) = 1` and `S(T) = S*(T) = 0`, the discounted sum of
survivor-year unrelated cost shifts plus EOL spike cost shifts equals the survival-gain-weighted
sum of effective marginal survival costs `μ_eff(t)`. -/
theorem abel_ttd_identity (T : ℕ) (β m K S Sstar : ℕ → ℝ)
    (hS0 : S 0 = 1) (hSstar0 : Sstar 0 = 1)
    (hST : S T = 0) (hSstarT : Sstar T = 0) :
    ∑ t ∈ Finset.range T,
        β t * ((Sstar t - S t) * m t + (decedents Sstar t - decedents S t) * K t) =
      ∑ t ∈ Finset.Ico 1 T, (Sstar t - S t) * muEff β m K t := by
  let Δ : ℕ → ℝ := fun t => Sstar t - S t
  have hΔ0 : Δ 0 = 0 := by simp [Δ, hS0, hSstar0]
  have hΔT : Δ T = 0 := by simp [Δ, hST, hSstarT]
  have hrew : ∀ t, β t * ((Sstar t - S t) * m t + (decedents Sstar t - decedents S t) * K t) =
      β t * m t * Δ t + (β t * K t) * (Δ t - Δ (t + 1)) := by
    intro t
    simp only [Δ, decedents]
    ring
  simp_rw [hrew, Finset.sum_add_distrib]
  rw [abel_summation_by_parts (fun t => β t * K t) Δ T]
  cases T with
  | zero => simp
  | succ n =>
    simp only [Nat.add_eq_zero, one_ne_zero, and_false, ↓reduceIte, hΔ0, hΔT,
      mul_zero, sub_zero, zero_add]
    have hfirst : ∑ t ∈ Finset.range (n + 1), β t * m t * Δ t =
        ∑ t ∈ Finset.Ico 1 (n + 1), β t * m t * Δ t := by
      rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (Nat.succ_pos n)]
      simp [hΔ0]
    rw [hfirst, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun t _ => ?_
    simp only [Δ, muEff]
    ring

/-- Equivalent form of the TTD identity over `Finset.range T` (the `t = 0` term vanishes
identically because `S*(0) - S(0) = 0`). -/
theorem abel_ttd_identity_range (T : ℕ) (β m K S Sstar : ℕ → ℝ)
    (hS0 : S 0 = 1) (hSstar0 : Sstar 0 = 1)
    (hST : S T = 0) (hSstarT : Sstar T = 0) :
    ∑ t ∈ Finset.range T,
        β t * ((Sstar t - S t) * m t + (decedents Sstar t - decedents S t) * K t) =
      ∑ t ∈ Finset.range T, (Sstar t - S t) * muEff β m K t := by
  rw [abel_ttd_identity T β m K S Sstar hS0 hSstar0 hST hSstarT]
  cases T with
  | zero => simp
  | succ n =>
    rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot (Nat.succ_pos n)]
    simp [hS0, hSstar0]

/-- If an intervention improves survival (`S*(t) ≥ S(t)`) and effective marginal survival cost is
nonnegative (`μ_eff(t) ≥ 0`) at every interior period `1 ≤ t < T`, then total TTD-adjusted
unrelated survival cost is nonnegative. -/
theorem abel_ttd_nonneg_of_muEff_nonneg (T : ℕ) (β m K S Sstar : ℕ → ℝ)
    (hS0 : S 0 = 1) (hSstar0 : Sstar 0 = 1)
    (hST : S T = 0) (hSstarT : Sstar T = 0)
    (hΔ : ∀ t ∈ Finset.Ico 1 T, 0 ≤ Sstar t - S t)
    (hμ : ∀ t ∈ Finset.Ico 1 T, 0 ≤ muEff β m K t) :
    0 ≤ ∑ t ∈ Finset.range T,
      β t * ((Sstar t - S t) * m t + (decedents Sstar t - decedents S t) * K t) := by
  rw [abel_ttd_identity T β m K S Sstar hS0 hSstar0 hST hSstarT]
  exact Finset.sum_nonneg fun t ht => mul_nonneg (hΔ t ht) (hμ t ht)

/-- Under geometric discounting `β(t) = δ^t` and constant EOL spike `K(t) = K₀`, the effective
marginal survival cost for `t ≥ 1` simplifies to `δ^(t - 1) * (δ * m(t) - (1 - δ) * K₀)`,
identifying the exact sign threshold `m(t) ≥ ((1 - δ)/δ) * K₀`. -/
theorem muEff_geometric_constant_eol (δ K0 : ℝ) (m : ℕ → ℝ) {t : ℕ} (ht : 1 ≤ t) :
    muEff (fun k => δ ^ k) m (fun _ => K0) t =
      δ ^ (t - 1) * (δ * m t - (1 - δ) * K0) := by
  unfold muEff
  dsimp only
  have hpow : δ ^ t = δ ^ (t - 1) * δ := by
    conv_lhs => rw [← Nat.sub_add_cancel ht, pow_succ]
  rw [hpow]
  ring

/-! ## Part 1B: Three-Perspective Multi-Payer Accounting & Zero-Cost Impossibility -/

/-- Societal net cost: `θ_soc = c₀ - S_rel + M_unrel`. -/
def thetaSoc (c0 S_rel M_unrel : ℝ) : ℝ := c0 - S_rel + M_unrel

/-- Non-capitated employer / ASO net cost over the retained horizon:
`θ_emp = c₀ - S_rel_emp + M_unrel_emp`. -/
def thetaEmp (c0 S_rel_emp M_unrel_emp : ℝ) : ℝ := c0 - S_rel_emp + M_unrel_emp

/-- Capitated MCO net cost, subtracting survival-year capitation revenue `B` and adding
risk-adjustment clawback `κ_RA * S_rel`:
`θ_MCO = c₀ - S_rel + M_unrel - B + κ_RA * S_rel`. -/
def thetaMCO (c0 S_rel M_unrel B_cap kappa_RA : ℝ) : ℝ :=
  c0 - S_rel + M_unrel - B_cap + kappa_RA * S_rel

/-- Non-capitated employer/ASO accounting is the special case of MCO accounting with zero
capitation revenue (`B = 0`) and zero risk-adjustment clawback (`κ_RA = 0`). -/
theorem thetaEmp_eq_thetaMCO_zero (c0 S_emp M_emp : ℝ) :
    thetaEmp c0 S_emp M_emp = thetaMCO c0 S_emp M_emp 0 0 := by
  unfold thetaEmp thetaMCO
  ring

/-- Equivalent risk-adjusted decomposition of MCO net cost:
`θ_MCO = c₀ - (1 - κ_RA) * S_rel + (M_unrel - B)`. -/
theorem thetaMCO_decomposition (c0 S_rel M_unrel B_cap kappa_RA : ℝ) :
    thetaMCO c0 S_rel M_unrel B_cap kappa_RA =
      c0 - (1 - kappa_RA) * S_rel + (M_unrel - B_cap) := by
  unfold thetaMCO
  ring

/-- **Zero-Cost Impossibility Corollary (Societal Perspective)**:
Even for a free intervention (`c₀ = 0`), societal net cost is nonnegative whenever unrelated
survival costs equal or exceed avoided disease-related costs (`M_unrel ≥ S_rel`). -/
theorem zero_cost_impossibility_soc {S_rel M_unrel : ℝ} (h : S_rel ≤ M_unrel) :
    0 ≤ thetaSoc 0 S_rel M_unrel := by
  unfold thetaSoc
  linarith

/-- Ratio form of Societal Zero-Cost Impossibility (`Ψ = M_unrel / S_rel ≥ 1`). -/
theorem zero_cost_impossibility_soc_ratio {S_rel M_unrel : ℝ}
    (hS : 0 < S_rel) (hΨ : 1 ≤ M_unrel / S_rel) :
    0 ≤ thetaSoc 0 S_rel M_unrel := by
  have h : S_rel ≤ M_unrel := by
    rwa [le_div_iff₀ hS, one_mul] at hΨ
  exact zero_cost_impossibility_soc h

/-- **Zero-Cost Impossibility Corollary (Employer/ASO Perspective)**:
Even at `c₀ = 0`, employer net cost is nonnegative whenever `M_emp / S_emp ≥ 1`. -/
theorem zero_cost_impossibility_emp_ratio {S_emp M_emp : ℝ}
    (hS : 0 < S_emp) (hΨ : 1 ≤ M_emp / S_emp) :
    0 ≤ thetaEmp 0 S_emp M_emp := by
  unfold thetaEmp
  have h : S_emp ≤ M_emp := by
    rwa [le_div_iff₀ hS, one_mul] at hΨ
  linarith

/-- **Zero-Cost Impossibility Corollary (Capitated MCO Perspective)**:
Even for a free intervention (`c₀ = 0`) under risk-adjustment clawback `κ ∈ [0, 1]`, MCO net
cost is nonnegative whenever the net unrelated-cost ratio satisfies
`Ψ_net = (M_unrel - B) / S_rel ≥ 1 - κ`. -/
theorem zero_cost_impossibility_mco {S_rel M_unrel B_cap kappa_RA : ℝ}
    (hS : 0 < S_rel) (_hκ0 : 0 ≤ kappa_RA) (_hκ1 : kappa_RA ≤ 1)
    (hΨ_net : 1 - kappa_RA ≤ (M_unrel - B_cap) / S_rel) :
    0 ≤ thetaMCO 0 S_rel M_unrel B_cap kappa_RA := by
  rw [thetaMCO_decomposition]
  have h : (1 - kappa_RA) * S_rel ≤ M_unrel - B_cap := by
    rwa [le_div_iff₀ hS] at hΨ_net
  linarith

/-- Strict positivity under positive intervention cost or strictly exceeding the MCO threshold. -/
theorem mco_pos_of_strict_threshold {c0 S_rel M_unrel B_cap kappa_RA : ℝ}
    (hc0 : 0 ≤ c0) (hS : 0 < S_rel)
    (hΨ_net : 1 - kappa_RA < (M_unrel - B_cap) / S_rel) :
    0 < thetaMCO c0 S_rel M_unrel B_cap kappa_RA := by
  rw [thetaMCO_decomposition]
  have h : (1 - kappa_RA) * S_rel < M_unrel - B_cap := by
    rwa [lt_div_iff₀ hS] at hΨ_net
  linarith

/-! ## Part 1C: Churn–Longevity Pincer (Non-Monotonicity Theorem) -/

/-- Payer net cost as a function of disenrollment/churn hazard `h`:
`f(h) = c₀ - A * exp(-h * t₁) + M * exp(-h * t₂)`. -/
noncomputable def churnNetCost (c0 A M t1 t2 h : ℝ) : ℝ :=
  c0 - A * Real.exp (-h * t1) + M * Real.exp (-h * t2)

/-- Exact derivative `f'(h) = A * t₁ * exp(-h * t₁) - M * t₂ * exp(-h * t₂)`. -/
noncomputable def churnNetCostDeriv (A M t1 t2 h : ℝ) : ℝ :=
  A * t1 * Real.exp (-h * t1) - M * t2 * Real.exp (-h * t2)

/-- Unique interior critical churn hazard `h* = log(M * t₂ / (A * t₁)) / (t₂ - t₁)`. -/
noncomputable def hStar (A M t1 t2 : ℝ) : ℝ :=
  Real.log (M * t2 / (A * t1)) / (t2 - t1)

/-- Closed-form maximum churn-induced gross offset at `h*`:
`A * (1 - t₁ / t₂) * (A * t₁ / (M * t₂)) ^ (t₁ / (t₂ - t₁))`. -/
noncomputable def pincerMaxOffset (A M t1 t2 : ℝ) : ℝ :=
  A * (1 - t1 / t2) * (A * t1 / (M * t2)) ^ (t1 / (t2 - t1))

/-- Exact derivative of `churnNetCost` at every `h ∈ ℝ`. -/
theorem hasDerivAt_churnNetCost (c0 A M t1 t2 h : ℝ) :
    HasDerivAt (churnNetCost c0 A M t1 t2) (churnNetCostDeriv A M t1 t2 h) h := by
  unfold churnNetCost churnNetCostDeriv
  have h1 : HasDerivAt (fun x : ℝ => -x * t1) (-t1) h := by
    simpa [one_mul] using (hasDerivAt_id h).neg.mul_const t1
  have h2 : HasDerivAt (fun x : ℝ => -x * t2) (-t2) h := by
    simpa [one_mul] using (hasDerivAt_id h).neg.mul_const t2
  have he1 : HasDerivAt (fun x : ℝ => A * Real.exp (-x * t1))
      (A * (Real.exp (-h * t1) * (-t1))) h :=
    (h1.exp).const_mul A
  have he2 : HasDerivAt (fun x : ℝ => M * Real.exp (-x * t2))
      (M * (Real.exp (-h * t2) * (-t2))) h :=
    (h2.exp).const_mul M
  have htot := ((hasDerivAt_const h c0).sub he1).add he2
  convert htot using 1
  ring

/-- The critical churn hazard `h*` is strictly positive whenever `0 < t₁ < t₂` and
`0 < A * t₁ < M * t₂`. -/
theorem hStar_pos {A M t1 t2 : ℝ} (ht1 : 0 < t1) (ht12 : t1 < t2)
    (hA : 0 < A) (hMt : A * t1 < M * t2) :
    0 < hStar A M t1 t2 := by
  unfold hStar
  have hAt1 : 0 < A * t1 := mul_pos hA ht1
  have hratio : 1 < M * t2 / (A * t1) := (one_lt_div hAt1).2 hMt
  exact div_pos (Real.log_pos hratio) (sub_pos.2 ht12)

/-- Factorization of `f'(h)` via `exp(-h * t₂)` for sign analysis. -/
lemma churnNetCostDeriv_factor (A M t1 t2 h : ℝ) :
    churnNetCostDeriv A M t1 t2 h =
      Real.exp (-h * t2) * (A * t1 * Real.exp (h * (t2 - t1)) - M * t2) := by
  unfold churnNetCostDeriv
  have hexp : Real.exp (-h * t2) * Real.exp (h * (t2 - t1)) = Real.exp (-h * t1) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc
    A * t1 * Real.exp (-h * t1) - M * t2 * Real.exp (-h * t2)
        = A * t1 * (Real.exp (-h * t2) * Real.exp (h * (t2 - t1))) -
            M * t2 * Real.exp (-h * t2) := by rw [hexp]
    _ = Real.exp (-h * t2) * (A * t1 * Real.exp (h * (t2 - t1)) - M * t2) := by ring

/-- `exp(h* * (t₂ - t₁)) = (M * t₂) / (A * t₁)`. -/
lemma exp_hStar_mul_sub {A M t1 t2 : ℝ} (ht1 : 0 < t1) (ht12 : t1 < t2)
    (hA : 0 < A) (hM : 0 < M) :
    Real.exp (hStar A M t1 t2 * (t2 - t1)) = M * t2 / (A * t1) := by
  unfold hStar
  have hdiff : t2 - t1 ≠ 0 := ne_of_gt (sub_pos.2 ht12)
  rw [div_mul_cancel₀ _ hdiff]
  have ht2 : 0 < t2 := lt_trans ht1 ht12
  exact Real.exp_log (div_pos (mul_pos hM ht2) (mul_pos hA ht1))

/-- **Part (a) of the Churn–Longevity Pincer**: `h*` is the unique critical point of `f`. -/
theorem churnNetCostDeriv_eq_zero_iff {A M t1 t2 : ℝ} (ht1 : 0 < t1) (ht12 : t1 < t2)
    (hA : 0 < A) (hM : 0 < M) (h : ℝ) :
    churnNetCostDeriv A M t1 t2 h = 0 ↔ h = hStar A M t1 t2 := by
  have ht2 : 0 < t2 := lt_trans ht1 ht12
  have hAt1 : 0 < A * t1 := mul_pos hA ht1
  have hMt2 : 0 < M * t2 := mul_pos hM ht2
  have hdiff : t2 - t1 ≠ 0 := ne_of_gt (sub_pos.2 ht12)
  rw [churnNetCostDeriv_factor]
  constructor
  · intro hzero
    rcases mul_eq_zero.1 hzero with hexp0 | hfac
    · exact absurd hexp0 (ne_of_gt (Real.exp_pos _))
    · have heq : Real.exp (h * (t2 - t1)) = M * t2 / (A * t1) := by
        rw [eq_div_iff (ne_of_gt hAt1)]
        linarith
      have hlog : h * (t2 - t1) = Real.log (M * t2 / (A * t1)) := by
        rw [← heq, Real.log_exp]
      unfold hStar
      exact (eq_div_iff hdiff).2 hlog
  · intro hh
    rw [hh, exp_hStar_mul_sub ht1 ht12 hA hM]
    have hmul : A * t1 * (M * t2 / (A * t1)) = M * t2 :=
      mul_div_cancel₀ _ (ne_of_gt hAt1)
    rw [hmul, sub_self, mul_zero]

/-- Derivative is strictly negative for `h < h*`. -/
theorem churnNetCostDeriv_neg_of_lt_hStar {A M t1 t2 h : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M)
    (hh : h < hStar A M t1 t2) :
    churnNetCostDeriv A M t1 t2 h < 0 := by
  rw [churnNetCostDeriv_factor]
  have hAt1 : 0 < A * t1 := mul_pos hA ht1
  have hdiff : 0 < t2 - t1 := sub_pos.2 ht12
  have hlt : h * (t2 - t1) < hStar A M t1 t2 * (t2 - t1) :=
    mul_lt_mul_of_pos_right hh hdiff
  have hexp_lt : Real.exp (h * (t2 - t1)) < M * t2 / (A * t1) := by
    rw [← exp_hStar_mul_sub ht1 ht12 hA hM]
    exact Real.exp_lt_exp.2 hlt
  have hfac_neg : A * t1 * Real.exp (h * (t2 - t1)) - M * t2 < 0 := by
    have hmul : A * t1 * Real.exp (h * (t2 - t1)) < M * t2 := by
      rwa [lt_div_iff₀ hAt1, mul_comm] at hexp_lt
    linarith
  exact mul_neg_of_pos_of_neg (Real.exp_pos _) hfac_neg

/-- Derivative is strictly positive for `h > h*`. -/
theorem churnNetCostDeriv_pos_of_gt_hStar {A M t1 t2 h : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M)
    (hh : hStar A M t1 t2 < h) :
    0 < churnNetCostDeriv A M t1 t2 h := by
  rw [churnNetCostDeriv_factor]
  have hAt1 : 0 < A * t1 := mul_pos hA ht1
  have hdiff : 0 < t2 - t1 := sub_pos.2 ht12
  have hgt : hStar A M t1 t2 * (t2 - t1) < h * (t2 - t1) :=
    mul_lt_mul_of_pos_right hh hdiff
  have hexp_gt : M * t2 / (A * t1) < Real.exp (h * (t2 - t1)) := by
    rw [← exp_hStar_mul_sub ht1 ht12 hA hM]
    exact Real.exp_lt_exp.2 hgt
  have hfac_pos : 0 < A * t1 * Real.exp (h * (t2 - t1)) - M * t2 := by
    have hmul : M * t2 < Real.exp (h * (t2 - t1)) * (A * t1) :=
      (div_lt_iff₀ hAt1).1 hexp_gt
    linarith
  exact mul_pos (Real.exp_pos _) hfac_pos

/-- **Part (b1)**: `f` is strictly decreasing on `[0, h*]`. -/
theorem churnNetCost_strictAntiOn_Icc (c0 : ℝ) {A M t1 t2 : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M) :
    StrictAntiOn (churnNetCost c0 A M t1 t2) (Set.Icc 0 (hStar A M t1 t2)) := by
  apply strictAntiOn_of_deriv_neg (convex_Icc 0 (hStar A M t1 t2))
  · exact (continuous_iff_continuousAt.2 fun h =>
      (hasDerivAt_churnNetCost c0 A M t1 t2 h).continuousAt).continuousOn
  · intro x hx
    rw [interior_Icc] at hx
    rw [(hasDerivAt_churnNetCost c0 A M t1 t2 x).deriv]
    exact churnNetCostDeriv_neg_of_lt_hStar ht1 ht12 hA hM hx.2

/-- **Part (b2)**: `f` is strictly increasing on `[h*, ∞)`. -/
theorem churnNetCost_strictMonoOn_Ici (c0 : ℝ) {A M t1 t2 : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M) :
    StrictMonoOn (churnNetCost c0 A M t1 t2) (Set.Ici (hStar A M t1 t2)) := by
  apply strictMonoOn_of_deriv_pos (convex_Ici (hStar A M t1 t2))
  · exact (continuous_iff_continuousAt.2 fun h =>
      (hasDerivAt_churnNetCost c0 A M t1 t2 h).continuousAt).continuousOn
  · intro x hx
    rw [interior_Ici] at hx
    rw [(hasDerivAt_churnNetCost c0 A M t1 t2 x).deriv]
    exact churnNetCostDeriv_pos_of_gt_hStar ht1 ht12 hA hM hx

/-- **Part (b3)**: `h*` is the global minimizer of `f` on `[0, ∞)`. -/
theorem churnNetCost_global_min_on_nonneg (c0 : ℝ) {A M t1 t2 : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M) (hMt : A * t1 < M * t2)
    {h : ℝ} (hh : 0 ≤ h) :
    churnNetCost c0 A M t1 t2 (hStar A M t1 t2) ≤ churnNetCost c0 A M t1 t2 h := by
  have hhStar : 0 < hStar A M t1 t2 := hStar_pos ht1 ht12 hA hMt
  rcases le_or_gt h (hStar A M t1 t2) with hle | hgt
  · rcases eq_or_lt_of_le hle with rfl | hlt
    · exact le_rfl
    · exact le_of_lt ((churnNetCost_strictAntiOn_Icc c0 ht1 ht12 hA hM)
        ⟨hh, hle⟩ ⟨le_of_lt hhStar, le_rfl⟩ hlt)
  · exact le_of_lt ((churnNetCost_strictMonoOn_Ici c0 ht1 ht12 hA hM)
      Set.left_mem_Ici (Set.mem_Ici.2 (le_of_lt hgt)) hgt)

/-- **Part (c1) — Exact Closed-Form Minimum Value**:
`f(h*) = c₀ - A * (1 - t₁ / t₂) * (A * t₁ / (M * t₂)) ^ (t₁ / (t₂ - t₁))`. -/
theorem churnNetCost_at_hStar_closed_form (c0 : ℝ) {A M t1 t2 : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M) :
    churnNetCost c0 A M t1 t2 (hStar A M t1 t2) =
      c0 - pincerMaxOffset A M t1 t2 := by
  have ht2 : 0 < t2 := lt_trans ht1 ht12
  have hAt1 : 0 < A * t1 := mul_pos hA ht1
  have hMt2 : 0 < M * t2 := mul_pos hM ht2
  have hderiv0 : churnNetCostDeriv A M t1 t2 (hStar A M t1 t2) = 0 :=
    (churnNetCostDeriv_eq_zero_iff ht1 ht12 hA hM (hStar A M t1 t2)).2 rfl
  have hMexp : M * Real.exp (-hStar A M t1 t2 * t2) =
      A * (t1 / t2) * Real.exp (-hStar A M t1 t2 * t1) := by
    unfold churnNetCostDeriv at hderiv0
    have hmul : M * t2 * Real.exp (-hStar A M t1 t2 * t2) =
        A * t1 * Real.exp (-hStar A M t1 t2 * t1) := by linarith
    have ht2ne : t2 ≠ 0 := ne_of_gt ht2
    calc
      M * Real.exp (-hStar A M t1 t2 * t2)
          = (M * t2 * Real.exp (-hStar A M t1 t2 * t2)) / t2 := by field_simp; ring
      _ = (A * t1 * Real.exp (-hStar A M t1 t2 * t1)) / t2 := by rw [hmul]
      _ = A * (t1 / t2) * Real.exp (-hStar A M t1 t2 * t1) := by ring
  have hexp_rpow : Real.exp (-hStar A M t1 t2 * t1) =
      (A * t1 / (M * t2)) ^ (t1 / (t2 - t1)) := by
    rw [Real.rpow_def_of_pos (div_pos hAt1 hMt2)]
    congr 1
    unfold hStar
    have hlog_inv : Real.log (A * t1 / (M * t2)) = -Real.log (M * t2 / (A * t1)) := by
      rw [← Real.log_inv, inv_div]
    rw [hlog_inv]
    ring
  unfold churnNetCost pincerMaxOffset
  rw [hMexp, hexp_rpow]
  ring

/-- **Part (c2) — Pincer Impossibility Bound**:
If upfront intervention cost exceeds `pincerMaxOffset A M t₁ t₂`, then net cost is strictly
positive for EVERY churn hazard `h ≥ 0` (no payer at any churn rate can save money). -/
theorem churn_pincer_impossibility_bound {c0 A M t1 t2 : ℝ}
    (ht1 : 0 < t1) (ht12 : t1 < t2) (hA : 0 < A) (hM : 0 < M) (hMt : A * t1 < M * t2)
    (hc0 : pincerMaxOffset A M t1 t2 < c0) :
    ∀ h : ℝ, 0 ≤ h → 0 < churnNetCost c0 A M t1 t2 h := by
  intro h hh
  have hmin := churnNetCost_global_min_on_nonneg c0 ht1 ht12 hA hM hMt hh
  rw [churnNetCost_at_hStar_closed_form c0 ht1 ht12 hA hM] at hmin
  linarith

/-- **Part (d) — Constructive Private-to-Medicare Offloading Paradox Witness**:
There exist positive parameters `(c₀, A, M, t₁, t₂)` such that:
- Zero-churn societal / Medicare net cost is strictly positive (`f(0) > 0`),
- High-churn Medicaid net cost is strictly positive (`c₀ > 0` and `∀ h ≥ H, f(h) > 0`),
- YET intermediate-churn commercial net cost at `h* > 0` is strictly negative (`f(h*) < 0`),
  because the commercial payer harvests early disease offsets while offloading late-life
  unrelated survival costs to Medicare. -/
theorem private_to_medicare_offloading_witness :
    ∃ c0 A M t1 t2 : ℝ,
      0 < c0 ∧ 0 < A ∧ 0 < M ∧ 0 < t1 ∧ t1 < t2 ∧ A * t1 < M * t2 ∧
      0 < churnNetCost c0 A M t1 t2 0 ∧
      churnNetCost c0 A M t1 t2 (hStar A M t1 t2) < 0 ∧
      (∃ H : ℝ, hStar A M t1 t2 < H ∧ ∀ h : ℝ, H ≤ h → 0 < churnNetCost c0 A M t1 t2 h) := by
  refine ⟨5, 40, 50, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, ?_, ?_, ?_⟩
  · unfold churnNetCost
    norm_num
  · rw [churnNetCost_at_hStar_closed_form (5 : ℝ) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)]
    unfold pincerMaxOffset
    norm_num
  · refine ⟨Real.log 8, ?_, ?_⟩
    · unfold hStar
      have harg : (50 : ℝ) * 2 / (40 * 1) = 5 / 2 := by norm_num
      rw [harg]
      have hdiv : Real.log (5 / 2 : ℝ) / (2 - 1) = Real.log (5 / 2 : ℝ) := by norm_num
      rw [hdiv]
      exact Real.log_lt_log (by norm_num) (by norm_num)
    · intro h hh
      unfold churnNetCost
      have hexp1 : Real.exp (-h * 1) ≤ (1 : ℝ) / 8 := by
        have hneg : -h * 1 ≤ -Real.log 8 := by linarith
        calc
          Real.exp (-h * 1) ≤ Real.exp (-Real.log 8) := Real.exp_le_exp.2 hneg
          _ = (Real.exp (Real.log 8))⁻¹ := Real.exp_neg _
          _ = 1 / 8 := by rw [Real.exp_log (by norm_num), one_div]
      have hpos2 : 0 < 50 * Real.exp (-h * 2) := mul_pos (by norm_num) (Real.exp_pos _)
      linarith

end PreventiveTargetingParadox.Theorem1
