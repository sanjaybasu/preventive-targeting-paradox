import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.Calculus.Monotone
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Data.Real.Pi.Bounds
import Mathlib.MeasureTheory.Integral.IntegralEqImproper
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic

/-!
# Theorem 3: Winner's Curse Sandwich and Exact Gaussian Mills-Ratio Sign-Flip Phase Transition

This module formalizes:
1. **Distribution-Free Winner's Curse Sandwich**:
   On any finite probability space with conditionally unbiased estimation error `ε`,
   `𝔼[min(θ̃, 0)] ≤ 𝔼[min(θ, 0)] ≤ 𝔼[1{θ̃ < 0} θ]`,
   proved both in general fiber-unbiased single-space form and in independent product-space form,
   together with the **Null-World Mirage Corollary** (`θ ≡ μ ≥ 0` produces strictly negative
   apparent in-sample cost while realized cost stays nonnegative).
2. **Exact Gaussian Mills-Ratio Bounds (Birnbaum Lower & Komatsu–Boyd Upper)**:
   Defining `φ(t) = exp(-t²/2) / √(2π)`, `Φ̄(z) = ∫_{t > z} φ(t) dt`, and `M(z) = Φ̄(z) / φ(z)`,
   proves by exact calculus for all `z > 0`:
   `2 / (√(z² + 4) + z) < M(z) < 2 / (√(z² + 8/π) + z)`.
3. **Exact Gaussian Mills-Ratio Sign-Flip Phase Transition Sandwich**:
   For `ΔC(ρ) = μ Φ̄(γ √ρ) - σ √ρ φ(γ √ρ)` with `μ > 0, σ > 0, γ = μ / σ`:
   - `ΔC(ρ) > 0` for all `0 ≤ ρ ≤ μ² / (μ² + σ²)` (equivalently noise SD `τ ≥ σ² / μ`);
   - `ΔC(ρ) < 0` for all `μ² / (μ² + (2/π) σ²) ≤ ρ`;
   - `ΔC` is strictly decreasing on `[0, 1]`, yielding a unique sign-flip threshold
     `ρ* ∈ (μ² / (μ² + σ²), μ² / (μ² + (2/π) σ²))` with `ΔC(ρ*) = 0`.
-/

namespace PreventiveTargetingParadox.Theorem3

open scoped BigOperators
open Real Set Filter MeasureTheory

/-! ## Part 3A: Distribution-Free Winner's Curse Sandwich -/

/-- Realized net cost of selecting everyone whose noisy estimate satisfies `θ̃_i < 0`. -/
noncomputable def realizedCost {ι : Type*} (θ θtilde : ι → ℝ) (i : ι) : ℝ :=
  if θtilde i < 0 then θ i else 0

/-- Pointwise right-hand inequality: `min(θ, 0) ≤ 1{θ̃ < 0} * θ` holds everywhere without any
unbiasedness assumption. -/
lemma min_zero_le_realizedCost (θval θtildeVal : ℝ) :
    min θval 0 ≤ if θtildeVal < 0 then θval else 0 := by
  split_ifs
  · exact min_le_left _ _
  · exact min_le_right _ _

/-- Right-hand side of the Winner's Curse Sandwich: oracle cost is always `≤` realized cost
of any noisy threshold rule `1{θ̃ < 0}`. -/
theorem winners_curse_right_ineq {ι : Type*} (s : Finset ι) (w θ θtilde : ι → ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) :
    ∑ i ∈ s, w i * min (θ i) 0 ≤ ∑ i ∈ s, w i * realizedCost θ θtilde i := by
  refine Finset.sum_le_sum fun i hi => ?_
  exact mul_le_mul_of_nonneg_left (min_zero_le_realizedCost (θ i) (θtilde i)) (hw i hi)

/-- **Distribution-Free Winner's Curse Sandwich (Conditional Sublevel Unbiasedness Form)**:
Whenever estimation noise `ε` has zero weighted expectation on the truly cost-saving subpopulation
`{i ∈ s | θ_i < 0}`, both sides of the Winner's Curse sandwich hold:
`𝔼[min(θ + ε, 0)] ≤ 𝔼[min(θ, 0)] ≤ 𝔼[1{θ + ε < 0} θ]`. -/
theorem winners_curse_sandwich_conditional {ι : Type*} [DecidableEq ι]
    (s : Finset ι) (w θ ε : ι → ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i)
    (h_unbiased : ∑ i ∈ s.filter (fun i => θ i < 0), w i * ε i = 0) :
    ∑ i ∈ s, w i * min (θ i + ε i) 0 ≤ ∑ i ∈ s, w i * min (θ i) 0 ∧
    ∑ i ∈ s, w i * min (θ i) 0 ≤ ∑ i ∈ s, w i * realizedCost θ (fun i => θ i + ε i) i := by
  refine ⟨?_, winners_curse_right_ineq s w θ (fun i => θ i + ε i) hw⟩
  rw [← Finset.sum_filter_add_sum_filter_not s (fun i => θ i < 0) (fun i => w i * min (θ i + ε i) 0)]
  rw [← Finset.sum_filter_add_sum_filter_not s (fun i => θ i < 0) (fun i => w i * min (θ i) 0)]
  have hneg : ∑ i ∈ s.filter (fun i => θ i < 0), w i * min (θ i + ε i) 0 ≤
      ∑ i ∈ s.filter (fun i => θ i < 0), w i * min (θ i) 0 := by
    calc
      ∑ i ∈ s.filter (fun i => θ i < 0), w i * min (θ i + ε i) 0
          ≤ ∑ i ∈ s.filter (fun i => θ i < 0), w i * (θ i + ε i) := by
            refine Finset.sum_le_sum fun i hi => ?_
            exact mul_le_mul_of_nonneg_left (min_le_left _ _) (hw i (Finset.mem_of_mem_filter i hi))
      _ = (∑ i ∈ s.filter (fun i => θ i < 0), w i * θ i) +
            ∑ i ∈ s.filter (fun i => θ i < 0), w i * ε i := by
            rw [← Finset.sum_add_distrib]
            refine Finset.sum_congr rfl fun i _ => ?_
            ring
      _ = ∑ i ∈ s.filter (fun i => θ i < 0), w i * min (θ i) 0 := by
            rw [h_unbiased, add_zero]
            refine Finset.sum_congr rfl fun i hi => ?_
            have hlt : θ i < 0 := (Finset.mem_filter.1 hi).2
            rw [min_eq_left (le_of_lt hlt)]
  have hpos : ∑ i ∈ s.filter (fun i => ¬ θ i < 0), w i * min (θ i + ε i) 0 ≤
      ∑ i ∈ s.filter (fun i => ¬ θ i < 0), w i * min (θ i) 0 := by
    refine Finset.sum_le_sum fun i hi => ?_
    have hge : 0 ≤ θ i := not_lt.1 (Finset.mem_filter.1 hi).2
    rw [min_eq_right hge]
    exact mul_le_mul_of_nonneg_left (min_le_right _ _) (hw i (Finset.mem_of_mem_filter i hi))
  exact add_le_add hneg hpos

/-- **Distribution-Free Winner's Curse Sandwich (Fiberwise Conditional Unbiasedness Form)**:
If `∑_{i ∈ s : θ_i = v} w_i ε_i = 0` for every value `v ∈ ℝ` (i.e. `𝔼[ε | θ] = 0`), then
`𝔼[min(θ + ε, 0)] ≤ 𝔼[min(θ, 0)] ≤ 𝔼[1{θ + ε < 0} θ]`. -/
theorem winners_curse_sandwich_fiber_unbiased {ι : Type*} [DecidableEq ι]
    (s : Finset ι) (w θ ε : ι → ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i)
    (h_fiber : ∀ v : ℝ, ∑ i ∈ s.filter (fun i => θ i = v), w i * ε i = 0) :
    ∑ i ∈ s, w i * min (θ i + ε i) 0 ≤ ∑ i ∈ s, w i * min (θ i) 0 ∧
    ∑ i ∈ s, w i * min (θ i) 0 ≤ ∑ i ∈ s, w i * realizedCost θ (fun i => θ i + ε i) i := by
  apply winners_curse_sandwich_conditional s w θ ε hw
  let sneg := s.filter (fun i => θ i < 0)
  have hmaps : ∀ i ∈ sneg, θ i ∈ sneg.image θ := fun i hi => Finset.mem_image_of_mem θ hi
  rw [← Finset.sum_fiberwise_of_maps_to hmaps (fun i => w i * ε i)]
  refine Finset.sum_eq_zero fun v hv => ?_
  rcases Finset.mem_image.1 hv with ⟨i0, hi0, rfl⟩
  have hv_neg : θ i0 < 0 := (Finset.mem_filter.1 hi0).2
  have hfeq : sneg.filter (fun i => θ i = θ i0) = s.filter (fun i => θ i = θ i0) := by
    ext i
    simp only [sneg, Finset.mem_filter]
    constructor
    · rintro ⟨⟨his, _⟩, heq⟩; exact ⟨his, heq⟩
    · rintro ⟨his, heq⟩; exact ⟨⟨his, heq ▸ hv_neg⟩, heq⟩
  rw [hfeq]
  exact h_fiber (θ i0)

/-- **Distribution-Free Winner's Curse Sandwich (Product-Space Independent / Cond-Unbiased Noise)**:
For finite patient types `a ∈ s₁` with weights `p a ≥ 0` and noise realizations `b ∈ s₂` with
conditional weights `q a b ≥ 0`, `∑_b q a b = 1`, and `∑_b q a b * ε a b = 0`,
`𝔼[min(θ + ε, 0)] ≤ 𝔼[min(θ, 0)] ≤ 𝔼[1{θ + ε < 0} θ]`. -/
theorem winners_curse_sandwich_product {α β : Type*} (s1 : Finset α) (s2 : Finset β)
    (p : α → ℝ) (q : α → β → ℝ) (θ : α → ℝ) (ε : α → β → ℝ)
    (hp : ∀ a ∈ s1, 0 ≤ p a)
    (hq : ∀ a ∈ s1, ∀ b ∈ s2, 0 ≤ q a b)
    (hq_sum : ∀ a ∈ s1, ∑ b ∈ s2, q a b = 1)
    (hε_zero : ∀ a ∈ s1, ∑ b ∈ s2, q a b * ε a b = 0) :
    (∑ a ∈ s1, p a * ∑ b ∈ s2, q a b * min (θ a + ε a b) 0) ≤
      (∑ a ∈ s1, p a * min (θ a) 0) ∧
    (∑ a ∈ s1, p a * min (θ a) 0) ≤
      (∑ a ∈ s1, p a * ∑ b ∈ s2, q a b * (if θ a + ε a b < 0 then θ a else 0)) := by
  constructor
  · refine Finset.sum_le_sum fun a ha => ?_
    refine mul_le_mul_of_nonneg_left ?_ (hp a ha)
    apply le_min
    · calc
        ∑ b ∈ s2, q a b * min (θ a + ε a b) 0
            ≤ ∑ b ∈ s2, q a b * (θ a + ε a b) :=
              Finset.sum_le_sum fun b hb =>
                mul_le_mul_of_nonneg_left (min_le_left _ _) (hq a ha b hb)
        _ = (∑ b ∈ s2, q a b) * θ a + ∑ b ∈ s2, q a b * ε a b := by
              rw [Finset.sum_mul, ← Finset.sum_add_distrib]
              refine Finset.sum_congr rfl fun b _ => ?_; ring
        _ = θ a := by rw [hq_sum a ha, hε_zero a ha]; ring
    · calc
        ∑ b ∈ s2, q a b * min (θ a + ε a b) 0
            ≤ ∑ b ∈ s2, q a b * 0 :=
              Finset.sum_le_sum fun b hb =>
                mul_le_mul_of_nonneg_left (min_le_right _ _) (hq a ha b hb)
        _ = 0 := by simp
  · refine Finset.sum_le_sum fun a ha => ?_
    refine mul_le_mul_of_nonneg_left ?_ (hp a ha)
    calc
      min (θ a) 0 = (∑ b ∈ s2, q a b) * min (θ a) 0 := by rw [hq_sum a ha, one_mul]
      _ = ∑ b ∈ s2, q a b * min (θ a) 0 := Finset.sum_mul _ _ _
      _ ≤ ∑ b ∈ s2, q a b * (if θ a + ε a b < 0 then θ a else 0) :=
          Finset.sum_le_sum fun b hb =>
            mul_le_mul_of_nonneg_left (min_zero_le_realizedCost (θ a) (θ a + ε a b)) (hq a ha b hb)

/-- **Null-World Mirage Corollary**:
If true net cost is identically `θ ≡ μ ≥ 0` everywhere, yet estimation noise creates at least one
positively weighted outcome with `θ̃_{i₀} < 0`, then apparent in-sample net cost is strictly
negative (`< 0`, mirage of savings) while realized net cost is nonnegative (`≥ 0`, and strictly
positive whenever `μ > 0`). -/
theorem null_world_mirage {ι : Type*} (s : Finset ι) (w θ θtilde : ι → ℝ) {μ : ℝ}
    (hw : ∀ i ∈ s, 0 ≤ w i) (hμ : 0 ≤ μ) (hθ : ∀ i ∈ s, θ i = μ)
    (hneg : ∃ i0 ∈ s, 0 < w i0 ∧ θtilde i0 < 0) :
    (∑ i ∈ s, w i * min (θtilde i) 0 < 0) ∧
    (0 ≤ ∑ i ∈ s, w i * realizedCost θ θtilde i) := by
  rcases hneg with ⟨i0, hi0, hw0, ht0⟩
  constructor
  · have hle : ∀ i ∈ s, w i * min (θtilde i) 0 ≤ 0 := fun i hi =>
      mul_nonpos_of_nonneg_of_nonpos (hw i hi) (min_le_right _ _)
    have hlt : w i0 * min (θtilde i0) 0 < 0 := by
      rw [min_eq_left (le_of_lt ht0)]
      exact mul_neg_of_pos_of_neg hw0 ht0
    have hsum_lt := Finset.sum_lt_sum hle ⟨i0, hi0, hlt⟩
    simpa using hsum_lt
  · refine Finset.sum_nonneg fun i hi => ?_
    refine mul_nonneg (hw i hi) ?_
    unfold realizedCost
    split_ifs
    · rw [hθ i hi]; exact hμ
    · exact le_rfl

/-! ## Part 3B: Exact Gaussian Mills-Ratio Bounds & Sign-Flip Phase Transition Sandwich -/

/-- Standard normal probability density function `φ(t) = exp(-t²/2) / √(2π)`. -/
noncomputable def stdNormalPdf (t : ℝ) : ℝ :=
  Real.exp (-t ^ 2 / 2) / Real.sqrt (2 * Real.pi)

/-- Unnormalized Gaussian upper tail `G(z) = ∫_{t > z} exp(-t²/2) dt`. -/
noncomputable def gaussTailUnnorm (z : ℝ) : ℝ :=
  ∫ t in Set.Ioi z, Real.exp (-t ^ 2 / 2)

/-- Standard normal upper tail (complementary CDF) `Φ̄(z) = ∫_{t > z} φ(t) dt`. -/
noncomputable def stdNormalTail (z : ℝ) : ℝ :=
  ∫ t in Set.Ioi z, stdNormalPdf t

/-- Standard normal Mills ratio `M(z) = Φ̄(z) / φ(z)`. -/
noncomputable def millsRatio (z : ℝ) : ℝ :=
  stdNormalTail z / stdNormalPdf z

/-- Realized plug-in net cost under Gaussian true net cost `θ ~ N(μ, σ²)` and score reliability
`ρ = σ² / (σ² + τ²)`: `ΔC(ρ) = μ Φ̄((μ/σ)√ρ) - σ √ρ φ((μ/σ)√ρ)`. -/
noncomputable def gaussianPluginNetCost (μ σ ρ : ℝ) : ℝ :=
  μ * stdNormalTail ((μ / σ) * Real.sqrt ρ) -
    σ * Real.sqrt ρ * stdNormalPdf ((μ / σ) * Real.sqrt ρ)

/-- Helper: `exp(-t²/2)` is integrable on `ℝ`. -/
lemma integrable_exp_neg_half_sq :
    Integrable (fun t : ℝ => Real.exp (-t ^ 2 / 2)) := by
  have h := integrable_exp_neg_mul_sq (by norm_num : (0 : ℝ) < 1 / 2)
  convert h using 1
  ext t
  congr 1
  ring

/-- `G(0) = ∫_{t > 0} exp(-t²/2) dt = √(2π) / 2`. -/
lemma gaussTailUnnorm_zero : gaussTailUnnorm 0 = Real.sqrt (2 * Real.pi) / 2 := by
  unfold gaussTailUnnorm
  have h := integral_gaussian_Ioi (1 / 2 : ℝ)
  have hfn : (fun t : ℝ => Real.exp (-t ^ 2 / 2)) = (fun t : ℝ => Real.exp (-(1 / 2 : ℝ) * t ^ 2)) := by
    ext t; congr 1; ring
  rw [hfn, h]
  have hdiv : Real.pi / (1 / 2 : ℝ) = 2 * Real.pi := by ring
  rw [hdiv]

/-- `Φ̄(z) = G(z) / √(2π)`. -/
lemma stdNormalTail_eq (z : ℝ) :
    stdNormalTail z = gaussTailUnnorm z / Real.sqrt (2 * Real.pi) := by
  unfold stdNormalTail stdNormalPdf gaussTailUnnorm
  rw [MeasureTheory.integral_div]

/-- `Φ̄(0) = 1 / 2`. -/
lemma stdNormalTail_zero : stdNormalTail 0 = 1 / 2 := by
  rw [stdNormalTail_eq, gaussTailUnnorm_zero]
  have hpos : 0 < Real.sqrt (2 * Real.pi) := Real.sqrt_pos.2 (by positivity)
  have hne : Real.sqrt (2 * Real.pi) ≠ 0 := ne_of_gt hpos
  field_simp
  ring

/-- `M(z) = G(z) / exp(-z²/2)`. -/
lemma millsRatio_eq (z : ℝ) :
    millsRatio z = gaussTailUnnorm z / Real.exp (-z ^ 2 / 2) := by
  unfold millsRatio stdNormalPdf
  rw [stdNormalTail_eq]
  have hpos : 0 < Real.sqrt (2 * Real.pi) := Real.sqrt_pos.2 (by positivity)
  have hne : Real.sqrt (2 * Real.pi) ≠ 0 := ne_of_gt hpos
  field_simp

/-- Fundamental identity splitting `G(z) = G(0) - ∫_0^z exp(-t²/2) dt` for all `z ∈ ℝ`. -/
lemma gaussTailUnnorm_eq_sub_intervalIntegral (z : ℝ) :
    gaussTailUnnorm z = gaussTailUnnorm 0 - ∫ t in (0 : ℝ)..z, Real.exp (-t ^ 2 / 2) := by
  unfold gaussTailUnnorm
  let f : ℝ → ℝ := fun t => Real.exp (-t ^ 2 / 2)
  have hf : Integrable f := integrable_exp_neg_half_sq
  rcases le_total 0 z with hz | hz
  · have hunion : ∫ t in Set.Ioi (0 : ℝ), f t =
        (∫ t in Set.Ioc (0 : ℝ) z, f t) + (∫ t in Set.Ioi z, f t) := by
      rw [← Set.Ioc_union_Ioi_eq_Ioi hz]
      exact MeasureTheory.setIntegral_union Set.Ioc_disjoint_Ioi_same measurableSet_Ioi
        hf.integrableOn hf.integrableOn
    rw [intervalIntegral.integral_of_le hz]
    linarith
  · have hunion : ∫ t in Set.Ioi z, f t =
        (∫ t in Set.Ioc z (0 : ℝ), f t) + (∫ t in Set.Ioi (0 : ℝ), f t) := by
      rw [← Set.Ioc_union_Ioi_eq_Ioi hz]
      exact MeasureTheory.setIntegral_union Set.Ioc_disjoint_Ioi_same measurableSet_Ioi
        hf.integrableOn hf.integrableOn
    rw [intervalIntegral.integral_symm, intervalIntegral.integral_of_le hz]
    linarith

/-- Exact derivative of the unnormalized Gaussian upper tail: `G'(z) = -exp(-z²/2)`. -/
lemma hasDerivAt_gaussTailUnnorm (z : ℝ) :
    HasDerivAt gaussTailUnnorm (-Real.exp (-z ^ 2 / 2)) z := by
  have hfun : gaussTailUnnorm = fun u => gaussTailUnnorm 0 - ∫ t in (0 : ℝ)..u, Real.exp (-t ^ 2 / 2) :=
    funext gaussTailUnnorm_eq_sub_intervalIntegral
  rw [hfun]
  have hcont : Continuous (fun t : ℝ => Real.exp (-t ^ 2 / 2)) := by fun_prop
  have hint : IntervalIntegrable (fun t : ℝ => Real.exp (-t ^ 2 / 2)) volume 0 z :=
    hcont.intervalIntegrable 0 z
  have hmeas : StronglyMeasurableAtFilter (fun t : ℝ => Real.exp (-t ^ 2 / 2)) (nhds z) volume :=
    hcont.stronglyMeasurableAtFilter volume (nhds z)
  have hftc := intervalIntegral.integral_hasDerivAt_right hint hmeas hcont.continuousAt
  simpa using (hasDerivAt_const z (gaussTailUnnorm 0)).sub hftc

/-- Exact derivative of the standard normal upper tail: `Φ̄'(z) = -φ(z)`. -/
lemma hasDerivAt_stdNormalTail (z : ℝ) :
    HasDerivAt stdNormalTail (-stdNormalPdf z) z := by
  have hfun : stdNormalTail = fun u => gaussTailUnnorm u / Real.sqrt (2 * Real.pi) :=
    funext stdNormalTail_eq
  rw [hfun]
  have h := (hasDerivAt_gaussTailUnnorm z).div_const (Real.sqrt (2 * Real.pi))
  convert h using 1
  unfold stdNormalPdf
  ring

/-- Exact derivative of the standard normal density: `φ'(z) = -z * φ(z)`. -/
lemma hasDerivAt_stdNormalPdf (z : ℝ) :
    HasDerivAt stdNormalPdf (-z * stdNormalPdf z) z := by
  unfold stdNormalPdf
  have h1 : HasDerivAt (fun t : ℝ => -t ^ 2 / 2) (-z) z := by
    have hsq : HasDerivAt (fun t : ℝ => t ^ 2) (2 * z) z := by
      simpa using hasDerivAt_pow 2 z
    have hneg := hsq.neg.div_const 2
    convert hneg using 1
    ring
  have h2 := (h1.exp).div_const (Real.sqrt (2 * Real.pi))
  convert h2 using 1
  ring

/-- `exp(-t²/2) → 0` as `t → ∞`. -/
lemma tendsto_exp_neg_half_sq_atTop :
    Tendsto (fun t : ℝ => Real.exp (-t ^ 2 / 2)) atTop (nhds 0) := by
  refine squeeze_zero' (Eventually.of_forall fun t => le_of_lt (Real.exp_pos _)) ?_
    tendsto_inv_atTop_zero
  filter_upwards [eventually_ge_atTop (1 : ℝ)] with t ht
  have hle : t ≤ Real.exp (t ^ 2 / 2) := by
    have h1 := Real.add_one_le_exp (t ^ 2 / 2)
    have h2 : t ≤ t ^ 2 / 2 + 1 := by nlinarith
    linarith
  have hexp_eq : Real.exp (-t ^ 2 / 2) = (Real.exp (t ^ 2 / 2))⁻¹ := by
    rw [show -t ^ 2 / 2 = -(t ^ 2 / 2) by ring, Real.exp_neg]
  rw [hexp_eq]
  exact inv_anti₀ (by linarith) hle

/-- Elementary upper bound `G(z) ≤ exp(-z²/2) / z` for `z > 0`, giving `G(z) → 0` at `∞`. -/
lemma gaussTailUnnorm_le_div {z : ℝ} (hz : 0 < z) :
    gaussTailUnnorm z ≤ Real.exp (-z ^ 2 / 2) / z := by
  have hint_t : IntegrableOn (fun t : ℝ => t * Real.exp (-t ^ 2 / 2)) (Set.Ioi z) volume := by
    have hdom : Integrable (fun t : ℝ => Real.exp (-(1 / 4 : ℝ) * t ^ 2)) :=
      integrable_exp_neg_mul_sq (by norm_num : (0 : ℝ) < 1 / 4)
    refine hdom.integrableOn.mono' (by fun_prop : AEStronglyMeasurable _ _) ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with t ht
    have ht_pos : 0 < t := lt_trans hz ht
    rw [Real.norm_of_nonneg (mul_nonneg (le_of_lt ht_pos) (le_of_lt (Real.exp_pos _)))]
    have h1 : t ≤ Real.exp (t ^ 2 / 4) := by
      have hlin : t ≤ t ^ 2 / 4 + 1 := by nlinarith [sq_nonneg (t - 2)]
      linarith [Real.add_one_le_exp (t ^ 2 / 4)]
    calc
      t * Real.exp (-t ^ 2 / 2)
          ≤ Real.exp (t ^ 2 / 4) * Real.exp (-t ^ 2 / 2) :=
            mul_le_mul_of_nonneg_right h1 (le_of_lt (Real.exp_pos _))
      _ = Real.exp (-(1 / 4 : ℝ) * t ^ 2) := by
            rw [← Real.exp_add]; congr 1; ring
  have hftc : ∫ t in Set.Ioi z, t * Real.exp (-t ^ 2 / 2) = Real.exp (-z ^ 2 / 2) := by
    have hderiv : ∀ t ∈ Set.Ici z,
        HasDerivAt (fun u : ℝ => -Real.exp (-u ^ 2 / 2)) (t * Real.exp (-t ^ 2 / 2)) t := by
      intro t _
      have h1 : HasDerivAt (fun u : ℝ => -u ^ 2 / 2) (-t) t := by
        have hsq : HasDerivAt (fun u : ℝ => u ^ 2) (2 * t) t := by
          simpa using hasDerivAt_pow 2 t
        convert hsq.neg.div_const 2 using 1; ring
      convert h1.exp.neg using 1; ring
    have htend : Tendsto (fun u : ℝ => -Real.exp (-u ^ 2 / 2)) atTop (nhds 0) := by
      simpa using tendsto_exp_neg_half_sq_atTop.neg
    have hres := MeasureTheory.integral_Ioi_of_hasDerivAt_of_tendsto' hderiv hint_t htend
    linarith
  have hcomp : ∫ t in Set.Ioi z, Real.exp (-t ^ 2 / 2) ≤
      ∫ t in Set.Ioi z, (t * Real.exp (-t ^ 2 / 2)) / z := by
    refine MeasureTheory.setIntegral_mono_on integrable_exp_neg_half_sq.integrableOn
      (hint_t.div_const z) measurableSet_Ioi fun t ht => ?_
    have hzt : z ≤ t := le_of_lt ht
    rw [le_div_iff₀ hz]
    nlinarith [Real.exp_pos (-t ^ 2 / 2)]
  calc
    gaussTailUnnorm z
        ≤ ∫ t in Set.Ioi z, (t * Real.exp (-t ^ 2 / 2)) / z := hcomp
    _ = (∫ t in Set.Ioi z, t * Real.exp (-t ^ 2 / 2)) / z := MeasureTheory.integral_div z _
    _ = Real.exp (-z ^ 2 / 2) / z := by rw [hftc]

/-- `G(z) → 0` as `z → ∞`. -/
lemma tendsto_gaussTailUnnorm_atTop : Tendsto gaussTailUnnorm atTop (nhds 0) := by
  refine squeeze_zero'
    (Eventually.of_forall fun z =>
      MeasureTheory.setIntegral_nonneg measurableSet_Ioi fun _ _ => le_of_lt (Real.exp_pos _))
    ?_ tendsto_inv_atTop_zero
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with z hz
  have h1 := gaussTailUnnorm_le_div hz
  have h2 : Real.exp (-z ^ 2 / 2) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith)
  have h3 : Real.exp (-z ^ 2 / 2) / z ≤ 1 / z := div_le_div_of_nonneg_right h2 (le_of_lt hz)
  rw [one_div] at h3
  exact le_trans h1 h3

/-- Unified Mills-ratio comparison function:
`D_c(z) = G(z) - (2 / c) * (√(z² + c) - z) * exp(-z²/2)`.
Note that `(2 / c) * (√(z² + c) - z) = 2 / (√(z² + c) + z)`. -/
noncomputable def millsDiff (c z : ℝ) : ℝ :=
  gaussTailUnnorm z - (2 / c) * (Real.sqrt (z ^ 2 + c) - z) * Real.exp (-z ^ 2 / 2)

/-- Rationalization identity `(2 / c) * (√(z² + c) - z) = 2 / (√(z² + c) + z)`. -/
lemma two_div_sqrt_add_eq {c : ℝ} (hc : 0 < c) (z : ℝ) :
    2 / (Real.sqrt (z ^ 2 + c) + z) = (2 / c) * (Real.sqrt (z ^ 2 + c) - z) := by
  have hrad : 0 < z ^ 2 + c := by positivity
  have hsq : Real.sqrt (z ^ 2 + c) ^ 2 = z ^ 2 + c := Real.sq_sqrt (le_of_lt hrad)
  have hgt : |z| < Real.sqrt (z ^ 2 + c) := by
    rw [← Real.sqrt_sq_eq_abs]
    exact Real.sqrt_lt_sqrt (sq_nonneg _) (by linarith)
  have hsum_pos : 0 < Real.sqrt (z ^ 2 + c) + z := by
    have := neg_abs_le z
    linarith
  have hsum_ne : Real.sqrt (z ^ 2 + c) + z ≠ 0 := ne_of_gt hsum_pos
  have hc_ne : c ≠ 0 := ne_of_gt hc
  field_simp
  nlinarith [hsq]

/-- Exact derivative of `D_c(z)` for any `c > 0` and `z ∈ ℝ`:
`D_c'(z) = -(exp(-z²/2) / (c √(z² + c))) * ((2 z² + c - 2) √(z² + c) - 2 z (z² + c - 1))`. -/
lemma hasDerivAt_millsDiff {c : ℝ} (hc : 0 < c) (z : ℝ) :
    HasDerivAt (millsDiff c)
      (-(Real.exp (-z ^ 2 / 2) / (c * Real.sqrt (z ^ 2 + c))) *
        ((2 * z ^ 2 + c - 2) * Real.sqrt (z ^ 2 + c) - 2 * z * (z ^ 2 + c - 1))) z := by
  unfold millsDiff
  have hrad_pos : 0 < z ^ 2 + c := by positivity
  have hrad_ne : z ^ 2 + c ≠ 0 := ne_of_gt hrad_pos
  have hsqrt_pos : 0 < Real.sqrt (z ^ 2 + c) := Real.sqrt_pos.2 hrad_pos
  have hsqrt_ne : Real.sqrt (z ^ 2 + c) ≠ 0 := ne_of_gt hsqrt_pos
  have hsq : Real.sqrt (z ^ 2 + c) ^ 2 = z ^ 2 + c := Real.sq_sqrt (le_of_lt hrad_pos)
  have hc_ne : c ≠ 0 := ne_of_gt hc
  have hG := hasDerivAt_gaussTailUnnorm z
  have hinner : HasDerivAt (fun u : ℝ => u ^ 2 + c) (2 * z) z := by
    simpa using (hasDerivAt_pow 2 z).add_const c
  have hsqrt : HasDerivAt (fun u : ℝ => Real.sqrt (u ^ 2 + c))
      (2 * z / (2 * Real.sqrt (z ^ 2 + c))) z := hinner.sqrt hrad_ne
  have hu : HasDerivAt (fun u : ℝ => (2 / c) * (Real.sqrt (u ^ 2 + c) - u))
      ((2 / c) * (2 * z / (2 * Real.sqrt (z ^ 2 + c)) - 1)) z :=
    (hsqrt.sub (hasDerivAt_id z)).const_mul (2 / c)
  have hexp : HasDerivAt (fun u : ℝ => Real.exp (-u ^ 2 / 2)) (-z * Real.exp (-z ^ 2 / 2)) z := by
    have h1 : HasDerivAt (fun u : ℝ => -u ^ 2 / 2) (-z) z := by
      have hsq' : HasDerivAt (fun u : ℝ => u ^ 2) (2 * z) z := by simpa using hasDerivAt_pow 2 z
      convert hsq'.neg.div_const 2 using 1; ring
    convert h1.exp using 1; ring
  have hprod := hu.mul hexp
  have htot := hG.sub hprod
  convert htot using 1
  have halg : -((1 : ℝ) / (c * Real.sqrt (z ^ 2 + c))) *
        ((2 * z ^ 2 + c - 2) * Real.sqrt (z ^ 2 + c) - 2 * z * (z ^ 2 + c - 1)) =
      -1 - (2 / c * (2 * z / (2 * Real.sqrt (z ^ 2 + c)) - 1) +
        2 / c * (Real.sqrt (z ^ 2 + c) - z) * (-z)) := by
    have hcube : Real.sqrt (z ^ 2 + c) ^ 3 = (z ^ 2 + c) * Real.sqrt (z ^ 2 + c) := by
      rw [pow_succ, hsq]
    field_simp
    ring_nf
    rw [hsq, hcube]
    ring
  calc
    -(Real.exp (-z ^ 2 / 2) / (c * Real.sqrt (z ^ 2 + c))) *
        ((2 * z ^ 2 + c - 2) * Real.sqrt (z ^ 2 + c) - 2 * z * (z ^ 2 + c - 1))
        = Real.exp (-z ^ 2 / 2) * (-((1 : ℝ) / (c * Real.sqrt (z ^ 2 + c))) *
            ((2 * z ^ 2 + c - 2) * Real.sqrt (z ^ 2 + c) - 2 * z * (z ^ 2 + c - 1))) := by ring
    _ = Real.exp (-z ^ 2 / 2) * (-1 - (2 / c * (2 * z / (2 * Real.sqrt (z ^ 2 + c)) - 1) +
            2 / c * (Real.sqrt (z ^ 2 + c) - z) * (-z))) := by rw [halg]
    _ = -Real.exp (-z ^ 2 / 2) -
          (2 / c * (2 * z / (2 * Real.sqrt (z ^ 2 + c)) - 1) * Real.exp (-z ^ 2 / 2) +
            2 / c * (Real.sqrt (z ^ 2 + c) - z) * (-z * Real.exp (-z ^ 2 / 2))) := by ring

/-- `D_c(z) → 0` as `z → ∞` for every `c > 0`. -/
lemma tendsto_millsDiff_atTop {c : ℝ} (hc : 0 < c) :
    Tendsto (millsDiff c) atTop (nhds 0) := by
  unfold millsDiff
  have hsecond : Tendsto
      (fun z : ℝ => (2 / c) * (Real.sqrt (z ^ 2 + c) - z) * Real.exp (-z ^ 2 / 2))
      atTop (nhds 0) := by
    have hlim : Tendsto (fun z : ℝ => (2 : ℝ) * z⁻¹) atTop (nhds 0) := by
      simpa using (tendsto_inv_atTop_zero (𝕜 := ℝ)).const_mul (2 : ℝ)
    refine squeeze_zero' ?_ ?_ hlim
    · refine Eventually.of_forall fun z => ?_
      rw [← two_div_sqrt_add_eq hc z]
      have hgt : |z| < Real.sqrt (z ^ 2 + c) := by
        rw [← Real.sqrt_sq_eq_abs]
        exact Real.sqrt_lt_sqrt (sq_nonneg _) (by linarith)
      have hsum_pos : 0 < Real.sqrt (z ^ 2 + c) + z := by
        have := neg_abs_le z; linarith
      exact mul_nonneg (div_nonneg (by norm_num) (le_of_lt hsum_pos)) (le_of_lt (Real.exp_pos _))
    · filter_upwards [eventually_gt_atTop (0 : ℝ)] with z hz
      rw [← two_div_sqrt_add_eq hc z]
      have hsum_ge : z ≤ Real.sqrt (z ^ 2 + c) + z := by linarith [Real.sqrt_nonneg (z ^ 2 + c)]
      have h1 : 2 / (Real.sqrt (z ^ 2 + c) + z) ≤ 2 / z :=
        div_le_div_of_nonneg_left (by norm_num) hz hsum_ge
      have h2 : Real.exp (-z ^ 2 / 2) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith)
      have h3 : 2 / (Real.sqrt (z ^ 2 + c) + z) * Real.exp (-z ^ 2 / 2) ≤ 2 / z * 1 :=
        mul_le_mul h1 h2 (le_of_lt (Real.exp_pos _)) (div_nonneg (by norm_num) (le_of_lt hz))
      simpa [div_eq_mul_inv] using h3
  have hsub := tendsto_gaussTailUnnorm_atTop.sub hsecond
  simpa using hsub

/-- **Birnbaum Lower Bound on the Mills Ratio**:
For all `z ≥ 0` (in particular all `z > 0`), `2 / (√(z² + 4) + z) < M(z)`. -/
theorem millsRatio_lower_bound {z : ℝ} (hz : 0 ≤ z) :
    2 / (Real.sqrt (z ^ 2 + 4) + z) < millsRatio z := by
  have hanti : StrictAntiOn (millsDiff 4) (Set.Ici 0) := by
    apply strictAntiOn_of_deriv_neg (convex_Ici 0)
    · exact (continuous_iff_continuousAt.2 fun x =>
        (hasDerivAt_millsDiff (by norm_num : (0 : ℝ) < 4) x).continuousAt).continuousOn
    · intro x hx
      rw [interior_Ici] at hx
      have hx_pos : 0 < x := Set.mem_Ioi.1 hx
      rw [(hasDerivAt_millsDiff (by norm_num : (0 : ℝ) < 4) x).deriv]
      have hsqrt_pos : 0 < Real.sqrt (x ^ 2 + 4) := Real.sqrt_pos.2 (by positivity)
      have hsq : Real.sqrt (x ^ 2 + 4) ^ 2 = x ^ 2 + 4 := Real.sq_sqrt (by positivity)
      have hfac_pos : 0 < Real.exp (-x ^ 2 / 2) / (4 * Real.sqrt (x ^ 2 + 4)) :=
        div_pos (Real.exp_pos _) (by positivity)
      have hnum_pos : 0 < (2 * x ^ 2 + 4 - 2) * Real.sqrt (x ^ 2 + 4) - 2 * x * (x ^ 2 + 4 - 1) := by
        have hA_pos : 0 < (2 * x ^ 2 + 4 - 2) * Real.sqrt (x ^ 2 + 4) :=
          mul_pos (by nlinarith) hsqrt_pos
        have hB_nonneg : 0 ≤ 2 * x * (x ^ 2 + 4 - 1) :=
          mul_nonneg (by linarith) (by nlinarith)
        have hsq_gt : (2 * x * (x ^ 2 + 4 - 1)) ^ 2 <
            ((2 * x ^ 2 + 4 - 2) * Real.sqrt (x ^ 2 + 4)) ^ 2 := by
          have hexp : ((2 * x ^ 2 + 4 - 2) * Real.sqrt (x ^ 2 + 4)) ^ 2 =
              (2 * x ^ 2 + 2) ^ 2 * (x ^ 2 + 4) := by
            rw [mul_pow, hsq]; ring
          rw [hexp]
          nlinarith [sq_nonneg x, sq_nonneg (x ^ 2)]
        nlinarith [hA_pos, hB_nonneg, hsq_gt]
      nlinarith
  have hD_pos : 0 < millsDiff 4 z := by
    have hlt : millsDiff 4 (z + 1) < millsDiff 4 z :=
      hanti (Set.mem_Ici.2 hz) (Set.mem_Ici.2 (by linarith)) (by linarith)
    have hge0 : 0 ≤ millsDiff 4 (z + 1) := by
      refine le_of_tendsto (tendsto_millsDiff_atTop (by norm_num : (0 : ℝ) < 4)) ?_
      filter_upwards [eventually_ge_atTop (z + 1)] with w hw
      rcases eq_or_lt_of_le hw with rfl | hgt
      · exact le_rfl
      · exact le_of_lt (hanti (Set.mem_Ici.2 (by linarith)) (Set.mem_Ici.2 (by linarith)) hgt)
    linarith
  rw [millsRatio_eq, lt_div_iff₀ (Real.exp_pos _), two_div_sqrt_add_eq (by norm_num : (0 : ℝ) < 4) z]
  unfold millsDiff at hD_pos
  linarith

/-- **Komatsu–Boyd Upper Bound on the Mills Ratio**:
For all `z > 0`, `M(z) < 2 / (√(z² + 8/π) + z)`. -/
theorem millsRatio_upper_bound {z : ℝ} (hz : 0 < z) :
    millsRatio z < 2 / (Real.sqrt (z ^ 2 + 8 / Real.pi) + z) := by
  set c := 8 / Real.pi
  have hpi3 : (3 : ℝ) < Real.pi := Real.pi_gt_three
  have hpi4 : Real.pi < 4 := Real.pi_lt_four
  have hpi_pos : 0 < Real.pi := by linarith
  have hc_gt2 : 2 < c := by
    simp only [c]
    rw [lt_div_iff₀ hpi_pos]
    linarith
  have hc_lt4 : c < 4 := by
    simp only [c]
    rw [div_lt_iff₀ hpi_pos]
    linarith
  have hc_pos : 0 < c := by linarith
  set z0 := (c - 2) / Real.sqrt (4 - c)
  have hz0_pos : 0 < z0 := div_pos (by linarith) (Real.sqrt_pos.2 (by linarith))
  have hz0_sq : z0 ^ 2 = (c - 2) ^ 2 / (4 - c) := by
    simp only [z0, div_pow, Real.sq_sqrt (by linarith : 0 ≤ 4 - c)]
  have hD0 : millsDiff c 0 = 0 := by
    unfold millsDiff
    rw [gaussTailUnnorm_zero]
    simp only [sq, mul_zero, zero_add, sub_zero, neg_zero, zero_div, Real.exp_zero, mul_one]
    have hsqrt_c : Real.sqrt c = 2 * Real.sqrt (2 / Real.pi) := by
      have hc_eq : c = 4 * (2 / Real.pi) := by simp only [c]; ring
      rw [hc_eq, Real.sqrt_mul (by norm_num), show Real.sqrt 4 = 2 by norm_num]
    have hprod : Real.sqrt (2 * Real.pi) / 2 = (2 / c) * Real.sqrt c := by
      have hpos1 : 0 ≤ Real.sqrt (2 * Real.pi) / 2 := by positivity
      have hpos2 : 0 ≤ (2 / c) * Real.sqrt c := by positivity
      have hsq_eq : (Real.sqrt (2 * Real.pi) / 2) ^ 2 = ((2 / c) * Real.sqrt c) ^ 2 := by
        rw [div_pow, mul_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt (le_of_lt hc_pos)]
        simp only [c]
        have hpi_ne : Real.pi ≠ 0 := ne_of_gt hpi_pos
        field_simp
        ring
      nlinarith
    linarith
  -- Key algebraic identity for the sign of D_c'(x)
  have hdiff_sq : ∀ x : ℝ,
      ((2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c)) ^ 2 - (2 * x * (x ^ 2 + c - 1)) ^ 2 =
        c * (4 - c) * (z0 ^ 2 - x ^ 2) := by
    intro x
    have hsq : Real.sqrt (x ^ 2 + c) ^ 2 = x ^ 2 + c := Real.sq_sqrt (by positivity)
    rw [mul_pow, hsq, hz0_sq]
    have h4c : 4 - c ≠ 0 := by linarith
    field_simp
    ring
  have hanti : StrictAntiOn (millsDiff c) (Set.Icc 0 z0) := by
    apply strictAntiOn_of_deriv_neg (convex_Icc 0 z0)
    · exact (continuous_iff_continuousAt.2 fun x =>
        (hasDerivAt_millsDiff hc_pos x).continuousAt).continuousOn
    · intro x hx
      rw [interior_Icc] at hx
      rw [(hasDerivAt_millsDiff hc_pos x).deriv]
      have hfac_pos : 0 < Real.exp (-x ^ 2 / 2) / (c * Real.sqrt (x ^ 2 + c)) :=
        div_pos (Real.exp_pos _) (mul_pos hc_pos (Real.sqrt_pos.2 (by positivity)))
      have hA_pos : 0 < (2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c) :=
        mul_pos (by nlinarith) (Real.sqrt_pos.2 (by positivity))
      have hB_nonneg : 0 ≤ 2 * x * (x ^ 2 + c - 1) :=
        mul_nonneg (by linarith [hx.1]) (by nlinarith)
      have hsq_pos : 0 < ((2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c)) ^ 2 -
          (2 * x * (x ^ 2 + c - 1)) ^ 2 := by
        rw [hdiff_sq x]
        have hx2 : x ^ 2 < z0 ^ 2 := by nlinarith [hx.1, hx.2]
        exact mul_pos (mul_pos hc_pos (by linarith)) (by linarith)
      have hnum_pos : 0 < (2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c) -
          2 * x * (x ^ 2 + c - 1) := by nlinarith
      nlinarith
  have hmono : StrictMonoOn (millsDiff c) (Set.Ici z0) := by
    apply strictMonoOn_of_deriv_pos (convex_Ici z0)
    · exact (continuous_iff_continuousAt.2 fun x =>
        (hasDerivAt_millsDiff hc_pos x).continuousAt).continuousOn
    · intro x hx
      rw [interior_Ici] at hx
      have hx_gt : z0 < x := Set.mem_Ioi.1 hx
      rw [(hasDerivAt_millsDiff hc_pos x).deriv]
      have hfac_pos : 0 < Real.exp (-x ^ 2 / 2) / (c * Real.sqrt (x ^ 2 + c)) :=
        div_pos (Real.exp_pos _) (mul_pos hc_pos (Real.sqrt_pos.2 (by positivity)))
      have hA_pos : 0 < (2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c) :=
        mul_pos (by nlinarith) (Real.sqrt_pos.2 (by positivity))
      have hB_nonneg : 0 ≤ 2 * x * (x ^ 2 + c - 1) :=
        mul_nonneg (by linarith [hx_gt, hz0_pos]) (by nlinarith)
      have hsq_neg : ((2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c)) ^ 2 -
          (2 * x * (x ^ 2 + c - 1)) ^ 2 < 0 := by
        rw [hdiff_sq x]
        have hx2 : z0 ^ 2 < x ^ 2 := by nlinarith [hx_gt, hz0_pos]
        exact mul_neg_of_pos_of_neg (mul_pos hc_pos (by linarith)) (by linarith)
      have hnum_neg : (2 * x ^ 2 + c - 2) * Real.sqrt (x ^ 2 + c) -
          2 * x * (x ^ 2 + c - 1) < 0 := by nlinarith
      nlinarith
  have hD_neg : millsDiff c z < 0 := by
    rcases le_or_gt z z0 with hle | hgt
    · have := hanti ⟨le_rfl, le_of_lt hz0_pos⟩ ⟨le_of_lt hz, hle⟩ hz
      linarith [hD0]
    · have hlt : millsDiff c z < millsDiff c (z + 1) :=
        hmono (Set.mem_Ici.2 (le_of_lt hgt)) (Set.mem_Ici.2 (by linarith)) (by linarith)
      have hle0 : millsDiff c (z + 1) ≤ 0 := by
        refine ge_of_tendsto (tendsto_millsDiff_atTop hc_pos) ?_
        filter_upwards [eventually_ge_atTop (z + 1)] with w hw
        rcases eq_or_lt_of_le hw with rfl | hwgt
        · exact le_rfl
        · exact le_of_lt (hmono (Set.mem_Ici.2 (by linarith)) (Set.mem_Ici.2 (by linarith)) hwgt)
      linarith
  rw [millsRatio_eq, div_lt_iff₀ (Real.exp_pos _), two_div_sqrt_add_eq hc_pos z]
  unfold millsDiff at hD_neg
  linarith

/-- **Lower Regime of the Gaussian Sign-Flip Phase Transition**:
For `μ > 0`, `σ > 0`, and any score reliability `0 ≤ ρ ≤ μ² / (μ² + σ²)`, realized plug-in
net cost is strictly positive (`ΔC(ρ) > 0`). -/
theorem gaussianPluginNetCost_pos_of_le_lower_threshold {μ σ ρ : ℝ}
    (hμ : 0 < μ) (hσ : 0 < σ) (hρ0 : 0 ≤ ρ) (hρ : ρ ≤ μ ^ 2 / (μ ^ 2 + σ ^ 2)) :
    0 < gaussianPluginNetCost μ σ ρ := by
  unfold gaussianPluginNetCost
  rcases eq_or_lt_of_le hρ0 with rfl | hρ_pos
  · simp only [Real.sqrt_zero, mul_zero, zero_mul, sub_zero, stdNormalTail_zero]
    linarith
  · set γ := μ / σ
    have hγ : 0 < γ := div_pos hμ hσ
    set z := γ * Real.sqrt ρ
    have hz : 0 < z := mul_pos hγ (Real.sqrt_pos.2 hρ_pos)
    have hz_sq : z ^ 2 = γ ^ 2 * ρ := by
      simp only [z, mul_pow, Real.sq_sqrt (le_of_lt hρ_pos)]
    have hρ_le_γ : ρ ≤ γ ^ 2 / (γ ^ 2 + 1) := by
      have hrew : μ ^ 2 / (μ ^ 2 + σ ^ 2) = γ ^ 2 / (γ ^ 2 + 1) := by
        simp only [γ]
        have hσ_ne : σ ≠ 0 := ne_of_gt hσ
        field_simp
      rwa [hrew] at hρ
    have hρ_lt_one : ρ < 1 := by
      have h1 : γ ^ 2 / (γ ^ 2 + 1) < 1 := (div_lt_one (by positivity)).2 (by linarith)
      linarith
    have hM := millsRatio_lower_bound (le_of_lt hz)
    have hbound : z / γ ^ 2 ≤ 2 / (Real.sqrt (z ^ 2 + 4) + z) := by
      have hden1 : 0 < γ ^ 2 := sq_pos_of_pos hγ
      have hden2 : 0 < Real.sqrt (z ^ 2 + 4) + z := by positivity
      rw [div_le_div_iff₀ hden1 hden2]
      have h2γ : 0 < 2 * γ ^ 2 - z ^ 2 := by nlinarith [hz_sq, hρ_lt_one]
      have hsq : Real.sqrt (z ^ 2 + 4) ^ 2 = z ^ 2 + 4 := Real.sq_sqrt (by positivity)
      have hprod_le : ρ * (γ ^ 2 + 1) ≤ γ ^ 2 := (le_div_iff₀ (by positivity)).1 hρ_le_γ
      have hsq_le : (z * Real.sqrt (z ^ 2 + 4)) ^ 2 ≤ (2 * γ ^ 2 - z ^ 2) ^ 2 := by
        rw [mul_pow, hsq]
        nlinarith [hz_sq, hprod_le]
      have hlhs_nonneg : 0 ≤ z * Real.sqrt (z ^ 2 + 4) := by positivity
      nlinarith
    have hM_gt : z / γ ^ 2 < millsRatio z := lt_of_le_of_lt hbound hM
    have hpdf_pos : 0 < stdNormalPdf z := by
      unfold stdNormalPdf
      exact div_pos (Real.exp_pos _) (Real.sqrt_pos.2 (by positivity))
    have htail_eq : stdNormalTail z = millsRatio z * stdNormalPdf z := by
      unfold millsRatio
      exact (div_mul_cancel₀ _ (ne_of_gt hpdf_pos)).symm
    rw [htail_eq]
    have hγM : Real.sqrt ρ < γ * millsRatio z := by
      have h1 : z < γ ^ 2 * millsRatio z := by
        rwa [div_lt_iff₀ (sq_pos_of_pos hγ), mul_comm] at hM_gt
      simp only [z] at h1
      nlinarith
    have hμ_eq : μ = σ * γ := by simp only [γ]; exact (mul_div_cancel₀ μ (ne_of_gt hσ)).symm
    rw [hμ_eq]
    have hfac : σ * γ * (millsRatio z * stdNormalPdf z) - σ * Real.sqrt ρ * stdNormalPdf z =
        σ * stdNormalPdf z * (γ * millsRatio z - Real.sqrt ρ) := by ring
    rw [hfac]
    exact mul_pos (mul_pos hσ hpdf_pos) (by linarith)

/-- **Noise-SD Form of the Lower Regime**:
When `ρ = σ² / (σ² + τ²)` and estimation noise standard deviation satisfies `τ ≥ σ² / μ`,
realized plug-in net cost is strictly positive (`ΔC(ρ) > 0`). -/
theorem gaussianPluginNetCost_pos_of_noise_sd {μ σ τ : ℝ}
    (hμ : 0 < μ) (hσ : 0 < σ) (hτ : σ ^ 2 / μ ≤ τ) :
    0 < gaussianPluginNetCost μ σ (σ ^ 2 / (σ ^ 2 + τ ^ 2)) := by
  have hτ_pos : 0 < τ := lt_of_lt_of_le (div_pos (sq_pos_of_pos hσ) hμ) hτ
  have hρ0 : 0 ≤ σ ^ 2 / (σ ^ 2 + τ ^ 2) := div_nonneg (sq_nonneg _) (by positivity)
  have hρ_le : σ ^ 2 / (σ ^ 2 + τ ^ 2) ≤ μ ^ 2 / (μ ^ 2 + σ ^ 2) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have hmul : σ ^ 2 ≤ μ * τ := by
      rwa [div_le_iff₀ hμ, mul_comm] at hτ
    nlinarith [sq_nonneg (μ * τ - σ ^ 2)]
  exact gaussianPluginNetCost_pos_of_le_lower_threshold hμ hσ hρ0 hρ_le

/-- **Upper Regime of the Gaussian Sign-Flip Phase Transition**:
For `μ > 0`, `σ > 0`, and any score reliability `μ² / (μ² + (2/π) σ²) ≤ ρ`, realized plug-in
net cost is strictly negative (`ΔC(ρ) < 0`). -/
theorem gaussianPluginNetCost_neg_of_ge_upper_threshold {μ σ ρ : ℝ}
    (hμ : 0 < μ) (hσ : 0 < σ) (hρ : μ ^ 2 / (μ ^ 2 + (2 / Real.pi) * σ ^ 2) ≤ ρ) :
    gaussianPluginNetCost μ σ ρ < 0 := by
  unfold gaussianPluginNetCost
  have hpi_pos : 0 < Real.pi := Real.pi_pos
  have hthr_pos : 0 < μ ^ 2 / (μ ^ 2 + (2 / Real.pi) * σ ^ 2) := div_pos (sq_pos_of_pos hμ) (by positivity)
  have hρ_pos : 0 < ρ := lt_of_lt_of_le hthr_pos hρ
  set γ := μ / σ
  have hγ : 0 < γ := div_pos hμ hσ
  set z := γ * Real.sqrt ρ
  have hz : 0 < z := mul_pos hγ (Real.sqrt_pos.2 hρ_pos)
  have hz_sq : z ^ 2 = γ ^ 2 * ρ := by
    simp only [z, mul_pow, Real.sq_sqrt (le_of_lt hρ_pos)]
  have hρ_ge_γ : γ ^ 2 / (γ ^ 2 + 2 / Real.pi) ≤ ρ := by
    have hrew : μ ^ 2 / (μ ^ 2 + (2 / Real.pi) * σ ^ 2) = γ ^ 2 / (γ ^ 2 + 2 / Real.pi) := by
      simp only [γ]
      have hσ_ne : σ ≠ 0 := ne_of_gt hσ
      field_simp
      ring
    rwa [hrew] at hρ
  have hM := millsRatio_upper_bound hz
  have hbound : 2 / (Real.sqrt (z ^ 2 + 8 / Real.pi) + z) ≤ z / γ ^ 2 := by
    have hden1 : 0 < Real.sqrt (z ^ 2 + 8 / Real.pi) + z := by positivity
    have hden2 : 0 < γ ^ 2 := sq_pos_of_pos hγ
    rw [div_le_div_iff₀ hden1 hden2]
    rcases le_or_gt (2 * γ ^ 2 - z ^ 2) 0 with hle | hgt
    · have hpos : 0 < z * Real.sqrt (z ^ 2 + 8 / Real.pi) := by positivity
      linarith
    · have hsq : Real.sqrt (z ^ 2 + 8 / Real.pi) ^ 2 = z ^ 2 + 8 / Real.pi :=
        Real.sq_sqrt (by positivity)
      have hprod_ge : γ ^ 2 ≤ ρ * (γ ^ 2 + 2 / Real.pi) :=
        (div_le_iff₀ (by positivity)).1 hρ_ge_γ
      have hsq_le : (2 * γ ^ 2 - z ^ 2) ^ 2 ≤ (z * Real.sqrt (z ^ 2 + 8 / Real.pi)) ^ 2 := by
        have hmul_sq : (z * Real.sqrt (z ^ 2 + 8 / Real.pi)) ^ 2 =
            z ^ 2 * Real.sqrt (z ^ 2 + 8 / Real.pi) ^ 2 := by ring
        rw [hmul_sq, hsq]
        have h4 : 4 * γ ^ 4 ≤ 4 * z ^ 2 * (γ ^ 2 + 2 / Real.pi) := by
          rw [hz_sq]
          have hmul := mul_le_mul_of_nonneg_left hprod_ge (by positivity : (0 : ℝ) ≤ 4 * γ ^ 2)
          linarith
        have hdiff : z ^ 2 * (z ^ 2 + 8 / Real.pi) - (2 * γ ^ 2 - z ^ 2) ^ 2 =
            4 * z ^ 2 * (γ ^ 2 + 2 / Real.pi) - 4 * γ ^ 4 := by ring
        linarith
      have hrhs_nonneg : 0 ≤ z * Real.sqrt (z ^ 2 + 8 / Real.pi) := by positivity
      nlinarith
  have hM_lt : millsRatio z < z / γ ^ 2 := lt_of_lt_of_le hM hbound
  have hpdf_pos : 0 < stdNormalPdf z := by
    unfold stdNormalPdf
    exact div_pos (Real.exp_pos _) (Real.sqrt_pos.2 (by positivity))
  have htail_eq : stdNormalTail z = millsRatio z * stdNormalPdf z := by
    unfold millsRatio
    exact (div_mul_cancel₀ _ (ne_of_gt hpdf_pos)).symm
  rw [htail_eq]
  have hγM : γ * millsRatio z < Real.sqrt ρ := by
    have h1 : γ ^ 2 * millsRatio z < z := by
      rwa [lt_div_iff₀ (sq_pos_of_pos hγ), mul_comm] at hM_lt
    simp only [z] at h1
    nlinarith
  have hμ_eq : μ = σ * γ := by simp only [γ]; exact (mul_div_cancel₀ μ (ne_of_gt hσ)).symm
  rw [hμ_eq]
  have hfac : σ * γ * (millsRatio z * stdNormalPdf z) - σ * Real.sqrt ρ * stdNormalPdf z =
      σ * stdNormalPdf z * (γ * millsRatio z - Real.sqrt ρ) := by ring
  rw [hfac]
  exact mul_neg_of_pos_of_neg (mul_pos hσ hpdf_pos) (by linarith)

/-- Exact derivative of `ΔC(ρ)` for `ρ > 0`:
`ΔC'(ρ) = -(σ φ((μ/σ)√ρ) / (2 √ρ)) * (1 + (μ/σ)² (1 - ρ))`,
which is strictly negative on `(0, 1]`. -/
lemma hasDerivAt_gaussianPluginNetCost {μ σ ρ : ℝ} (hσ : 0 < σ) (hρ : 0 < ρ) :
    HasDerivAt (gaussianPluginNetCost μ σ)
      (-(σ * stdNormalPdf ((μ / σ) * Real.sqrt ρ) / (2 * Real.sqrt ρ)) *
        (1 + (μ / σ) ^ 2 * (1 - ρ))) ρ := by
  unfold gaussianPluginNetCost
  set γ := μ / σ
  have hρ_ne : ρ ≠ 0 := ne_of_gt hρ
  have hsqrt_pos : 0 < Real.sqrt ρ := Real.sqrt_pos.2 hρ
  have hsqrt_ne : Real.sqrt ρ ≠ 0 := ne_of_gt hsqrt_pos
  have hsq : Real.sqrt ρ ^ 2 = ρ := Real.sq_sqrt (le_of_lt hρ)
  have hsqrt_deriv : HasDerivAt Real.sqrt (1 / (2 * Real.sqrt ρ)) ρ := Real.hasDerivAt_sqrt hρ_ne
  have hz_deriv : HasDerivAt (fun r => γ * Real.sqrt r) (γ * (1 / (2 * Real.sqrt ρ))) ρ :=
    hsqrt_deriv.const_mul γ
  have htail_comp : HasDerivAt (fun r => stdNormalTail (γ * Real.sqrt r))
      (-stdNormalPdf (γ * Real.sqrt ρ) * (γ * (1 / (2 * Real.sqrt ρ)))) ρ := by
    simpa [Function.comp_def] using (hasDerivAt_stdNormalTail (γ * Real.sqrt ρ)).comp ρ hz_deriv
  have htail : HasDerivAt (fun r => μ * stdNormalTail (γ * Real.sqrt r))
      (μ * (-stdNormalPdf (γ * Real.sqrt ρ) * (γ * (1 / (2 * Real.sqrt ρ))))) ρ :=
    htail_comp.const_mul μ
  have hpdf_comp : HasDerivAt (fun r => stdNormalPdf (γ * Real.sqrt r))
      ((-(γ * Real.sqrt ρ) * stdNormalPdf (γ * Real.sqrt ρ)) * (γ * (1 / (2 * Real.sqrt ρ)))) ρ := by
    simpa [Function.comp_def] using (hasDerivAt_stdNormalPdf (γ * Real.sqrt ρ)).comp ρ hz_deriv
  have hsecond : HasDerivAt (fun r => σ * Real.sqrt r * stdNormalPdf (γ * Real.sqrt r))
      ((σ * (1 / (2 * Real.sqrt ρ))) * stdNormalPdf (γ * Real.sqrt ρ) +
        (σ * Real.sqrt ρ) *
          ((-(γ * Real.sqrt ρ) * stdNormalPdf (γ * Real.sqrt ρ)) * (γ * (1 / (2 * Real.sqrt ρ))))) ρ :=
    (hsqrt_deriv.const_mul σ).mul hpdf_comp
  have htot := htail.sub hsecond
  convert htot using 1
  have hσ_ne : σ ≠ 0 := ne_of_gt hσ
  have hμ_eq : μ = σ * γ := by simp only [γ]; exact (mul_div_cancel₀ μ hσ_ne).symm
  rw [hμ_eq]
  have halg : -(σ / (2 * Real.sqrt ρ)) * (1 + γ ^ 2 * (1 - ρ)) =
      σ * γ * (-(γ * (1 / (2 * Real.sqrt ρ)))) -
        (σ * (1 / (2 * Real.sqrt ρ)) + σ * Real.sqrt ρ * (-(γ * Real.sqrt ρ) * (γ * (1 / (2 * Real.sqrt ρ))))) := by
    have h4 : Real.sqrt ρ ^ 4 = ρ ^ 2 := by
      rw [show (4 : ℕ) = 2 * 2 by rfl, pow_mul, hsq]
    field_simp
    ring_nf
    rw [hsq, h4]
    ring
  calc
    -(σ * stdNormalPdf (γ * Real.sqrt ρ) / (2 * Real.sqrt ρ)) * (1 + γ ^ 2 * (1 - ρ))
        = stdNormalPdf (γ * Real.sqrt ρ) * (-(σ / (2 * Real.sqrt ρ)) * (1 + γ ^ 2 * (1 - ρ))) := by ring
    _ = stdNormalPdf (γ * Real.sqrt ρ) *
          (σ * γ * (-(γ * (1 / (2 * Real.sqrt ρ)))) -
            (σ * (1 / (2 * Real.sqrt ρ)) +
              σ * Real.sqrt ρ * (-(γ * Real.sqrt ρ) * (γ * (1 / (2 * Real.sqrt ρ)))))) := by rw [halg]
    _ = σ * γ * (-stdNormalPdf (γ * Real.sqrt ρ) * (γ * (1 / (2 * Real.sqrt ρ)))) -
          (σ * (1 / (2 * Real.sqrt ρ)) * stdNormalPdf (γ * Real.sqrt ρ) +
            σ * Real.sqrt ρ * (-(γ * Real.sqrt ρ) * stdNormalPdf (γ * Real.sqrt ρ) *
              (γ * (1 / (2 * Real.sqrt ρ))))) := by ring

/-- Continuity of `ΔC(ρ)` on `ℝ`. -/
lemma continuous_gaussianPluginNetCost (μ σ : ℝ) :
    Continuous (gaussianPluginNetCost μ σ) := by
  unfold gaussianPluginNetCost
  have htail_cont : Continuous stdNormalTail :=
    continuous_iff_continuousAt.2 fun x => (hasDerivAt_stdNormalTail x).continuousAt
  have hpdf_cont : Continuous stdNormalPdf :=
    continuous_iff_continuousAt.2 fun x => (hasDerivAt_stdNormalPdf x).continuousAt
  have hz_cont : Continuous (fun ρ : ℝ => (μ / σ) * Real.sqrt ρ) :=
    continuous_const.mul Real.continuous_sqrt
  exact (continuous_const.mul (htail_cont.comp hz_cont)).sub
    ((continuous_const.mul Real.continuous_sqrt).mul (hpdf_cont.comp hz_cont))

/-- `ΔC(ρ)` is strictly decreasing on `[0, 1]`. -/
theorem gaussianPluginNetCost_strictAntiOn {μ σ : ℝ} (hσ : 0 < σ) :
    StrictAntiOn (gaussianPluginNetCost μ σ) (Set.Icc 0 1) := by
  apply strictAntiOn_of_deriv_neg (convex_Icc 0 1)
  · exact (continuous_gaussianPluginNetCost μ σ).continuousOn
  · intro ρ hρ
    rw [interior_Icc] at hρ
    rw [(hasDerivAt_gaussianPluginNetCost hσ hρ.1).deriv]
    have hpdf_pos : 0 < stdNormalPdf ((μ / σ) * Real.sqrt ρ) := by
      unfold stdNormalPdf
      exact div_pos (Real.exp_pos _) (Real.sqrt_pos.2 (by positivity))
    have hfac_pos : 0 < σ * stdNormalPdf ((μ / σ) * Real.sqrt ρ) / (2 * Real.sqrt ρ) :=
      div_pos (mul_pos hσ hpdf_pos) (mul_pos (by norm_num) (Real.sqrt_pos.2 hρ.1))
    have hbr_pos : 0 < 1 + (μ / σ) ^ 2 * (1 - ρ) := by
      have : 0 ≤ (μ / σ) ^ 2 * (1 - ρ) := mul_nonneg (sq_nonneg _) (by linarith [hρ.2])
      linarith
    nlinarith

/-- **Exact Gaussian Mills-Ratio Sign-Flip Phase Transition Sandwich Theorem**:
For any `μ > 0` and `σ > 0`, let `ρ_lo = μ² / (μ² + σ²)` and `ρ_hi = μ² / (μ² + (2/π) σ²)`.
Then `0 < ρ_lo < ρ_hi < 1`, and there exists a **unique** sign-flip threshold `ρ* ∈ [0, 1]`
satisfying:
1. `ΔC(ρ*) = 0`;
2. `ρ_lo < ρ* < ρ_hi`;
3. `ΔC(ρ) > 0` for all `ρ ∈ [0, ρ*)` and `ΔC(ρ) < 0` for all `ρ ∈ (ρ*, 1]`. -/
theorem gaussian_mills_sign_flip_phase_transition {μ σ : ℝ} (hμ : 0 < μ) (hσ : 0 < σ) :
    let ρ_lo := μ ^ 2 / (μ ^ 2 + σ ^ 2)
    let ρ_hi := μ ^ 2 / (μ ^ 2 + (2 / Real.pi) * σ ^ 2)
    0 < ρ_lo ∧ ρ_lo < ρ_hi ∧ ρ_hi < 1 ∧
    (∃! ρstar : ℝ, ρstar ∈ Set.Icc (0 : ℝ) 1 ∧ gaussianPluginNetCost μ σ ρstar = 0) ∧
    (∃ ρstar : ℝ, ρ_lo < ρstar ∧ ρstar < ρ_hi ∧ gaussianPluginNetCost μ σ ρstar = 0 ∧
      (∀ ρ ∈ Set.Ico 0 ρstar, 0 < gaussianPluginNetCost μ σ ρ) ∧
      (∀ ρ ∈ Set.Ioc ρstar 1, gaussianPluginNetCost μ σ ρ < 0)) := by
  intro ρ_lo ρ_hi
  have hpi3 : (3 : ℝ) < Real.pi := Real.pi_gt_three
  have hpi_pos : 0 < Real.pi := by linarith
  have h2pi_lt1 : 2 / Real.pi < 1 := (div_lt_one hpi_pos).2 (by linarith)
  have hρlo_pos : 0 < ρ_lo := div_pos (sq_pos_of_pos hμ) (by positivity)
  have hρlo_lt_hi : ρ_lo < ρ_hi := by
    refine div_lt_div_of_pos_left (sq_pos_of_pos hμ) (by positivity) ?_
    nlinarith [sq_pos_of_pos hσ, h2pi_lt1]
  have hρhi_lt1 : ρ_hi < 1 := (div_lt_one (by positivity)).2 (by
    have : 0 < (2 / Real.pi) * σ ^ 2 := mul_pos (div_pos (by norm_num) hpi_pos) (sq_pos_of_pos hσ)
    linarith)
  have hpos_lo : 0 < gaussianPluginNetCost μ σ ρ_lo :=
    gaussianPluginNetCost_pos_of_le_lower_threshold hμ hσ (le_of_lt hρlo_pos) le_rfl
  have hneg_hi : gaussianPluginNetCost μ σ ρ_hi < 0 :=
    gaussianPluginNetCost_neg_of_ge_upper_threshold hμ hσ le_rfl
  have hzero_mem : (0 : ℝ) ∈ Set.Icc (gaussianPluginNetCost μ σ ρ_hi)
      (gaussianPluginNetCost μ σ ρ_lo) := ⟨le_of_lt hneg_hi, le_of_lt hpos_lo⟩
  have hivt := intermediate_value_Icc' (le_of_lt hρlo_lt_hi)
    (continuous_gaussianPluginNetCost μ σ).continuousOn hzero_mem
  rcases hivt with ⟨ρstar, ⟨hlo_le, hhi_le⟩, hzero⟩
  have hlo_lt : ρ_lo < ρstar := by
    rcases eq_or_lt_of_le hlo_le with rfl | hlt
    · linarith
    · exact hlt
  have hhi_lt : ρstar < ρ_hi := by
    rcases eq_or_lt_of_le hhi_le with rfl | hlt
    · linarith
    · exact hlt
  have hstar_mem : ρstar ∈ Set.Icc (0 : ℝ) 1 := ⟨by linarith, by linarith⟩
  have hanti := gaussianPluginNetCost_strictAntiOn (μ := μ) hσ
  refine ⟨hρlo_pos, hρlo_lt_hi, hρhi_lt1, ?_, ?_⟩
  · refine ⟨ρstar, ⟨hstar_mem, hzero⟩, ?_⟩
    rintro r ⟨hr_mem, hr_zero⟩
    rcases lt_trichotomy r ρstar with hlt | rfl | hgt
    · have := hanti hr_mem hstar_mem hlt; linarith
    · rfl
    · have := hanti hstar_mem hr_mem hgt; linarith
  · refine ⟨ρstar, hlo_lt, hhi_lt, hzero, ?_, ?_⟩
    · intro ρ hρ
      have hρ_mem : ρ ∈ Set.Icc (0 : ℝ) 1 := ⟨hρ.1, by linarith [hρ.2]⟩
      have := hanti hρ_mem hstar_mem hρ.2
      linarith
    · intro ρ hρ
      have hρ_mem : ρ ∈ Set.Icc (0 : ℝ) 1 := ⟨by linarith [hρ.1], hρ.2⟩
      have := hanti hstar_mem hρ_mem hρ.1
      linarith

end PreventiveTargetingParadox.Theorem3
