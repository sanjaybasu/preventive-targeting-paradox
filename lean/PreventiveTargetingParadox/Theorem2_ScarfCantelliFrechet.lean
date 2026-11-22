import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

/-!
# Theorem 2: Sharp Scarf–Cantelli Ceiling, Lower-Bounded Tightening, and Fréchet–Hoeffding Coupling

This module formalizes:
1. **Sharp Scarf–Cantelli Oracle Savings Ceiling**: over any finite probability space with mean
   `μ > 0` and variance `σ² ≥ 0`, exact quadratic duality proves
   `-∑ w_i min(θ_i, 0) ≤ (√(μ² + σ²) - μ) / 2 = σ² / (2 (√(μ² + σ²) + μ)) ≤ σ² / (4 μ)`,
   together with an explicit two-point attainment witness on `Fin 2`.
2. **Finite-Support Bhatia–Davis Variance Bound & Lower-Bounded Tightening**:
   - Bhatia–Davis inequality `σ² ≤ (U - μ)(μ - L)` for `L ≤ θ_i ≤ U`;
   - Exact lower-bounded Scarf tightening `-∑ w_i min(θ_i, 0) ≤ (-L * σ²) / ((μ - L)² + σ²)`
     under the exact active-lower-bound regime `L² ≤ μ² + σ²` (`L < 0, μ > 0`), with strict
     improvement over the unconstrained Scarf ceiling whenever `L² < μ² + σ²` and `σ² > 0`,
     an explicit two-point attainment witness on `Fin 2`, and a sharpness theorem proving that
     whenever `L² > μ² + σ²` and `σ > 0` the unconstrained Scarf witness lies in `[L, ∞)` and
     strictly exceeds `(-L * σ²) / ((μ - L)² + σ²)`.
3. **Submodular Fréchet–Hoeffding Coupling Inequality**: positive comonotonic coupling between
   avoided disease cost `S` and unrelated survival cost `M` weakly reduces achievable oracle
   savings `-𝔼[min(θ, 0)]`.
-/

namespace PreventiveTargetingParadox.Theorem2

open scoped BigOperators
open Real

/-! ## Part 2A: Sharp Scarf–Cantelli Oracle Savings Ceiling -/

/-- Exact Scarf quadratic dual majorant: for every `x ∈ ℝ` and `d > 0`,
`-min(x, 0) = max(-x, 0) ≤ (x - d)² / (4 d)`.
Equality holds precisely at `x = -d` and `x = d`. -/
lemma neg_min_le_quadratic_majorant (x d : ℝ) (hd : 0 < d) :
    -min x 0 ≤ (x - d) ^ 2 / (4 * d) := by
  rcases le_total x 0 with hx0 | hx0
  · rw [min_eq_left hx0]
    have hdiff : (x - d) ^ 2 / (4 * d) - (-x) = (x + d) ^ 2 / (4 * d) := by
      field_simp
      ring
    have hnn : 0 ≤ (x + d) ^ 2 / (4 * d) := div_nonneg (sq_nonneg _) (by linarith)
    linarith
  · rw [min_eq_right hx0, neg_zero]
    exact div_nonneg (sq_nonneg _) (by linarith)

/-- Weighted second-moment parallel-axis identity on any finite probability space. -/
lemma weighted_sq_dev_eq {ι : Type*} (s : Finset ι) (w θ : ι → ℝ)
    (hw_sum : ∑ i ∈ s, w i = 1) (d : ℝ) :
    let μ := ∑ i ∈ s, w i * θ i
    ∑ i ∈ s, w i * (θ i - d) ^ 2 = (∑ i ∈ s, w i * (θ i - μ) ^ 2) + (μ - d) ^ 2 := by
  intro μ
  have hmean0 : ∑ i ∈ s, w i * (θ i - μ) = 0 := by
    calc
      ∑ i ∈ s, w i * (θ i - μ)
          = (∑ i ∈ s, w i * θ i) - (∑ i ∈ s, w i * μ) := by
            rw [← Finset.sum_sub_distrib]
            refine Finset.sum_congr rfl fun i _ => ?_
            ring
      _ = μ - (∑ i ∈ s, w i) * μ := by rw [← Finset.sum_mul]
      _ = 0 := by rw [hw_sum, one_mul, sub_self]
  have hexpand : ∀ i, w i * (θ i - d) ^ 2 =
      w i * (θ i - μ) ^ 2 + (2 * (μ - d)) * (w i * (θ i - μ)) + w i * (μ - d) ^ 2 := by
    intro i; ring
  calc
    ∑ i ∈ s, w i * (θ i - d) ^ 2
        = ∑ i ∈ s, (w i * (θ i - μ) ^ 2 + (2 * (μ - d)) * (w i * (θ i - μ)) +
            w i * (μ - d) ^ 2) := Finset.sum_congr rfl fun i _ => hexpand i
    _ = (∑ i ∈ s, w i * (θ i - μ) ^ 2) +
          (2 * (μ - d)) * (∑ i ∈ s, w i * (θ i - μ)) +
          (∑ i ∈ s, w i) * (μ - d) ^ 2 := by
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul]
    _ = (∑ i ∈ s, w i * (θ i - μ) ^ 2) + (μ - d) ^ 2 := by
        rw [hmean0, hw_sum]
        ring

/-- **Sharp Scarf–Cantelli Upper Bound**:
For any finite probability distribution with mean `μ > 0` and variance `σ²`,
`-∑ w_i min(θ_i, 0) ≤ (√(μ² + σ²) - μ) / 2`. -/
theorem scarf_cantelli_upper_bound {ι : Type*} (s : Finset ι) (w θ : ι → ℝ)
    (hw_nonneg : ∀ i ∈ s, 0 ≤ w i) (hw_sum : ∑ i ∈ s, w i = 1)
    (hμ : 0 < ∑ i ∈ s, w i * θ i) :
    let μ := ∑ i ∈ s, w i * θ i;
    let σsq := ∑ i ∈ s, w i * (θ i - μ) ^ 2;
    -∑ i ∈ s, w i * min (θ i) 0 ≤ (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 := by
  intro μ σsq
  have hσ_nonneg : 0 ≤ σsq :=
    Finset.sum_nonneg fun i hi => mul_nonneg (hw_nonneg i hi) (sq_nonneg _)
  have hrad_pos : 0 < μ ^ 2 + σsq := by nlinarith
  set d := Real.sqrt (μ ^ 2 + σsq)
  have hd_pos : 0 < d := Real.sqrt_pos.2 hrad_pos
  have hd_sq : d ^ 2 = μ ^ 2 + σsq := Real.sq_sqrt (le_of_lt hrad_pos)
  have hstep1 : -∑ i ∈ s, w i * min (θ i) 0 ≤ ∑ i ∈ s, w i * ((θ i - d) ^ 2 / (4 * d)) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun i hi => ?_
    have hrew : -(w i * min (θ i) 0) = w i * (-min (θ i) 0) := by ring
    rw [hrew]
    exact mul_le_mul_of_nonneg_left (neg_min_le_quadratic_majorant (θ i) d hd_pos) (hw_nonneg i hi)
  have hstep2 : ∑ i ∈ s, w i * ((θ i - d) ^ 2 / (4 * d)) =
      (σsq + (μ - d) ^ 2) / (4 * d) := by
    calc
      ∑ i ∈ s, w i * ((θ i - d) ^ 2 / (4 * d))
          = ∑ i ∈ s, (w i * (θ i - d) ^ 2) / (4 * d) :=
            Finset.sum_congr rfl fun i _ => by ring
      _ = (∑ i ∈ s, w i * (θ i - d) ^ 2) / (4 * d) := by rw [← Finset.sum_div]
      _ = (σsq + (μ - d) ^ 2) / (4 * d) := by rw [weighted_sq_dev_eq s w θ hw_sum d]
  have hstep3 : (σsq + (μ - d) ^ 2) / (4 * d) = (d - μ) / 2 := by
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    nlinarith [hd_sq]
  linarith

/-- Rationalized form of the Scarf–Cantelli ceiling:
`(√(μ² + σ²) - μ) / 2 = σ² / (2 * (√(μ² + σ²) + μ))`. -/
theorem scarf_cantelli_rationalized_eq {μ σsq : ℝ} (hμ : 0 < μ) (hσ : 0 ≤ σsq) :
    (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 =
      σsq / (2 * (Real.sqrt (μ ^ 2 + σsq) + μ)) := by
  have hrad : 0 ≤ μ ^ 2 + σsq := by nlinarith
  have hd_nonneg : 0 ≤ Real.sqrt (μ ^ 2 + σsq) := Real.sqrt_nonneg _
  have hden : Real.sqrt (μ ^ 2 + σsq) + μ ≠ 0 := by linarith
  have hsq : Real.sqrt (μ ^ 2 + σsq) ^ 2 = μ ^ 2 + σsq := Real.sq_sqrt hrad
  field_simp
  nlinarith [hsq]

/-- Elementary quadratic-ratio upper bound:
`(√(μ² + σ²) - μ) / 2 ≤ σ² / (4 * μ)`. -/
theorem scarf_cantelli_le_quad {μ σsq : ℝ} (hμ : 0 < μ) (hσ : 0 ≤ σsq) :
    (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 ≤ σsq / (4 * μ) := by
  rw [scarf_cantelli_rationalized_eq hμ hσ]
  have hrad_ge : μ ≤ Real.sqrt (μ ^ 2 + σsq) := by
    calc
      μ = Real.sqrt (μ ^ 2) := (Real.sqrt_sq (le_of_lt hμ)).symm
      _ ≤ Real.sqrt (μ ^ 2 + σsq) := Real.sqrt_le_sqrt (by linarith)
  have hden_pos : 0 < 4 * μ := by linarith
  have hden_le : 4 * μ ≤ 2 * (Real.sqrt (μ ^ 2 + σsq) + μ) := by linarith
  exact div_le_div_of_nonneg_left hσ hden_pos hden_le

/-- Combined Scarf–Cantelli ceiling chain uniting the upper bound, rationalized identity,
and `σ² / (4 μ)` corollary. -/
theorem scarf_cantelli_ceiling_chain {ι : Type*} (s : Finset ι) (w θ : ι → ℝ)
    (hw_nonneg : ∀ i ∈ s, 0 ≤ w i) (hw_sum : ∑ i ∈ s, w i = 1)
    (hμ : 0 < ∑ i ∈ s, w i * θ i) :
    let μ := ∑ i ∈ s, w i * θ i;
    let σsq := ∑ i ∈ s, w i * (θ i - μ) ^ 2;
    -∑ i ∈ s, w i * min (θ i) 0 ≤ (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 ∧
    (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 = σsq / (2 * (Real.sqrt (μ ^ 2 + σsq) + μ)) ∧
    σsq / (2 * (Real.sqrt (μ ^ 2 + σsq) + μ)) ≤ σsq / (4 * μ) := by
  intro μ σsq
  have hσ : 0 ≤ σsq :=
    Finset.sum_nonneg fun i hi => mul_nonneg (hw_nonneg i hi) (sq_nonneg _)
  have h1 := scarf_cantelli_upper_bound s w θ hw_nonneg hw_sum hμ
  have h2 := scarf_cantelli_rationalized_eq hμ hσ
  have h3 := scarf_cantelli_le_quad hμ hσ
  exact ⟨h1, h2, by linarith⟩

/-- **Exact Attainment Witness (Scarf–Cantelli Ceiling)**:
For any given `μ > 0` and `σ > 0`, there exists an explicit two-point probability distribution
on `Fin 2` with mean `μ` and variance `σ²` attaining exact equality
`-∑ w_i min(θ_i, 0) = (√(μ² + σ²) - μ) / 2`. -/
theorem scarf_cantelli_attainment_witness {μ σ : ℝ} (hμ : 0 < μ) (hσ : 0 < σ) :
    ∃ (w θ : Fin 2 → ℝ),
      (∀ i, 0 ≤ w i) ∧
      (∑ i : Fin 2, w i = 1) ∧
      (∑ i : Fin 2, w i * θ i = μ) ∧
      (∑ i : Fin 2, w i * (θ i - μ) ^ 2 = σ ^ 2) ∧
      (-∑ i : Fin 2, w i * min (θ i) 0 = (Real.sqrt (μ ^ 2 + σ ^ 2) - μ) / 2) := by
  set d := Real.sqrt (μ ^ 2 + σ ^ 2)
  have hrad_pos : 0 < μ ^ 2 + σ ^ 2 := by positivity
  have hd_pos : 0 < d := Real.sqrt_pos.2 hrad_pos
  have hd_sq : d ^ 2 = μ ^ 2 + σ ^ 2 := Real.sq_sqrt (le_of_lt hrad_pos)
  have hd_gt_μ : μ < d := by
    have : μ = Real.sqrt (μ ^ 2) := (Real.sqrt_sq (le_of_lt hμ)).symm
    rw [this]
    exact Real.sqrt_lt_sqrt (sq_nonneg _) (by nlinarith)
  refine ⟨![(d - μ) / (2 * d), (d + μ) / (2 * d)], ![-d, d], ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i
    · exact div_nonneg (by linarith) (by linarith)
    · exact div_nonneg (by linarith) (by linarith)
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    ring
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    ring
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    nlinarith [hd_sq]
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    rw [min_eq_left (by linarith : -d ≤ 0), min_eq_right (le_of_lt hd_pos)]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    ring

/-! ## Part 2B: Finite-Support Bhatia–Davis Bound & Lower-Bounded Tightening -/

/-- **Bhatia–Davis Variance Bound**: on any finite probability space supported in `[L, U]`,
the variance satisfies `σ² ≤ (U - μ)(μ - L)`. -/
theorem bhatia_davis_variance_bound {ι : Type*} (s : Finset ι) (w θ : ι → ℝ) {L U : ℝ}
    (hw_nonneg : ∀ i ∈ s, 0 ≤ w i) (hw_sum : ∑ i ∈ s, w i = 1)
    (hsupp : ∀ i ∈ s, L ≤ θ i ∧ θ i ≤ U) :
    let μ := ∑ i ∈ s, w i * θ i
    ∑ i ∈ s, w i * (θ i - μ) ^ 2 ≤ (U - μ) * (μ - L) := by
  intro μ
  have hmean0 : ∑ i ∈ s, w i * (θ i - μ) = 0 := by
    calc
      ∑ i ∈ s, w i * (θ i - μ)
          = (∑ i ∈ s, w i * θ i) - (∑ i ∈ s, w i) * μ := by
            rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
            refine Finset.sum_congr rfl fun i _ => ?_
            ring
      _ = 0 := by rw [hw_sum, one_mul, sub_self]
  have hnonneg : 0 ≤ ∑ i ∈ s, w i * ((U - θ i) * (θ i - L)) :=
    Finset.sum_nonneg fun i hi =>
      mul_nonneg (hw_nonneg i hi)
        (mul_nonneg (sub_nonneg.2 (hsupp i hi).2) (sub_nonneg.2 (hsupp i hi).1))
  have hident : ∑ i ∈ s, w i * ((U - θ i) * (θ i - L)) =
      (U - μ) * (μ - L) - ∑ i ∈ s, w i * (θ i - μ) ^ 2 := by
    have hpt : ∀ i, w i * ((U - θ i) * (θ i - L)) =
        w i * ((U - μ) * (μ - L)) - w i * (θ i - μ) ^ 2 +
        (U + L - 2 * μ) * (w i * (θ i - μ)) := by
      intro i; ring
    calc
      ∑ i ∈ s, w i * ((U - θ i) * (θ i - L))
          = ∑ i ∈ s, (w i * ((U - μ) * (μ - L)) - w i * (θ i - μ) ^ 2 +
              (U + L - 2 * μ) * (w i * (θ i - μ))) := Finset.sum_congr rfl fun i _ => hpt i
      _ = (∑ i ∈ s, w i) * ((U - μ) * (μ - L)) - (∑ i ∈ s, w i * (θ i - μ) ^ 2) +
            (U + L - 2 * μ) * (∑ i ∈ s, w i * (θ i - μ)) := by
          rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.mul_sum]
      _ = (U - μ) * (μ - L) - ∑ i ∈ s, w i * (θ i - μ) ^ 2 := by
          rw [hw_sum, hmean0]
          ring
  linarith

/-- Lower-bounded quadratic dual majorant on `[L, ∞)`:
when `L < 0` and `x₊ ≥ -L`, the quadratic `q(x) = (-L)/(x₊ - L)² * (x - x₊)²` majorizes
`-min(x, 0)` for all `x ≥ L`, with exact equality at `x = L` and `x = x₊`. -/
lemma neg_min_le_lower_bounded_majorant {x L x_plus : ℝ}
    (hL : L < 0) (hx_plus : -L ≤ x_plus) (hx : L ≤ x) :
    -min x 0 ≤ (-L) / (x_plus - L) ^ 2 * (x - x_plus) ^ 2 := by
  have hden_pos : 0 < x_plus - L := by linarith
  have hden_sq : 0 < (x_plus - L) ^ 2 := sq_pos_of_pos hden_pos
  rcases le_total x 0 with hx0 | hx0
  · rw [min_eq_left hx0]
    have hdiff : (-L) / (x_plus - L) ^ 2 * (x - x_plus) ^ 2 - (-x) =
        (x - L) * ((x_plus + L) / (x_plus - L) + (-L) / (x_plus - L) ^ 2 * (x - L)) := by
      field_simp
      ring
    have hu : 0 ≤ x - L := sub_nonneg.2 hx
    have ht1 : 0 ≤ (x_plus + L) / (x_plus - L) := div_nonneg (by linarith) (le_of_lt hden_pos)
    have ht2 : 0 ≤ (-L) / (x_plus - L) ^ 2 * (x - L) :=
      mul_nonneg (div_nonneg (by linarith) (le_of_lt hden_sq)) hu
    have hnn : 0 ≤ (-L) / (x_plus - L) ^ 2 * (x - x_plus) ^ 2 - (-x) := by
      rw [hdiff]
      exact mul_nonneg hu (add_nonneg ht1 ht2)
    linarith
  · rw [min_eq_right hx0, neg_zero]
    exact mul_nonneg (div_nonneg (by linarith) (le_of_lt hden_sq)) (sq_nonneg _)

/-- **Exact Lower-Bounded Scarf–Cantelli Tightening**:
When `∀ i ∈ s, L ≤ θ_i` with `L < 0`, `μ > 0`, and the active-lower-bound regime condition
`L² ≤ μ² + σ²` (equivalently `-L ≤ √(μ² + σ²)`),
`-∑ w_i min(θ_i, 0) ≤ (-L * σ²) / ((μ - L)² + σ²)`. -/
theorem lower_bounded_scarf_upper_bound {ι : Type*} (s : Finset ι) (w θ : ι → ℝ) {L : ℝ}
    (hw_nonneg : ∀ i ∈ s, 0 ≤ w i) (hw_sum : ∑ i ∈ s, w i = 1)
    (hL : L < 0) (hθ_ge : ∀ i ∈ s, L ≤ θ i)
    (hμ : 0 < ∑ i ∈ s, w i * θ i)
    (hreg : L ^ 2 ≤ (∑ i ∈ s, w i * θ i) ^ 2 +
      ∑ i ∈ s, w i * (θ i - ∑ j ∈ s, w j * θ j) ^ 2) :
    let μ := ∑ i ∈ s, w i * θ i;
    let σsq := ∑ i ∈ s, w i * (θ i - μ) ^ 2;
    -∑ i ∈ s, w i * min (θ i) 0 ≤ (-L * σsq) / ((μ - L) ^ 2 + σsq) := by
  intro μ σsq
  have hσ_nonneg : 0 ≤ σsq :=
    Finset.sum_nonneg fun i hi => mul_nonneg (hw_nonneg i hi) (sq_nonneg _)
  have hμL : 0 < μ - L := by linarith
  set x_plus := μ + σsq / (μ - L)
  have hx_plus : -L ≤ x_plus := by
    have hnum : 0 ≤ (μ - L) * (μ + L) + σsq := by nlinarith [hreg]
    have hrew : x_plus + L = ((μ - L) * (μ + L) + σsq) / (μ - L) := by
      simp only [x_plus]
      field_simp
      ring
    have hpos : 0 ≤ x_plus + L := by
      rw [hrew]
      exact div_nonneg hnum (le_of_lt hμL)
    linarith
  have hstep1 : -∑ i ∈ s, w i * min (θ i) 0 ≤
      ∑ i ∈ s, w i * ((-L) / (x_plus - L) ^ 2 * (θ i - x_plus) ^ 2) := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_le_sum fun i hi => ?_
    have hrew : -(w i * min (θ i) 0) = w i * (-min (θ i) 0) := by ring
    rw [hrew]
    exact mul_le_mul_of_nonneg_left
      (neg_min_le_lower_bounded_majorant hL hx_plus (hθ_ge i hi)) (hw_nonneg i hi)
  have hstep2 : ∑ i ∈ s, w i * ((-L) / (x_plus - L) ^ 2 * (θ i - x_plus) ^ 2) =
      (-L) / (x_plus - L) ^ 2 * (σsq + (μ - x_plus) ^ 2) := by
    calc
      ∑ i ∈ s, w i * ((-L) / (x_plus - L) ^ 2 * (θ i - x_plus) ^ 2)
          = ∑ i ∈ s, (-L) / (x_plus - L) ^ 2 * (w i * (θ i - x_plus) ^ 2) :=
            Finset.sum_congr rfl fun i _ => by ring
      _ = (-L) / (x_plus - L) ^ 2 * (∑ i ∈ s, w i * (θ i - x_plus) ^ 2) := by
            rw [← Finset.mul_sum]
      _ = (-L) / (x_plus - L) ^ 2 * (σsq + (μ - x_plus) ^ 2) := by
            rw [weighted_sq_dev_eq s w θ hw_sum x_plus]
  have hstep3 : (-L) / (x_plus - L) ^ 2 * (σsq + (μ - x_plus) ^ 2) =
      (-L * σsq) / ((μ - L) ^ 2 + σsq) := by
    have hμL_ne : μ - L ≠ 0 := ne_of_gt hμL
    have hden_pos : 0 < (μ - L) ^ 2 + σsq := by positivity
    have hden_ne : (μ - L) ^ 2 + σsq ≠ 0 := ne_of_gt hden_pos
    have hxL : x_plus - L = ((μ - L) ^ 2 + σsq) / (μ - L) := by
      simp only [x_plus]
      field_simp
      ring
    have hμx : μ - x_plus = -σsq / (μ - L) := by
      simp only [x_plus]
      field_simp
    rw [hxL, hμx]
    field_simp
    ring
  linarith

/-- Corollary: whenever the maximum possible per-patient savings `-L` does not exceed the
mean net cost `μ` (`-L ≤ μ`), the active regime condition `L² ≤ μ² + σ²` holds automatically. -/
theorem lower_bounded_scarf_of_neg_L_le_mu {ι : Type*} (s : Finset ι) (w θ : ι → ℝ) {L : ℝ}
    (hw_nonneg : ∀ i ∈ s, 0 ≤ w i) (hw_sum : ∑ i ∈ s, w i = 1)
    (hL : L < 0) (hθ_ge : ∀ i ∈ s, L ≤ θ i)
    (hμ : 0 < ∑ i ∈ s, w i * θ i)
    (hLμ : -L ≤ ∑ i ∈ s, w i * θ i) :
    let μ := ∑ i ∈ s, w i * θ i;
    let σsq := ∑ i ∈ s, w i * (θ i - μ) ^ 2;
    -∑ i ∈ s, w i * min (θ i) 0 ≤ (-L * σsq) / ((μ - L) ^ 2 + σsq) := by
  have hσ_nonneg : 0 ≤ ∑ i ∈ s, w i * (θ i - ∑ j ∈ s, w j * θ j) ^ 2 :=
    Finset.sum_nonneg fun i hi => mul_nonneg (hw_nonneg i hi) (sq_nonneg _)
  have hreg : L ^ 2 ≤ (∑ i ∈ s, w i * θ i) ^ 2 +
      ∑ i ∈ s, w i * (θ i - ∑ j ∈ s, w j * θ j) ^ 2 := by nlinarith
  exact lower_bounded_scarf_upper_bound s w θ hw_nonneg hw_sum hL hθ_ge hμ hreg

/-- The lower-bounded formula `(-L * σ²) / ((μ - L)² + σ²)` is always `≤` the unconstrained
Scarf–Cantelli ceiling `(√(μ² + σ²) - μ) / 2`, with strict inequality whenever `σ² > 0`
and `L² ≠ μ² + σ²`. -/
theorem lower_bounded_scarf_le_unconstrained {L μ σsq : ℝ}
    (hL : L < 0) (hμ : 0 < μ) (hσ : 0 ≤ σsq) :
    (-L * σsq) / ((μ - L) ^ 2 + σsq) ≤ (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 := by
  rw [scarf_cantelli_rationalized_eq hμ hσ]
  set d := Real.sqrt (μ ^ 2 + σsq)
  have hd_pos : 0 < d := Real.sqrt_pos.2 (by nlinarith)
  have hd_sq : d ^ 2 = μ ^ 2 + σsq := Real.sq_sqrt (by nlinarith)
  have hμL : 0 < μ - L := by linarith
  have hden1 : 0 < (μ - L) ^ 2 + σsq := by nlinarith
  have hden2 : 0 < 2 * (d + μ) := by linarith
  rw [div_le_div_iff₀ hden1 hden2]
  have hkey : 0 ≤ σsq * (d + L) ^ 2 := mul_nonneg hσ (sq_nonneg _)
  nlinarith [hd_sq, hkey]

/-- Strict gap between the unconstrained Scarf ceiling and `(-L * σ²) / ((μ - L)² + σ²)`
whenever `σ² > 0` and `-L ≠ √(μ² + σ²)` (i.e. `L² ≠ μ² + σ²`). -/
theorem lower_bounded_scarf_lt_unconstrained {L μ σsq : ℝ}
    (hL : L < 0) (hμ : 0 < μ) (hσ : 0 < σsq) (hne : L ^ 2 ≠ μ ^ 2 + σsq) :
    (-L * σsq) / ((μ - L) ^ 2 + σsq) < (Real.sqrt (μ ^ 2 + σsq) - μ) / 2 := by
  rw [scarf_cantelli_rationalized_eq hμ (le_of_lt hσ)]
  set d := Real.sqrt (μ ^ 2 + σsq)
  have hd_pos : 0 < d := Real.sqrt_pos.2 (by nlinarith)
  have hd_sq : d ^ 2 = μ ^ 2 + σsq := Real.sq_sqrt (by nlinarith)
  have hdL_ne : d + L ≠ 0 := by
    intro hzero
    have hL_eq : L = -d := by linarith
    apply hne
    rw [hL_eq]
    nlinarith [hd_sq]
  have hμL : 0 < μ - L := by linarith
  have hden1 : 0 < (μ - L) ^ 2 + σsq := by nlinarith
  have hden2 : 0 < 2 * (d + μ) := by linarith
  rw [div_lt_div_iff₀ hden1 hden2]
  have hkey : 0 < σsq * (d + L) ^ 2 := mul_pos hσ (sq_pos_of_ne_zero hdL_ne)
  nlinarith [hd_sq, hkey]

/-- **Exact Attainment Witness (Lower-Bounded Scarf–Cantelli Tightening)**:
For any `L < 0`, `μ > 0`, `σ > 0`, there exists an explicit two-point probability distribution
on `Fin 2` supported in `[L, ∞)` with mean `μ` and variance `σ²` attaining exact equality
`-∑ w_i min(θ_i, 0) = (-L * σ²) / ((μ - L)² + σ²)`. -/
theorem lower_bounded_scarf_attainment_witness {L μ σ : ℝ}
    (hL : L < 0) (hμ : 0 < μ) (hσ : 0 < σ) :
    ∃ (w θ : Fin 2 → ℝ),
      (∀ i, 0 ≤ w i) ∧
      (∑ i : Fin 2, w i = 1) ∧
      (∀ i, L ≤ θ i) ∧
      (∑ i : Fin 2, w i * θ i = μ) ∧
      (∑ i : Fin 2, w i * (θ i - μ) ^ 2 = σ ^ 2) ∧
      (-∑ i : Fin 2, w i * min (θ i) 0 = (-L * σ ^ 2) / ((μ - L) ^ 2 + σ ^ 2)) := by
  have hμL : 0 < μ - L := by linarith
  have hden_pos : 0 < (μ - L) ^ 2 + σ ^ 2 := by positivity
  set x_plus := μ + σ ^ 2 / (μ - L)
  have hx_pos : 0 < x_plus := by
    simp only [x_plus]
    have : 0 < σ ^ 2 / (μ - L) := div_pos (sq_pos_of_pos hσ) hμL
    linarith
  refine ⟨![σ ^ 2 / ((μ - L) ^ 2 + σ ^ 2), (μ - L) ^ 2 / ((μ - L) ^ 2 + σ ^ 2)],
    ![L, x_plus], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i
    · exact div_nonneg (sq_nonneg _) (le_of_lt hden_pos)
    · exact div_nonneg (sq_nonneg _) (le_of_lt hden_pos)
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hden_ne : (μ - L) ^ 2 + σ ^ 2 ≠ 0 := ne_of_gt hden_pos
    field_simp
    ring
  · intro i
    fin_cases i
    · simp
    · simp only [Fin.isValue, Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero]
      linarith
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      x_plus]
    have hμL_ne : μ - L ≠ 0 := ne_of_gt hμL
    have hden_ne : (μ - L) ^ 2 + σ ^ 2 ≠ 0 := ne_of_gt hden_pos
    field_simp
    ring
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      x_plus]
    have hμL_ne : μ - L ≠ 0 := ne_of_gt hμL
    have hden_ne : (μ - L) ^ 2 + σ ^ 2 ≠ 0 := ne_of_gt hden_pos
    field_simp
    ring
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    rw [min_eq_left (le_of_lt hL), min_eq_right (le_of_lt hx_pos)]
    ring

/-- **Exact Regime Boundary Sharpness**:
Whenever `μ² + σ² < L²` (i.e., `L < -√(μ² + σ²)`) and `σ > 0`, the lower bound `L` is
non-binding: the unconstrained Scarf two-point distribution lies strictly inside `[L, ∞)` and
achieves oracle savings strictly greater than `(-L * σ²) / ((μ - L)² + σ²)`. This establishes
that `L² ≤ μ² + σ²` is both necessary and sufficient for `lower_bounded_scarf_upper_bound`. -/
theorem lower_bounded_scarf_regime_sharpness {L μ σ : ℝ}
    (hL : L < 0) (hμ : 0 < μ) (hσ : 0 < σ) (hgt : μ ^ 2 + σ ^ 2 < L ^ 2) :
    ∃ (w θ : Fin 2 → ℝ),
      (∀ i, 0 ≤ w i) ∧
      (∑ i : Fin 2, w i = 1) ∧
      (∀ i, L ≤ θ i) ∧
      (∑ i : Fin 2, w i * θ i = μ) ∧
      (∑ i : Fin 2, w i * (θ i - μ) ^ 2 = σ ^ 2) ∧
      (-L * σ ^ 2) / ((μ - L) ^ 2 + σ ^ 2) < -∑ i : Fin 2, w i * min (θ i) 0 := by
  set d := Real.sqrt (μ ^ 2 + σ ^ 2)
  have hrad_pos : 0 < μ ^ 2 + σ ^ 2 := by positivity
  have hd_pos : 0 < d := Real.sqrt_pos.2 hrad_pos
  have hd_sq : d ^ 2 = μ ^ 2 + σ ^ 2 := Real.sq_sqrt (le_of_lt hrad_pos)
  have hd_lt_negL : d < -L := by
    have : -L = Real.sqrt (L ^ 2) := by
      rw [Real.sqrt_sq_eq_abs, abs_of_neg hL]
    rw [this]
    exact Real.sqrt_lt_sqrt (le_of_lt hrad_pos) hgt
  have hd_gt_μ : μ < d := by
    calc
      μ = Real.sqrt (μ ^ 2) := (Real.sqrt_sq (le_of_lt hμ)).symm
      _ < d := Real.sqrt_lt_sqrt (sq_nonneg _) (by nlinarith)
  refine ⟨![(d - μ) / (2 * d), (d + μ) / (2 * d)], ![-d, d], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    fin_cases i
    · exact div_nonneg (by linarith) (by linarith)
    · exact div_nonneg (by linarith) (by linarith)
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    ring
  · intro i
    fin_cases i
    · simp only [Fin.isValue, Fin.mk_zero, Matrix.cons_val_zero]; linarith
    · simp only [Fin.isValue, Fin.mk_one, Matrix.cons_val_one, Matrix.cons_val_zero]; linarith
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    ring
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    have hd_ne : d ≠ 0 := ne_of_gt hd_pos
    field_simp
    nlinarith [hd_sq]
  · simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons]
    rw [min_eq_left (by linarith : -d ≤ 0), min_eq_right (le_of_lt hd_pos)]
    have hval : -((d - μ) / (2 * d) * -d + (d + μ) / (2 * d) * 0) = (d - μ) / 2 := by
      have hd_ne : d ≠ 0 := ne_of_gt hd_pos
      field_simp
      ring
    rw [hval]
    exact lower_bounded_scarf_lt_unconstrained hL hμ (sq_pos_of_pos hσ) (ne_of_gt hgt)

/-! ## Part 2C: Submodular Fréchet–Hoeffding Coupling Inequality -/

/-- **Submodular Fréchet–Hoeffding Coupling Inequality**:
For all `c₀, s₁, s₂, m₁, m₂ ∈ ℝ` with `s₁ ≤ s₂` and `m₁ ≤ m₂`,
`min(c₀ - s₁ + m₂, 0) + min(c₀ - s₂ + m₁, 0) ≤ min(c₀ - s₁ + m₁, 0) + min(c₀ - s₂ + m₂, 0)`. -/
theorem frechet_hoeffding_submodular_min (c0 s1 s2 m1 m2 : ℝ)
    (hs : s1 ≤ s2) (hm : m1 ≤ m2) :
    min (c0 - s1 + m2) 0 + min (c0 - s2 + m1) 0 ≤
      min (c0 - s1 + m1) 0 + min (c0 - s2 + m2) 0 := by
  have h1 : min (c0 - s1 + m2) 0 ≤ c0 - s1 + m2 := min_le_left _ _
  have h2 : min (c0 - s1 + m2) 0 ≤ 0 := min_le_right _ _
  have h3 : min (c0 - s2 + m1) 0 ≤ c0 - s2 + m1 := min_le_left _ _
  have h4 : min (c0 - s2 + m1) 0 ≤ 0 := min_le_right _ _
  rcases le_total (c0 - s1 + m1) 0 with h11 | h11 <;>
  rcases le_total (c0 - s2 + m2) 0 with h22 | h22
  · rw [min_eq_left h11, min_eq_left h22]; linarith
  · rw [min_eq_left h11, min_eq_right h22]; linarith
  · rw [min_eq_right h11, min_eq_left h22]; linarith
  · rw [min_eq_right h11, min_eq_right h22]; linarith

/-- Equivalent oracle-savings form: positive comonotonic coupling `(s₁, m₁), (s₂, m₂)` weakly
reduces achievable oracle savings `-min(θ, 0)` relative to countermonotonic coupling
`(s₁, m₂), (s₂, m₁)`. -/
theorem frechet_hoeffding_comonotonic_reduces_savings (c0 s1 s2 m1 m2 : ℝ)
    (hs : s1 ≤ s2) (hm : m1 ≤ m2) :
    (-min (c0 - s1 + m1) 0) + (-min (c0 - s2 + m2) 0) ≤
      (-min (c0 - s1 + m2) 0) + (-min (c0 - s2 + m1) 0) := by
  linarith [frechet_hoeffding_submodular_min c0 s1 s2 m1 m2 hs hm]

/-- Equal-weight 2-patient expectation corollary: replacing countermonotonic pairing with
comonotonic pairing between avoided disease offset `S` and unrelated survival cost `M` weakly
reduces average oracle savings `-(1/2) * (min θ₁ 0 + min θ₂ 0)`. -/
theorem frechet_hoeffding_two_patient_expectation (c0 s1 s2 m1 m2 : ℝ)
    (hs : s1 ≤ s2) (hm : m1 ≤ m2) :
    -(1 / 2 : ℝ) * (min (c0 - s1 + m1) 0 + min (c0 - s2 + m2) 0) ≤
      -(1 / 2 : ℝ) * (min (c0 - s1 + m2) 0 + min (c0 - s2 + m1) 0) := by
  linarith [frechet_hoeffding_submodular_min c0 s1 s2 m1 m2 hs hm]

end PreventiveTargetingParadox.Theorem2
