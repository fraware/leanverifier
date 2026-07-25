import LeanVerifier.Property.ClassificationRobustness
import Mathlib.Tactic.Linarith

/-!
# Central L-infinity margin robustness theorem

Decision boundary is the strict half-space `score > 0`.
The equality case `|score(x)| = ε · ‖w‖₁` is intentionally excluded (not certified).
-/

namespace LeanVerifier
namespace AffineBinaryClassifier

variable {n : Nat}

/-- Local robustness at reference point `x` for radius `ε`. -/
def LocallyRobustAt (M : AffineBinaryClassifier n) (x : Vec n) (ε : Rat) : Prop :=
  ∀ y : Vec n, InLinfBall x ε y → classify M y = classify M x

private theorem weightL1_nonneg (M : AffineBinaryClassifier n) : 0 ≤ weightL1 M := by
  simp [weightL1, Vec.l1Norm]
  exact Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- Absolute strict-margin theorem.
If `|score(x)| > ε · ‖w‖₁` with `ε ≥ 0`, classification is invariant on the ball.
Points with `score = 0` are never classified `true` (explicit zero-boundary treatment). -/
theorem affine_margin_robustness (M : AffineBinaryClassifier n) (x : Vec n) (ε : Rat)
    (hε : 0 ≤ ε) (hmargin : ε * weightL1 M < |score M x|) :
    LocallyRobustAt M x ε := by
  intro y hy
  rcases lt_trichotomy (score M x) 0 with hlt | heq | hgt
  · -- score x < 0
    have habs : |score M x| = -(score M x) := abs_of_neg hlt
    have hstrict : score M x < -(ε * weightL1 M) := by
      linarith [hmargin, habs]
    have hyf := classify_false_of_margin M x y ε hε hy hstrict
    have hxf : classify M x = false := by
      change decide (0 < score M x) = false
      exact decide_eq_false (not_lt.mpr (le_of_lt hlt))
    exact hyf.trans hxf.symm
  · -- score x = 0 contradicts strict margin
    have habs0 : |score M x| = 0 := by rw [heq, abs_zero]
    have hnonneg : 0 ≤ ε * weightL1 M := mul_nonneg hε (weightL1_nonneg M)
    linarith [hmargin, habs0, hnonneg]
  · -- 0 < score x
    have habs : |score M x| = score M x := abs_of_pos hgt
    have hstrict : ε * weightL1 M < score M x := by
      rwa [habs] at hmargin
    have hyt := classify_true_of_margin M x y ε hε hy hstrict
    have hxt : classify M x = true := by
      change decide (0 < score M x) = true
      exact decide_eq_true hgt
    exact hyt.trans hxt.symm

/-- Equivalent packaging used by generated instances. -/
theorem classify_eq_of_strict_margin (M : AffineBinaryClassifier n) (x y : Vec n) (ε : Rat)
    (hε : 0 ≤ ε) (hy : InLinfBall x ε y)
    (hmargin : ε * weightL1 M < |score M x|) :
    classify M y = classify M x :=
  affine_margin_robustness M x ε hε hmargin y hy

end AffineBinaryClassifier
end LeanVerifier
