import LeanVerifier.Model.AffineBinary
import LeanVerifier.Property.LinfBall
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Score Lipschitz bound under L-infinity perturbations
-/

namespace LeanVerifier
namespace AffineBinaryClassifier

variable {n : Nat}

theorem score_sub_score (M : AffineBinaryClassifier n) (x y : Vec n) :
    score M y - score M x = ∑ i : Fin n, M.weights i * (y i - x i) := by
  simp only [score]
  calc
    ((∑ i : Fin n, M.weights i * y i) + M.bias) -
        ((∑ i : Fin n, M.weights i * x i) + M.bias)
      = (∑ i : Fin n, M.weights i * y i) - (∑ i : Fin n, M.weights i * x i) := by
            ring
    _ = ∑ i : Fin n, (M.weights i * y i - M.weights i * x i) := by
          rw [Finset.sum_sub_distrib]
    _ = ∑ i : Fin n, M.weights i * (y i - x i) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          ring

theorem abs_score_sub_score_le (M : AffineBinaryClassifier n) (x y : Vec n) (ε : Rat)
    (_hε : 0 ≤ ε) (hy : InLinfBall x ε y) :
    |score M y - score M x| ≤ ε * weightL1 M := by
  have hterm : ∀ i : Fin n, |M.weights i * (y i - x i)| ≤ |M.weights i| * ε := by
    intro i
    have hi := hy i
    calc
      |M.weights i * (y i - x i)| = |M.weights i| * |y i - x i| := by simp [abs_mul]
      _ ≤ |M.weights i| * ε := mul_le_mul_of_nonneg_left hi (abs_nonneg _)
  have hsum :
      |∑ i : Fin n, M.weights i * (y i - x i)| ≤ ∑ i : Fin n, |M.weights i| * ε :=
    calc
      |∑ i : Fin n, M.weights i * (y i - x i)| ≤
          ∑ i : Fin n, |M.weights i * (y i - x i)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i : Fin n, |M.weights i| * ε := Finset.sum_le_sum fun i _ => hterm i
  have hfactor : (∑ i : Fin n, |M.weights i| * ε) = ε * weightL1 M := by
    simp only [weightL1, Vec.l1Norm]
    have sum_mul_right :
        ∀ (s : Finset (Fin n)) (f : Fin n → Rat) (c : Rat),
          ∑ i ∈ s, f i * c = (∑ i ∈ s, f i) * c := by
      intro s f c
      refine Finset.induction_on s ?base ?step
      · simp
      · intro a s ha ih
        simp [Finset.sum_insert ha, ih, add_mul]
    have := sum_mul_right Finset.univ (fun i => |M.weights i|) ε
    rw [this, mul_comm]
  calc
    |score M y - score M x| = |∑ i : Fin n, M.weights i * (y i - x i)| := by
      rw [score_sub_score]
    _ ≤ ∑ i : Fin n, |M.weights i| * ε := hsum
    _ = ε * weightL1 M := hfactor

theorem classify_true_of_margin (M : AffineBinaryClassifier n) (x y : Vec n) (ε : Rat)
    (hε : 0 ≤ ε) (hy : InLinfBall x ε y)
    (hmargin : ε * weightL1 M < score M x) :
    classify M y = true := by
  have hlip := abs_score_sub_score_le M x y ε hε hy
  have hlower : score M x - ε * weightL1 M ≤ score M y := by
    have := neg_le_of_abs_le hlip
    linarith
  have hypos : 0 < score M y := by linarith
  simpa [classify] using hypos

theorem classify_false_of_margin (M : AffineBinaryClassifier n) (x y : Vec n) (ε : Rat)
    (hε : 0 ≤ ε) (hy : InLinfBall x ε y)
    (hmargin : score M x < -(ε * weightL1 M)) :
    classify M y = false := by
  have hlip := abs_score_sub_score_le M x y ε hε hy
  have hupper : score M y ≤ score M x + ε * weightL1 M := by
    have := le_of_abs_le hlip
    linarith
  have hynonpos : ¬ 0 < score M y := by linarith
  change decide (0 < score M y) = false
  exact decide_eq_false hynonpos

end AffineBinaryClassifier
end LeanVerifier
