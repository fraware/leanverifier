import LeanVerifier.Data.Vector
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Affine binary classifier over exact rationals

`score` and `classify` are total and shape-checked by construction via `Fin n`.
-/

namespace LeanVerifier

/-- Affine binary classifier: weights in Rat^n and bias in Rat. -/
structure AffineBinaryClassifier (n : Nat) where
  /-- Weight vector in `Rat^n`. -/
  weights : Vec n
  /-- Affine bias term. -/
  bias : Rat
  deriving DecidableEq

namespace AffineBinaryClassifier

variable {n : Nat}

/-- Exact affine score. -/
def score (M : AffineBinaryClassifier n) (x : Vec n) : Rat :=
  (∑ i : Fin n, M.weights i * x i) + M.bias

/-- Classification is true iff the score is strictly positive. -/
def classify (M : AffineBinaryClassifier n) (x : Vec n) : Bool :=
  decide (0 < score M x)

/-- Convenience: L1 norm of weights. -/
def weightL1 (M : AffineBinaryClassifier n) : Rat :=
  Vec.l1Norm M.weights

@[simp] theorem score_add_bias (M : AffineBinaryClassifier n) (x : Vec n) :
    score M x = (∑ i : Fin n, M.weights i * x i) + M.bias := rfl

/-- Unit: zero model scores to zero. -/
theorem score_zero_model (x : Vec n) :
    score ⟨fun _ => 0, 0⟩ x = 0 := by
  simp [score]

/-- Repeated-weight unit model (guards historical indexOf-style bugs). -/
def repeatedWeightModel : AffineBinaryClassifier 3 :=
  ⟨fun
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => -2,
   0⟩

/-- All-ones input for `repeatedWeightModel` (score cancels to 0). -/
def repeatedWeightInputOnes : Vec 3 := fun _ => 1

/-- Input focusing the first coordinate of `repeatedWeightModel`. -/
def repeatedWeightInputFocus : Vec 3 :=
  fun
  | ⟨0, _⟩ => 2
  | _ => 0

theorem repeated_weight_score_ones :
    score repeatedWeightModel repeatedWeightInputOnes = 0 := by
  native_decide

theorem repeated_weight_classify_ones :
    classify repeatedWeightModel repeatedWeightInputOnes = false := by
  native_decide

theorem repeated_weight_score_focus :
    score repeatedWeightModel repeatedWeightInputFocus = 2 := by
  native_decide

theorem repeated_weight_classify_focus :
    classify repeatedWeightModel repeatedWeightInputFocus = true := by
  native_decide

theorem classify_iff_pos_score (M : AffineBinaryClassifier n) (x : Vec n) :
    classify M x = true ↔ 0 < score M x := by
  simp [classify]

end AffineBinaryClassifier

end LeanVerifier
