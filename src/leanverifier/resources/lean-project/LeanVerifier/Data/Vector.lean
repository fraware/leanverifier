import Mathlib.Data.Rat.Defs
import Mathlib.Data.Fintype.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Algebra.Order.Ring.Abs

/-!
# Finite-dimensional rational vectors

`Vec n` is an exact function `Fin n → Rat`. No floats, no partial indexing.
-/

namespace LeanVerifier

/-- Exact rational vector of length `n`. -/
abbrev Vec (n : Nat) := Fin n → Rat

namespace Vec

/-- L1 norm as a sum of absolute coordinates. -/
def l1Norm {n : Nat} (x : Vec n) : Rat :=
  ∑ i : Fin n, |x i|

@[simp] theorem l1Norm_zero {n : Nat} : l1Norm (fun _ : Fin n => (0 : Rat)) = 0 := by
  simp [l1Norm]

end Vec

end LeanVerifier
