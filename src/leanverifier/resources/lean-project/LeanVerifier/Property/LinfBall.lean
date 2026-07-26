import LeanVerifier.Data.Vector

/-!
# Closed L-infinity balls over rational vectors
-/

namespace LeanVerifier

/-- `y` lies in the closed L-infinity ball of radius `ε` around `x`. -/
def InLinfBall {n : Nat} (x : Vec n) (ε : Rat) (y : Vec n) : Prop :=
  ∀ i : Fin n, |y i - x i| ≤ ε

theorem InLinfBall.refl {n : Nat} (x : Vec n) {ε : Rat} (hε : 0 ≤ ε) :
    InLinfBall x ε x := by
  intro i
  simpa [sub_self, abs_zero] using hε

theorem InLinfBall.mono_radius {n : Nat} {x y : Vec n} {ε δ : Rat}
    (hle : ε ≤ δ) (hy : InLinfBall x ε y) : InLinfBall x δ y := by
  intro i
  exact (hy i).trans hle

end LeanVerifier
