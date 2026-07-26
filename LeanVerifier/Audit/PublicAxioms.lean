import LeanVerifier.Theorem.AffineMarginRobustness

/-!
# Public axiom inventory for supported theorems

Supported theorems may depend only on Lean's standard classical axioms:
`propext`, `Classical.choice`, and `Quot.sound`.

Custom axioms, `sorryAx`, dedicated native-decide axioms, and compiler-trust
flags are forbidden. `scripts/audit_axioms.py` enforces this gate in CI for
both library declarations and concrete generated modules.
-/

namespace LeanVerifier.Audit

open AffineBinaryClassifier

/-- Force elaboration of the central theorem for axiom printing. -/
theorem affine_margin_robustness_axioms_anchor
    {n : Nat} (M : AffineBinaryClassifier n) (x : Vec n) (ε : Rat)
    (hε : 0 ≤ ε) (hmargin : ε * weightL1 M < |score M x|) :
    LocallyRobustAt M x ε :=
  affine_margin_robustness M x ε hε hmargin

#print axioms affine_margin_robustness
#print axioms abs_score_sub_score_le
#print axioms classify_true_of_margin
#print axioms classify_false_of_margin
#print axioms repeated_weight_score_ones
#print axioms repeated_weight_classify_ones
#print axioms repeated_weight_score_focus
#print axioms repeated_weight_classify_focus

end LeanVerifier.Audit
