import Mathlib
import FormalVerifML.base.definitions

namespace FormalVerifML

/--
Compute the Euclidean (L2) distance between two vectors represented as Array Float.
--/
def distL2 (x y : Array Float) : Float :=
  let pairs := Array.zip x y
  Float.sqrt <| pairs.foldl (fun acc (xi, yi) => acc + (xi - yi) * (xi - yi)) 0.0

/--
Robust classification property:
A classification function f is robust at level ε if any two inputs within ε (L2 norm) yield the same output.
--/
def robustClass (f : Array Float → Nat) (ε : Float) : Prop :=
  ∀ (x x' : Array Float), distL2 x x' < ε → f x = f x'

/--
Interpretability property:
A function f is interpretable if small changes in the input (bounded by δ) yield
changes in the output that are bounded by η.
--/
def interpretable (f : Array Float → Array Float) (δ η : Float) : Prop :=
  ∀ (x x' : Array Float), distL2 x x' < δ → distL2 (f x) (f x') < η

/--
Monotonicity property:
A function f is monotonic in a designated feature if, when that feature is increased,
the output does not decrease.
--/
def monotonic (f : Array Float → Float) (feature_index : Nat) : Prop :=
  ∀ (x : Array Float) (δ : Float), f (x.modify feature_index (fun v => v + δ)) ≥ f x

/--
Sensitivity analysis property:
The change in the output of a function f is bounded by a constant L times the change in the input.
--/
def sensitivity (f : Array Float → Array Float) (L : Float) : Prop :=
  ∀ (x x' : Array Float), distL2 (f x) (f x') ≤ L * distL2 x x'

/--
Counterfactual fairness:
A classifier f is counterfactually fair if, under a counterfactual transformation cf of an individual,
the classifier's output remains unchanged.
--/
def counterfactual_fairness (Individual : Type) (f : Individual → Nat) (cf : Individual → Individual) : Prop :=
  ∀ ind, f ind = f (cf ind)

end FormalVerifML
