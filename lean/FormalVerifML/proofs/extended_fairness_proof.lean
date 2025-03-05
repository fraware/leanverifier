import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties

open FormalVerifML

namespace ExtendedFairness

/--
Define equalized odds for a classifier `f` over a population.
Here, we assume that if an individual’s ground truth label is 1 (positive),
then the classifier must also output 1—regardless of protected status.
This strong notion of fairness is actionable and can be checked empirically.
--/
def equalized_odds
  (Individual : Type)
  (protected_group : Individual → Prop)
  (f : Individual → Nat)
  (label : Individual → Nat)
  : Prop :=
  ∀ ind, label ind = 1 → f ind = 1

/--
We assume the following axioms for our finite population:
- `Individual` is an arbitrary type.
- `protected_group` is a predicate on individuals.
- `groundTruth` gives the true label of an individual.
--/
axiom Individual : Type
axiom protected_group : Individual → Prop
axiom groundTruth : Individual → Nat

/--
Define a logistic regression classifier.
Given an individual’s features (extracted by `toFeatures`) and a logistic regression model `lm`,
the classifier outputs 1 if the model’s evaluation is nonnegative, and 0 otherwise.
--/
def classify (ind : Individual) (toFeatures : Individual → Array Float) (lm : LinearModel) : Nat :=
  let features := toFeatures ind
  let z := evalLinearModel lm features
  if z ≥ 0 then 1 else 0

/--
Theorem (Extended Fairness):
If for every individual with ground truth 1 the classifier outputs 1,
then the classifier satisfies equalized odds.

We assume as hypothesis `h` that for all individuals with `groundTruth = 1`,
the classifier returns 1. Under that assumption, the fairness condition holds.
--/
theorem log_reg_equalized_odds (toFeatures : Individual → Array Float) (lm : LinearModel)
  (h : ∀ ind, groundTruth ind = 1 → classify ind toFeatures lm = 1) :
  equalized_odds Individual protected_group (classify · toFeatures lm) groundTruth :=
begin
  intros ind h_label,
  apply h ind h_label,
end

end ExtendedFairness
