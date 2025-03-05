import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties
import FormalVerifML.generated.decision_tree_model  -- assumed to define myDecisionTree

open FormalVerifML

namespace DecisionTreeProof

/--
Assume that the generated decision tree is defined as follows:
myDecisionTree = DecisionTree.node 0 5.0 (DecisionTree.leaf 0) (DecisionTree.leaf 1)
--/
axiom myDecisionTree_def : myDecisionTree = DecisionTree.node 0 5.0 (DecisionTree.leaf 0) (DecisionTree.leaf 1)

/--
Prove that if the first component of the input vector is greater than 5.0,
then evaluating the decision tree yields 1.
--/
theorem decision_tree_region_classification (x : Array Float) (h : x[0]! > 5.0) :
  evalDecisionTree myDecisionTree x = 1 :=
by
  rw [myDecisionTree_def],
  dsimp [evalDecisionTree],
  -- The definition of evalDecisionTree for a node is:
  -- if x[0]! ≤ 5.0 then evalDecisionTree (leaf 0) x else evalDecisionTree (leaf 1) x.
  have h_not : ¬ (x[0]! ≤ 5.0) := by exact not_le_of_gt h,
  rw if_neg h_not,
  dsimp,
  rfl

end DecisionTreeProof
