import FormalVerifML.definitions
import FormalVerifML.ml_properties
import FormalVerifML.generated.example_model  -- assumed to define exampleNeuralNet

open FormalVerifML

namespace AdversarialRobustnessExample

/--
We assume the following two axioms:
1. Lipschitz continuity: The difference in the first output component is bounded by 50 times the L2 distance.
2. Margin condition: For any input, if the first output is nonnegative then it is at least 0.1, and if negative then at most -0.1.
--/
axiom example_net_lipschitz : ∀ x x' : Array Float,
  |(evalNeuralNet exampleNeuralNet x)[0]! - (evalNeuralNet exampleNeuralNet x')[0]!| ≤ 50 * distL2 x x'
axiom example_net_margin : ∀ x : Array Float,
  if (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0 then (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0.1
  else (evalNeuralNet exampleNeuralNet x)[0]! ≤ -0.1

/--
Define the classification function based on exampleNeuralNet:
If the first output component is nonnegative, classify as 1; otherwise, 0.
--/
def classify (x : Array Float) : Nat :=
  if (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0.0 then 1 else 0

/--
Prove that if ε ≤ 0.001 then for any two inputs x and x' within ε (in L2 distance), the classifier remains unchanged.
--/
theorem tiny_epsilon_robust (ε : Float) (hε : ε ≤ 0.001) :
  robustClass classify ε :=
begin
  intros x x' hdist,
  -- Let L = 50 (by assumption) so that the change in the first output is bounded by 50*ε.
  have diff_bound := example_net_lipschitz x x',
  have L_bound : 50 * ε ≤ 50 * 0.001, from mul_le_mul_of_nonneg_right hε (by norm_num),
  have diff_small : |(evalNeuralNet exampleNeuralNet x)[0]! - (evalNeuralNet exampleNeuralNet x')[0]!| < 50 * 0.001,
  { exact lt_of_le_of_lt diff_bound (by norm_num) },
  -- Now, consider the two cases for the sign of (evalNeuralNet exampleNeuralNet x)[0]!.
  by_cases hpos : (evalNeuralNet exampleNeuralNet x)[0]! ≥ 0.0,
  { have margin := example_net_margin x,
    rw if_pos hpos,
    rw if_pos,
    { -- Since the output of x is at least 0.1 and the perturbation is less than 0.05,
      -- the output for x' must remain nonnegative.
      have : (evalNeuralNet exampleNeuralNet x')[0]! ≥ (evalNeuralNet exampleNeuralNet x)[0]! - 50 * 0.001,
      { -- This follows from the triangle inequality.
        apply sub_le_self,
      },
      have nonneg_x' : (evalNeuralNet exampleNeuralNet x')[0]! ≥ 0.1 - 50 * 0.001,
      { linarith [margin] },
      have pos_bound : 0.1 - 50 * 0.001 > 0,
      { norm_num },
      -- Thus, classify x = 1 and classify x' = 1.
      refl },
    { -- Provide the proof that (evalNeuralNet exampleNeuralNet x')[0]! ≥ 0.0 given the margin.
      have : (evalNeuralNet exampleNeuralNet x')[0]! ≥ 0.1 - 50 * 0.001, by linarith [example_net_margin x],
      linarith } },
  { rw if_neg hpos,
    rw if_neg,
    { -- In the negative case, a similar argument shows that the output remains ≤ -0.1.
      have margin := example_net_margin x,
      have : (evalNeuralNet exampleNeuralNet x')[0]! ≤ (evalNeuralNet exampleNeuralNet x)[0]! + 50 * ε,
      { apply le_trans _ (example_net_lipschitz x x').le, },
      have neg_bound : (evalNeuralNet exampleNeuralNet x')[0]! ≤ -0.1 + 50 * 0.001,
      { linarith [margin] },
      have : -0.1 + 50 * 0.001 < 0, by norm_num,
      refl },
    { -- Provide the proof that (evalNeuralNet exampleNeuralNet x')[0]! < 0.
      have : (evalNeuralNet exampleNeuralNet x')[0]! ≤ -0.1 + 50 * ε, by linarith [example_net_lipschitz x x'],
      linarith } }
end

end AdversarialRobustnessExample
