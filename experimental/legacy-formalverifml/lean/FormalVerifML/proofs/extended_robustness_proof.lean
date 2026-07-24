import FormalVerifML.base.definitions
import FormalVerifML.base.ml_properties
import FormalVerifML.generated.example_model  -- assumed to define exampleNeuralNet

open FormalVerifML

namespace ExtendedRobustness

/--
Assume that each layer is Lipschitz continuous with constant 2.
(This is a realistic assumption if, for example, the weights are normalized.)
--/
axiom layer_lipschitz : ∀ (l : LayerType) (x y : Array Float),
  distL2 (evalLayer l x) (evalLayer l y) ≤ 2 * distL2 x y

/--
Prove that the neural network is Lipschitz continuous with constant 2^(n),
where n is the number of layers.
--/
theorem neural_net_lipschitz (nn : NeuralNet) (x x' : Array Float) :
  distL2 (evalNeuralNet nn x) (evalNeuralNet nn x') ≤ (2^(nn.layers.length)) * distL2 x x' :=
begin
  induction nn.layers with
  | nil =>
    simp [evalNeuralNet],
  | cons l ls ih =>
    have h_lip : distL2 (evalLayer l x) (evalLayer l x') ≤ 2 * distL2 x x' :=
      layer_lipschitz l x x',
    let y := evalLayer l x,
    let y' := evalLayer l x',
    have h_comp : distL2 (evalNeuralNet { layers := ls, ..nn } y) (evalNeuralNet { layers := ls, ..nn } y')
         ≤ (2^(ls.length)) * distL2 y y', from ih y y',
    calc
      distL2 (evalNeuralNet nn x) (evalNeuralNet nn x')
          = distL2 (evalNeuralNet { layers := ls, ..nn } (evalLayer l x))
                      (evalNeuralNet { layers := ls, ..nn } (evalLayer l x')) := rfl
      ... ≤ (2^(ls.length)) * distL2 (evalLayer l x) (evalLayer l x') : h_comp
      ... ≤ (2^(ls.length)) * (2 * distL2 x x')              : mul_le_mul_of_nonneg_right h_lip (by norm_num)
      ... = (2^(ls.length + 1)) * distL2 x x'                  : by rw [← Nat.add_one, ← pow_succ']
end

end ExtendedRobustness
