import Mathlib

namespace FormalVerifML

/--
A generic feed-forward neural network with a fixed input and output dimension.
Layers are modeled as either linear transformations or activation functions.
--/
inductive LayerType
| linear (weight : Array (Array Float)) (bias : Array Float)
| relu
| sigmoid
| tanh
-- Additional activation layers may be added here.

open LayerType

/--
Structure representing a feed-forward neural network.
--/
structure NeuralNet where
  inputDim  : Nat
  outputDim : Nat
  layers    : List LayerType

/--
Evaluate a single linear layer on an input vector.
This naive implementation computes a dot product for each output neuron and adds the bias.
--/
def evalLinear (w : Array (Array Float)) (b : Array Float) (x : Array Float) : Array Float :=
  let mut out := #[]
  for i in [0 : w.size] do
    let row := w[i]!
    let rowVal := row.foldl (fun acc (w_ij : Float) => acc + w_ij * x[row.indexOf w_ij]!) 0.0
    out := out.push (rowVal + b[i]!)
  out

/--
Evaluate an activation function on an input vector.
--/
def evalActivation (layer : LayerType) (x : Array Float) : Array Float :=
  match layer with
  | relu    => x.map (fun v => if v < 0.0 then 0.0 else v)
  | sigmoid => x.map (fun v => 1.0 / (1.0 + Float.exp (-v)))
  | tanh    => x.map (fun v => Float.tanh v)
  | _       => x  -- For non-activation layers, return x unchanged

/--
Evaluate one layer of the network, dispatching between linear and activation functions.
--/
def evalLayer (l : LayerType) (x : Array Float) : Array Float :=
  match l with
  | linear w b => evalLinear w b x
  | relu       => evalActivation relu x
  | sigmoid    => evalActivation sigmoid x
  | tanh       => evalActivation tanh x

/--
Evaluate the entire neural network on an input vector by sequentially applying each layer.
--/
def evalNeuralNet (nn : NeuralNet) (x : Array Float) : Array Float :=
  nn.layers.foldl (fun acc layer => evalLayer layer acc) x

/--
A linear (logistic) model represented by a weight vector and a bias.
--/
structure LinearModel where
  inputDim : Nat
  weights  : Array Float
  bias     : Float

/--
Evaluate a linear model on an input vector.
--/
def evalLinearModel (lm : LinearModel) (x : Array Float) : Float :=
  let dot := (Array.zip x lm.weights).foldl (fun s (xi, wi) => s + xi * wi) 0.0
  dot + lm.bias

/--
A decision tree is either a leaf with a classification label or a node that splits on a feature.
--/
inductive DecisionTree
| leaf (label : Nat)
| node (feature_index : Nat) (threshold : Float) (left : DecisionTree) (right : DecisionTree)

open DecisionTree

/--
Evaluate a decision tree on an input vector.
At a node, if the specified feature is less than or equal to the threshold, follow the left branch; otherwise, follow the right branch.
--/
def evalDecisionTree : DecisionTree → Array Float → Nat
| leaf label, _ => label
| node fi th left right, x =>
    if x[fi]! ≤ th then evalDecisionTree left x else evalDecisionTree right x

end FormalVerifML
