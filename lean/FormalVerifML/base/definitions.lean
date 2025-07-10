/--
# FormalVerifML Base Definitions

This file contains the core definitions for machine learning models and their
evaluation functions in the FormalVerifML framework. It provides formal
mathematical representations of neural networks, linear models, and decision trees.

## Overview

The definitions in this file establish the foundation for formal verification
of machine learning models. Each model type is defined with precise mathematical
structures that can be used for proving properties such as robustness, fairness,
and interpretability.

## Key Components

- **LayerType**: Inductive type for neural network layers
- **NeuralNet**: Structure for feed-forward neural networks
- **LinearModel**: Structure for linear/logistic regression models
- **DecisionTree**: Inductive type for decision tree models
- **Evaluation Functions**: Functions to compute model outputs

## Mathematical Foundation

All definitions are based on standard mathematical concepts:
- Linear algebra for weight matrices and bias vectors
- Real analysis for activation functions
- Discrete mathematics for decision tree structures

Author: FormalVerifML Team
License: MIT
Version: 2.0.0
--/

import Mathlib

namespace FormalVerifML

/--
# Layer Types

Inductive type representing different types of neural network layers.
Each layer type corresponds to a specific mathematical operation.

## Layer Types

- **linear**: Linear transformation with weight matrix and bias vector
- **relu**: Rectified Linear Unit activation function
- **sigmoid**: Sigmoid activation function
- **tanh**: Hyperbolic tangent activation function

## Mathematical Definition

For a layer with input x and output y:
- **linear**: y = Wx + b where W is weight matrix, b is bias vector
- **relu**: y_i = max(0, x_i) for each component i
- **sigmoid**: y_i = 1/(1 + exp(-x_i)) for each component i
- **tanh**: y_i = tanh(x_i) for each component i
--/
inductive LayerType
| linear (weight : Array (Array Float)) (bias : Array Float)
| relu
| sigmoid
| tanh
-- Additional activation layers may be added here as needed

open LayerType

/--
# Neural Network Structure

Structure representing a feed-forward neural network with fixed input
and output dimensions. The network consists of a sequence of layers
that are applied sequentially to transform input to output.

## Fields

- **inputDim**: Dimension of input vectors
- **outputDim**: Dimension of output vectors
- **layers**: List of layers to be applied sequentially

## Mathematical Definition

A neural network f: ℝⁿ → ℝᵐ is defined as:
f(x) = fₖ ∘ fₖ₋₁ ∘ ... ∘ f₁(x)

where each fᵢ is a layer function and k is the number of layers.

## Invariants

- All layers must be compatible (output dimension of layer i = input dimension of layer i+1)
- Final layer output dimension must equal network output dimension
--/
structure NeuralNet where
  inputDim  : Nat
  outputDim : Nat
  layers    : List LayerType

/--
# Linear Layer Evaluation

Evaluate a single linear layer on an input vector.
This function computes the matrix-vector product and adds the bias.

## Mathematical Definition

For weight matrix W ∈ ℝᵐˣⁿ, bias vector b ∈ ℝᵐ, and input x ∈ ℝⁿ:
evalLinear(W, b, x) = Wx + b

## Implementation

The function computes the dot product for each output neuron and adds
the corresponding bias term. This is a naive implementation that can
be optimized for performance in practice.

## Parameters

- **w**: Weight matrix as 2D array
- **b**: Bias vector as 1D array
- **x**: Input vector as 1D array

## Returns

Output vector after linear transformation

## Preconditions

- w.size > 0 (at least one output neuron)
- w[0].size > 0 (at least one input feature)
- b.size = w.size (bias dimension matches output dimension)
- x.size = w[0].size (input dimension matches weight matrix columns)
--/
def evalLinear (w : Array (Array Float)) (b : Array Float) (x : Array Float) : Array Float :=
  let mut out := #[]
  for i in [0 : w.size] do
    let row := w[i]!
    let rowVal := row.foldl (fun acc (w_ij : Float) => acc + w_ij * x[row.indexOf w_ij]!) 0.0
    out := out.push (rowVal + b[i]!)
  out

/--
# Activation Function Evaluation

Evaluate an activation function on an input vector.
This function applies the activation function element-wise to each component
of the input vector.

## Mathematical Definitions

- **ReLU**: f(x) = max(0, x)
- **Sigmoid**: f(x) = 1/(1 + exp(-x))
- **Tanh**: f(x) = tanh(x)

## Parameters

- **layer**: Layer type specifying the activation function
- **x**: Input vector

## Returns

Output vector after applying activation function

## Properties

- ReLU is piecewise linear and preserves non-negative values
- Sigmoid maps any real number to (0, 1)
- Tanh maps any real number to (-1, 1)
--/
def evalActivation (layer : LayerType) (x : Array Float) : Array Float :=
  match layer with
  | relu    => x.map (fun v => if v < 0.0 then 0.0 else v)
  | sigmoid => x.map (fun v => 1.0 / (1.0 + Float.exp (-v)))
  | tanh    => x.map (fun v => Float.tanh v)
  | _       => x  -- For non-activation layers, return x unchanged

/--
# Layer Evaluation

Evaluate one layer of the network, dispatching between linear and activation functions.
This function determines the layer type and applies the appropriate evaluation function.

## Parameters

- **l**: Layer to evaluate
- **x**: Input vector

## Returns

Output vector after layer evaluation

## Behavior

- For linear layers: applies matrix multiplication and bias addition
- For activation layers: applies element-wise activation function
- For other layer types: returns input unchanged (fallback behavior)
--/
def evalLayer (l : LayerType) (x : Array Float) : Array Float :=
  match l with
  | linear w b => evalLinear w b x
  | relu       => evalActivation relu x
  | sigmoid    => evalActivation sigmoid x
  | tanh       => evalActivation tanh x

/--
# Neural Network Evaluation

Evaluate the entire neural network on an input vector by sequentially
applying each layer. This is the main function for computing network outputs.

## Mathematical Definition

For a neural network with layers [l₁, l₂, ..., lₖ] and input x:
evalNeuralNet(network, x) = lₖ ∘ lₖ₋₁ ∘ ... ∘ l₁(x)

## Parameters

- **nn**: Neural network structure
- **x**: Input vector

## Returns

Output vector after passing through all layers

## Implementation

Uses functional programming with foldl to apply layers sequentially.
Each layer's output becomes the input to the next layer.

## Preconditions

- x.size = nn.inputDim (input dimension matches network specification)
- All layers are compatible (output dimension matches next layer input dimension)

## Postconditions

- Result size = nn.outputDim (output dimension matches network specification)
--/
def evalNeuralNet (nn : NeuralNet) (x : Array Float) : Array Float :=
  nn.layers.foldl (fun acc layer => evalLayer layer acc) x

/--
# Linear Model Structure

Structure representing a linear (logistic) model with a weight vector
and bias term. This is used for linear regression and logistic regression.

## Fields

- **inputDim**: Dimension of input vectors
- **weights**: Weight vector for linear combination
- **bias**: Bias term (scalar)

## Mathematical Definition

A linear model f: ℝⁿ → ℝ is defined as:
f(x) = wᵀx + b

where w ∈ ℝⁿ is the weight vector and b ∈ ℝ is the bias.

## Applications

- Linear regression: f(x) = wᵀx + b
- Logistic regression: P(y=1|x) = σ(wᵀx + b) where σ is sigmoid function
--/
structure LinearModel where
  inputDim : Nat
  weights  : Array Float
  bias     : Float

/--
# Linear Model Evaluation

Evaluate a linear model on an input vector.
This function computes the dot product of weights and input, then adds the bias.

## Mathematical Definition

For weight vector w ∈ ℝⁿ, bias b ∈ ℝ, and input x ∈ ℝⁿ:
evalLinearModel(model, x) = wᵀx + b

## Parameters

- **lm**: Linear model structure
- **x**: Input vector

## Returns

Scalar output of linear model

## Implementation

Uses Array.zip to pair weights and inputs, then computes dot product
using foldl. Finally adds the bias term.

## Preconditions

- x.size = lm.weights.size (input dimension matches weight vector dimension)
- x.size = lm.inputDim (input dimension matches model specification)
--/
def evalLinearModel (lm : LinearModel) (x : Array Float) : Float :=
  let dot := (Array.zip x lm.weights).foldl (fun s (xi, wi) => s + xi * wi) 0.0
  dot + lm.bias

/--
# Decision Tree Structure

Inductive type representing a decision tree for classification.
A decision tree is either a leaf node with a classification label
or an internal node that splits on a feature.

## Mathematical Definition

A decision tree T is defined recursively:
- Leaf node: T = leaf(c) where c is the classification label
- Internal node: T = node(f, t, T_left, T_right) where:
  - f is the feature index to split on
  - t is the threshold value
  - T_left is the subtree for x[f] ≤ t
  - T_right is the subtree for x[f] > t

## Properties

- Each internal node has exactly two children
- Leaf nodes have no children
- The tree structure is finite and acyclic
--/
inductive DecisionTree
| leaf (label : Nat)
| node (feature_index : Nat) (threshold : Float) (left : DecisionTree) (right : DecisionTree)

open DecisionTree

/--
# Decision Tree Evaluation

Evaluate a decision tree on an input vector.
This function traverses the tree from root to leaf, making decisions
at each internal node based on feature values.

## Mathematical Definition

For a decision tree T and input vector x:
evalDecisionTree(T, x) =
  if T is leaf(c) then c
  else if x[T.feature_index] ≤ T.threshold then evalDecisionTree(T.left, x)
  else evalDecisionTree(T.right, x)

## Parameters

- **tree**: Decision tree to evaluate
- **x**: Input vector

## Returns

Classification label (natural number)

## Algorithm

1. If current node is a leaf, return the label
2. If current node is internal:
   - Compare x[feature_index] with threshold
   - If ≤ threshold, recurse on left subtree
   - If > threshold, recurse on right subtree

## Preconditions

- x.size > tree.feature_index (input dimension sufficient for all features used in tree)
- All feature indices in tree are valid (≥ 0 and < input dimension)

## Properties

- Deterministic: same input always produces same output
- Fast evaluation: O(depth of tree) time complexity
- Interpretable: decision path can be traced from root to leaf
--/
def evalDecisionTree : DecisionTree → Array Float → Nat
| leaf label, _ => label
| node fi th left right, x =>
    if x[fi]! ≤ th then evalDecisionTree left x else evalDecisionTree right x

/--
# Model Type Union

Union type representing all supported model types in the framework.
This allows for generic operations that work on any model type.

## Model Types

- **NeuralNet**: Feed-forward neural networks
- **LinearModel**: Linear/logistic regression models
- **DecisionTree**: Decision tree classifiers

## Usage

This type is used for generic model evaluation and verification
functions that can work with any supported model type.
--/
inductive ModelType
| neuralNet (model : NeuralNet)
| linearModel (model : LinearModel)
| decisionTree (model : DecisionTree)

/--
# Generic Model Evaluation

Evaluate any model type on an input vector.
This function dispatches to the appropriate evaluation function
based on the model type.

## Parameters

- **model**: Model of any supported type
- **x**: Input vector

## Returns

Model output (vector for neural nets, scalar for linear models, label for trees)

## Implementation

Uses pattern matching to determine model type and call appropriate
evaluation function.

## Preconditions

- Input dimension matches model requirements
- Input vector is valid for the model type
--/
def evalModel : ModelType → Array Float → Array Float
| ModelType.neuralNet model => evalNeuralNet model
| ModelType.linearModel model => #[evalLinearModel model]
| ModelType.decisionTree model => #[Float.ofNat (evalDecisionTree model)]

/--
# Model Input Dimension

Get the input dimension for any model type.
This function provides a uniform interface for accessing model input dimensions.

## Parameters

- **model**: Model of any supported type

## Returns

Input dimension as natural number

## Usage

Useful for validation and ensuring input compatibility across different model types.
--/
def getInputDim : ModelType → Nat
| ModelType.neuralNet model => model.inputDim
| ModelType.linearModel model => model.inputDim
| ModelType.decisionTree _ => 0  -- Decision trees don't have fixed input dimension

/--
# Model Output Dimension

Get the output dimension for any model type.
This function provides a uniform interface for accessing model output dimensions.

## Parameters

- **model**: Model of any supported type

## Returns

Output dimension as natural number

## Usage

Useful for validation and ensuring output compatibility across different model types.
--/
def getOutputDim : ModelType → Nat
| ModelType.neuralNet model => model.outputDim
| ModelType.linearModel _ => 1  -- Linear models output scalar
| ModelType.decisionTree _ => 1  -- Decision trees output single label

end FormalVerifML
