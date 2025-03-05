import FormalVerifML.base.definitions

namespace FormalVerifML

/--
A convolutional layer for a ConvNet.
This layer contains a 2D filter (kernel), a stride (uniform in both dimensions), and a symmetric padding.
--/
structure ConvLayer where
  filter  : Array (Array Float)  -- 2D kernel
  stride  : Nat                  -- Stride length
  padding : Nat                  -- Zero-padding size
  deriving Inhabited

/--
Pad a 2D matrix with zeros.
Adds `pad` rows at the top and bottom, and `pad` zeros to the left and right of each row.
--/
def padMatrix (input : Array (Array Float)) (pad : Nat) : Array (Array Float) :=
  let H := input.size
  let W := if H > 0 then (input[0]!).size else 0
  let paddedRow := Array.mkArray (W + 2 * pad) 0.0
  let topBottom := Array.mkArray pad paddedRow
  let middle := input.map (λ row =>
    let leftPad  := Array.mkArray pad 0.0
    let rightPad := Array.mkArray pad 0.0
    leftPad ++ row ++ rightPad)
  topBottom ++ middle ++ topBottom

/--
Perform a simple 2D convolution on an input matrix using a given ConvLayer.
This implementation computes the convolution sum over each valid window.
--/
def conv2d (layer : ConvLayer) (input : Array (Array Float)) : Array (Array Float) :=
  let padded := padMatrix input layer.padding
  let H := padded.size
  let W := if H > 0 then (padded[0]!).size else 0
  let filterHeight := layer.filter.size
  let filterWidth  := if filterHeight > 0 then (layer.filter[0]!).size else 0
  let outHeight := ((H - filterHeight) / layer.stride) + 1
  let outWidth  := ((W - filterWidth) / layer.stride) + 1
  let mutable output := Array.mkEmpty outHeight
  for i in List.range outHeight do
    let mutable rowRes := Array.mkEmpty outWidth
    for j in List.range outWidth do
      let mutable sum := 0.0
      for k in List.range filterHeight do
        for l in List.range filterWidth do
          let a := (padded.getD (i * layer.stride + k) (Array.mkEmpty 0)).getD (j * layer.stride + l) 0.0
          let b := (layer.filter.getD k (Array.mkEmpty 0)).getD l 0.0
          sum := sum + a * b
      rowRes := rowRes.push sum
    output := output.push rowRes
  output

/--
Skeleton for a Convolutional Neural Network.
The network consists of an input dimension, an output dimension, a list of convolutional layers,
and a list of fully-connected (FC) layers. Each FC layer is represented as a pair (weight matrix, bias vector).
--/
structure ConvNet where
  inputDim   : Nat
  outputDim  : Nat
  convLayers : List ConvLayer
  fcLayers   : List (Array (Array Float) × Array Float)

/--
Evaluate the ConvNet on a given 2D input.
First applies the convolutional layers, then flattens the result, and finally applies each FC layer.
--/
def evalConvNet (cnn : ConvNet) (x : Array (Array Float)) : Array Float :=
  let conv_output := cnn.convLayers.foldl (λ acc layer => conv2d layer acc) x
  let flattened : Array Float := conv_output.foldl (λ acc row => acc ++ row) #[]
  cnn.fcLayers.foldl (λ acc (w, b) =>
    let dot := (Array.zip acc b).foldl (λ s (a, bi) => s + a * bi) 0.0
    #[dot]
  ) flattened

/--
An RNN cell for a Recurrent Neural Network.
It contains a weight matrix for the current input, one for the previous hidden state, and a bias vector.
--/
structure RNNCell where
  weight_input  : Array (Array Float)
  weight_hidden : Array (Array Float)
  bias          : Array Float

/--
Evaluate a single RNN cell given an input vector and a hidden state.
Uses the existing evalLinear function for both parts and returns the element-wise sum.
--/
def evalRNNCell (cell : RNNCell) (x h : Array Float) : Array Float :=
  let input_part  := evalLinear cell.weight_input cell.bias x
  let hidden_part := evalLinear cell.weight_hidden cell.bias h
  input_part.zipWith (· + ·) hidden_part

/--
Skeleton for a Recurrent Neural Network.
Consists of an input dimension, a hidden state dimension, an output dimension, a list of RNN cells,
and a final FC layer.
--/
structure RecurrentNet where
  inputDim  : Nat
  hiddenDim : Nat
  outputDim : Nat
  cells     : List RNNCell
  fcLayer   : (Array (Array Float) × Array Float)

/--
Evaluate the RecurrentNet on a sequence of input vectors.
Starts with a zero hidden state, updates it with the first available RNN cell, and then applies the FC layer.
--/
def evalRecurrentNet (rn : RecurrentNet) (xs : List (Array Float)) : Array Float :=
  let initial_hidden : Array Float := Array.mkArray rn.hiddenDim 0.0
  let final_hidden := xs.foldl (λ h x =>
    match rn.cells with
    | []       => h
    | cell :: _ => evalRNNCell cell x h
  ) initial_hidden
  let (w, b) := rn.fcLayer
  evalLinearModel { inputDim := rn.hiddenDim, weights := w.foldl (λ acc row => acc ++ row) #[], bias := 0.0 } final_hidden

/--
Helper: Multiply two matrices A and B.
Assumes A : m×n and B : n×p. Returns the m×p product.
--/
def matrixMul (A B : Array (Array Float)) : Array (Array Float) :=
  let m := A.size
  let n := if m > 0 then (A[0]!).size else 0
  let p := if B.size > 0 then (B[0]!).size else 0
  let mutable result := Array.mkEmpty m
  for i in List.range m do
    let mutable row := Array.mkEmpty p
    for j in List.range p do
      let mutable sum := 0.0
      for k in List.range n do
         sum := sum + (A.getD i (Array.mkEmpty 0)).getD k 0.0 * (B.getD k (Array.mkEmpty 0)).getD j 0.0
      row := row.push sum
    result := result.push row
  result

/--
Helper: Transpose a matrix.
--/
def transpose (M : Array (Array Float)) : Array (Array Float) :=
  let m := M.size
  let n := if m > 0 then (M[0]!).size else 0
  let mutable res := Array.mkEmpty n
  for j in List.range n do
    let mutable row := Array.mkEmpty m
    for i in List.range m do
      row := row.push (M.getD i (Array.mkEmpty 0)).getD j 0.0
    res := res.push row
  res

/--
Helper: Compute the softmax of a vector.
--/
def softmax (v : Array Float) : Array Float :=
  let exps := v.map (λ x => Float.exp x)
  let sumExp := exps.foldl (· + ·) 0.0
  exps.map (λ x => x / sumExp)

/--
Structure for a Transformer model.
For this actionable implementation, we include weight matrices for query, key, and value projections.
--/
structure Transformer where
  inputDim   : Nat
  numHeads   : Nat
  numLayers  : Nat
  W_q : Array (Array Float)
  W_k : Array (Array Float)
  W_v : Array (Array Float)
  deriving Inhabited

/--
Evaluate the Transformer model using scaled dot-product attention.
This minimal implementation computes query, key, and value projections, then performs attention.
--/
def evalTransformer (tr : Transformer) (x : Array (Array Float)) : Array (Array Float) :=
  let queries := matrixMul x tr.W_q
  let keys    := matrixMul x tr.W_k
  let values  := matrixMul x tr.W_v
  let keyT    := transpose keys
  let scores  := matrixMul queries keyT
  let scale   := 1.0 / Float.sqrt (Float.ofNat tr.inputDim)
  let scaled  := scores.map (λ row => row.map (λ s => s * scale))
  let attnWeights := scaled.map softmax
  matrixMul attnWeights values

end FormalVerifML
