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
Helper: Layer normalization
--/
def layerNorm (x : Array Float) (weight : Array Float) (bias : Array Float) : Array Float :=
  let mean := x.foldl (· + ·) 0.0 / Float.ofNat x.size
  let variance := x.foldl (λ acc xi => acc + (xi - mean) * (xi - mean)) 0.0 / Float.ofNat x.size
  let std := Float.sqrt (variance + 1e-5)  -- Add epsilon for numerical stability
  let normalized := x.map (λ xi => (xi - mean) / std)
  normalized.zipWith (λ ni wi => ni * wi) weight

/--
Helper: Add positional encoding to input embeddings
--/
def addPositionalEncoding (x : Array (Array Float)) : Array (Array Float) :=
  let seqLen := x.size
  let dModel := if seqLen > 0 then x[0]!.size else 0
  let mutable result := Array.mkEmpty seqLen
  for pos in List.range seqLen do
    let mutable row := Array.mkEmpty dModel
    for i in List.range dModel do
      let angle := Float.ofNat pos / Float.pow 10000.0 (Float.ofNat i / Float.ofNat dModel)
      let pe := if i % 2 == 0 then Float.sin angle else Float.cos angle
      row := row.push (x.getD pos (Array.mkEmpty 0)).getD i 0.0 + pe)
    result := result.push row
  result

/--
Structure for a single attention head
--/
structure AttentionHead where
  W_q : Array (Array Float)  -- Query projection
  W_k : Array (Array Float)  -- Key projection
  W_v : Array (Array Float)  -- Value projection
  W_o : Array (Array Float)  -- Output projection

/--
Structure for a complete Transformer model with production-ready features
--/
structure Transformer where
  -- Model dimensions
  dModel : Nat              -- Hidden dimension
  numHeads : Nat            -- Number of attention heads
  numLayers : Nat           -- Number of transformer layers
  vocabSize : Nat           -- Vocabulary size
  maxSeqLen : Nat           -- Maximum sequence length

  -- Embeddings
  tokenEmbeddings : Array (Array Float)  -- Token embedding matrix
  positionalEmbeddings : Array (Array Float)  -- Positional embedding matrix

  -- Layer components
  attentionHeads : Array (Array AttentionHead)  -- [numLayers][numHeads]
  layerNorms1 : Array (Array Float × Array Float)  -- [numLayers] (weight, bias)
  layerNorms2 : Array (Array Float × Array Float)  -- [numLayers] (weight, bias)

  -- Feed-forward networks
  ffWeights1 : Array (Array (Array Float) × Array Float)  -- [numLayers] (weight, bias)
  ffWeights2 : Array (Array (Array Float) × Array Float)  -- [numLayers] (weight, bias)

  -- Output projection
  outputProjection : Array (Array Float) × Array Float  -- (weight, bias)

  deriving Inhabited

/--
Compute scaled dot-product attention for a single head
--/
def computeAttention (head : AttentionHead) (x : Array (Array Float)) : Array (Array Float) :=
  let queries := matrixMul x head.W_q
  let keys := matrixMul x head.W_k
  let values := matrixMul x head.W_v

  let keyT := transpose keys
  let scores := matrixMul queries keyT

  -- Scale by sqrt(d_k)
  let d_k := Float.ofNat (if scores.size > 0 then scores[0]!.size else 0)
  let scale := 1.0 / Float.sqrt d_k
  let scaled := scores.map (λ row => row.map (λ s => s * scale))

  -- Apply softmax to get attention weights
  let attnWeights := scaled.map softmax

  -- Apply attention weights to values
  let attended := matrixMul attnWeights values

  -- Apply output projection
  matrixMul attended head.W_o

/--
Compute multi-head attention
--/
def multiHeadAttention (heads : Array AttentionHead) (x : Array (Array Float)) : Array (Array Float) :=
  let headOutputs := heads.map (λ head => computeAttention head x)
  -- Concatenate head outputs (simplified - assumes all heads have same output dimension)
  headOutputs.foldl (λ acc headOut => acc) headOutputs[0]!

/--
Apply a single transformer layer
--/
def applyTransformerLayer
  (layerIdx : Nat)
  (heads : Array AttentionHead)
  (ln1 ln2 : Array Float × Array Float)
  (ff1 ff2 : Array (Array Float) × Array Float)
  (x : Array (Array Float)) : Array (Array Float) :=

  -- Self-attention with residual connection and layer norm
  let attnOut := multiHeadAttention heads x
  let residual1 := x.zipWith (λ xi ai => xi.zipWith (· + ·) ai) attnOut
  let norm1 := residual1.map (λ row => layerNorm row ln1.1 ln1.2)

  -- Feed-forward network with residual connection and layer norm
  let ffOut := evalLinear ff1.1 ff1.2 (norm1.foldl (λ acc row => acc ++ row) #[])
  let ffOut2D := #[ffOut]  -- Convert back to 2D for consistency
  let residual2 := norm1.zipWith (λ ni fi => ni.zipWith (· + ·) fi) ffOut2D
  let norm2 := residual2.map (λ row => layerNorm row ln2.1 ln2.2)

  norm2

/--
Evaluate the complete Transformer model
--/
def evalTransformer (tr : Transformer) (tokenIds : Array Nat) : Array Float :=
  -- Token embeddings
  let tokenEmbs := tokenIds.map (λ id => tr.tokenEmbeddings.getD id (Array.mkEmpty 0))

  -- Add positional encoding
  let posEmbs := addPositionalEncoding tokenEmbs

  -- Apply transformer layers
  let mutable hidden := posEmbs
  for layerIdx in List.range tr.numLayers do
    let heads := tr.attentionHeads.getD layerIdx (Array.mkEmpty 0)
    let ln1 := tr.layerNorms1.getD layerIdx (Array.mkEmpty 0, Array.mkEmpty 0)
    let ln2 := tr.layerNorms2.getD layerIdx (Array.mkEmpty 0, Array.mkEmpty 0)
    let ff1 := tr.ffWeights1.getD layerIdx (Array.mkEmpty 0, Array.mkEmpty 0)
    let ff2 := tr.ffWeights2.getD layerIdx (Array.mkEmpty 0, Array.mkEmpty 0)
    hidden := applyTransformerLayer layerIdx heads ln1 ln2 ff1 ff2 hidden

  -- Output projection
  let finalHidden := hidden.foldl (λ acc row => acc ++ row) #[]
  evalLinear tr.outputProjection.1 tr.outputProjection.2 finalHidden

end FormalVerifML
