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
Compute the L-infinity distance between two vectors.
--/
def distLInf (x y : Array Float) : Float :=
  let pairs := Array.zip x y
  pairs.foldl (fun acc (xi, yi) => Float.max acc (Float.abs (xi - yi))) 0.0

/--
Robust classification property:
A classification function f is robust at level ε if any two inputs within ε (L2 norm) yield the same output.
--/
def robustClass (f : Array Float → Nat) (ε : Float) : Prop :=
  ∀ (x x' : Array Float), distL2 x x' < ε → f x = f x'

/--
Robust classification property using L-infinity norm:
A classification function f is robust at level ε if any two inputs within ε (L-infinity norm) yield the same output.
--/
def robustClassLInf (f : Array Float → Nat) (ε : Float) : Prop :=
  ∀ (x x' : Array Float), distLInf x x' < ε → f x = f x'

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

/--
Transformer-specific properties
--/

/--
Attention robustness property:
The attention mechanism is robust if small perturbations to input tokens
result in bounded changes to attention weights.
--/
def attentionRobust (attention_fn : Array (Array Float) → Array (Array Float)) (ε δ : Float) : Prop :=
  ∀ (x x' : Array (Array Float)),
  (∀ i, distL2 x[i]! x'[i]! < ε) →
  ∀ i j, |(attention_fn x)[i]![j]! - (attention_fn x')[i]![j]!| < δ

/--
Sequence invariance property:
The model output is invariant to permutations of input tokens that don't change meaning.
--/
def sequenceInvariant (f : Array Nat → Array Float) (perm : Array Nat → Array Nat) : Prop :=
  ∀ (tokens : Array Nat), f tokens = f (perm tokens)

/--
Causal masking property:
For causal language models, the output at position i should only depend on tokens at positions ≤ i.
--/
def causalMasking (f : Array Nat → Array Float) : Prop :=
  ∀ (tokens1 tokens2 : Array Nat) (i : Nat),
  (∀ j, j ≤ i → tokens1[j]! = tokens2[j]!) →
  (f tokens1)[i]! = (f tokens2)[i]!

/--
Positional encoding invariance:
The model should be invariant to absolute position shifts (up to a certain limit).
--/
def positionalInvariance (f : Array Nat → Array Float) (shift : Nat) (max_shift : Nat) : Prop :=
  shift ≤ max_shift →
  ∀ (tokens : Array Nat),
  let shifted := tokens.map (λ t => t + shift)
  distL2 (f tokens) (f shifted) < 0.1  -- Small tolerance

/--
Attention head diversity:
Different attention heads should attend to different aspects of the input.
--/
def attentionHeadDiversity (heads : Array (Array (Array Float) → Array (Array Float))) : Prop :=
  ∀ (i j : Nat) (x : Array (Array Float)),
  i ≠ j → i < heads.size → j < heads.size →
  let attn_i := heads[i]! x
  let attn_j := heads[j]! x
  ∃ k l, |attn_i[k]![l]! - attn_j[k]![l]!| > 0.1

/--
Gradient-based interpretability:
The model's output should be interpretable through gradient-based methods.
--/
def gradientInterpretable (f : Array Float → Float) (ε : Float) : Prop :=
  ∀ (x : Array Float) (i : Nat),
  let grad_i := (f (x.modify i (λ v => v + ε)) - f (x.modify i (λ v => v - ε))) / (2 * ε)
  Float.abs grad_i < 100.0  -- Bounded gradients

/--
Fairness across attention heads:
All attention heads should treat different demographic groups fairly.
--/
def attentionFairness (attention_fn : Array (Array Float) → Array (Array Float))
  (demographic_feature : Nat) : Prop :=
  ∀ (x : Array (Array Float)) (group1 group2 : Nat),
  let x1 := x.modify demographic_feature (λ _ => Float.ofNat group1)
  let x2 := x.modify demographic_feature (λ _ => Float.ofNat group2)
  let attn1 := attention_fn x1
  let attn2 := attention_fn x2
  distL2 (attn1.foldl (λ acc row => acc ++ row) #[])
         (attn2.foldl (λ acc row => acc ++ row) #[]) < 0.1

/--
Memory efficiency property:
The attention mechanism should not consume excessive memory for long sequences.
--/
def memoryEfficient (attention_fn : Array (Array Float) → Array (Array Float)) : Prop :=
  ∀ (x : Array (Array Float)),
  let seq_len := x.size
  let d_model := if seq_len > 0 then x[0]!.size else 0
  -- Memory usage should be O(seq_len * d_model) not O(seq_len^2)
  seq_len * d_model ≤ 1000000  -- 1M parameter limit

/--
Token importance consistency:
Important tokens should consistently receive high attention across different inputs.
--/
def tokenImportanceConsistency (attention_fn : Array (Array Float) → Array (Array Float)) : Prop :=
  ∀ (x1 x2 : Array (Array Float)) (important_pos : Nat),
  let attn1 := attention_fn x1
  let attn2 := attention_fn x2
  let importance1 := attn1.foldl (λ acc row => acc + row[important_pos]!) 0.0
  let importance2 := attn2.foldl (λ acc row => acc + row[important_pos]!) 0.0
  |importance1 - importance2| < 0.5

end FormalVerifML
