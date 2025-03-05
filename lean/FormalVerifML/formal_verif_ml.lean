--------------------------------------------------------------------------------
-- Top-level entry point for the FormalVerifML project.
-- This file imports:
--   - Base definitions and property definitions.
--   - Auto-generated models (from the translator).
--   - Extended proof scripts.
--------------------------------------------------------------------------------
import Mathlib
import lean.FormalVerifML.base.definitions   -- NeuralNet, LinearModel, DecisionTree, etc.
import lean.FormalVerifML.base.ml_properties -- Robustness, fairness, etc.
import lean.FormalVerifML.base.advanced_tactics
import lean.FormalVerifML.base.advanced_models
import lean.FormalVerifML.base.symbolic_models

-- Auto-generated models:
import lean.FormalVerifML.generated.example_model          -- original NN model
import lean.FormalVerifML.generated.another_nn_model         -- generated from another_nn.json
import lean.FormalVerifML.generated.log_reg_model            -- generated from log_reg.json
import lean.FormalVerifML.generated.decision_tree_model      -- generated from decision_tree.json

-- Proof scripts:
import lean.FormalVerifML.proofs.example_robustness_proof
import lean.FormalVerifML.proofs.example_fairness_proof
import lean.FormalVerifML.proofs.extended_robustness_proof
import lean.FormalVerifML.proofs.extended_fairness_proof
import lean.FormalVerifML.proofs.decision_tree_proof

open FormalVerifML

/--
  A trivial theorem to ensure the project builds.
--/
theorem project_builds_successfully : True :=
  trivial
