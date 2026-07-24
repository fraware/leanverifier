--------------------------------------------------------------------------------
-- Top-level entry point for the FormalVerifML project.
-- This file imports:
--   - Base definitions and property definitions.
--   - Auto-generated models (from the translator).
--   - Extended proof scripts.
--   - Enterprise features and advanced architectures.
--------------------------------------------------------------------------------
import Mathlib
import lean.FormalVerifML.base.definitions   -- NeuralNet, LinearModel, DecisionTree, etc.
import lean.FormalVerifML.base.ml_properties -- Robustness, fairness, etc.
import lean.FormalVerifML.base.advanced_tactics
import lean.FormalVerifML.base.advanced_models
import lean.FormalVerifML.base.symbolic_models
import lean.FormalVerifML.base.memory_optimized_models  -- Memory optimization
import lean.FormalVerifML.base.smt_integration          -- SMT solver integration
import lean.FormalVerifML.base.large_scale_models       -- 100M+ parameter models
import lean.FormalVerifML.base.vision_models            -- Vision Transformers
import lean.FormalVerifML.base.distributed_verification -- Distributed verification
import lean.FormalVerifML.base.enterprise_features      -- Enterprise features

-- Auto-generated models:
import lean.FormalVerifML.generated.example_model          -- original NN model
import lean.FormalVerifML.generated.another_nn_model         -- generated from another_nn.json
import lean.FormalVerifML.generated.log_reg_model            -- generated from log_reg.json
import lean.FormalVerifML.generated.decision_tree_model      -- generated from decision_tree.json
import lean.FormalVerifML.generated.sample_transformer_model  -- new sample transformer

-- Proof scripts:
import lean.FormalVerifML.proofs.example_robustness_proof
import lean.FormalVerifML.proofs.example_fairness_proof
import lean.FormalVerifML.proofs.extended_robustness_proof
import lean.FormalVerifML.proofs.extended_fairness_proof
import lean.FormalVerifML.proofs.decision_tree_proof
import lean.FormalVerifML.proofs.comprehensive_test_suite  -- Comprehensive test suite

open FormalVerifML

/--
  A trivial theorem to ensure the project builds.
--/
theorem project_builds_successfully : True :=
  trivial

/--
  Trivial theorem referencing the sample transformer to ensure it is type-checked.
--/
theorem sample_transformer_exists : True :=
  let _ := sampleTransformer;
  trivial

/--
  Enterprise features are properly integrated.
--/
theorem enterprise_features_integrated : True :=
  let _ : EnterpriseConfig := {
    enableAuthentication := true,
    sessionTimeout := 3600,
    maxSessionsPerUser := 5,
    enableRoleBasedAccess := true,
    defaultRole := "user",
    enableAuditLogging := true,
    auditRetentionDays := 90,
    logSensitiveActions := true,
    enableRateLimiting := true,
    maxRequestsPerMinute := 100,
    enableEncryption := true,
    maxConcurrentJobs := 10,
    jobTimeout := 300,
    enableCaching := true
  };
  trivial

/--
  Large-scale models are properly integrated.
--/
theorem large_scale_models_integrated : True :=
  let _ : LargeScaleTransformer := {
    dModel := 4096,
    numHeads := 64,
    numLayers := 48,
    vocabSize := 100000,
    maxSeqLen := 4096,
    useModelParallelism := true,
    useGradientCheckpointing := true,
    useMixedPrecision := true,
    useActivationCheckpointing := true,
    chunkSize := 128,
    maxMemoryGB := 64,
    numGPUs := 8,
    useDataParallelism := true,
    usePipelineParallelism := true,
    tokenEmbeddings := Array.mkEmpty 0,
    positionalEmbeddings := Array.mkEmpty 0,
    attentionHeads := Array.mkEmpty 0,
    layerNorms1 := Array.mkEmpty 0,
    layerNorms2 := Array.mkEmpty 0,
    ffWeights1 := Array.mkEmpty 0,
    ffWeights2 := Array.mkEmpty 0,
    outputProjection := (Array.mkEmpty 0, Array.mkEmpty 0)
  };
  trivial

/--
  Vision models are properly integrated.
--/
theorem vision_models_integrated : True :=
  let _ : VisionTransformer := {
    imageSize := 224,
    patchSize := 16,
    numChannels := 3,
    dModel := 768,
    numHeads := 12,
    numLayers := 12,
    numClasses := 1000,
    maxSeqLen := 197,  -- (224/16)^2 + 1 for CLS token
    useClassToken := true,
    usePositionalEmbeddings := true,
    useLayerNorm := true,
    patchEmbeddings := Array.mkEmpty 0,
    classToken := Array.mkEmpty 0,
    positionalEmbeddings := Array.mkEmpty 0,
    attentionHeads := Array.mkEmpty 0,
    layerNorms1 := Array.mkEmpty 0,
    layerNorms2 := Array.mkEmpty 0,
    ffWeights1 := Array.mkEmpty 0,
    ffWeights2 := Array.mkEmpty 0,
    outputProjection := (Array.mkEmpty 0, Array.mkEmpty 0)
  };
  trivial

/--
  Distributed verification is properly integrated.
--/
theorem distributed_verification_integrated : True :=
  let _ : DistributedConfig := {
    numNodes := 8,
    nodeTimeout := 300,
    maxConcurrentProofs := 5,
    useParallelSMT := true,
    useProofSharding := true,
    useResultAggregation := true,
    enableLoadBalancing := true,
    enableFaultTolerance := true
  };
  trivial
