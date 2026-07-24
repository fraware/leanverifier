#!/usr/bin/env python3
"""
FormalVerifML Model Translator

This module provides functionality to convert machine learning models from JSON format
to Lean 4 formal verification code. It supports various model types including neural
networks, transformers, decision trees, and linear models.

Author: FormalVerifML Team
License: MIT
Version: 2.0.0
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTranslator:
    """
    Main translator class for converting ML models to Lean 4 code.

    This class handles the conversion of various machine learning model types
    from JSON format to Lean 4 formal verification code. It supports neural
    networks, transformers, decision trees, and linear models.

    Attributes:
        supported_types: List of supported model types
        output_directory: Directory for generated Lean files
    """

    def __init__(self, output_directory: Optional[str] = None):
        """
        Initialize the ModelTranslator.

        Args:
            output_directory: Directory to save generated Lean files.
                             If None, uses current directory.
        """
        self.supported_types = [
            "neural_net",
            "transformer",
            "decision_tree",
            "linear_model",
            "vision_transformer",
        ]
        self.output_directory = (
            Path(output_directory) if output_directory else Path.cwd()
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized ModelTranslator with output directory: {self.output_directory}"
        )

    def validate_model_json(self, model_data: Dict[str, Any]) -> bool:
        """
        Validate the structure of a model JSON.

        Args:
            model_data: Dictionary containing model data

        Returns:
            True if model is valid, False otherwise

        Raises:
            ValueError: If model type is not supported or required fields are missing
        """
        if not isinstance(model_data, dict):
            raise ValueError("Model data must be a dictionary")

        model_type = model_data.get("model_type", "")
        if not model_type:
            raise ValueError("Model type is required")

        if model_type not in self.supported_types:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {', '.join(self.supported_types)}"
            )

        # Validate based on model type
        if model_type == "neural_net":
            self._validate_neural_net(model_data)
        elif model_type == "transformer":
            self._validate_transformer(model_data)
        elif model_type == "decision_tree":
            self._validate_decision_tree(model_data)
        elif model_type == "linear_model":
            self._validate_linear_model(model_data)

        logger.info(f"Model validation passed for type: {model_type}")
        return True

    def _validate_neural_net(self, model_data: Dict[str, Any]) -> None:
        """Validate neural network model structure."""
        required_fields = ["input_dim", "output_dim", "layers"]
        for field in required_fields:
            if field not in model_data:
                raise ValueError(f"Neural network missing required field: {field}")

        if not isinstance(model_data["layers"], list):
            raise ValueError("Layers must be a list")

        for i, layer in enumerate(model_data["layers"]):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer {i} must be a dictionary")
            if "type" not in layer:
                raise ValueError(f"Layer {i} missing 'type' field")

    def _validate_transformer(self, model_data: Dict[str, Any]) -> None:
        """Validate transformer model structure."""
        required_fields = ["d_model", "num_heads", "num_layers", "vocab_size"]
        for field in required_fields:
            if field not in model_data:
                raise ValueError(f"Transformer missing required field: {field}")

    def _validate_decision_tree(self, model_data: Dict[str, Any]) -> None:
        """Validate decision tree model structure."""
        if "tree" not in model_data:
            raise ValueError("Decision tree missing 'tree' field")

    def _validate_linear_model(self, model_data: Dict[str, Any]) -> None:
        """Validate linear model structure."""
        required_fields = ["input_dim", "weights", "bias"]
        for field in required_fields:
            if field not in model_data:
                raise ValueError(f"Linear model missing required field: {field}")

    def generate_neural_net_code(self, model_json: Dict[str, Any]) -> str:
        """
        Generate Lean code for a neural network model.

        Args:
            model_json: Dictionary containing neural network data

        Returns:
            Generated Lean code as a string

        Raises:
            ValueError: If model data is invalid
        """
        name = model_json.get("name", "exampleNeuralNet")
        input_dim = model_json["input_dim"]
        output_dim = model_json["output_dim"]
        layers = model_json["layers"]

        logger.info(f"Generating neural network code for: {name}")

        lean_lines = []
        lean_lines.append("import FormalVerifML.base.definitions")
        lean_lines.append("namespace FormalVerifML")
        lean_lines.append(f"-- Auto-generated Neural Net definition for {name}\n")

        # Generate layer expressions
        layer_exprs = []
        for idx, layer in enumerate(layers):
            layer_type = layer["type"]

            if layer_type == "linear":
                weight = layer["weight"]
                bias = layer["bias"]
                weight_str = self._format_weight_matrix(weight)
                bias_str = self._format_bias_vector(bias)
                layer_exprs.append(f"LayerType.linear {weight_str} {bias_str}")

            elif layer_type == "relu":
                layer_exprs.append("LayerType.relu")
            elif layer_type == "sigmoid":
                layer_exprs.append("LayerType.sigmoid")
            elif layer_type == "tanh":
                layer_exprs.append("LayerType.tanh")
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Generate model definition
        layers_str = ",\n    ".join(layer_exprs)
        lean_lines.append(f"def {name} : NeuralNet :=")
        lean_lines.append(f"  {{ inputDim  := {input_dim},")
        lean_lines.append(f"    outputDim := {output_dim},")
        lean_lines.append(f"    layers    := [\n      {layers_str}\n    ] }}")
        lean_lines.append("end FormalVerifML")

        return "\n".join(lean_lines)

    def generate_transformer_code(self, model_json: Dict[str, Any]) -> str:
        """
        Generate Lean code for a transformer model.

        Args:
            model_json: Dictionary containing transformer data

        Returns:
            Generated Lean code as a string
        """
        name = model_json.get("name", "exampleTransformer")

        logger.info(f"Generating transformer code for: {name}")

        lean_lines = []
        lean_lines.append("import FormalVerifML.base.advanced_models")
        lean_lines.append("namespace FormalVerifML")
        lean_lines.append(f"-- Auto-generated Transformer definition for {name}\n")

        # Extract transformer parameters
        d_model = model_json.get("d_model", 512)
        num_heads = model_json.get("num_heads", 8)
        num_layers = model_json.get("num_layers", 6)
        vocab_size = model_json.get("vocab_size", 30000)
        max_seq_len = model_json.get("max_seq_len", 512)

        # Generate embeddings
        token_emb_str = self._format_embeddings(model_json.get("token_embeddings", []))
        pos_emb_str = self._format_embeddings(
            model_json.get("positional_embeddings", [])
        )

        # Generate attention heads
        heads_str = self._format_attention_heads(model_json.get("attention_heads", []))

        # Generate layer norms
        ln1_str = self._format_layer_norms(model_json.get("layer_norms1", []))
        ln2_str = self._format_layer_norms(model_json.get("layer_norms2", []))

        # Generate feed-forward networks
        ff1_str = self._format_feed_forward(model_json.get("ff_weights1", []))
        ff2_str = self._format_feed_forward(model_json.get("ff_weights2", []))

        # Generate output projection
        output_proj = model_json.get("output_projection", {})
        out_weight_str = self._format_weight_matrix(output_proj.get("weight", []))
        out_bias_str = self._format_bias_vector(output_proj.get("bias", []))

        # Generate model definition
        lean_lines.append(f"def {name} : Transformer :=")
        lean_lines.append(f"  {{ dModel := {d_model},")
        lean_lines.append(f"    numHeads := {num_heads},")
        lean_lines.append(f"    numLayers := {num_layers},")
        lean_lines.append(f"    vocabSize := {vocab_size},")
        lean_lines.append(f"    maxSeqLen := {max_seq_len},")
        lean_lines.append(f"    tokenEmbeddings := {token_emb_str},")
        lean_lines.append(f"    positionalEmbeddings := {pos_emb_str},")
        lean_lines.append(f"    attentionHeads := {heads_str},")
        lean_lines.append(f"    layerNorms1 := {ln1_str},")
        lean_lines.append(f"    layerNorms2 := {ln2_str},")
        lean_lines.append(f"    ffWeights1 := {ff1_str},")
        lean_lines.append(f"    ffWeights2 := {ff2_str},")
        lean_lines.append(
            f"    outputProjection := ({out_weight_str}, {out_bias_str}) }}"
        )
        lean_lines.append("end FormalVerifML")

        return "\n".join(lean_lines)

    def generate_log_reg_code(self, model_json: Dict[str, Any]) -> str:
        """
        Generate Lean code for a logistic regression model.

        Args:
            model_json: Dictionary containing logistic regression data

        Returns:
            Generated Lean code as a string
        """
        name = model_json.get("name", "myLogRegModel")
        input_dim = model_json["input_dim"]
        weights = model_json["weights"]
        bias = model_json["bias"]

        logger.info(f"Generating logistic regression code for: {name}")

        lean_lines = []
        lean_lines.append("import FormalVerifML.base.definitions")
        lean_lines.append("namespace FormalVerifML")
        lean_lines.append(
            f"-- Auto-generated Logistic Regression definition for {name}\n"
        )

        weights_str = f"#[{', '.join(str(w) for w in weights)}]"

        lean_lines.append(f"def {name} : LinearModel :=")
        lean_lines.append(f"  {{ inputDim := {input_dim},")
        lean_lines.append(f"    weights  := {weights_str},")
        lean_lines.append(f"    bias     := {bias} }}")
        lean_lines.append("end FormalVerifML")

        return "\n".join(lean_lines)

    def generate_decision_tree_code(self, model_json: Dict[str, Any]) -> str:
        """
        Generate Lean code for a decision tree model.

        Args:
            model_json: Dictionary containing decision tree data

        Returns:
            Generated Lean code as a string
        """
        name = model_json.get("name", "myDecisionTree")
        tree = model_json["tree"]

        logger.info(f"Generating decision tree code for: {name}")

        lean_lines = []
        lean_lines.append("import FormalVerifML.base.definitions")
        lean_lines.append("namespace FormalVerifML")
        lean_lines.append(f"-- Auto-generated Decision Tree definition for {name}\n")

        def gen_tree(node: Dict[str, Any]) -> str:
            if "leaf" in node:
                return f"DecisionTree.leaf {node['leaf']}"
            else:
                feature_index = node["feature_index"]
                threshold = node["threshold"]
                left = gen_tree(node["left"])
                right = gen_tree(node["right"])
                return f"DecisionTree.node {feature_index} {threshold} {left} {right}"

        tree_expr = gen_tree(tree)

        lean_lines.append(f"def {name} : DecisionTree :=")
        lean_lines.append(f"  {tree_expr}")
        lean_lines.append("end FormalVerifML")

        return "\n".join(lean_lines)

    def _format_weight_matrix(self, weights: List[List[float]]) -> str:
        """Format weight matrix for Lean code."""
        if not weights:
            return "[]"

        weight_str = "[\n    "
        weight_str += ",\n    ".join(
            f"#[{', '.join(str(wij) for wij in row)}]" for row in weights
        )
        weight_str += "\n  ]"
        return weight_str

    def _format_bias_vector(self, bias: List[float]) -> str:
        """Format bias vector for Lean code."""
        return f"#[{', '.join(str(bi) for bi in bias)}]"

    def _format_embeddings(self, embeddings: List[List[float]]) -> str:
        """Format embeddings for Lean code."""
        if not embeddings:
            return "[]"

        emb_str = "[\n    "
        emb_str += ",\n    ".join(
            f"#[{', '.join(str(w) for w in emb)}]" for emb in embeddings
        )
        emb_str += "\n  ]"
        return emb_str

    def _format_attention_heads(
        self, attention_heads: List[List[Dict[str, Any]]]
    ) -> str:
        """Format attention heads for Lean code."""
        if not attention_heads:
            return "[]"

        heads_str = "[\n    "
        for layer_idx, layer_heads in enumerate(attention_heads):
            heads_str += "[\n      "
            for head_idx, head in enumerate(layer_heads):
                w_q = head.get("W_q", [])
                w_k = head.get("W_k", [])
                w_v = head.get("W_v", [])
                w_o = head.get("W_o", [])

                w_q_str = self._format_weight_matrix(w_q)
                w_k_str = self._format_weight_matrix(w_k)
                w_v_str = self._format_weight_matrix(w_v)
                w_o_str = self._format_weight_matrix(w_o)

                heads_str += f"AttentionHead.mk {w_q_str} {w_k_str} {w_v_str} {w_o_str}"
                if head_idx < len(layer_heads) - 1:
                    heads_str += ",\n      "
            heads_str += "\n    ]"
            if layer_idx < len(attention_heads) - 1:
                heads_str += ",\n    "
        heads_str += "\n  ]"
        return heads_str

    def _format_layer_norms(self, layer_norms: List[Dict[str, List[float]]]) -> str:
        """Format layer norms for Lean code."""
        if not layer_norms:
            return "[]"

        ln_str = "[\n    "
        ln_str += ",\n    ".join(
            f"(#[{', '.join(str(w) for w in ln['weight'])}], #[{', '.join(str(b) for b in ln['bias'])}])"
            for ln in layer_norms
        )
        ln_str += "\n  ]"
        return ln_str

    def _format_feed_forward(self, ff_weights: List[Dict[str, Any]]) -> str:
        """Format feed-forward weights for Lean code."""
        if not ff_weights:
            return "[]"

        ff_str = "[\n    "
        ff_str += ",\n    ".join(
            f"([\n      "
            + ",\n      ".join(
                f"#[{', '.join(str(w) for w in row)}]" for row in ff["weight"]
            )
            + f"\n    ], #[{', '.join(str(b) for b in ff['bias'])}])"
            for ff in ff_weights
        )
        ff_str += "\n  ]"
        return ff_str

    def translate_model(
        self, model_json_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Translate a model from JSON to Lean code.

        Args:
            model_json_path: Path to the JSON model file
            output_path: Path for the output Lean file. If None, auto-generates.

        Returns:
            Path to the generated Lean file

        Raises:
            FileNotFoundError: If model JSON file doesn't exist
            ValueError: If model data is invalid
        """
        # Load and validate model JSON
        if not os.path.exists(model_json_path):
            raise FileNotFoundError(f"Model JSON file not found: {model_json_path}")

        with open(model_json_path, "r") as f:
            model_data = json.load(f)

        self.validate_model_json(model_data)

        # Generate Lean code based on model type
        model_type = model_data["model_type"]
        if model_type == "neural_net":
            lean_code = self.generate_neural_net_code(model_data)
        elif model_type == "transformer":
            lean_code = self.generate_transformer_code(model_data)
        elif model_type == "decision_tree":
            lean_code = self.generate_decision_tree_code(model_data)
        elif model_type == "linear_model":
            lean_code = self.generate_log_reg_code(model_data)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Determine output path
        if output_path is None:
            model_name = model_data.get("name", "model")
            output_path = str(self.output_directory / f"{model_name}_model.lean")
        else:
            output_path = str(Path(output_path))

        # Write Lean code to file
        with open(output_path, "w") as f:
            f.write(lean_code)

        logger.info(f"Generated Lean code: {output_path}")
        return str(output_path)


def main():
    """Main entry point for the model translator."""
    parser = argparse.ArgumentParser(
        description="Convert machine learning models from JSON to Lean 4 code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model_json model.json --output_lean model.lean
  %(prog)s --model_json neural_net.json --output_lean neural_net_model.lean
  %(prog)s --model_json transformer.json --output_lean transformer_model.lean
        """,
    )

    parser.add_argument(
        "--model_json", required=True, help="Path to the JSON model file"
    )

    parser.add_argument(
        "--output_lean",
        help="Path for the output Lean file (auto-generated if not specified)",
    )

    parser.add_argument("--output_dir", help="Directory for generated Lean files")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize translator
        translator = ModelTranslator(output_directory=args.output_dir)

        # Translate model
        output_path = translator.translate_model(
            model_json_path=args.model_json, output_path=args.output_lean
        )

        print(f"✅ Successfully generated Lean code: {output_path}")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
