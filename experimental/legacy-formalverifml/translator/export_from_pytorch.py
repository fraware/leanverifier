"""
Enhanced PyTorch model export script that supports real transformer models.
This script can export models from HuggingFace transformers library and custom PyTorch models.
"""

import torch
import torch.nn as nn
import json
import argparse
import os
import numpy as np
from typing import Dict, List, Any, Optional


def extract_linear_layer(layer: nn.Linear) -> Dict[str, Any]:
    """Extract weights and bias from a linear layer."""
    return {
        "weight": layer.weight.detach().cpu().numpy().tolist(),
        "bias": (
            layer.bias.detach().cpu().numpy().tolist() if layer.bias is not None else []
        ),
    }


def extract_attention_head(
    attention_layer: nn.MultiheadAttention, head_idx: int, d_model: int, num_heads: int
) -> Dict[str, Any]:
    """Extract a single attention head from a multi-head attention layer."""
    d_k = d_model // num_heads

    # Extract query, key, value projections
    q_weight = attention_layer.in_proj_weight[:d_model, :].detach().cpu().numpy()
    k_weight = (
        attention_layer.in_proj_weight[d_model : 2 * d_model, :].detach().cpu().numpy()
    )
    v_weight = attention_layer.in_proj_weight[2 * d_model :, :].detach().cpu().numpy()

    # Extract output projection
    o_weight = attention_layer.out_proj.weight.detach().cpu().numpy()

    # For each head, extract the corresponding slice
    start_idx = head_idx * d_k
    end_idx = (head_idx + 1) * d_k

    return {
        "W_q": q_weight[start_idx:end_idx, :].tolist(),
        "W_k": k_weight[start_idx:end_idx, :].tolist(),
        "W_v": v_weight[start_idx:end_idx, :].tolist(),
        "W_o": o_weight[:, start_idx:end_idx].tolist(),
    }


def extract_layer_norm(layer: nn.LayerNorm) -> Dict[str, Any]:
    """Extract weights and bias from a layer normalization layer."""
    return {
        "weight": layer.weight.detach().cpu().numpy().tolist(),
        "bias": (
            layer.bias.detach().cpu().numpy().tolist() if layer.bias is not None else []
        ),
    }


def extract_transformer_model(model: nn.Module) -> Dict[str, Any]:
    """Extract a complete transformer model architecture and weights."""

    # Try to identify if this is a HuggingFace transformer
    if hasattr(model, "config"):
        config = model.config
        d_model = getattr(config, "hidden_size", 512)
        num_heads = getattr(config, "num_attention_heads", 8)
        num_layers = getattr(config, "num_hidden_layers", 6)
        vocab_size = getattr(config, "vocab_size", 30000)
        max_seq_len = getattr(config, "max_position_embeddings", 512)
    else:
        # Default values for custom models
        d_model = 512
        num_heads = 8
        num_layers = 6
        vocab_size = 30000
        max_seq_len = 512

    # Extract token embeddings
    token_embeddings = []
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        token_embeddings = (
            model.embeddings.word_embeddings.weight.detach().cpu().numpy().tolist()
        )
    elif hasattr(model, "embedding"):
        token_embeddings = model.embedding.weight.detach().cpu().numpy().tolist()

    # Extract positional embeddings
    positional_embeddings = []
    if hasattr(model, "embeddings") and hasattr(
        model.embeddings, "position_embeddings"
    ):
        positional_embeddings = (
            model.embeddings.position_embeddings.weight.detach().cpu().numpy().tolist()
        )

    # Extract attention heads for each layer
    attention_heads = []
    layer_norms1 = []
    layer_norms2 = []
    ff_weights1 = []
    ff_weights2 = []

    # Try to find transformer layers
    layers = []
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = model.encoder.layer
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = model.transformer.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        # Fallback: look for any module that might be transformer layers
        for name, module in model.named_modules():
            if "layer" in name.lower() and len(list(module.children())) > 0:
                layers = list(module.children())
                break

    for layer in layers:
        layer_heads = []

        # Extract attention heads
        if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
            attention_layer = layer.attention.self
            for head_idx in range(num_heads):
                head = extract_attention_head(
                    attention_layer, head_idx, d_model, num_heads
                )
                layer_heads.append(head)
        elif hasattr(layer, "self_attn"):
            attention_layer = layer.self_attn
            for head_idx in range(num_heads):
                head = extract_attention_head(
                    attention_layer, head_idx, d_model, num_heads
                )
                layer_heads.append(head)

        attention_heads.append(layer_heads)

        # Extract layer norms
        if (
            hasattr(layer, "attention")
            and hasattr(layer.attention, "output")
            and hasattr(layer.attention.output, "LayerNorm")
        ):
            ln1 = extract_layer_norm(layer.attention.output.LayerNorm)
            layer_norms1.append(ln1)
        elif hasattr(layer, "norm1"):
            ln1 = extract_layer_norm(layer.norm1)
            layer_norms1.append(ln1)

        if hasattr(layer, "output") and hasattr(layer.output, "LayerNorm"):
            ln2 = extract_layer_norm(layer.output.LayerNorm)
            layer_norms2.append(ln2)
        elif hasattr(layer, "norm2"):
            ln2 = extract_layer_norm(layer.norm2)
            layer_norms2.append(ln2)

        # Extract feed-forward networks
        if hasattr(layer, "intermediate") and hasattr(layer.intermediate, "dense"):
            ff1 = extract_linear_layer(layer.intermediate.dense)
            ff_weights1.append(ff1)
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "fc1"):
            ff1 = extract_linear_layer(layer.mlp.fc1)
            ff_weights1.append(ff1)

        if hasattr(layer, "output") and hasattr(layer.output, "dense"):
            ff2 = extract_linear_layer(layer.output.dense)
            ff_weights2.append(ff2)
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
            ff2 = extract_linear_layer(layer.mlp.fc2)
            ff_weights2.append(ff2)

    # Extract output projection
    output_projection = {"weight": [], "bias": []}
    if hasattr(model, "classifier"):
        output_projection = extract_linear_layer(model.classifier)
    elif hasattr(model, "head"):
        output_projection = extract_linear_layer(model.head)
    elif hasattr(model, "output_projection"):
        output_projection = extract_linear_layer(model.output_projection)

    return {
        "model_type": "transformer",
        "name": "exportedTransformer",
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "token_embeddings": token_embeddings,
        "positional_embeddings": positional_embeddings,
        "attention_heads": attention_heads,
        "layer_norms1": layer_norms1,
        "layer_norms2": layer_norms2,
        "ff_weights1": ff_weights1,
        "ff_weights2": ff_weights2,
        "output_projection": output_projection,
    }


def export_model(model: nn.Module, output_path: str, model_type: Optional[str] = None):
    """
    Export a PyTorch model to JSON format.

    Args:
        model: PyTorch model to export
        output_path: Path to save the JSON file
        model_type: Optional model type override
    """

    # Determine model type
    if model_type is None:
        # Try to auto-detect transformer models
        if any("transformer" in str(type(model)).lower() for _ in [1]):
            model_type = "transformer"
        elif any("bert" in str(type(model)).lower() for _ in [1]):
            model_type = "transformer"
        elif any("gpt" in str(type(model)).lower() for _ in [1]):
            model_type = "transformer"
        else:
            model_type = "NN"  # Default to neural network

    if model_type == "transformer":
        export_dict = extract_transformer_model(model)
    else:
        # Original neural network export logic
        layers = []
        for layer in model.modules():
            if layer == model:
                continue
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().numpy().tolist()
                bias = (
                    layer.bias.detach().cpu().numpy().tolist()
                    if layer.bias is not None
                    else []
                )
                layers.append({"type": "linear", "weight": weight, "bias": bias})
            elif isinstance(layer, nn.ReLU):
                layers.append({"type": "relu"})
            elif isinstance(layer, nn.Sigmoid):
                layers.append({"type": "sigmoid"})
            elif isinstance(layer, nn.Tanh):
                layers.append({"type": "tanh"})

        # Infer dimensions
        input_dim = None
        output_dim = None
        for layer in layers:
            if layer["type"] == "linear":
                if input_dim is None:
                    input_dim = len(layer["weight"][0])
                output_dim = len(layer["weight"])

        if input_dim is None or output_dim is None:
            raise ValueError(
                "Could not infer input or output dimensions from the model layers."
            )

        export_dict = {
            "model_type": "NN",
            "name": os.path.splitext(os.path.basename(output_path))[0],
            "input_dim": input_dim,
            "output_dim": output_dim,
            "layers": layers,
        }

    with open(output_path, "w") as f:
        json.dump(export_dict, f, indent=2)

    print(f"Exported {model_type} model to {output_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to JSON format for Lean verification."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the PyTorch model file (.pth)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the JSON file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["NN", "transformer"],
        help="Model type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--from_huggingface",
        action="store_true",
        help="Load model from HuggingFace transformers",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model name (if using --from_huggingface)",
    )

    args = parser.parse_args()

    if args.from_huggingface:
        if not args.model_name:
            raise ValueError("--model_name is required when using --from_huggingface")

        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(args.model_name)
            print(f"Loaded HuggingFace model: {args.model_name}")
        except ImportError:
            print("Please install transformers: pip install transformers")
            return
    else:
        # Load local PyTorch model
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

        model = torch.load(args.model_path, map_location="cpu")
        if isinstance(model, dict):
            # Handle state dict format
            if "model" in model:
                model = model["model"]
            elif "state_dict" in model:
                model = model["state_dict"]

        print(f"Loaded local model from: {args.model_path}")

    export_model(model, args.output_path, args.model_type)


if __name__ == "__main__":
    main()
