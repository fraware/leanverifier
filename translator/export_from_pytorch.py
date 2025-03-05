"""
This script loads a full PyTorch model (saved with torch.save),
extracts its architecture and weights (currently supporting Linear and ReLU layers),
and outputs JSON in the required schema.
"""

import torch
import json
import argparse
import os


def export_model(model, output_path):
    layers = []
    # Use model.modules() but skip the top-level container.
    # We assume the model is an instance of torch.nn.Module with a sequential structure.
    for layer in model.modules():
        # Skip the top-level module.
        if layer == model:
            continue
        if isinstance(layer, torch.nn.Linear):
            weight = layer.weight.detach().cpu().numpy().tolist()
            bias = layer.bias.detach().cpu().numpy().tolist()
            layers.append({"type": "linear", "weight": weight, "bias": bias})
        elif isinstance(layer, torch.nn.ReLU):
            layers.append({"type": "relu"})
        # Extend here for other layer types (e.g., Sigmoid, Tanh, etc.)

    # Infer input_dim and output_dim from the first and last linear layers.
    input_dim = None
    output_dim = None
    for layer in layers:
        if layer["type"] == "linear":
            if input_dim is None:
                # Assume weight shape: [out_features, in_features]
                input_dim = len(layer["weight"][0])
            output_dim = len(
                layer["weight"]
            )  # last encountered linear's output dimension

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
    print(f"Exported model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export a PyTorch model to Lean JSON format."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved PyTorch model file (saved with torch.save).",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output JSON file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file '{args.model_path}' does not exist.")

    # Load the model (assumes the full model was saved, not just a state_dict)
    model = torch.load(args.model_path, map_location=torch.device("cpu"))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Loaded object is not a torch.nn.Module. Ensure you saved the full model with torch.save."
        )

    model.eval()  # Ensure model is in evaluation mode

    export_model(model, args.output)


if __name__ == "__main__":
    main()
