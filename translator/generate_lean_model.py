import json
import argparse
import os


def generate_neural_net_code(model_json) -> str:
    name = model_json.get("name", "exampleNeuralNet")
    input_dim = model_json["input_dim"]
    output_dim = model_json["output_dim"]
    layers = model_json["layers"]

    lean_lines = []
    lean_lines.append("import FormalVerifML.base.definitions")
    lean_lines.append("namespace FormalVerifML")
    lean_lines.append(f"-- Auto-generated Neural Net definition for {name}\n")

    layer_exprs = []
    for idx, layer in enumerate(layers):
        ltype = layer["type"]
        if ltype == "linear":
            weight = layer["weight"]
            bias = layer["bias"]
            weight_str = "["
            weight_str += ", ".join(
                f"#[{', '.join(str(wij) for wij in row)}]" for row in weight
            )
            weight_str += "]"
            bias_str = f"#[{', '.join(str(bi) for bi in bias)}]"
            layer_exprs.append(f"LayerType.linear {weight_str} {bias_str}")
        elif ltype == "relu":
            layer_exprs.append("LayerType.relu")
        elif ltype == "sigmoid":
            layer_exprs.append("LayerType.sigmoid")
        elif ltype == "tanh":
            layer_exprs.append("LayerType.tanh")
        else:
            raise ValueError(f"Unsupported layer type: {ltype}")

    layers_str = ",\n    ".join(layer_exprs)
    lean_lines.append(f"def {name} : NeuralNet :=")
    lean_lines.append(f"  {{ inputDim  := {input_dim},")
    lean_lines.append(f"    outputDim := {output_dim},")
    lean_lines.append(f"    layers    := [\n      {layers_str}\n    ] }}")
    lean_lines.append("end FormalVerifML")
    return "\n".join(lean_lines)


def generate_log_reg_code(model_json) -> str:
    name = model_json.get("name", "myLogRegModel")
    input_dim = model_json["input_dim"]
    weights = model_json["weights"]
    bias = model_json["bias"]
    weights_str = f"#[{', '.join(str(w) for w in weights)}]"

    lean_lines = []
    lean_lines.append("import FormalVerifML.base.definitions")
    lean_lines.append("namespace FormalVerifML")
    lean_lines.append(f"-- Auto-generated Logistic Regression Model for {name}\n")
    lean_lines.append(f"def {name} : LinearModel :=")
    lean_lines.append(f"  {{ inputDim := {input_dim},")
    lean_lines.append(f"    weights  := {weights_str},")
    lean_lines.append(f"    bias     := {bias} }}")
    lean_lines.append("end FormalVerifML")
    return "\n".join(lean_lines)


def generate_decision_tree_code(model_json) -> str:
    name = model_json.get("name", "myDecisionTree")

    # If the node is a leaf, then the JSON will have a key "leaf"
    def gen_tree(node):
        if "leaf" in node:
            return f"(DecisionTree.leaf {node['leaf']})"
        else:
            left = gen_tree(node["left"])
            right = gen_tree(node["right"])
            return (
                f"(DecisionTree.node {node['feature_index']} {node['threshold']} "
                f"{left} {right})"
            )

    tree_code = gen_tree(model_json)

    lean_lines = []
    lean_lines.append("import FormalVerifML.base.definitions")
    lean_lines.append("namespace FormalVerifML")
    lean_lines.append(f"-- Auto-generated Decision Tree Model for {name}\n")
    lean_lines.append(f"def {name} : DecisionTree :=")
    lean_lines.append(f"  {tree_code}")
    lean_lines.append("end FormalVerifML")
    return "\n".join(lean_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Lean code for an ML model.")
    parser.add_argument(
        "--model_json",
        type=str,
        required=True,
        help="Path to the JSON file describing the model.",
    )
    parser.add_argument(
        "--output_lean", type=str, required=True, help="Path to the output Lean file."
    )
    args = parser.parse_args()

    with open(args.model_json, "r") as f:
        model_data = json.load(f)

    model_type = model_data.get("model_type")
    if model_type == "NN":
        lean_code = generate_neural_net_code(model_data)
    elif model_type == "log_reg":
        lean_code = generate_log_reg_code(model_data)
    elif model_type == "decision_tree":
        lean_code = generate_decision_tree_code(model_data)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_lean), exist_ok=True)
    with open(args.output_lean, "w") as f:
        f.write(lean_code)

    print(f"Generated Lean code at {args.output_lean}.")


if __name__ == "__main__":
    main()
