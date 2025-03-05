from flask import Flask, request, render_template, jsonify, url_for
import subprocess
import os
import json
import graphviz

app = Flask(__name__)


def generate_model_graph(json_path, output_image_base):
    """
    Generate a visualization of the model architecture from a JSON file.
    Supports model_type "NN" and "decision_tree". The resulting image is saved
    as a PNG file at output_image_base + ".png" and its path is returned.
    """
    with open(json_path, "r") as f:
        model_data = json.load(f)
    model_type = model_data.get("model_type", "")
    dot = graphviz.Digraph(comment="Model Architecture", format="png")

    if model_type == "NN":
        # Neural Network visualization: create sequential graph.
        input_dim = model_data.get("input_dim", "?")
        output_dim = model_data.get("output_dim", "?")
        dot.node("input", f"Input\n(dim={input_dim})")
        layers = model_data.get("layers", [])
        for i, layer in enumerate(layers):
            layer_type = layer.get("type", "unknown")
            label = layer_type
            if layer_type == "linear":
                weight = layer.get("weight", [])
                if weight and isinstance(weight, list) and len(weight) > 0:
                    out_dim = len(weight)
                    in_dim = len(weight[0]) if weight[0] else "?"
                    label += f"\n({in_dim}→{out_dim})"
            node_name = f"layer{i}"
            dot.node(node_name, label)
            if i == 0:
                dot.edge("input", node_name)
            else:
                dot.edge(f"layer{i-1}", node_name)
        dot.node("output", f"Output\n(dim={output_dim})")
        if layers:
            dot.edge(f"layer{len(layers)-1}", "output")
        else:
            dot.edge("input", "output")
    elif model_type == "decision_tree":
        # Decision Tree visualization: build recursively.
        def add_tree_nodes(tree, parent=None):
            if "leaf" in tree:
                label = f"Leaf: {tree['leaf']}"
                node_id = f"leaf_{tree['leaf']}_{os.urandom(2).hex()}"
                dot.node(node_id, label)
                if parent:
                    dot.edge(parent, node_id)
            else:
                feature_index = tree.get("feature_index", "?")
                threshold = tree.get("threshold", "?")
                label = f"X[{feature_index}] ≤ {threshold}?"
                node_id = f"node_{feature_index}_{threshold}_{os.urandom(2).hex()}"
                dot.node(node_id, label)
                if parent:
                    dot.edge(parent, node_id)
                if "left" in tree:
                    add_tree_nodes(tree["left"], parent=node_id)
                if "right" in tree:
                    add_tree_nodes(tree["right"], parent=node_id)

        add_tree_nodes(model_data)
    else:
        dot.node("error", "Unknown model type")

    # Render the graph to a PNG file.
    output_path = dot.render(filename=output_image_base, cleanup=True)
    return output_path  # Returns path ending with ".png"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("model_json")
        responses = []
        for file in files:
            if file:
                model_name = file.filename.split(".")[0]
                model_json_path = os.path.join("/app/translator", f"{model_name}.json")
                file.save(model_json_path)

                lean_out_path = os.path.join(
                    "/app/lean/generated", f"{model_name}_model.lean"
                )
                translator_script = os.path.join(
                    "/app/translator", "generate_lean_model.py"
                )

                cmd = [
                    "python",
                    translator_script,
                    "--model_json",
                    model_json_path,
                    "--output_lean",
                    lean_out_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    responses.append(
                        f"Translator failed for {model_name}:\n{result.stderr}"
                    )
                    continue

                build_cmd = ["lean", "--make", "/app/lean/formal_verif_ml.lean"]
                build_result = subprocess.run(build_cmd, capture_output=True, text=True)
                if build_result.returncode != 0:
                    responses.append(
                        f"Lean build failed for {model_name}:\n{build_result.stderr}"
                    )
                    continue

                responses.append(
                    f"Model {model_name} imported and Lean proofs compiled successfully!"
                )
        return jsonify({"results": responses})
    else:
        return render_template("index.html")


@app.route("/visualize", methods=["GET"])
def visualize():
    # Retrieve model name from query parameter; default to 'exported_model'
    model_name = request.args.get("model", "exported_model")
    json_path = os.path.join("/app/translator", f"{model_name}.json")
    if not os.path.exists(json_path):
        return f"Model JSON file {model_name}.json not found.", 404
    # Generate visualization image and save it in the static folder.
    output_image_base = os.path.join("/app/static", "model_graph")
    try:
        image_path = generate_model_graph(json_path, output_image_base)
    except Exception as e:
        return f"Error generating visualization: {e}", 500
    # Pass the image URL to the template.
    image_filename = os.path.basename(image_path)
    return render_template(
        "model_visualization.html",
        image_url=url_for("static", filename=image_filename),
        model_name=model_name,
    )


@app.route("/logs", methods=["GET"])
def logs():
    log_path = "/app/logs/proof.log"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    else:
        return "No logs available."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
