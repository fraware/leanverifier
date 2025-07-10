#!/usr/bin/env python3
"""
FormalVerifML Web Interface

A Flask-based web application for uploading, visualizing, and verifying machine
learning models using the FormalVerifML framework. Provides an intuitive interface
for model management, verification, and result visualization.

Author: FormalVerifML Team
License: MIT
Version: 2.0.0
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback

from flask import Flask, request, render_template, jsonify, url_for, send_file
from flask_cors import CORS
import graphviz
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/webapp.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Handles model architecture visualization using Graphviz.

    This class generates visual representations of machine learning models
    including neural networks, decision trees, and transformers.
    """

    def __init__(self, output_dir: str = "static"):
        """
        Initialize the ModelVisualizer.

        Args:
            output_dir: Directory to save generated visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.supported_formats = ["png", "svg", "pdf"]

        logger.info(
            f"Initialized ModelVisualizer with output directory: {self.output_dir}"
        )

    def generate_model_graph(
        self, json_path: str, output_image_base: str, format: str = "png"
    ) -> str:
        """
        Generate a visualization of the model architecture from a JSON file.

        Args:
            json_path: Path to the model JSON file
            output_image_base: Base name for the output image file
            format: Output format (png, svg, pdf)

        Returns:
            Path to the generated visualization file

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If model type is not supported
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Model JSON file not found: {json_path}")

        if format not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        # Load model data
        with open(json_path, "r") as f:
            model_data = json.load(f)

        model_type = model_data.get("model_type", "")
        logger.info(f"Generating visualization for model type: {model_type}")

        # Create Graphviz diagram
        dot = graphviz.Digraph(
            comment="Model Architecture", format=format, engine="dot"
        )
        dot.attr(rankdir="TB", size="8,8", dpi="300")

        # Generate visualization based on model type
        if model_type == "neural_net":
            self._visualize_neural_network(dot, model_data)
        elif model_type == "decision_tree":
            self._visualize_decision_tree(dot, model_data)
        elif model_type == "transformer":
            self._visualize_transformer(dot, model_data)
        elif model_type == "linear_model":
            self._visualize_linear_model(dot, model_data)
        else:
            dot.node("error", f"Unknown model type: {model_type}")

        # Render the graph
        output_path = self.output_dir / f"{output_image_base}.{format}"
        dot.render(filename=str(output_path), cleanup=True)

        logger.info(f"Generated visualization: {output_path}")
        return str(output_path)

    def _visualize_neural_network(
        self, dot: graphviz.Digraph, model_data: Dict[str, Any]
    ) -> None:
        """Visualize neural network architecture."""
        input_dim = model_data.get("input_dim", "?")
        output_dim = model_data.get("output_dim", "?")

        # Input layer
        dot.node(
            "input",
            f"Input\n(dim={input_dim})",
            shape="box",
            style="filled",
            fillcolor="lightblue",
        )

        # Hidden layers
        layers = model_data.get("layers", [])
        for i, layer in enumerate(layers):
            layer_type = layer.get("type", "unknown")
            label = layer_type.upper()

            # Add layer details
            if layer_type == "linear":
                weight = layer.get("weight", [])
                if weight and isinstance(weight, list) and len(weight) > 0:
                    out_dim = len(weight)
                    in_dim = len(weight[0]) if weight[0] else "?"
                    label += f"\n({in_dim}→{out_dim})"

            node_name = f"layer{i}"
            dot.node(
                node_name, label, shape="box", style="filled", fillcolor="lightgreen"
            )

            # Connect layers
            if i == 0:
                dot.edge("input", node_name)
            else:
                dot.edge(f"layer{i-1}", node_name)

        # Output layer
        dot.node(
            "output",
            f"Output\n(dim={output_dim})",
            shape="box",
            style="filled",
            fillcolor="lightcoral",
        )

        if layers:
            dot.edge(f"layer{len(layers)-1}", "output")
        else:
            dot.edge("input", "output")

    def _visualize_decision_tree(
        self, dot: graphviz.Digraph, model_data: Dict[str, Any]
    ) -> None:
        """Visualize decision tree structure."""

        def add_tree_nodes(tree: Dict[str, Any], parent: Optional[str] = None) -> None:
            if "leaf" in tree:
                label = f"Leaf: {tree['leaf']}"
                node_id = f"leaf_{tree['leaf']}_{os.urandom(2).hex()}"
                dot.node(
                    node_id,
                    label,
                    shape="ellipse",
                    style="filled",
                    fillcolor="lightyellow",
                )
                if parent:
                    dot.edge(parent, node_id)
            else:
                feature_index = tree.get("feature_index", "?")
                threshold = tree.get("threshold", "?")
                label = f"X[{feature_index}] ≤ {threshold}?"
                node_id = f"node_{feature_index}_{threshold}_{os.urandom(2).hex()}"
                dot.node(
                    node_id,
                    label,
                    shape="diamond",
                    style="filled",
                    fillcolor="lightcyan",
                )
                if parent:
                    dot.edge(parent, node_id)
                if "left" in tree:
                    add_tree_nodes(tree["left"], parent=node_id)
                if "right" in tree:
                    add_tree_nodes(tree["right"], parent=node_id)

        add_tree_nodes(model_data.get("tree", {}))

    def _visualize_transformer(
        self, dot: graphviz.Digraph, model_data: Dict[str, Any]
    ) -> None:
        """Visualize transformer architecture."""
        d_model = model_data.get("d_model", "?")
        num_heads = model_data.get("num_heads", "?")
        num_layers = model_data.get("num_layers", "?")

        # Input embedding
        dot.node(
            "input",
            f"Input Embedding\n(d_model={d_model})",
            shape="box",
            style="filled",
            fillcolor="lightblue",
        )

        # Transformer layers
        for i in range(num_layers):
            layer_name = f"layer{i}"
            dot.node(
                layer_name,
                f"Transformer Layer {i+1}\n({num_heads} heads)",
                shape="box",
                style="filled",
                fillcolor="lightgreen",
            )

            if i == 0:
                dot.edge("input", layer_name)
            else:
                dot.edge(f"layer{i-1}", layer_name)

        # Output projection
        dot.node(
            "output",
            "Output Projection",
            shape="box",
            style="filled",
            fillcolor="lightcoral",
        )
        dot.edge(f"layer{num_layers-1}", "output")

    def _visualize_linear_model(
        self, dot: graphviz.Digraph, model_data: Dict[str, Any]
    ) -> None:
        """Visualize linear model architecture."""
        input_dim = model_data.get("input_dim", "?")

        dot.node(
            "input",
            f"Input\n(dim={input_dim})",
            shape="box",
            style="filled",
            fillcolor="lightblue",
        )
        dot.node(
            "linear",
            "Linear Layer",
            shape="box",
            style="filled",
            fillcolor="lightgreen",
        )
        dot.node(
            "output", "Output", shape="box", style="filled", fillcolor="lightcoral"
        )

        dot.edge("input", "linear")
        dot.edge("linear", "output")


class ModelProcessor:
    """
    Handles model processing and verification.

    This class manages the conversion of models from JSON to Lean code
    and coordinates the verification process.
    """

    def __init__(
        self,
        translator_path: str = "translator",
        lean_path: str = "lean",
        output_path: str = "lean/FormalVerifML/generated",
    ):
        """
        Initialize the ModelProcessor.

        Args:
            translator_path: Path to translator scripts
            lean_path: Path to Lean code
            output_path: Path for generated Lean files
        """
        self.translator_path = Path(translator_path)
        self.lean_path = Path(lean_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized ModelProcessor with paths: "
            f"translator={self.translator_path}, lean={self.lean_path}"
        )

    def process_model(self, model_json_path: str, model_name: str) -> Dict[str, Any]:
        """
        Process a model through the complete pipeline.

        Args:
            model_json_path: Path to the model JSON file
            model_name: Name of the model

        Returns:
            Dictionary containing processing results

        Raises:
            FileNotFoundError: If required files don't exist
            subprocess.CalledProcessError: If external commands fail
        """
        logger.info(f"Processing model: {model_name}")

        try:
            # Step 1: Generate Lean code
            lean_out_path = self.output_path / f"{model_name}_model.lean"
            translator_script = self.translator_path / "generate_lean_model.py"

            if not translator_script.exists():
                raise FileNotFoundError(
                    f"Translator script not found: {translator_script}"
                )

            cmd = [
                "python",
                str(translator_script),
                "--model_json",
                model_json_path,
                "--output_lean",
                str(lean_out_path),
            ]

            logger.info(f"Running translator: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            # Step 2: Build Lean project
            build_cmd = ["lake", "build"]
            logger.info(f"Building Lean project: {' '.join(build_cmd)}")
            build_result = subprocess.run(
                build_cmd, capture_output=True, text=True, timeout=600
            )

            if build_result.returncode != 0:
                raise subprocess.CalledProcessError(
                    build_result.returncode,
                    build_cmd,
                    build_result.stdout,
                    build_result.stderr,
                )

            return {
                "success": True,
                "model_name": model_name,
                "lean_file": str(lean_out_path),
                "translator_output": result.stdout,
                "build_output": build_result.stdout,
            }

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out: {e}")
            return {
                "success": False,
                "model_name": model_name,
                "error": f"Command timed out: {e}",
                "command": str(e.cmd),
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return {
                "success": False,
                "model_name": model_name,
                "error": e.stderr,
                "command": str(e.cmd),
                "return_code": e.returncode,
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"success": False, "model_name": model_name, "error": str(e)}


class Config:
    """Configuration management for the web application."""

    def __init__(self):
        """Initialize configuration with default values."""
        self.MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size
        self.UPLOAD_FOLDER = "uploads"
        self.ALLOWED_EXTENSIONS = {"json"}
        self.DEBUG = os.getenv("FLASK_ENV") == "development"
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 5000))
        self.SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

        # Create necessary directories
        Path(self.UPLOAD_FOLDER).mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        Path("static").mkdir(exist_ok=True)


def create_app(config: Optional[Config] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Configuration object. If None, uses default config.

    Returns:
        Configured Flask application
    """
    if config is None:
        config = Config()

    app = Flask(__name__)
    app.config.from_object(config)

    # Enable CORS for development
    CORS(app)

    # Initialize components
    visualizer = ModelVisualizer()
    processor = ModelProcessor()

    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed."""
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
        )

    @app.route("/", methods=["GET", "POST"])
    def index():
        """Main page for model upload and processing."""
        if request.method == "POST":
            try:
                files = request.files.getlist("model_json")
                if not files or all(not file.filename for file in files):
                    return jsonify({"error": "No files selected"}), 400

                responses = []
                for file in files:
                    if file and file.filename and allowed_file(file.filename):
                        # Secure filename and save
                        filename = secure_filename(file.filename)
                        model_name = filename.rsplit(".", 1)[0]
                        model_json_path = os.path.join(
                            app.config["UPLOAD_FOLDER"], filename
                        )
                        file.save(model_json_path)

                        logger.info(f"Uploaded model: {filename}")

                        # Process model
                        result = processor.process_model(model_json_path, model_name)
                        responses.append(result)

                        if result["success"]:
                            logger.info(f"Successfully processed model: {model_name}")
                        else:
                            logger.error(f"Failed to process model: {model_name}")
                    else:
                        responses.append(
                            {
                                "success": False,
                                "model_name": file.filename if file else "unknown",
                                "error": "Invalid file type or no filename",
                            }
                        )

                return jsonify({"results": responses})

            except RequestEntityTooLarge:
                return jsonify({"error": "File too large"}), 413
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            return render_template("index.html")

    @app.route("/visualize", methods=["GET"])
    def visualize():
        """Generate and display model visualization."""
        try:
            model_name = request.args.get("model", "exported_model")
            json_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{model_name}.json")

            if not os.path.exists(json_path):
                return f"Model JSON file {model_name}.json not found.", 404

            # Generate visualization
            output_image_base = os.path.join("static", "model_graph")
            image_path = visualizer.generate_model_graph(json_path, output_image_base)

            # Get image URL
            image_filename = os.path.basename(image_path)
            image_url = url_for("static", filename=image_filename)

            return render_template(
                "model_visualization.html",
                image_url=image_url,
                model_name=model_name,
            )

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return f"Error generating visualization: {e}", 500

    @app.route("/logs", methods=["GET"])
    def logs():
        """Display application logs."""
        try:
            log_path = "logs/webapp.log"
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    content = f.read()
                return f"<pre>{content}</pre>"
            else:
                return "No logs available."
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return f"Error reading logs: {e}", 500

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
            }
        )

    @app.route("/api/models", methods=["GET"])
    def list_models():
        """List available models."""
        try:
            upload_dir = Path(app.config["UPLOAD_FOLDER"])
            models = []

            for json_file in upload_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        model_data = json.load(f)

                    models.append(
                        {
                            "name": json_file.stem,
                            "type": model_data.get("model_type", "unknown"),
                            "upload_time": datetime.fromtimestamp(
                                json_file.stat().st_mtime
                            ).isoformat(),
                            "size": json_file.stat().st_size,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error reading model {json_file}: {e}")

            return jsonify({"models": models})

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    return app


def main():
    """Main entry point for the web application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="FormalVerifML Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --host 0.0.0.0 --port 5000
  %(prog)s --debug
  %(prog)s --config production
        """,
    )

    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind to (default: 5000)"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--config",
        choices=["development", "production", "testing"],
        default="development",
        help="Configuration environment (default: development)",
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["FLASK_ENV"] = args.config

    # Create and run application
    app = create_app()

    if args.debug:
        app.config["DEBUG"] = True
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting FormalVerifML web interface on {args.host}:{args.port}")
    logger.info(f"Configuration: {args.config}")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down web interface")
    except Exception as e:
        logger.error(f"Error starting web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
