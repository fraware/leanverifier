# Developer Guide

Welcome to the Formal Verification of Machine Learning Models in Lean project. This guide provides developers with details on the architecture, code structure, and instructions for extending the framework.

## Project Structure

```bash
formal_verif_ml/
├── docs/
│ ├── developer_guide.md
│ ├── user_guide.md
│ └── tutorials.md
├── lean/
│ ├── base/
│ │ ├── advanced_models.lean
│ │ ├── advanced_tactics.lean
│ │ ├── definitions.lean
│ │ ├── ml_properties.lean
│ │ └── symbolic_models.lean
│ ├── generated/
│ │ ├── another_nn_model.lean
│ │ ├── decision_tree_model.lean
│ │ ├── example_model.lean
│ │ └── log_reg_model.lean
│ ├── proofs/
│ │ ├── decision_tree_proof.lean
│ │ ├── example_fairness_proof.lean
│ │ ├── example_robustness_proof.lean
│ │ ├── extended_fairness_proof.lean
│ │ └── extended_robustness_proof.lean
│ └── formal_verif_ml.lean
├── translator/
│ ├── another_nn.json
│ ├── decision_tree.json
│ ├── export_from_pytorch.py
│ ├── generate_lean_model.py
│ ├── log_reg.json
│ └── requirements.txt
├── webapp/
│ ├── app.py
│ ├── static/
│ │ └── model_graph.png
│ └── templates/
│ ├── index.html
│ └── model_visualization.html
├── lakefile.lean
├── Dockerfile
└── .github/
└── workflows/
└── lean_ci.yml
```

## Key Components

### Lean Code

- **Base Definitions (`lean/base/`)**:  
  Contains the core data structures for neural networks, linear models, decision trees, and additional advanced model types.
- **Properties (`ml_properties.lean`)**:  
  Formal definitions for robustness, fairness, interpretability, monotonicity, sensitivity, and counterfactual fairness.
- **Advanced Tactics (`advanced_tactics.lean`)**:  
  Custom tactics for handling piecewise reasoning and placeholders for external SMT solver integration.
- **Generated Files (`lean/generated/`)**:  
  Auto-generated Lean definitions produced by the Python translator. Each file corresponds to a different model (e.g., `another_nn_model.lean`, `log_reg_model.lean`).

- **Proof Scripts (`lean/proofs/`)**:  
  Extended proof scripts demonstrating how to verify various properties. These include examples of bounding proofs, case-splitting for ReLU, and induction proofs for decision trees.

### Python Translator

- **generate_lean_model.py**:  
  Converts model definitions in JSON into Lean code. Supports multiple model types such as neural networks, logistic regression, and decision trees.
- **export_from_pytorch.py**:  
  An adapter that loads a full PyTorch model (saved with torch.save), extracts its architecture and weights (currently supporting Linear and ReLU layers), and outputs JSON in the required schema.

### Web Interface

- **Flask App (`webapp/app.py`)**:  
  Provides a user-friendly web interface for uploading model JSON files, initiating Lean code generation, and viewing proof logs.
- **Templates & Static Files**:  
  HTML templates for the home page, model visualization, and proof logs. Static assets support visualization.

## Extending the Framework

### Adding New Model Types

1. **JSON Schema**:

   - Extend the JSON schema (in `translator/`) to include new fields for additional model types (e.g., Convolutional layers, LSTM layers).

2. **Translator Updates**:

   - Update `generate_lean_model.py` to parse new JSON fields and generate the corresponding Lean code.

3. **Lean Definitions**:
   - Add new structures and evaluation functions in `lean/base/advanced_models.lean` as needed.

### Extending Proof Capabilities

- **Custom Tactics**:  
  Develop or integrate advanced tactics in `lean/base/advanced_tactics.lean` to automate non-linear and piecewise reasoning.

- **New Properties**:  
  Implement additional property definitions (e.g., interpretability, monotonicity, counterfactual fairness) in `ml_properties.lean`.

- **Proof Scripts**:  
  Create new proof scripts in `lean/proofs/` to demonstrate and validate these properties.

### Testing and CI/CD

- Use the provided GitHub Actions workflow (`.github/workflows/lean_ci.yml`) as a starting point.
- Add new tests for the translator and web interface.
- Consider stress tests for larger models and additional proof logs.

## Contributing

We welcome contributions! Before submitting a pull request:

- Please review the code structure and style guidelines.
- Ensure your changes pass the CI/CD tests using Lake.
- Update documentation as needed.

For further questions, please reach out via the repository’s issue tracker.
