# Formal Verification of Machine Learning Models in Lean

Welcome to the **Formal Verification of Machine Learning Models in Lean** project. This repository provides a framework for specifying and proving properties—such as robustness, fairness, and interpretability—of machine learning models using Lean 4.

A live, interactive webpage is available at: [proof-pipeline-interactor.lovable.app](https://proof-pipeline-interactor.lovable.app)

## Overview

In high-stakes applications (e.g., healthcare, finance, autonomous systems), ensuring that machine learning models meet strict reliability and fairness properties is essential. This project provides:

- **Lean Library**: Formal definitions for neural networks, linear models, decision trees, and advanced models (ConvNets, RNNs, Transformers), along with properties like adversarial robustness, fairness, interpretability, monotonicity, and sensitivity analysis.
- **Model Translator**: A Python-based tool that exports trained models (e.g., from PyTorch) to a JSON schema and automatically generates corresponding Lean code.
- **Web Interface**: A Flask application for uploading models, triggering Lean verification, visualizing model architectures (using Graphviz), and viewing proof logs.
- **CI/CD Pipeline**: A reproducible, Dockerized environment using Lean 4’s Lake build system with GitHub Actions for continuous integration and deployment.

## Features

- **Formal Verification**: Prove key properties of ML models including adversarial robustness and fairness.
- **Advanced Model Support**: Extendable to support convolutional networks, recurrent architectures, transformers, and even symbolic models.
- **Interactive Web Portal**: Upload model JSON files, view generated Lean code, trigger Lean proof compilation, and visualize the model architecture.
- **Automated Build Pipeline**: Docker and GitHub Actions integration for reliable, reproducible builds.

## Quick Start

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/fraware/formal_verif_ml.git
   cd formal_verif_ml
   ```

2. **Build the Docker Image:**

   ```bash
   docker build -t formal-ml .

   ```

3. **Run the Container:**

   ```bash
   docker run -p 5000:5000 formal-ml
   ```

4. **Access the Web Interface:**

Open http://localhost:5000 in your browser.

For detailed usage and contribution guidelines, please refer to the User Guide and Developer Guide.

## Contributing

Contributions, improvements, and bug reports are welcome. Please see the `docs/` folder for additional developer guidelines and contribution standards.

## License

This project is licensed under the MIT License.
