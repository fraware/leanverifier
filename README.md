# FormalVerifML: Formal Verification of Machine Learning Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Lean 4](https://img.shields.io/badge/Lean-4-green.svg)](https://leanprover.github.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> **Enterprise-grade formal verification framework for machine learning models with support for large-scale transformers, vision models, and distributed verification.**

<p align="center">
  <img src=".github/assets/FormalVerifML.png" alt="FormalVerifML Logo" width="200"/>
</p>

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

FormalVerifML is a state-of-the-art framework for formally verifying machine learning models using Lean 4. It provides comprehensive support for verifying properties such as robustness, fairness, interpretability, and safety across a wide range of model architectures.

### Mission

To provide **mathematically rigorous verification** of ML models for high-stakes applications in healthcare, finance, autonomous systems, and other critical domains where model reliability is paramount.

### What Makes Us Different

- **Mathematical Rigor**: Uses Lean 4 theorem prover for formal mathematical proofs
- **Production Ready**: Enterprise features with multi-user support, audit logging, and security
- **Scalable**: Supports models up to 100M+ parameters with distributed verification
- **Comprehensive**: Vision transformers, large-scale models, and advanced architectures
- **Automated**: SMT solver integration for automated proof generation

## Key Features

### Model Support

- **Neural Networks**: Feed-forward, convolutional, recurrent architectures
- **Transformers**: Full transformer support with multi-head attention
- **Vision Models**: ViT, Swin Transformers, CLIP-style multi-modal models
- **Large-Scale Models**: 100M+ parameter models with distributed processing
- **Decision Trees**: Interpretable tree-based models
- **Linear Models**: Logistic regression and linear classifiers

### Verification Properties

- **Robustness**: Adversarial robustness and input perturbation resistance
- **Fairness**: Demographic parity, equalized odds, individual fairness
- **Interpretability**: Attention analysis, feature attribution verification
- **Safety**: Causal masking, sequence invariance, memory efficiency
- **Performance**: Memory optimization, distributed verification

### Enterprise Features

- **Multi-User Support**: Role-based access control and session management
- **Audit Logging**: Comprehensive activity tracking and compliance
- **Security**: Rate limiting, encryption, and input validation
- **Distributed Processing**: Multi-node verification with fault tolerance
- **Monitoring**: Real-time performance metrics and health checks

## Architecture

```
FormalVerifML/
├── lean/                          # Lean 4 formal verification code
│   ├── FormalVerifML/
│   │   ├── base/                  # Core definitions and properties
│   │   ├── generated/             # Auto-generated model definitions
│   │   └── proofs/                # Verification proof scripts
├── translator/                    # Model translation and testing
│   ├── export_from_pytorch.py    # PyTorch model export
│   ├── generate_lean_model.py    # JSON to Lean code generation
│   └── test_*.py                 # Comprehensive test suites
├── webapp/                       # Web interface and visualization
├── docs/                         # Documentation and guides
└── .github/                      # CI/CD and workflows
```

### Data Flow

1. **Model Export**: PyTorch/HuggingFace models → JSON format
2. **Code Generation**: JSON → Lean 4 definitions
3. **Verification**: Lean 4 → Formal proofs of properties
4. **Results**: Web interface visualization and reports

## Quick Start

### Prerequisites

- **Docker** (recommended) or **Python 3.9+** and **Lean 4**
- **8GB+ RAM** for large model verification
- **Modern web browser** for the interface

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/fraware/formal_verif_ml.git
cd formal_verif_ml

# Build and run with Docker
docker build -t formalverifml .
docker run -p 5000:5000 -v $(pwd)/models:/app/models formalverifml

# Access the web interface
open http://localhost:5000
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/fraware/formal_verif_ml.git
cd formal_verif_ml

# Install Python dependencies
pip install -r translator/requirements.txt

# Install Lean 4 (see https://leanprover.github.io/lean4/doc/setup.html)
# Then build the project
lake build

# Run the web interface
python webapp/app.py
```

## Usage

### 1. Export Your Model

```python
# Export a PyTorch model
python translator/export_from_pytorch.py \
    --model_path your_model.pth \
    --output_path model.json \
    --model_type transformer
```

### 2. Generate Lean Code

```python
# Convert JSON to Lean definitions
python translator/generate_lean_model.py \
    --model_json model.json \
    --output_lean lean/FormalVerifML/generated/my_model.lean
```

### 3. Verify Properties

```bash
# Build and verify with Lean
lake build
lake exe FormalVerifML
```

### 4. Web Interface

Upload your model JSON files through the web interface at `http://localhost:5000` to:

- Visualize model architecture
- Generate Lean code automatically
- Run verification proofs
- View detailed logs and results

## Documentation

### User Guides

- **[User Guide](docs/user_guide.md)**: Getting started and basic usage
- **[Developer Guide](docs/developer_guide.md)**: Architecture and extension guide

### API Reference

- **[Lean API](lean/FormalVerifML/base/)**: Core definitions and properties
- **[Python API](translator/)**: Model translation and testing tools
- **[Web API](webapp/)**: Web interface and visualization

## Testing

### Run All Tests

```bash
# Comprehensive test suite
python translator/run_comprehensive_tests.py

# Enterprise feature tests
python translator/test_enterprise_features.py

# HuggingFace model tests
python translator/test_huggingface_models.py
```

### Test Categories

- ✅ **Model Loading**: PyTorch and HuggingFace model compatibility
- ✅ **Code Generation**: JSON to Lean translation accuracy
- ✅ **Verification**: Property verification correctness
- ✅ **Performance**: Memory usage and execution time
- ✅ **Enterprise**: Multi-user, security, and audit features

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/fraware/formal_verif_ml.git
cd formal_verif_ml

# Install development dependencies
pip install -r translator/requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

### Code Standards

- **Python**: Follow PEP 8 with type hints
- **Lean**: Use Lean 4 style guide and mathlib conventions
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: 90%+ test coverage required

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Lean Community**: For the excellent theorem prover
- **HuggingFace**: For transformer model support
- **PyTorch Team**: For the deep learning framework
- **Contributors**: All who have helped improve this project

## Support

- **Issues**: [GitHub Issues](https://github.com/fraware/formal_verif_ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fraware/formal_verif_ml/discussions)
- **Documentation**: [Project Wiki](https://github.com/fraware/formal_verif_ml/wiki)

---

**Made with ❤️ by the FormalVerifML Team**
