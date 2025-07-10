# FormalVerifML Developer Guide

> **Comprehensive guide for developers extending and contributing to the FormalVerifML framework**

## Table of Contents

- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
- [Code Organization](#code-organization)
- [Development Setup](#development-setup)
- [Extending the Framework](#extending-the-framework)
- [Testing Guidelines](#testing-guidelines)
- [Code Standards](#code-standards)
- [Performance Optimization](#performance-optimization)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Introduction

This guide is designed for developers who want to extend, contribute to, or understand the internal architecture of the FormalVerifML framework. It provides detailed information about the codebase structure, development practices, and extension points.

### Target Audience

- **ML Engineers**: Adding new model types and verification properties
- **Formal Verification Experts**: Extending proof capabilities and tactics
- **DevOps Engineers**: Setting up CI/CD and deployment pipelines
- **Researchers**: Implementing novel verification techniques

### Prerequisites

- **Python 3.9+** with development tools
- **Lean 4** with mathlib
- **Git** and version control experience
- **Docker** for containerized development
- **Understanding** of formal verification concepts

## Architecture Overview

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Export  │    │  Code Generation │    │   Verification  │
│                 │    │                 │    │                 │
│ • PyTorch       │───▶│ • JSON → Lean   │───▶│ • Lean 4        │
│ • HuggingFace   │    │ • Type checking │    │ • SMT Solvers   │
│ • Custom        │    │ • Validation    │    │ • Proof tactics │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Test Suites   │    │   Results &     │
│                 │    │                 │    │   Reports       │
│ • Upload        │    │ • Unit Tests    │    │ • Visualization │
│ • Visualization │    │ • Integration   │    │ • Logs          │
│ • Monitoring    │    │ • Performance   │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. **Model Translation Layer**

- **Purpose**: Convert ML models to formal representations
- **Components**: PyTorch export, JSON schema, Lean code generation
- **Key Files**: `translator/export_from_pytorch.py`, `translator/generate_lean_model.py`

#### 2. **Formal Verification Engine**

- **Purpose**: Execute formal proofs of model properties
- **Components**: Lean 4 definitions, SMT integration, proof tactics
- **Key Files**: `lean/FormalVerifML/base/`, `lean/FormalVerifML/proofs/`

#### 3. **Web Interface**

- **Purpose**: User-friendly interaction with the framework
- **Components**: Flask app, visualization, monitoring
- **Key Files**: `webapp/app.py`, `webapp/templates/`

#### 4. **Testing Infrastructure**

- **Purpose**: Ensure correctness and performance
- **Components**: Unit tests, integration tests, performance benchmarks
- **Key Files**: `translator/test_*.py`, `tests/`

## Code Organization

### Project Structure

```
FormalVerifML/
├── 📁 lean/                          # Lean 4 formal verification code
│   └── 📁 FormalVerifML/
│       ├── 📁 base/                  # Core definitions and properties
│       │   ├── 📄 definitions.lean           # Basic ML model structures
│       │   ├── 📄 advanced_models.lean       # Transformer and advanced models
│       │   ├── 📄 ml_properties.lean         # Verification property definitions
│       │   ├── 📄 advanced_tactics.lean      # Custom proof tactics
│       │   ├── 📄 symbolic_models.lean       # Symbolic reasoning support
│       │   ├── 📄 memory_optimized_models.lean # Memory optimization
│       │   ├── 📄 smt_integration.lean       # SMT solver integration
│       │   ├── 📄 large_scale_models.lean    # 100M+ parameter support
│       │   ├── 📄 vision_models.lean         # Vision transformer support
│       │   ├── 📄 distributed_verification.lean # Distributed processing
│       │   └── 📄 enterprise_features.lean   # Enterprise features
│       ├── 📁 generated/             # Auto-generated model definitions
│       │   ├── 📄 example_model.lean         # Sample neural network
│       │   ├── 📄 another_nn_model.lean      # Additional neural network
│       │   ├── 📄 log_reg_model.lean         # Logistic regression
│       │   ├── 📄 decision_tree_model.lean   # Decision tree
│       │   └── 📄 sample_transformer_model.lean # Transformer example
│       ├── 📁 proofs/                # Verification proof scripts
│       │   ├── 📄 example_robustness_proof.lean
│       │   ├── 📄 example_fairness_proof.lean
│       │   ├── 📄 extended_robustness_proof.lean
│       │   ├── 📄 extended_fairness_proof.lean
│       │   ├── 📄 decision_tree_proof.lean
│       │   └── 📄 comprehensive_test_suite.lean
│       └── 📄 formal_verif_ml.lean   # Main entry point
├── 📁 translator/                    # Model translation and testing
│   ├── 📄 export_from_pytorch.py    # PyTorch model export
│   ├── 📄 generate_lean_model.py    # JSON to Lean code generation
│   ├── 📄 run_comprehensive_tests.py # Comprehensive test runner
│   ├── 📄 test_huggingface_models.py # HuggingFace model testing
│   ├── 📄 test_enterprise_features.py # Enterprise feature testing
│   ├── 📄 requirements.txt          # Python dependencies
│   └── 📄 *.json                    # Sample model definitions
├── 📁 webapp/                       # Web interface and visualization
│   ├── 📄 app.py                    # Flask application
│   ├── 📁 templates/                # HTML templates
│   │   ├── 📄 index.html            # Main interface
│   │   └── 📄 model_visualization.html # Model visualization
│   └── 📁 static/                   # Static assets
├── 📁 docs/                         # Documentation
│   ├── 📄 user_guide.md             # User documentation
│   ├── 📄 developer_guide.md        # This file
│   └── 📄 improvement_roadmap.md    # Development roadmap
├── 📁 .github/                      # CI/CD and workflows
│   └── 📁 workflows/
│       └── 📄 lean_ci.yml           # GitHub Actions CI
├── 📄 lakefile.lean                 # Lean build configuration
├── 📄 lake-manifest.json            # Lean dependencies
├── 📄 lean-toolchain                # Lean version specification
├── 📄 Dockerfile                    # Docker container definition
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # Project overview
```

### Key Design Principles

#### 1. **Modularity**

- Each component has a single responsibility
- Clear interfaces between modules
- Minimal coupling between components

#### 2. **Extensibility**

- Plugin architecture for new model types
- Configurable verification properties
- Customizable proof tactics

#### 3. **Performance**

- Memory optimization for large models
- Parallel processing where possible
- Efficient data structures

#### 4. **Reliability**

- Comprehensive testing
- Formal verification of core components
- Error handling and recovery

## 🛠️ Development Setup

### Prerequisites Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-dev python3.9-venv
sudo apt-get install -y build-essential git curl

# Install Lean 4
curl -sL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source ~/.bashrc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/fraware/formal_verif_ml.git
cd formal_verif_ml

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r translator/requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Build Lean project
lake build

# Run tests
python -m pytest tests/
```

### IDE Configuration

#### VS Code Setup

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "lean4.serverEnv": {
    "LEAN_SRC_PATH": "${workspaceFolder}/lean"
  }
}
```

#### PyCharm Setup

1. **Configure Python Interpreter**: Point to `venv/bin/python`
2. **Install Plugins**: Python, Docker, Git
3. **Configure Testing**: Use pytest framework
4. **Setup Code Style**: Use Black formatter

### Docker Development

```bash
# Build development image
docker build -f Dockerfile.dev -t formalverifml-dev .

# Run development container
docker run -it --rm \
    -v $(pwd):/app \
    -p 5000:5000 \
    -p 8000:8000 \
    formalverifml-dev

# Inside container
cd /app
lake build
python -m pytest tests/
```

## 🔧 Extending the Framework

### Adding New Model Types

#### 1. **Define Model Structure**

```lean
-- lean/FormalVerifML/base/definitions.lean

/--
New model type structure.
--/
structure NewModelType where
  -- Model parameters
  inputDim : Nat
  outputDim : Nat
  hiddenDim : Nat

  -- Model weights
  weights : Array (Array Float)
  bias : Array Float

  -- Configuration
  config : ModelConfig

  deriving Inhabited
```

#### 2. **Implement Evaluation Function**

```lean
/--
Evaluate the new model type.
--/
def evalNewModelType (model : NewModelType) (input : Array Float) : Array Float :=
  -- Implementation here
  let linear := evalLinear model.weights model.bias input
  -- Apply any additional transformations
  linear
```

#### 3. **Add to Translator**

```python
# translator/generate_lean_model.py

def generate_new_model_code(model_json) -> str:
    """Generate Lean code for new model type."""
    name = model_json.get("name", "newModel")
    input_dim = model_json["input_dim"]
    output_dim = model_json["output_dim"]

    lean_lines = []
    lean_lines.append("import FormalVerifML.base.definitions")
    lean_lines.append("namespace FormalVerifML")
    lean_lines.append(f"-- Auto-generated NewModelType definition for {name}\n")

    # Generate model definition
    lean_lines.append(f"def {name} : NewModelType :=")
    lean_lines.append(f"  {{ inputDim := {input_dim},")
    lean_lines.append(f"    outputDim := {output_dim},")
    # Add other fields...
    lean_lines.append("  }")
    lean_lines.append("end FormalVerifML")

    return "\n".join(lean_lines)
```

#### 4. **Add Export Support**

```python
# translator/export_from_pytorch.py

def extract_new_model(model: nn.Module) -> Dict[str, Any]:
    """Extract new model type from PyTorch model."""
    return {
        "model_type": "new_model",
        "name": "exportedNewModel",
        "input_dim": get_input_dim(model),
        "output_dim": get_output_dim(model),
        "weights": extract_weights(model),
        "bias": extract_bias(model),
        # Add other fields...
    }
```

### Adding New Verification Properties

#### 1. **Define Property**

```lean
-- lean/FormalVerifML/base/ml_properties.lean

/--
New verification property.
--/
def newProperty (model : ModelType) (ε δ : Float) : Prop :=
  ∀ (x x' : Array Float),
  (∀ i, distL2 x[i]! x'[i]! < ε) →
  ∀ i, |(evalModel model x)[i]! - (evalModel model x')[i]!| < δ
```

#### 2. **Implement Verification Logic**

```lean
-- lean/FormalVerifML/proofs/new_property_proof.lean

/--
Proof that a specific model satisfies the new property.
--/
theorem modelSatisfiesNewProperty (model : ModelType) (ε δ : Float) :
  newProperty model ε δ := by
  -- Proof implementation here
  sorry  -- Placeholder
```

#### 3. **Add to Test Suite**

```python
# translator/test_new_property.py

def test_new_property():
    """Test the new verification property."""
    # Load test model
    model = load_test_model()

    # Run verification
    result = verify_property(model, "new_property", epsilon=0.1, delta=0.05)

    # Assert results
    assert result.status == "verified"
    assert result.confidence > 0.95
```

### Adding New Proof Tactics

#### 1. **Define Custom Tactic**

```lean
-- lean/FormalVerifML/base/advanced_tactics.lean

/--
Custom tactic for specific proof patterns.
--/
macro "custom_tactic" : tactic => `(tactic|
  -- Tactic implementation
  apply some_theorem
  simp
  assumption
)
```

#### 2. **Use in Proofs**

```lean
theorem example_proof : some_property := by
  custom_tactic
  -- Additional proof steps
  done
```

## 🧪 Testing Guidelines

### Testing Strategy

#### 1. **Unit Tests**

- **Coverage**: 90%+ code coverage required
- **Framework**: pytest
- **Location**: `tests/unit/`

```python
# tests/unit/test_model_export.py

import pytest
from translator.export_from_pytorch import extract_linear_layer

def test_extract_linear_layer():
    """Test linear layer extraction."""
    # Setup
    layer = create_mock_linear_layer()

    # Execute
    result = extract_linear_layer(layer)

    # Assert
    assert "weight" in result
    assert "bias" in result
    assert len(result["weight"]) > 0
```

#### 2. **Integration Tests**

- **Purpose**: Test component interactions
- **Framework**: pytest with fixtures
- **Location**: `tests/integration/`

```python
# tests/integration/test_end_to_end.py

def test_model_export_to_verification():
    """Test complete pipeline from export to verification."""
    # Export model
    model_json = export_pytorch_model("test_model.pth")

    # Generate Lean code
    lean_code = generate_lean_code(model_json)

    # Run verification
    result = run_verification(lean_code)

    # Assert results
    assert result.success
    assert len(result.properties) > 0
```

#### 3. **Performance Tests**

- **Purpose**: Ensure performance requirements
- **Framework**: pytest-benchmark
- **Location**: `tests/performance/`

```python
# tests/performance/test_large_models.py

def test_large_model_verification(benchmark):
    """Benchmark large model verification."""
    model = create_large_model(1000000)  # 1M parameters

    def verify_model():
        return run_verification(model)

    result = benchmark(verify_model)

    # Assert performance requirements
    assert result.stats.mean < 60.0  # < 60 seconds
    assert result.stats.max < 120.0  # < 2 minutes
```

#### 4. **Property-Based Tests**

- **Purpose**: Test with random inputs
- **Framework**: hypothesis
- **Location**: `tests/property/`

```python
# tests/property/test_model_properties.py

from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=1, max_size=100))
def test_model_robustness(input_data):
    """Test model robustness with random inputs."""
    model = create_test_model()

    # Add small perturbation
    perturbed = [x + 0.01 for x in input_data]

    # Get predictions
    pred1 = model.predict(input_data)
    pred2 = model.predict(perturbed)

    # Assert robustness
    assert abs(pred1 - pred2) < 0.1
```

### Test Organization

```
tests/
├── 📁 unit/                    # Unit tests
│   ├── 📁 translator/          # Translator unit tests
│   ├── 📁 webapp/              # Web interface unit tests
│   └── 📁 lean/                # Lean code unit tests
├── 📁 integration/             # Integration tests
│   ├── 📄 test_pipeline.py     # End-to-end pipeline
│   ├── 📄 test_api.py          # API integration
│   └── 📄 test_web_interface.py # Web interface integration
├── 📁 performance/             # Performance tests
│   ├── 📄 test_memory.py       # Memory usage tests
│   ├── 📄 test_speed.py        # Speed tests
│   └── 📄 test_scalability.py  # Scalability tests
├── 📁 property/                # Property-based tests
│   ├── 📄 test_robustness.py   # Robustness properties
│   ├── 📄 test_fairness.py     # Fairness properties
│   └── 📄 test_safety.py       # Safety properties
├── 📁 fixtures/                # Test fixtures
│   ├── 📄 models.py            # Model fixtures
│   ├── 📄 data.py              # Data fixtures
│   └── 📄 configs.py           # Configuration fixtures
└── 📄 conftest.py              # pytest configuration
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/

# Run with coverage
python -m pytest tests/ --cov=translator --cov=webapp --cov-report=html

# Run performance tests
python -m pytest tests/performance/ --benchmark-only

# Run property tests
python -m pytest tests/property/ --hypothesis-profile=ci
```

## Code Standards

### Python Standards

#### 1. **Style Guide**

- **Formatter**: Black (line length: 88)
- **Linter**: pylint (score: 9.0+)
- **Type Hints**: Required for all functions
- **Docstrings**: Google style

```python
def extract_model_weights(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract weights from a PyTorch model.

    Args:
        model: PyTorch model to extract weights from.

    Returns:
        Dictionary mapping layer names to weight arrays.

    Raises:
        ValueError: If model has no parameters.
    """
    if not list(model.parameters()):
        raise ValueError("Model has no parameters")

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy()

    return weights
```

#### 2. **Error Handling**

- **Exceptions**: Use specific exception types
- **Logging**: Use structured logging
- **Validation**: Validate inputs early

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_model_export(model_path: str, output_path: str) -> Optional[Dict[str, Any]]:
    """Safely export a model with error handling."""
    try:
        # Validate inputs
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = torch.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Export model
        result = export_model(model, output_path)
        logger.info(f"Exported model to {output_path}")

        return result

    except Exception as e:
        logger.error(f"Model export failed: {str(e)}")
        return None
```

### Lean Standards

#### 1. **Style Guide**

- **Naming**: Use camelCase for definitions, snake_case for variables
- **Documentation**: Use `/--` for documentation comments
- **Structure**: Group related definitions together

```lean
/--
Extract weights from a linear layer.
-/
def extractLinearWeights (layer : LinearLayer) : Array (Array Float) :=
  layer.weights

/--
Extract bias from a linear layer.
-/
def extractLinearBias (layer : LinearLayer) : Array Float :=
  layer.bias
```

#### 2. **Proof Organization**

- **Structure**: Clear proof structure with comments
- **Tactics**: Use appropriate tactics for the proof
- **Documentation**: Document complex proof steps

```lean
theorem modelRobustness (model : ModelType) (ε δ : Float) :
  robust model ε δ := by
  -- Apply robustness definition
  unfold robust

  -- Use triangle inequality
  apply triangleInequality

  -- Apply model properties
  apply modelProperties

  -- Complete proof
  done
```

### Git Standards

#### 1. **Commit Messages**

- **Format**: Conventional Commits
- **Scope**: Component being changed
- **Description**: Clear, concise description

```bash
feat(translator): add support for vision transformers

- Add ViT model export functionality
- Support for patch embeddings and positional encoding
- Add vision-specific verification properties

Closes #123
```

#### 2. **Branch Naming**

- **Feature**: `feature/vision-transformer-support`
- **Bugfix**: `bugfix/memory-leak-in-export`
- **Hotfix**: `hotfix/critical-security-issue`

#### 3. **Pull Request Process**

1. **Create branch** from main
2. **Implement changes** with tests
3. **Update documentation**
4. **Run all tests**
5. **Create PR** with description
6. **Code review** required
7. **Merge** after approval

## Performance Optimization

### Memory Optimization

#### 1. **Large Model Handling**

```python
def optimize_memory_usage(model_size: int) -> Dict[str, Any]:
    """Optimize memory usage for large models."""
    if model_size > 100_000_000:  # 100M parameters
        return {
            "use_chunking": True,
            "chunk_size": 64,
            "use_sparse_attention": True,
            "use_gradient_checkpointing": True
        }
    return {"use_chunking": False}
```

#### 2. **Streaming Processing**

```python
def stream_model_verification(model_path: str, chunk_size: int = 1024):
    """Stream model verification to reduce memory usage."""
    with open(model_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield process_chunk(chunk)
```

### Speed Optimization

#### 1. **Parallel Processing**

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def parallel_verification(models: List[str], max_workers: int = None):
    """Run verification in parallel."""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(verify_model, models))

    return results
```

#### 2. **Caching**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_model_verification(model_hash: str, properties: str):
    """Cache verification results."""
    return verify_model_by_hash(model_hash, properties)
```

### Profiling and Monitoring

#### 1. **Performance Profiling**

```python
import cProfile
import pstats

def profile_verification(model_path: str):
    """Profile verification performance."""
    profiler = cProfile.Profile()
    profiler.enable()

    # Run verification
    result = verify_model(model_path)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

    return result
```

#### 2. **Memory Monitoring**

```python
import psutil
import tracemalloc

def monitor_memory_usage():
    """Monitor memory usage during verification."""
    tracemalloc.start()

    # Run verification
    result = verify_model("large_model.json")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
        "result": result
    }
```

## Deployment

### Docker Deployment

#### 1. **Production Dockerfile**

```dockerfile
# Dockerfile.prod
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Lean 4
RUN curl -sL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY translator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build Lean project
RUN lake build

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "webapp/app.py", "--host", "0.0.0.0", "--port", "5000"]
```

#### 2. **Docker Compose**

```yaml
# docker-compose.yml
version: "3.8"

services:
  formalverifml:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - MAX_WORKERS=4
    restart: unless-stopped
```

### Kubernetes Deployment

#### 1. **Deployment Configuration**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: formalverifml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: formalverifml
  template:
    metadata:
      labels:
        app: formalverifml
    spec:
      containers:
        - name: formalverifml
          image: formalverifml:latest
          ports:
            - containerPort: 5000
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
            limits:
              memory: "8Gi"
              cpu: "4"
          env:
            - name: FLASK_ENV
              value: "production"
```

#### 2. **Service Configuration**

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: formalverifml-service
spec:
  selector:
    app: formalverifml
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

### CI/CD Pipeline

#### 1. **GitHub Actions**

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r translator/requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: |
          docker build -t formalverifml:${{ github.sha }} .
      - name: Deploy to production
        run: |
          # Deployment commands
```

## Contributing

### Contribution Process

#### 1. **Fork and Clone**

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/formal_verif_ml.git
cd formal_verif_ml

# Add upstream remote
git remote add upstream https://github.com/fraware/formal_verif_ml.git
```

#### 2. **Create Feature Branch**

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Add tests
# Update documentation
```

#### 3. **Commit and Push**

```bash
# Add changes
git add .

# Commit with conventional commit message
git commit -m "feat: add new verification property"

# Push to your fork
git push origin feature/your-feature-name
```

#### 4. **Create Pull Request**

- Go to GitHub and create a pull request
- Fill out the PR template
- Request review from maintainers

### Contribution Guidelines

#### 1. **Code Quality**

- Follow style guides (Black, pylint)
- Add comprehensive tests
- Update documentation
- Use type hints

#### 2. **Testing Requirements**

- Unit tests for new functionality
- Integration tests for complex features
- Performance tests for optimizations
- Property-based tests for verification

#### 3. **Documentation Requirements**

- Update user guide for new features
- Add API documentation
- Include usage examples
- Update architecture diagrams

#### 4. **Review Process**

- All PRs require review
- Address review comments
- Ensure CI/CD passes
- Update branch if needed

### Development Tools

#### 1. **Pre-commit Hooks**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/pylint
    rev: v2.15.0
    hooks:
      - id: pylint
        args: [--rcfile=.pylintrc]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

#### 2. **Development Scripts**

```bash
#!/bin/bash
# scripts/dev-setup.sh

# Setup development environment
python3.9 -m venv venv
source venv/bin/activate
pip install -r translator/requirements.txt
pip install -r requirements-dev.txt
pre-commit install
lake build
```

### Getting Help

#### 1. **Documentation**

- [User Guide](user_guide.md)
- [API Reference](api_reference.md)
- [Architecture Guide](architecture.md)

#### 2. **Community**

- [GitHub Issues](https://github.com/fraware/formal_verif_ml/issues)
- [GitHub Discussions](https://github.com/fraware/formal_verif_ml/discussions)
- [Discord Server](https://discord.gg/formalverifml)

---

**Ready to contribute?** Start with a [good first issue](https://github.com/fraware/formal_verif_ml/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or check our [contributing guidelines](CONTRIBUTING.md).
