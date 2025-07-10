# 🤝 Contributing to FormalVerifML

Thank you for your interest in contributing to FormalVerifML! This document provides comprehensive guidelines for contributing to the project.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.9+** with pip
- **Lean 4** with mathlib
- **Git** and GitHub account
- **Docker** (optional, for containerized development)
- **8GB+ RAM** for large model verification

### Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/formal_verif_ml.git
   cd formal_verif_ml
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/fraware/formal_verif_ml.git
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

### Environment Setup

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Build Lean project
lake build

# Run tests to verify setup
python -m pytest tests/
```

### IDE Configuration

#### VS Code

1. **Install Extensions**:

   - Python
   - Lean 4
   - GitLens
   - Pylint
   - Black Formatter

2. **Workspace Settings** (`.vscode/settings.json`):
   ```json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": true,
     "python.formatting.provider": "black",
     "python.testing.pytestEnabled": true,
     "python.testing.pytestArgs": ["tests/"],
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
       "source.organizeImports": true
     }
   }
   ```

#### PyCharm

1. **Configure Python Interpreter**: Point to `venv/bin/python`
2. **Install Plugins**: Python, Docker, Git, Lean 4
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
```

## 📏 Code Standards

### Python Standards

#### Style Guide

- **Formatter**: Black (line length: 88)
- **Linter**: Pylint (score: 9.0+)
- **Type Hints**: Required for all functions
- **Docstrings**: Google style

#### Example Code

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def process_model_data(
    model_path: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process model data with configuration.

    Args:
        model_path: Path to the model file.
        config: Optional configuration dictionary.

    Returns:
        Dictionary containing processed model data.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model data is invalid.

    Example:
        >>> config = {"normalize": True, "batch_size": 32}
        >>> result = process_model_data("model.pth", config)
        >>> print(result["status"])
        'success'
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Processing model: {model_path}")

    # Implementation here
    return {"status": "success", "data": processed_data}
```

#### Error Handling

```python
def safe_operation(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Perform operation with comprehensive error handling."""
    try:
        # Validate input
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        # Perform operation
        result = perform_operation(data)

        # Validate output
        if not result:
            raise RuntimeError("Operation returned empty result")

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return None
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
```

### Lean Standards

#### Style Guide

- **Naming**: Use camelCase for definitions, snake_case for variables
- **Documentation**: Use `/--` for documentation comments
- **Structure**: Group related definitions together

#### Example Code

```lean
/--
Process neural network layer with validation.
-/
def processLayer (layer : LayerType) (input : Array Float) : Array Float :=
  -- Validate input
  if input.size == 0 then
    #[]
  else
    -- Process layer
    match layer with
    | LayerType.linear w b => evalLinear w b input
    | LayerType.relu => evalActivation LayerType.relu input
    | _ => input
```

#### Proof Organization

```lean
theorem modelRobustness (model : NeuralNet) (ε δ : Float) :
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

#### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
# Format: <type>(<scope>): <description>

# Examples:
feat(translator): add support for vision transformers
fix(webapp): resolve memory leak in model processing
docs(readme): update installation instructions
test(verification): add property-based tests for robustness
refactor(core): simplify model evaluation logic
```

#### Branch Naming

```bash
# Feature branches
feature/vision-transformer-support
feature/memory-optimization

# Bug fix branches
bugfix/memory-leak-in-export
bugfix/lean-build-failure

# Hotfix branches
hotfix/critical-security-issue
hotfix/performance-regression
```

## 🧪 Testing Guidelines

### Test Categories

#### 1. Unit Tests

- **Coverage**: 90%+ required
- **Location**: `tests/unit/`
- **Framework**: pytest

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

#### 2. Integration Tests

- **Purpose**: Test component interactions
- **Location**: `tests/integration/`
- **Framework**: pytest with fixtures

```python
# tests/integration/test_pipeline.py
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

#### 3. Performance Tests

- **Purpose**: Ensure performance requirements
- **Location**: `tests/performance/`
- **Framework**: pytest-benchmark

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

#### 4. Property-Based Tests

- **Purpose**: Test with random inputs
- **Location**: `tests/property/`
- **Framework**: hypothesis

```python
# tests/property/test_robustness.py
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

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure tests pass**:

   ```bash
   python -m pytest tests/
   python -m pylint translator/ webapp/
   python -m mypy translator/ webapp/
   ```

2. **Update documentation**:

   - Update relevant docstrings
   - Update user guide if needed
   - Update API documentation

3. **Check code quality**:
   ```bash
   black --check translator/ webapp/
   isort --check-only translator/ webapp/
   bandit -r translator/ webapp/
   ```

### PR Guidelines

#### Title and Description

```markdown
## Title

feat(translator): add support for vision transformers

## Description

This PR adds comprehensive support for vision transformer models including:

- ViT (Vision Transformer) model export
- Swin Transformer support
- Multi-modal transformer integration
- Vision-specific verification properties

## Changes

- Add `VisionTransformer` structure to Lean definitions
- Implement vision model export in PyTorch translator
- Add vision-specific verification properties
- Update documentation with vision model examples

## Testing

- [x] Unit tests for vision transformer export
- [x] Integration tests for end-to-end pipeline
- [x] Performance tests for large vision models
- [x] Property-based tests for vision robustness

## Breaking Changes

None

## Related Issues

Closes #123
Related to #456
```

#### Review Process

1. **Self-review**: Review your own code before requesting review
2. **Request review**: Assign appropriate reviewers
3. **Address feedback**: Respond to all review comments
4. **Update PR**: Push additional commits if needed
5. **Merge**: Only merge after approval and CI passes

### Commit Guidelines

```bash
# Make atomic commits
git add translator/vision_export.py
git commit -m "feat(translator): add vision transformer export function"

git add tests/test_vision_export.py
git commit -m "test(translator): add tests for vision transformer export"

git add docs/vision_models.md
git commit -m "docs: add vision transformer documentation"
```

## 🐛 Issue Reporting

### Bug Reports

```markdown
## Bug Description

Brief description of the bug

## Steps to Reproduce

1. Load model from `example.json`
2. Run verification with robustness property
3. Observe memory error

## Expected Behavior

Verification should complete successfully

## Actual Behavior

Process crashes with "Out of memory" error

## Environment

- OS: Ubuntu 22.04
- Python: 3.9.12
- Lean: 4.0.0
- Memory: 16GB RAM

## Additional Information

- Model size: 50M parameters
- Error occurs only with large models
- Works fine with models < 10M parameters
```

### Feature Requests

```markdown
## Feature Description

Support for distributed verification across multiple nodes

## Use Case

Large models (>100M parameters) require significant computational resources

## Proposed Solution

Implement distributed verification with:

- Task sharding across nodes
- Load balancing
- Fault tolerance
- Result aggregation

## Alternatives Considered

- Cloud-based verification (rejected due to cost)
- Memory optimization only (insufficient for very large models)

## Implementation Plan

1. Design distributed architecture
2. Implement node communication
3. Add fault tolerance mechanisms
4. Create monitoring and logging
5. Update documentation
```

## 📚 Documentation

### Documentation Standards

#### Python Docstrings

```python
def verify_model_robustness(
    model: NeuralNet,
    epsilon: float,
    delta: float,
    num_samples: int = 1000
) -> RobustnessResult:
    """Verify model robustness against adversarial perturbations.

    This function performs formal verification of model robustness by checking
    that small perturbations to inputs don't cause large changes in outputs.

    Args:
        model: Neural network model to verify.
        epsilon: Maximum allowed input perturbation.
        delta: Maximum allowed output change.
        num_samples: Number of random samples to test.

    Returns:
        RobustnessResult containing verification status and statistics.

    Raises:
        ValueError: If epsilon or delta are negative.
        RuntimeError: If verification fails due to insufficient resources.

    Example:
        >>> model = load_model("model.json")
        >>> result = verify_model_robustness(model, epsilon=0.1, delta=0.05)
        >>> print(result.status)
        'verified'

    Note:
        This function uses formal verification techniques and may take
        significant time for large models.
    """
```

#### Lean Documentation

```lean
/--
Verify model robustness against adversarial perturbations.

This theorem establishes that a neural network model is robust
against input perturbations of magnitude at most ε, ensuring
output changes are bounded by δ.

## Mathematical Definition

For a model f: ℝⁿ → ℝᵐ, we say f is (ε, δ)-robust if:
∀x, x' ∈ ℝⁿ, ||x - x'||₂ ≤ ε → ||f(x) - f(x')||₂ ≤ δ

## Parameters

- **model**: Neural network to verify
- **epsilon**: Maximum input perturbation
- **delta**: Maximum output change

## Returns

Proof that the model satisfies the robustness property

## Usage

This theorem can be used to establish formal guarantees
about model behavior under adversarial attacks.
-/
theorem modelRobustness (model : NeuralNet) (ε δ : Float) :
  robust model ε δ := by
  -- Proof implementation
  sorry
```

### Documentation Updates

When making changes:

1. **Update docstrings** for modified functions
2. **Update user guide** for new features
3. **Update API documentation** for new endpoints
4. **Update examples** to reflect changes
5. **Update README** if needed

## 👥 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** and considerate of others
- **Be collaborative** and open to feedback
- **Be constructive** in criticism and suggestions
- **Be inclusive** and welcoming to new contributors

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions
- **Discord**: For real-time chat (link in README)

### Recognition

Contributors are recognized through:

- **GitHub Contributors** page
- **Release notes** for significant contributions
- **Documentation** acknowledgments
- **Community highlights** in blog posts

## 🎯 Getting Help

### Resources

- **[User Guide](docs/user_guide.md)**: Getting started and usage
- **[Developer Guide](docs/developer_guide.md)**: Architecture and development
- **[API Reference](docs/api_reference.md)**: Detailed API documentation
- **[Examples](examples/)**: Code examples and tutorials

### Support Channels

- **GitHub Issues**: [Create an issue](https://github.com/fraware/formal_verif_ml/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/fraware/formal_verif_ml/discussions)
- **Discord**: Join our community server
- **Email**: Contact the maintainers directly

### Mentorship

New contributors can:

- **Ask for help** in GitHub Discussions
- **Request mentorship** from experienced contributors
- **Start with good first issues** labeled as such
- **Join community calls** for guidance

---

**Thank you for contributing to FormalVerifML!** 🚀

Your contributions help make formal verification of machine learning models more accessible and reliable for everyone.
