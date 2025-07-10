# 📖 FormalVerifML User Guide

> **Complete guide to using FormalVerifML for formal verification of machine learning models**

## 📋 Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Model Export](#model-export)
- [Verification Process](#verification-process)
- [Web Interface](#web-interface)
- [Command Line Usage](#command-line-usage)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## 🎯 Introduction

FormalVerifML is a comprehensive framework for formally verifying machine learning models. This guide will walk you through the complete process from model export to verification results.

### 🎯 What You'll Learn

- How to export models from PyTorch/HuggingFace
- How to generate Lean 4 verification code
- How to run formal verification proofs
- How to interpret results and troubleshoot issues

### 🎯 Prerequisites

- **Python 3.9+** with pip
- **Docker** (recommended) or **Lean 4** installation
- **8GB+ RAM** for large model verification
- **Basic understanding** of machine learning models

## 🚀 Getting Started

### Option 1: Docker (Recommended)

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

### Option 2: Manual Installation

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

### 🧪 Verify Installation

```bash
# Run the comprehensive test suite
python translator/run_comprehensive_tests.py

# Expected output: All tests should pass
```

## 📤 Model Export

### Exporting PyTorch Models

```python
# Basic PyTorch model export
python translator/export_from_pytorch.py \
    --model_path your_model.pth \
    --output_path model.json \
    --model_type transformer
```

### Exporting HuggingFace Models

```python
# Export a HuggingFace transformer
python translator/export_from_pytorch.py \
    --model_name "bert-base-uncased" \
    --output_path bert_model.json \
    --model_type transformer
```

### Supported Model Types

| Model Type      | Description                          | Example                       |
| --------------- | ------------------------------------ | ----------------------------- |
| `transformer`   | Transformer models (BERT, GPT, etc.) | `bert-base-uncased`           |
| `neural_net`    | Feed-forward neural networks         | Custom PyTorch models         |
| `vision`        | Vision transformers (ViT, Swin)      | `google/vit-base-patch16-224` |
| `decision_tree` | Decision tree models                 | Scikit-learn trees            |

### Export Options

```bash
# Full export with all options
python translator/export_from_pytorch.py \
    --model_path model.pth \
    --output_path model.json \
    --model_type transformer \
    --include_weights true \
    --include_config true \
    --format json \
    --verbose true
```

### 📋 Export Configuration

Create a configuration file `export_config.json`:

```json
{
  "model_path": "your_model.pth",
  "output_path": "model.json",
  "model_type": "transformer",
  "include_weights": true,
  "include_config": true,
  "format": "json",
  "verbose": true,
  "max_seq_len": 512,
  "vocab_size": 30000
}
```

Then use:

```bash
python translator/export_from_pytorch.py --config export_config.json
```

## 🔍 Verification Process

### Step 1: Generate Lean Code

```bash
# Convert JSON model to Lean definitions
python translator/generate_lean_model.py \
    --model_json model.json \
    --output_lean lean/FormalVerifML/generated/my_model.lean
```

### Step 2: Run Verification

```bash
# Build the Lean project
lake build

# Run verification with specific properties
lake exe FormalVerifML --properties robustness,fairness
```

### Step 3: View Results

```bash
# Check verification results
cat verification_results.json

# View detailed logs
cat logs/verification.log
```

### 🔧 Verification Properties

| Property           | Description               | Command                         |
| ------------------ | ------------------------- | ------------------------------- |
| `robustness`       | Adversarial robustness    | `--properties robustness`       |
| `fairness`         | Fairness verification     | `--properties fairness`         |
| `interpretability` | Interpretability analysis | `--properties interpretability` |
| `safety`           | Safety properties         | `--properties safety`           |
| `all`              | All properties            | `--properties all`              |

### 📊 Verification Configuration

Create `verification_config.json`:

```json
{
  "properties": ["robustness", "fairness"],
  "epsilon": 0.1,
  "delta": 0.05,
  "timeout": 300,
  "memory_limit": "8GB",
  "smt_solver": "z3",
  "verbose": true
}
```

## 🌐 Web Interface

### Accessing the Interface

1. **Start the server**: `python webapp/app.py`
2. **Open browser**: Navigate to `http://localhost:5000`
3. **Upload models**: Use the file upload interface

### Interface Features

#### 📤 Model Upload

- **Drag & Drop**: Upload multiple JSON files
- **File Validation**: Automatic format checking
- **Progress Tracking**: Real-time upload progress

#### 🔍 Model Visualization

- **Architecture View**: Interactive model structure
- **Layer Details**: Click to see layer information
- **Export Options**: Save visualizations as PNG/SVG

#### 📊 Verification Dashboard

- **Property Status**: Visual indicators for each property
- **Progress Tracking**: Real-time verification progress
- **Results Summary**: Quick overview of verification results

#### 📋 Logs and Debugging

- **Live Logs**: Real-time log streaming
- **Error Details**: Detailed error messages
- **Debug Information**: Technical details for developers

### 🎨 Interface Customization

```python
# Customize web interface settings
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='/app/uploads',
    ALLOWED_EXTENSIONS={'json'},
    DEBUG=True
)
```

## 💻 Command Line Usage

### Basic Commands

```bash
# Export model
python translator/export_from_pytorch.py --model_path model.pth --output_path model.json

# Generate Lean code
python translator/generate_lean_model.py --model_json model.json --output_lean model.lean

# Run verification
lake build && lake exe FormalVerifML

# Run tests
python translator/run_comprehensive_tests.py
```

### Advanced Commands

```bash
# Batch processing
python translator/batch_process.py --input_dir models/ --output_dir results/

# Performance profiling
python translator/profile_verification.py --model model.json --iterations 10

# Memory analysis
python translator/memory_analysis.py --model model.json --max_memory 8GB
```

### 🔧 Command Line Options

#### Export Options

```bash
--model_path PATH          # Path to PyTorch model
--model_name NAME          # HuggingFace model name
--output_path PATH         # Output JSON path
--model_type TYPE          # Model type (transformer, neural_net, etc.)
--include_weights BOOL     # Include model weights
--include_config BOOL      # Include model configuration
--format FORMAT            # Output format (json, yaml)
--verbose BOOL             # Verbose output
```

#### Verification Options

```bash
--properties LIST          # Properties to verify
--epsilon FLOAT            # Robustness epsilon
--delta FLOAT              # Robustness delta
--timeout INT              # Verification timeout (seconds)
--memory_limit STRING      # Memory limit (e.g., "8GB")
--smt_solver STRING        # SMT solver to use
--parallel BOOL            # Enable parallel verification
--verbose BOOL             # Verbose output
```

## 🚀 Advanced Features

### 🏢 Enterprise Features

#### Multi-User Support

```bash
# Enable multi-user mode
python webapp/app.py --multi_user --auth_required

# Create user accounts
python translator/manage_users.py --create_user --username alice --role admin
```

#### Audit Logging

```bash
# Enable audit logging
python webapp/app.py --audit_logging --log_level INFO

# View audit logs
python translator/view_audit_logs.py --start_date 2024-01-01 --end_date 2024-01-31
```

#### Security Features

```bash
# Enable rate limiting
python webapp/app.py --rate_limit 100 --rate_window 60

# Enable encryption
python webapp/app.py --encryption --key_file keys/encryption.key
```

### 📈 Large-Scale Models

#### Distributed Verification

```bash
# Run distributed verification
python translator/distributed_verification.py \
    --nodes 4 \
    --model model.json \
    --properties all \
    --load_balance true
```

#### Memory Optimization

```bash
# Enable memory optimization
python translator/generate_lean_model.py \
    --model_json model.json \
    --output_lean model.lean \
    --memory_optimized \
    --chunk_size 128 \
    --sparse_attention
```

### 🎨 Vision Models

#### Vision Transformer Support

```bash
# Export vision transformer
python translator/export_from_pytorch.py \
    --model_name "google/vit-base-patch16-224" \
    --output_path vit_model.json \
    --model_type vision

# Verify vision properties
python translator/verify_vision_properties.py \
    --model vit_model.json \
    --properties attention_robustness,spatial_invariance
```

## 🔧 Troubleshooting

### Common Issues

#### Model Export Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`

```bash
# Solution: Install PyTorch
pip install torch torchvision torchaudio
```

**Problem**: `ValueError: Model type not supported`

```bash
# Solution: Check supported model types
python translator/export_from_pytorch.py --help

# Use correct model type
python translator/export_from_pytorch.py --model_type transformer
```

#### Verification Issues

**Problem**: `Lean build failed`

```bash
# Solution: Check Lean installation
lean --version

# Clean and rebuild
lake clean && lake build
```

**Problem**: `Out of memory error`

```bash
# Solution: Use memory optimization
python translator/generate_lean_model.py --memory_optimized --chunk_size 64

# Or use distributed verification
python translator/distributed_verification.py --nodes 2
```

#### Web Interface Issues

**Problem**: `Connection refused`

```bash
# Solution: Check if server is running
ps aux | grep app.py

# Restart server
python webapp/app.py --host 0.0.0.0 --port 5000
```

**Problem**: `File upload failed`

```bash
# Solution: Check file size limits
python webapp/app.py --max_file_size 32MB

# Check file format
python translator/validate_model.py --model_file model.json
```

### 🔍 Debug Mode

```bash
# Enable debug mode
python webapp/app.py --debug

# Enable verbose logging
python translator/export_from_pytorch.py --verbose --debug

# Generate debug report
python translator/debug_report.py --model model.json --output debug_report.txt
```

### 📊 Performance Monitoring

```bash
# Monitor memory usage
python translator/monitor_memory.py --model model.json --duration 300

# Profile verification time
python translator/profile_verification.py --model model.json --iterations 5

# Generate performance report
python translator/performance_report.py --output performance_report.json
```

## ❓ FAQ

### General Questions

**Q: What types of models does FormalVerifML support?**
A: FormalVerifML supports transformers, neural networks, vision models, decision trees, and linear models. See the [Model Support](#supported-model-types) section for details.

**Q: How accurate are the verification results?**
A: FormalVerifML uses mathematical proofs, so results are 100% accurate when verification succeeds. The framework provides formal guarantees.

**Q: Can I verify my own custom models?**
A: Yes! Export your model to JSON format and use the standard verification process. See [Model Export](#model-export) for details.

### Technical Questions

**Q: How much memory do I need?**
A: For small models (< 100M parameters): 4GB RAM. For large models (100M+ parameters): 8GB+ RAM. Use memory optimization for very large models.

**Q: How long does verification take?**
A: Small models: seconds to minutes. Large models: minutes to hours. Use distributed verification for faster results.

**Q: Can I run verification on GPU?**
A: Currently, verification runs on CPU. GPU support is planned for future releases.

### Enterprise Questions

**Q: How do I set up multi-user access?**
A: Enable multi-user mode and create user accounts. See [Enterprise Features](#enterprise-features) for details.

**Q: How do I enable audit logging?**
A: Use the `--audit_logging` flag when starting the web interface. See [Audit Logging](#audit-logging) for details.

**Q: Can I integrate with existing CI/CD pipelines?**
A: Yes! FormalVerifML provides command-line interfaces that integrate with any CI/CD system.

### Support

**Q: Where can I get help?**
A: Check the [Troubleshooting](#troubleshooting) section, GitHub Issues, or Documentation.

**Q: How do I report bugs?**
A: Use GitHub Issues with detailed reproduction steps and error messages.

**Q: Can I contribute to the project?**
A: Yes! See the [Contributing Guide](../CONTRIBUTING.md) for details.

---

**Need more help?** Check our [Documentation](../docs/) or [GitHub Issues](https://github.com/fraware/formal_verif_ml/issues).
