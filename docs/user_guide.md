# User Guide

This guide is designed for users who want to run and interact with the Formal Verification framework.

## Overview

The system provides a web interface and command-line tools to:

- Upload or export your machine learning model in JSON format.
- Automatically generate corresponding Lean definitions.
- Run formal proofs that verify properties such as robustness, fairness, interpretability, and more.
- Visualize model architectures and review proof logs.

## Prerequisites

- **Docker**: Ensure Docker is installed on your machine.
- **Web Browser**: For accessing the interactive web interface.
- **(Optional) Python 3.9+**: To run translator scripts locally.

## Running the System

### Using Docker

1. **Build the Docker Image:**

   ```bash
   docker build -t formal-ml .
   ```

2. **Run the Container:**

   ```bash
   docker run -p 5000:5000 formal-ml

   ```

3. **Access the Web Interface:**

Open your browser and navigate to http://localhost:5000. You will see:

- A form to upload one or more model JSON files.
- Links to visualize model architectures.
- A page to view proof logs.

### Uploading and Verifying a Model

1. **Upload Model JSON Files:**

- On the homepage, click **"Choose Files"** and select your JSON files (e.g., `another_nn.json`, `log_reg.json`, `decision_tree.json`).
- Click **Submit**.

2. **Translation and Proof Verification:**

- The system calls the Python translator to generate Lean code for each uploaded model.
- The Lean project is then built using Lean 4’s Lake build system.
- A JSON response will show whether the model was imported and proofs compiled successfully.

  3.**Viewing Results:**

- Use the **"Visualize Model Architecture"** link to see a graphical representation (currently a placeholder) of your model.
- Use the **"View Proof Logs"** link to see detailed logs in case any proofs fail.

### Command-Line Usage

You can also run the translator manually:

```bash
python translator/generate_lean_model.py --model_json path/to/model.json --output_lean lean/generated/model_name.lean
```

After generating the Lean file, compile the Lean project using:

```bash
lake build
```

## FAQs

- **What formats are supported?**

Currently, JSON is supported for representing models exported from ML frameworks.

- **How do I add new model types?**

Refer to the Developer Guide for instructions on extending the JSON schema and translator.

- **Where can I find more documentation?**
  See this guide and the Developer Guide for advanced usage and contribution details.
