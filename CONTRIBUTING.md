# Contributing to LeanVerifier

LeanVerifier is a narrowly scoped formal verification technical preview.
Contributions must preserve the supported claim boundary in [CLAIMS.md](CLAIMS.md) and [STATUS.md](STATUS.md).

## Scope rules

- Supported work: exact affine binary classifiers over ℚ, L∞ margin robustness, schemas, safe generator/CLI, tests, assurance docs, CI.
- Do not expand the supported surface to neural nets, transformers, vision, SMT, web upload, or enterprise features without a new evidence program and claim register update.
- No custom axioms, `sorry`, `admit`, or `Lean.trustCompiler` in supported theorems.
- Supported Lake build must not import `experimental/`.
- Do not re-enable the legacy web upload → `lake build` path.

## Setup

```bash
# Lean (from repository root)
lake update
lake build
python scripts/audit_axioms.py

# Python
python -m pip install -e ".[dev]"
python -m pytest
ruff check src/leanverifier tests scripts
mypy src/leanverifier
python scripts/verify_docs.py
```

Clone URL: `https://github.com/fraware/leanverifier.git`.

## Pull requests

Use the repository PR template. Required sections: Objective, Design, Threats, Tests, Docs, Claim impact, Contribution record.
Use “None” with an explanation when a section does not apply.

Authorship: add `Co-authored-by` only for material contributors.

### Reviewer separation

For formal-kernel, schema, generator, workflow, and release PRs, the implementer must not be the sole designated reviewer.

## Security

Read [SECURITY.md](SECURITY.md). Report generator path escapes, injection, or CI privilege issues privately via GitHub Security Advisories.

## Program process

Templates live under `docs/program/`. Release authorization follows [docs/release-process.md](docs/release-process.md).
