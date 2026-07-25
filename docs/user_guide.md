# User guide

## What LeanVerifier does

LeanVerifier checks one property for affine binary classifiers defined over exact rationals:

If `|score(x)| > ε · ‖w‖₁` with `ε ≥ 0`, then every input in the closed L∞ ball of radius `ε` around `x` receives the same classification as `x`, where classification is `true` iff `score > 0`.

## What it does not do

- It does not verify neural networks, transformers, or vision models.
- It does not claim floating-point or PyTorch equivalence.
- It does not expose a web upload interface (legacy path is quarantined under `experimental/legacy-formalverifml/webapp/`).
- Insufficient margin is `not-certified`, not a proof of vulnerability.

## Install

From the repository root (`https://github.com/fraware/leanverifier`):

```bash
# Lean library (once per machine / after toolchain changes)
lake update
lake build

# Python CLI
python -m pip install -e ".[dev]"
```

Requires Lean `v4.32.0` (see `lean-toolchain`) and Python 3.11+.

## Inputs

1. Model JSON conforming to `schemas/model.schema.json` (canonical rational strings).
2. Request JSON conforming to `schemas/request.schema.json` (reference input, ε, model digest).

See [model-format.md](model-format.md). Worked example: `examples/affine_binary/`.

## CLI

```bash
leanverifier validate-model examples/affine_binary/model.json
leanverifier validate-request examples/affine_binary/request.json
leanverifier generate --model examples/affine_binary/model.json \
  --request examples/affine_binary/request.json --out-dir /tmp/lv-out
leanverifier verify --model examples/affine_binary/model.json \
  --request examples/affine_binary/request.json \
  --output /tmp/lv-evidence
leanverifier inspect /tmp/lv-evidence
```

`verify` requires `--output`. It builds in an isolated temporary workspace under the pinned toolchain, runs the axiom audit, then writes an immutable evidence bundle (canonical inputs, generated Lean, logs, axiom report, result, checksums) before deleting the temp workspace.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | verified |
| 2 | not-certified (e.g. insufficient margin) |
| 3 | invalid input / schema |
| 4 | Lean build failure |
| 5 | axiom audit failure |
| 6 | digest mismatch |
| 7 | internal error |
| 8 | timeout |

## Reading results

The evidence bundle under `--output` is the authoritative record. `verification-result.json` includes claim, digests, generator version, Lean toolchain, mathlib revision, build/audit exit codes, axiom audit status, and `evidence_bundle_digest`. Use `leanverifier inspect <dir>` to recompute digests and reject tampering (`--rerun` optionally re-executes verify).
Treat `verified` as certification of the concrete margin instance only — not deployment readiness or floating-point correctness.

## Further reading

- [assurance-boundary.md](assurance-boundary.md) — in / out of scope
- [limitations.md](limitations.md) — known limits
- [threat-model.md](threat-model.md) — adversary model
- [SECURITY.md](../SECURITY.md) — supported surface and legacy quarantine
