# Developer guide

## Toolchain

| Pin | Location |
|-----|----------|
| Lean 4 stable `v4.32.0` | `lean-toolchain` |
| mathlib `v4.32.0` | `lakefile.toml` + committed `lake-manifest.json` |
| Python ≥ 3.11 | `pyproject.toml` (minimal locked deps; install with `pip install -e ".[dev]"`) |

Never use Lean RC toolchains on supported branches. Run all Lean commands from the repository root.

## Layout

| Path | Role |
|------|------|
| `LeanVerifier/` | Supported Lean sources |
| `LeanVerifier.lean` | Root import (must not import `experimental`) |
| `src/leanverifier/` | Generator + CLI (src layout; avoids Windows case collision with `LeanVerifier/`) |
| `schemas/` | JSON Schema (draft 2020-12) |
| `examples/affine_binary/` | Canonical worked example |
| `tests/` | unit / schema / integration / mutation / security |
| `experimental/` | Legacy FormalVerifML — excluded from supported build |
| `scripts/` | audits, docs verification, release scaffolding |
| `docs/` | User/developer guides + assurance case |

## Local checks

```bash
lake build
lake lint
python scripts/audit_axioms.py
python -m pip install -e ".[dev]"
python -m pytest
ruff check src/leanverifier tests scripts
mypy src/leanverifier
python scripts/verify_docs.py
git status --porcelain   # clean after build when .lake / caches are gitignored
```

## Adding formal content

1. Keep definitions exact over ℚ; no `Array Float`, bang indexing, `getD`, or silent zip truncate.
2. Prove unit theorems with `norm_num` where helpful; include repeated-weight examples.
3. Extend `Audit/PublicAxioms.lean` and ensure `scripts/audit_axioms.py` still passes.
4. Update `CLAIMS.md` if the public claim surface changes.

## Generator constraints

User-controlled values may appear only as rationals, lengths, digests, and comments.
Generated modules import only supported `LeanVerifier` modules.
Module names must match `GeneratedModel_[0-9a-f]{16}`.

## CI

`.github/workflows/ci.yml` is SHA-pinned with `permissions: contents: read` by default.
Do not use `pull_request_target` or `workflow_run` for untrusted tests.

RC release scaffolding lives in `.github/workflows/release-rc.yml` (see [release-process.md](release-process.md)).

## Experimental tree

`experimental/legacy-formalverifml/` retains FormalVerifML sources with a WARNING header.
It is never a Lake dependency of `LeanVerifier` and is not installed by the Python package.
