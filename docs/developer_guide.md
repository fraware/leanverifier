# Developer guide

## Toolchain

| Pin | Location |
|-----|----------|
| Python ≥ 3.11 | `pyproject.toml` + committed `uv.lock` (`uv sync --frozen --all-extras`) |
| Lean 4 stable `v4.32.1` | `lean-toolchain` (checkout) and `src/leanverifier/resources/lean-project/lean-toolchain` (wheel) |
| mathlib `v4.32.1` | `lakefile.toml` + committed `lake-manifest.json` |

Never use Lean RC toolchains on supported branches. Run Lake commands from the repository root for development; installed wheels materialize the packaged Lake project into an isolated cache.

## Layout

| Path | Role |
|------|------|
| `LeanVerifier/` | Supported Lean sources (checkout; keep in sync with packaged copy) |
| `LeanVerifier.lean` | Root import (must not import `experimental`) |
| `src/leanverifier/` | Generator + CLI (src layout; avoids Windows case collision with `LeanVerifier/`) |
| `src/leanverifier/resources/` | Packaged schemas, Lean project, axiom-audit script (`importlib.resources`) |
| `schemas/` | Checkout mirror of packaged JSON schemas (must match resource hashes) |
| `examples/affine_binary/` | Canonical worked example |
| `tests/` | unit / schema / integration / mutation / security |
| `experimental/` | Legacy FormalVerifML — excluded from supported build |
| `scripts/` | audits, docs verification, wheel smoke, release scaffolding |
| `docs/` | User/developer guides + assurance case |

## Local checks

```bash
lake build
lake lint
uv sync --frozen --all-extras
uv run python scripts/audit_axioms.py
uv run python -m pytest
uv run ruff check src/leanverifier tests scripts
uv run mypy src/leanverifier
uv run python scripts/verify_docs.py
uv build && uv run python scripts/smoke_wheel.py --dist dist
git status --porcelain   # clean after build when .lake / caches are gitignored
```

After editing checkout `schemas/` or Lean sources that ship in the wheel, refresh packaged copies and digests:

```bash
uv run python scripts/regen_resource_hashes.py
```

Digests are LF-canonical (see `.gitattributes`). Always regenerate on the machine that edited the files; do not hand-edit `hashes.json`.

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
