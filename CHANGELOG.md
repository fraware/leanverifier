# Changelog

All notable changes to the supported LeanVerifier surface are recorded here.
The project follows a narrow claim boundary; see [CLAIMS.md](CLAIMS.md).

## [Unreleased] — v0.1.0 technical preview (unauthorized)

Reconstruction of LeanVerifier as an exact affine L∞ margin checker.

### Added

- Supported Lean library (`LeanVerifier/`) for affine binary classifiers over ℚ and L∞ margin robustness
- Canonical JSON schemas (`schemas/`) and deterministic Python generator/CLI (`src/leanverifier/`)
- Axiom audit gate (`scripts/audit_axioms.py`) allowing only `propext`, `Classical.choice`, `Quot.sound`
- Assurance docs: STATUS, CLAIMS, SECURITY, TCB, threat model, limitations, release process
- CI scaffolding (SHA-pinned Actions) and RC release workflow scaffolding
- Example model/request under `examples/affine_binary/`

### Changed

- Toolchain pinned to Lean / mathlib `v4.32.1` (see `docs/program/r-06-lean-4.32.1-migration.md`)
- Active documentation rewritten to the single supported claim; former FormalVerifML marketing withdrawn

### Removed / quarantined

- Legacy FormalVerifML Lean tree, translator, and webapp moved under `experimental/legacy-formalverifml/` (unsupported; web upload path quarantined)
- Overclaimed enterprise summaries and stale test reports relocated under `docs/history/`

### Security

- Supported verify path uses an isolated temporary workspace, path containment, and fail-closed digest/margin/axiom gates
- Legacy Flask upload → translator → `lake build` must not be exposed

## Pre-history

Earlier FormalVerifML packaging and claims are **not** part of this changelog’s supported product line. See [docs/history/](docs/history/).
