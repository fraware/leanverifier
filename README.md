# LeanVerifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LeanVerifier** is a research technical preview for *exact* formal checks of affine binary classifiers over the rationals.

**v0.1.0 target — not authorized for release.** Read [STATUS.md](STATUS.md), [CLAIMS.md](CLAIMS.md), and [SECURITY.md](SECURITY.md) before use.

## Formal claim under validation (not Supported)

Given a dimensionally valid affine binary classifier over exact rationals ℚ, a reference input \(x\), and a nonnegative L∞ radius \(\varepsilon\), if the absolute score margin strictly exceeds \(\varepsilon \cdot \|w\|_1\), then the classification sign (strictly positive score) is invariant on the closed L∞ ball of radius \(\varepsilon\) around \(x\).

Implementation exists; assurance is incomplete — see [CLAIMS.md](CLAIMS.md) and [STATUS.md](STATUS.md). This is **not** a general ML verification product. Neural nets, transformers, vision models, SMT automation, distributed or enterprise features, and the legacy web upload path are unsupported historical material under [`experimental/legacy-formalverifml/`](experimental/legacy-formalverifml/).

## Quick start

### Prerequisites

- Lean 4 matching [`lean-toolchain`](lean-toolchain) (`leanprover/lean4:v4.32.1`) via [elan](https://github.com/leanprover/elan)
- Python 3.11+ for the generator and CLI
- Docker (optional) for a Lean-only container build

### Lean library

```bash
lake update
lake build
python scripts/audit_axioms.py
```

### Python package / CLI

```bash
uv sync --frozen --all-extras
uv run leanverifier validate-model examples/affine_binary/model.json
uv run leanverifier verify \
  --model examples/affine_binary/model.json \
  --request examples/affine_binary/request.json \
  --output /tmp/lv-evidence
uv run leanverifier inspect /tmp/lv-evidence
```

| Exit | Meaning |
|------|---------|
| 0 | `verified` |
| 2 | `not-certified` (e.g. insufficient margin) |
| 3 | invalid input / schema |
| 4 | Lean build failure |
| 5 | axiom audit failure |
| 6 | digest mismatch |
| 7 | internal error |
| 8 | timeout |

## Repository layout

```
LeanVerifier/          # Supported Lean library (checkout; mirrored into the wheel)
src/leanverifier/      # Supported Python package + CLI
src/leanverifier/resources/  # Packaged schemas, Lake project, axiom-audit script
schemas/               # Checkout mirror of packaged JSON schemas
examples/affine_binary/
tests/                 # unit, schema, integration, mutation, security
scripts/               # axiom audit, docs verification, wheel smoke, release helpers
uv.lock                # Frozen Python deps (runtime, dev, build)
docs/                  # guides + assurance case
experimental/          # Legacy FormalVerifML — not in supported build
```

Supported Lake targets must not import `experimental/`.

## Documentation

| Doc | Purpose |
|-----|---------|
| [STATUS.md](STATUS.md) | Program status and blockers |
| [CLAIMS.md](CLAIMS.md) | Implementation / assurance state for public claims |
| [SECURITY.md](SECURITY.md) | Supported surface and legacy web quarantine |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |
| [docs/user_guide.md](docs/user_guide.md) | Operator guide |
| [docs/developer_guide.md](docs/developer_guide.md) | Build, layout, contribution mechanics |
| [docs/model-format.md](docs/model-format.md) | Canonical rational JSON |
| [docs/assurance-boundary.md](docs/assurance-boundary.md) | In / out of scope |
| [docs/trusted-computing-base.md](docs/trusted-computing-base.md) | TCB inventory |
| [docs/threat-model.md](docs/threat-model.md) | Threats and controls |
| [docs/limitations.md](docs/limitations.md) | Known limits |
| [docs/release-process.md](docs/release-process.md) | RC and v0.1.0 authorization |
| [docs/history/README.md](docs/history/README.md) | Withdrawn FormalVerifML materials |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). On formal-kernel, schema, generator, workflow, and release PRs, the implementer must not be the sole designated reviewer.

## Citation

See [CITATION.cff](CITATION.cff). Prefer citing the specific commit or tag you evaluated.

## License

MIT — see [LICENSE](LICENSE).
