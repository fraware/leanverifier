# Trusted computing base (TCB)

| Component | Role | Notes |
|-----------|------|-------|
| Lean 4 kernel + elaborator | Proof checking | Pinned in `lean-toolchain` (`v4.32.0`) |
| mathlib (pinned revision) | Libraries used by supported theorems | `lakefile.toml` / `lake-manifest.json` (`v4.32.0`) |
| `LeanVerifier` supported modules | Formal model + theorem | No `experimental` imports |
| `scripts/audit_axioms.py` | Axiom gate | Fail closed |
| `leanverifier` Python package | Schema validation + generation + CLI orchestration | No eval/exec/shell |
| Host OS + elan installer | Toolchain bootstrap | Docker pins base image digest and elan tarball sha256 |
| GitHub Actions runners + SHA-pinned Actions | CI | Default `contents: read` |

Not in TCB: legacy FormalVerifML, webapp, translators, third-party ML frameworks, or unpinned Actions/tags.
