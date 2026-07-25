# Trusted computing base (TCB)

LeanVerifier assurance is layered. Removing a layer does not expand what a
`verified` result means; each layer has a distinct trust boundary.

Pinned toolchain: Lean `v4.32.0` (`lean-toolchain`); mathlib `v4.32.0`
(`lakefile.toml` / `lake-manifest.json`). Supported proofs use only
kernel-checkable tactics (no native evaluation decide tactics,
`decide +native`, or bitvector decide).

## 1. Logical soundness TCB

What must be sound for theorem *derivations* to be trusted:

| Component | Role |
|-----------|------|
| Lean 4 kernel | Type / proof checking |
| Logical axioms | Only `propext`, `Classical.choice`, `Quot.sound` on supported decls |
| Imported theorem dependencies | Supported `LeanVerifier` modules + pinned mathlib |

Secondary checking: `lean4checker --fresh` on supported modules (CI artifact).

Concrete generated modules (`eps_nonneg`, `margin_strict`, `certified`) and
CLAIMS-referenced public theorems (including `repeated_weight_*`) are audited
with `#print axioms` via `scripts/audit_axioms.py`. Any axiom outside the
allowlist fails closed.

## 2. Statement-correctness TCB

What must be correct for the *meaning* of a theorem to match the claim:

| Component | Role |
|-----------|------|
| Model / property definitions | `AffineBinaryClassifier`, `score`, `classify`, L∞ ball |
| Theorem statements | e.g. `affine_margin_robustness` |
| Reviewer judgment | Human review that statements match CLAIMS / STATUS |

Kernel success does not by itself prove the English claim is the intended one.

## 3. Manifest-correspondence TCB

What must be correct so JSON inputs match the Lean instance that was checked:

| Component | Role |
|-----------|------|
| JSON Schema + parser | `schemas/`, `leanverifier.manifest` |
| Canonicalizer | Rational grammar and digests |
| Generator | Deterministic Lean emission (`leanverifier.generator`) |
| Digest implementation | Model / request content addressing |

A sound proof of the wrong instance is still a correspondence failure.

## 4. Execution and provenance TCB

What must be trustworthy for builds and release evidence:

| Component | Role |
|-----------|------|
| Host OS | Process isolation, filesystem |
| Elan / Lean toolchain install | Bootstrap of the pinned compiler |
| Lake | Build orchestration |
| GitHub Actions (SHA-pinned) | CI execution; default `contents: read` |
| Packaging / attestation tools | Wheel/sdist and (future) RC provenance |

Docker pins base image digest and elan tarball sha256 where used.

## Explicitly out of TCB

Legacy FormalVerifML, webapp, translators, third-party ML frameworks, unpinned
Actions tags, and any experimental tree imports.

## Relation to `verified`

Status `verified` requires margin check, Lean compile of the concrete generated
module, and a **concrete** axiom audit of that module’s declarations (not only
library theorems). See [threat-model.md](threat-model.md).
