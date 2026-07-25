# LeanVerifier Claims Register

Every public capability mentioned in active documentation must appear here.

**Status vocabulary**

- **Implementation state:** `Implemented` | `Scaffolded` | `Experimental` | `Planned` | `Rejected`
- **Assurance state:** evidence quality only — `Internal proof audit passed` | `Validation incomplete` | `Axiom audit failed` | `Evidence incomplete` | `Installability failed` | `Unvalidated` | `Not applicable`

Reserve the word **Supported** for claims that pass the release checklist in `docs/release-process.md`. No LV-C* claim is Supported until remediation R-00–R-08 and independent review complete. Public posture: research implementation under validation (see `STATUS.md`).

## Core claims (v0.1.0 technical preview — not Supported)

| ID | Claim | Implementation state | Assurance state | Evidence | Assumptions | Negative evidence |
|----|-------|----------------------|-----------------|----------|-------------|-------------------|
| LV-C001 | Exact rational affine binary classifier (`score` / `classify`) | Implemented | Validation incomplete | `LeanVerifier/Model/AffineBinary.lean`; unit theorems incl. repeated weights (`norm_num`) | Exact ℚ arithmetic; dimension `n` matches weights/input | Float arrays; silent zip truncate; bang indexing |
| LV-C002 | L∞ ball perturbation bound `\|score(y)-score(x)\| ≤ ε · ‖w‖₁` | Implemented | Internal proof audit passed | `abs_score_sub_score_le` in `Property/ClassificationRobustness.lean` | `ε ≥ 0`; `y` in closed L∞ ball of radius `ε` | Other norms; FP rounding |
| LV-C003 | Strict margin ⇒ classification invariant on L∞ ball (incl. generated certificate) | Implemented | Validation incomplete | `affine_margin_robustness` in `Theorem/AffineMarginRobustness.lean`; generator emits `norm_num` instance proofs | `\|score(x)\| > ε · ‖w‖₁`; decision boundary is strict `score > 0` | Equality case (not certified); other classifiers; full positive verify evidence pending R-02/R-05 |
| LV-C004 | Public axiom surface limited to approved builtins | Implemented | Validation incomplete | `Audit/PublicAxioms.lean`; `scripts/audit_axioms.py` (library + generated); prohibited-tactic scan | Lean kernel soundness; allowlist `{propext, Classical.choice, Quot.sound}` | Custom axioms; `sorry`; remaining assurance-matrix coverage in R-05 |
| LV-C005 | Canonical JSON model/request/result schemas over rational strings | Implemented | Validation incomplete | Packaged `leanverifier.resources/schemas`; checkout `schemas/` mirror; `docs/model-format.md`; schema tests | UTF-8; canonical rational grammar | JSON floats; duplicate keys; assurance matrix incomplete (R-05) |
| LV-C006 | Deterministic safe Lean generation + CLI verify semantics | Implemented | Evidence incomplete | `src/leanverifier/`; CLI exit codes; integration/mutation/security tests | Isolated workspace; pinned toolchain offline after fetch; concrete axiom audit before `verified` | Web upload path; CLI discards evidence bundle (F-006); positive verify path under-tested (F-005) |

## Distribution and release (not Supported)

| ID | Claim | Implementation state | Assurance state | Notes |
|----|-------|----------------------|-----------------|-------|
| LV-D001 | Wheel / sdist installable verification | Implemented | Validation incomplete | Resources shipped; `scripts/smoke_wheel.py` + CI `wheel-smoke`; full Lake verify still needs elan in CI (R-05) |
| LV-D002 | Reproducible release reproduction + attestations | Scaffolded | Unvalidated | `reproduce_release.sh` / `package_rc_artifacts.sh` scaffolding; frozen `uv.lock` in R-04; no live RC (F-015–F-017) |

## Experimental (not supported)

| ID | Claim | Implementation state | Assurance state | Location |
|----|-------|----------------------|-----------------|----------|
| LV-X001 | Neural / transformer / vision formalizations | Experimental | Not applicable | `experimental/legacy-formalverifml/lean/` |
| LV-X002 | SMT automation integration | Experimental | Not applicable | legacy `smt_integration.lean` |
| LV-X003 | Distributed / enterprise verification stubs | Experimental | Not applicable | legacy enterprise modules |
| LV-X004 | Flask web upload → `lake build` | Experimental | Not applicable | `experimental/legacy-formalverifml/webapp/` — **do not expose** |
| LV-X005 | PyTorch / HuggingFace export translators | Experimental | Not applicable | `experimental/legacy-formalverifml/translator/` |

## Planned

| ID | Claim | Implementation state | Assurance state |
|----|-------|----------------------|-----------------|
| LV-P001 | Additional norms / multi-class affine margins | Planned | Unvalidated |
| LV-P002 | Independent attestation-backed binary releases beyond RC scaffolding | Planned (LV-10/11) | Unvalidated |

## Rejected / withdrawn

Former README claims of “production ready”, “enterprise-grade”, “encryption”, “100M+ parameters”, and “distributed verification” as product features are **withdrawn**. See `docs/history/`.
