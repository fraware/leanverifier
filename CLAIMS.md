# LeanVerifier Claims Register

Every public capability mentioned in active documentation must appear here.

**Status vocabulary**

- **Implementation state:** `Implemented` | `Scaffolded` | `Experimental` | `Planned` | `Rejected`
- **Assurance state:** evidence quality only — `Internal proof audit passed` | `Validation incomplete` | `Axiom audit failed` | `Evidence incomplete` | `Installability failed` | `Unvalidated` | `Not applicable`

Reserve the word **Supported** for claims that pass the signed release-authorization checklist in `docs/program/review/release-authorization-checklist.md` (and `docs/release-process.md` §14). No LV-C* claim is Supported today: R-00–R-07 landed on remediation branches; R-08 supplies review packets only — **RC is not authorized** and independent reviewers are TBD. Public posture: research implementation under validation (see `STATUS.md`).

## Core claims (v0.1.0 technical preview — not Supported)

| ID | Claim | Implementation state | Assurance state | Evidence | Assumptions | Negative evidence |
|----|-------|----------------------|-----------------|----------|-------------|-------------------|
| LV-C001 | Exact rational affine binary classifier (`score` / `classify`) | Implemented | Validation incomplete | `LeanVerifier/Model/AffineBinary.lean`; unit theorems incl. repeated weights (`norm_num`) | Exact ℚ arithmetic; dimension `n` matches weights/input | Float arrays; silent zip truncate; bang indexing |
| LV-C002 | L∞ ball perturbation bound `\|score(y)-score(x)\| ≤ ε · ‖w‖₁` | Implemented | Internal proof audit passed | `abs_score_sub_score_le` in `Property/ClassificationRobustness.lean` | `ε ≥ 0`; `y` in closed L∞ ball of radius `ε` | Other norms; FP rounding |
| LV-C003 | Strict margin ⇒ classification invariant on L∞ ball (incl. generated certificate) | Implemented | Validation incomplete | `affine_margin_robustness` in `Theorem/AffineMarginRobustness.lean`; generator emits `norm_num` instance proofs; Hypothesis L∞ grid | `\|score(x)\| > ε · ‖w‖₁`; decision boundary is strict `score > 0` | Equality case (not certified); other classifiers |
| LV-C004 | Public axiom surface limited to approved builtins | Implemented | Validation incomplete | `Audit/PublicAxioms.lean`; `scripts/audit_axioms.py` (library + generated); prohibited-tactic scan; mutation table | Lean kernel soundness; allowlist `{propext, Classical.choice, Quot.sound}` | Custom axioms; `sorry` |
| LV-C005 | Canonical JSON model/request/result schemas over rational strings | Implemented | Validation incomplete | Packaged `leanverifier.resources/schemas`; checkout `schemas/` mirror; `docs/model-format.md`; schema tests | UTF-8; canonical rational grammar | JSON floats; duplicate keys |
| LV-C006 | Deterministic safe Lean generation + CLI verify semantics | Implemented | Validation incomplete | `src/leanverifier/`; evidence bundle + inspect; integration/mutation/security/docs-commands | Isolated workspace; pinned toolchain offline after fetch; concrete axiom audit before `verified` | Web upload path; public RC attestation blocked until signed R-08 checklist |

## Distribution and release (not Supported)

| ID | Claim | Implementation state | Assurance state | Notes |
|----|-------|----------------------|-----------------|-------|
| LV-D001 | Wheel / sdist installable verification | Implemented | Validation incomplete | Resources shipped; CI `wheel-smoke` with `--require-verify`; Docker formal build |
| LV-D002 | Reproducible release reproduction + attestations | Implemented | Validation incomplete | Full axiom report (no `--skip-lean`); pinned `cyclonedx-bom` SBOM; `reproduce_release.sh` end-to-end; `release-rc.yml` attest+reproduce. **RC tag / publish blocked** until `docs/program/review/release-authorization-checklist.md` is signed (R-08 scaffolding only; no fake approvals) |

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
