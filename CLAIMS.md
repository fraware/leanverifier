# LeanVerifier Claims Register

Every public capability mentioned in active documentation must appear here.
Statuses: **Supported** | **Experimental** | **Planned** | **Rejected**.

## Supported (v0.1.0 technical preview)

| ID | Claim | Status | Evidence | Assumptions | Negative evidence |
|----|-------|--------|----------|-------------|-------------------|
| LV-C001 | Exact rational affine binary classifier (`score` / `classify`) | Supported | `LeanVerifier/Model/AffineBinary.lean`; unit theorems incl. repeated weights | Exact ℚ arithmetic; dimension `n` matches weights/input | Float arrays; silent zip truncate; bang indexing |
| LV-C002 | L∞ ball perturbation bound `\|score(y)-score(x)\| ≤ ε · ‖w‖₁` | Supported | `abs_score_sub_score_le` in `Property/ClassificationRobustness.lean` | `ε ≥ 0`; `y` in closed L∞ ball of radius `ε` | Other norms; FP rounding |
| LV-C003 | Strict margin ⇒ classification invariant on L∞ ball | Supported | `affine_margin_robustness` in `Theorem/AffineMarginRobustness.lean` | `\|score(x)\| > ε · ‖w‖₁`; decision boundary is strict `score > 0` | Equality case (not certified); other classifiers |
| LV-C004 | Public axiom surface limited to approved builtins | Supported | `Audit/PublicAxioms.lean`; `scripts/audit_axioms.py` | Lean kernel soundness | Custom axioms; `sorry`; `Lean.trustCompiler` |
| LV-C005 | Canonical JSON model/request/result schemas over rational strings | Supported | `schemas/*.schema.json`; `docs/model-format.md`; schema tests | UTF-8; canonical rational grammar | JSON floats; duplicate keys |
| LV-C006 | Deterministic safe Lean generation + CLI verify semantics | Supported | `src/leanverifier/`; CLI exit codes; integration/mutation/security tests | Isolated workspace; pinned toolchain offline after fetch | Web upload path; compiling into checkout |

## Experimental (not supported)

| ID | Claim | Status | Location |
|----|-------|--------|----------|
| LV-X001 | Neural / transformer / vision formalizations | Experimental | `experimental/legacy-formalverifml/lean/` |
| LV-X002 | SMT automation integration | Experimental | legacy `smt_integration.lean` |
| LV-X003 | Distributed / enterprise verification stubs | Experimental | legacy enterprise modules |
| LV-X004 | Flask web upload → `lake build` | Experimental / quarantined | `experimental/legacy-formalverifml/webapp/` — **do not expose** |
| LV-X005 | PyTorch / HuggingFace export translators | Experimental | `experimental/legacy-formalverifml/translator/` |

## Planned

| ID | Claim | Status |
|----|-------|--------|
| LV-P001 | Additional norms / multi-class affine margins | Planned |
| LV-P002 | Independent attestation-backed binary releases beyond RC scaffolding | Planned (LV-10/11) |

## Rejected / withdrawn

Former README claims of “production ready”, “enterprise-grade”, “encryption”, “100M+ parameters”, and “distributed verification” as product features are **withdrawn**. See `docs/history/`.
