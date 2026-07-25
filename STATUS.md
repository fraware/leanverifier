# LeanVerifier Status

**Version target:** v0.1.0 technical preview  
**Status date:** 2026-07-25  
**Program lead:** Mateo Petel  
**Repository:** https://github.com/fraware/leanverifier  
**Working branch:** `fix/r-03-input-and-path-hardening` (post-audit remediation R-03; based on R-00)

## Current claim posture

LeanVerifier is a **research implementation under validation**. It is **not** released, **not** independently validated, and **not** authorized for an RC tag or package publish.

The formal kernel implements exact rational affine binary classification with an L∞ margin-robustness theorem. Implementation of LV-C001–C006 exists in-tree; **assurance is incomplete or failed** for several of those claims (see `CLAIMS.md` Implementation / Assurance columns). Do not describe any LV-C* claim as Supported until the release checklist in `docs/release-process.md` passes after remediation R-00–R-08.

Everything else formerly advertised under FormalVerifML (neural nets, transformers, vision, SMT, distributed/enterprise web verification, PyTorch equivalence, floating-point correctness) is **unsupported** and lives under `experimental/legacy-formalverifml/` for historical reference only.

## Lifecycle

| Stage | State |
|-------|-------|
| Truth reset (LV-00) | Complete in working tree |
| Governance templates | Complete in tree; GitHub `master`→`main` rename + remote ruleset **blocked** (admin) |
| Build baseline (LV-01) | Lean 4.32.0 + mathlib `v4.32.0` pinned; SHA-pinned CI; Docker elan checksummed |
| Affine kernel (LV-02) | Implemented; assurance incomplete (F-002) |
| L∞ theorem + axiom audit (LV-03) | Library `#print axioms` path present; generated/native axiom gaps open (F-001, F-020) |
| Schemas (LV-04) | Implemented; R-03 hardening (dim≥1, resource limits) addressed |
| Generator (LV-05) | Implemented; Fin-pattern vectors; path symlink hardening (R-03); `native_decide` remains (F-001) |
| CLI (LV-06) | Implemented; timeout/process-group + env strip (R-03); evidence bundle incomplete (F-006) |
| Adversarial tests (LV-07) | Partial; R-03 security cases added; mutation/matrix remainder (F-013, F-014) |
| Legacy isolation (LV-08) | Complete under `experimental/` |
| Assurance docs (LV-09) | Present; TCB rewrite pending (F-020) |
| Post-audit remediation (R-00–R-08) | R-00 done locally; R-03 in this branch; R-01/R-02/R-04+ parallel or pending |
| RC release pipeline (LV-10) | Scaffolding only; **no RC tag**; live attestations blocked |
| Independent review (LV-11) | **Blocked** on external reviewers |
| v0.1.0 authorization | **Not authorized** — do not tag, publish, or promote |

## Security posture

- The legacy web upload → translator → `lake build` path is **quarantined** and must not be exposed.
- Supported verification uses an isolated temp workspace and a deterministic generator with pre-resolve symlink rejection and documented resource limits (R-03).
- See [SECURITY.md](SECURITY.md).

## What “verified” means (today)

Exit code `0` / status `verified` means only that the generated Lean module compiled under the pinned toolchain and the current axiom audit script passed for the concrete margin instance. That path still has known assurance gaps (native axioms, discarded evidence, incomplete positive tests). It does **not** mean model correctness in floating point, deployment readiness, Supported status, or any property outside the L∞ margin statement.

## Blockers (do not fabricate completion)

1. Close Critical/High audit findings F-001–F-020 via R-00–R-08 (tracker: `docs/program/findings-F001-F020.md`)
2. GitHub default branch rename `master` → `main` (admin)
3. Remote branch ruleset + real multi-party CODEOWNERS reviewers (pending handles in `docs/program/admin-actions.md`)
4. External LV-11 reviewer recruitment and findings resolution
5. Live GitHub Actions provenance/SBOM attestations only after R-08 authorization — **not before**
