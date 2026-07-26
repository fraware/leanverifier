# LeanVerifier Status

**Version target:** v0.1.0 technical preview  
**Status date:** 2026-07-26  
**Program lead:** Mateo Petel  
**Repository:** https://github.com/fraware/leanverifier  
**Working branch:** `recovery/post-audit-integrated` (R-08 tip `8349595` + local polish)

## Current claim posture

LeanVerifier is a **research implementation under validation**. It is **not** released, **not** independently validated, and **not** authorized for an RC tag or package publish.

The formal kernel implements exact rational affine binary classification with an L∞ margin-robustness theorem. Implementation of LV-C001–C006 exists in-tree; **assurance is incomplete** pending external review and the signed release-authorization checklist. Do **not** describe any LV-C* claim as Supported until that checklist in `docs/program/review/release-authorization-checklist.md` (and `docs/release-process.md` §14) passes after R-00–R-08.

Everything else formerly advertised under FormalVerifML (neural nets, transformers, vision, SMT, distributed/enterprise web verification, PyTorch equivalence, floating-point correctness) is **unsupported** and lives under `experimental/legacy-formalverifml/` for historical reference only.

## Lifecycle

| Stage | State |
|-------|-------|
| Truth reset (LV-00) | Complete in working tree |
| Governance templates | Complete in tree; default branch is **`main`**; remote ruleset still **blocked** (admin / CODEOWNERS) |
| Build baseline (LV-01) | Lean 4.32.1 + mathlib `v4.32.1` pinned; SHA-pinned CI; Docker elan checksummed |
| Affine kernel (LV-02) | Implemented; repeated-weight score equalities via `norm_num`; classify via kernel `decide` after rewrite (F-002; no `native_decide`) |
| L∞ theorem + axiom audit (LV-03) | Library + generated `#print axioms` path; leanchecker job (F-001/F-020 addressed in R-01) |
| Schemas (LV-04) | Implemented; R-03 hardening; packaged under `leanverifier.resources` (R-04) |
| Generator (LV-05) | Implemented; Fin-pattern vectors; path symlink hardening (R-03); `norm_num` instance proofs (F-001) |
| CLI (LV-06) | Implemented; required `--output` evidence bundle + `inspect` (R-02); packaged Lean materialize (R-04); concrete axiom audit (R-01) |
| Adversarial tests (LV-07) | Mutation table + Hypothesis L∞ invariance + security matrix (R-05) |
| Legacy isolation (LV-08) | Complete under `experimental/` |
| Assurance docs (LV-09) | Four-layer TCB rewritten (F-020 addressed in R-01) |
| Post-audit remediation (R-00–R-08) | Integrated on `recovery/post-audit-integrated` (R-08 tip); R-08 review/authorization **scaffolding** only — **RC still not authorized** |
| RC release pipeline (LV-10) | Reproducible packaging + reproduce job (R-07); **RC not authorized**; **no RC tag**; no PyPI publish |
| Independent review (LV-11) | Packets + checklist ready under `docs/program/review/`; **blocked** on external reviewers (handles TBD) |
| v0.1.0 authorization | **Not authorized** — do not tag, publish, or promote |

## Remediation branches (R-00–R-08)

Landed on dedicated branches (see `docs/program/r-08-remediation-handoff.md`). Linear tip: **R-08** (`8349595` on `review/r-08-release-authorization`), which contains R-07→R-05→R-06 lineage. Local integration branch: `recovery/post-audit-integrated`. R-01/R-02/R-04 were integrated in R-05; do not re-apply early parallels as siblings after R-07.

| PR | SHA (short) | Branch |
|----|-------------|--------|
| R-00 | `f09cc70` | `recovery/r-00-governance-ci` |
| R-01 | `c93639b` | `fix/r-01-kernel-only-generated-proofs` |
| R-02 | `cf88065` | `feature/r-02-evidence-bundle` |
| R-03 | `8e6c846` | `fix/r-03-input-and-path-hardening` |
| R-04 | `6de2a42` | `fix/r-04-distribution-integrity` |
| R-05 | `1f1af08` | `test/r-05-assurance-matrix` |
| R-06 | `2a78730` | `chore/r-06-lean-4.32.1` |
| R-07 | `3377529` | `release/r-07-reproducible-rc` |
| R-08 | `8349595` | `review/r-08-release-authorization` |

## Required reviewers (roles; handles TBD)

| Role | Handle | Packet |
|------|--------|--------|
| Lean / formal methods (≥2) | TBD | Formal |
| Application security | TBD | Generator |
| Python packaging | TBD | Generator / Release |
| Reproducible build | TBD | Release |
| Program lead / release authority | Mateo (`@fraware`) | Authorization checklist |

## Security posture

- The legacy web upload → translator → `lake build` path is **quarantined** and must not be exposed.
- Verification uses an isolated temp workspace and a deterministic generator with pre-resolve symlink rejection on generate and verify `--output` paths, plus documented resource limits (R-03).
- See [SECURITY.md](SECURITY.md).

## What “verified” means (today)

Exit code `0` / status `verified` means the generated Lean module compiled under the pinned toolchain and the axiom audit passed for both library theorems and the concrete generated declarations (`eps_nonneg`, `margin_strict`, `certified`), with allowlist `{propext, Classical.choice, Quot.sound}` only. It does **not** mean model correctness in floating point, deployment readiness, Supported status, independent validation, or any property outside the L∞ margin statement.

## Blockers (do not fabricate completion)

1. Mateo confirmation that Critical/High are closed on the integration tip; Mediums resolved or explicitly accepted (`docs/program/findings-F001-F020.md` + authorization checklist)
2. Remote branch ruleset + real multi-party CODEOWNERS reviewers (pending handles in `docs/program/admin-actions.md`)
3. External LV-11 reviewer recruitment, packet completion, and findings resolution
4. Three independent reproductions + artifacts regenerated after final code change
5. RC tag `v0.1.0-rc1`, PyPI publish, and public “released” language **only** after the signed checklist — **not before**
