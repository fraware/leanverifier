# Audit findings tracker (F-001 – F-020)

**Baseline commit:** `a9a030e4801c567079bef9f1cc91ca78106ead20`  
**Campaign:** Post-audit remediation R-00–R-08  
**Last updated:** 2026-07-26 (integrated on `recovery/post-audit-integrated` @ R-08 tip `8349595` + local polish)

Owner labels are program roles until real GitHub handles exist (`docs/program/admin-actions.md`). Status values: `Open` | `Partial` | `Addressed` | `Blocked`.

**RC authorization:** Findings marked Addressed on remediation branches are **not** a release approval. `v0.1.0-rc1` remains blocked until `docs/program/review/release-authorization-checklist.md` is signed (zero Critical/High confirmed, Mediums resolved or Mateo-accepted, three independent reproductions, external formal + security approval, artifacts regenerated after final code change).

| ID | Severity | Title | Owner | Target PR | Status | Notes |
|----|----------|-------|-------|-----------|--------|-------|
| F-001 | Critical | Generated proof escapes the declared axiom policy | Formal Lead | R-01 | Addressed | Generator uses `norm_num`; concrete generated axiom audit before `verified` |
| F-002 | High | Public model theorems also use native evaluation | Formal Lead | R-01 | Addressed | Score equalities via `norm_num`; classify via kernel `decide` after rewrite; inventoried in PublicAxioms + audit |
| F-003 | High | CI failed and was bypassed | Build/Release | R-00 / R-05 | Addressed | R-00 triggers + R-05 full job matrix (markers, lean4checker, docker, docs-*) |
| F-004 | Medium | The clean-worktree check is vacuous | Build/Release | R-00 / R-05 | Addressed | Non-vacuous clean-worktree + assurance matrix jobs |
| F-005 | High | The full positive verification path is not tested | Generator Lead / CLI | R-02 / R-05 | Addressed | Positive verify+inspect integration; CI `verify-positive` job |
| F-006 | High | The CLI destroys the verification evidence | Generator Lead / CLI | R-02 | Addressed | Required `--output` evidence bundle; `inspect` validates checksums |
| F-007 | High | The wheel is not a functional distribution | Generator Lead | R-04 | Addressed | Schemas + Lean + audit script via `importlib.resources`; hash-checked materialize (hashes regenerated after R-06 Lean pin) |
| F-008 | High | Python dependencies are not reproducibly locked | Build/Release | R-04 | Addressed | Committed `uv.lock`; CI/release use `uv sync --frozen --all-extras` |
| F-009 | Medium | Canonical rational handling has an approximation defect | Security / Generator | R-03 | Addressed | Exact `Fraction` only; denom > 1e6 tests |
| F-010 | Medium | Schemas permit zero-dimensional models | Generator Lead | R-03 | Addressed | `dimension` / `minItems` ≥ 1; no Fin 0 path |
| F-011 | Medium | Rational strings and input files are insufficiently bounded | Security Reviewer | R-03 | Addressed | Resource limits + stable codes in `limits.py` |
| F-012 | Medium | Symlink-root rejection is ineffective | Security Reviewer | R-03 | Addressed | `lstat` before `resolve` on generate and verify `--output`; CLI no early resolve |
| F-013 | High | Mutation tests do not test failure behavior | Build/Release | R-05 | Addressed | Real verify/audit/inspect mutation table |
| F-014 | Medium | Security tests cover only a small subset of the threat model | Security Reviewer | R-03 / R-05 | Addressed | R-03 cases + CI `security` / `security-windows`; Hypothesis L∞ invariance |
| F-015 | High | The release reproduction job performs no reproduction | Build/Release | R-07 | Addressed | `reproduce_release.sh` + `release-rc.yml` reproduce job (locks → lake → checker → wheel → verify → byte-compare) |
| F-016 | High | The packaged axiom report is not an axiom report | Build/Release / Formal | R-07 | Addressed | `package_rc_artifacts.sh` runs full `audit_axioms.py` (library + generated; no `--skip-lean`) |
| F-017 | Medium | The SBOM is a handwritten inventory scaffold | Build/Release | R-07 | Addressed | Pinned `cyclonedx-bom==7.3.0` via `scripts/generate_sbom.py` + Lean/mathlib enrichment |
| F-018 | Medium | Documentation verification does not execute documentation | Docs / Build | R-05 | Addressed | Split `docs-claims` / `docs-links` / `docs-commands` |
| F-019 | Medium | Claim status is premature | Program lead | R-00 | Addressed | CLAIMS Implementation/Assurance columns; no Supported until checklist |
| F-020 | High | The documented TCB omits native computation | Formal Lead | R-01 | Addressed | Four-layer TCB; no native decide; concrete-module audit before `verified` |

## Severity note

Severities follow the post-implementation audit at merge `a9a030e`. Medium findings that are still release-blocking before RC remain tracked here with High/Critical items through F-020.

## Campaign branch status (R-00–R-08)

Remediation work landed on branches summarized in `docs/program/r-08-remediation-handoff.md`.
Integration tip: **R-08** `8349595` (`review/r-08-release-authorization`), which contains the R-07→R-05→R-06 lineage (R-01/R-02/R-04 already merged into R-05). Local checkout: `recovery/post-audit-integrated`. R-08 adds review packets and the authorization checklist only — **no RC tag**.

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

## R-00 / R-01 scope

R-00: this tracker, claims downgrade (F-019), CI `master` push (F-003 partial), and
non-vacuous clean-worktree (F-004 partial).

R-01: F-001, F-002, F-020 (kernel-only proofs, concrete axiom audit, leanchecker,
four-layer TCB). Do not mark other findings Addressed from R-00/R-01 alone.
