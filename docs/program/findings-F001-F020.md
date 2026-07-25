# Audit findings tracker (F-001 – F-020)

**Baseline commit:** `a9a030e4801c567079bef9f1cc91ca78106ead20`  
**Campaign:** Post-audit remediation R-00–R-08  
**Last updated:** 2026-07-25 (R-03)

Owner labels are program roles until real GitHub handles exist (`docs/program/admin-actions.md`). Status values: `Open` | `Partial` | `Addressed` | `Blocked`.

| ID | Severity | Title | Owner | Target PR | Status | Notes |
|----|----------|-------|-------|-----------|--------|-------|
| F-001 | Critical | Generated proof escapes the declared axiom policy | Formal Lead | R-01 | Open | `native_decide` in generator templates |
| F-002 | High | Public model theorems also use native evaluation | Formal Lead | R-01 | Open | `repeated_weight_*` theorems |
| F-003 | High | CI failed and was bypassed | Build/Release | R-00 / R-05 | Partial | R-00: add `master` push trigger; green Python gates; full job matrix in R-05 |
| F-004 | Medium | The clean-worktree check is vacuous | Build/Release | R-00 / R-05 | Partial | R-00: lake + pytest + audits + generate outside checkout; full matrix in R-05 |
| F-005 | High | The full positive verification path is not tested | Generator Lead / CLI | R-02 / R-05 | Open | Positive `verify` + inspect integration |
| F-006 | High | The CLI destroys the verification evidence | Generator Lead / CLI | R-02 | Open | Required `--output` evidence bundle |
| F-007 | High | The wheel is not a functional distribution | Generator Lead | R-04 | Open | Package schemas + Lean via `importlib.resources` |
| F-008 | High | Python dependencies are not reproducibly locked | Build/Release | R-04 | Open | Commit `uv.lock`; frozen CI install |
| F-009 | Medium | Canonical rational handling has an approximation defect | Security / Generator | R-03 | Addressed | Exact `Fraction` only; denom > 1e6 tests |
| F-010 | Medium | Schemas permit zero-dimensional models | Generator Lead | R-03 | Addressed | `dimension` / `minItems` ≥ 1; no Fin 0 path |
| F-011 | Medium | Rational strings and input files are insufficiently bounded | Security Reviewer | R-03 | Addressed | Resource limits + stable codes in `limits.py` |
| F-012 | Medium | Symlink-root rejection is ineffective | Security Reviewer | R-03 | Addressed | `lstat` before `resolve`; CLI no early resolve |
| F-013 | High | Mutation tests do not test failure behavior | Build/Release | R-05 | Open | Real verify/audit/inspect mutations |
| F-014 | Medium | Security tests cover only a small subset of the threat model | Security Reviewer | R-03 / R-05 | Partial | R-03: UTF-8, timeout, env, symlink, limits; remainder in R-05 |
| F-015 | High | The release reproduction job performs no reproduction | Build/Release | R-07 | Open | Real reproduce job; no `--skip-lean` for release evidence |
| F-016 | High | The packaged axiom report is not an axiom report | Build/Release / Formal | R-07 | Open | Full axiom report in RC artifacts |
| F-017 | Medium | The SBOM is a handwritten inventory scaffold | Build/Release | R-07 | Open | Tool-generated CycloneDX/SPDX |
| F-018 | Medium | Documentation verification does not execute documentation | Docs / Build | R-05 | Open | `docs-commands` job |
| F-019 | Medium | Claim status is premature | Program lead | R-00 | Addressed | CLAIMS Implementation/Assurance columns; no Supported until checklist |
| F-020 | High | The documented TCB omits native computation | Formal Lead | R-01 | Open | Four-layer TCB rewrite; concrete-module audit |

## Severity note

Severities follow the post-implementation audit at merge `a9a030e`. Medium findings that are still release-blocking before RC remain tracked here with High/Critical items through F-020.

## R-00 scope

This document itself, claims downgrade (F-019), CI `master` push (F-003 partial), and non-vacuous clean-worktree (F-004 partial) are the R-00 deliverables. Do not mark other findings Addressed from R-00 alone.
