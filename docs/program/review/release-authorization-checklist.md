# Release authorization checklist (R-08)

**Target tag:** `v0.1.0-rc1`  
**Current authorization:** **NOT AUTHORIZED**  
**Program lead:** Mateo Petel  
**Last updated:** 2026-07-25

This checklist is the only gate that may authorize creating tag `v0.1.0-rc1`, publishing packages, or describing the project as released / independently validated. Completing remediation PRs R-00–R-07 and producing review **packets** does **not** satisfy this checklist.

**Hard constraints until every row below is signed:**

- Do **not** create or push tag `v0.1.0-rc1`
- Do **not** publish packages (PyPI or otherwise)
- Do **not** mark LV-C* claims as Supported in `CLAIMS.md`
- Do **not** invent external reviewer approvals

Related: [review/README.md](README.md), [../findings-F001-F020.md](../findings-F001-F020.md), [../../release-process.md](../../release-process.md), [../../../STATUS.md](../../../STATUS.md).

---

## Gate A — Findings severity

Source of truth: `docs/program/findings-F001-F020.md`.

| Gate | Requirement | Status | Evidence / notes |
|------|-------------|--------|------------------|
| A1 | Zero **Critical** findings remaining open | Pending Mateo confirmation | Tracker lists F-001 as Addressed on remediation branches; confirm on integration tip before sign-off |
| A2 | Zero **High** findings remaining open | Pending Mateo confirmation | F-002–F-008, F-013, F-015–F-016, F-020 marked Addressed on branches — not a substitute for Mateo sign-off |
| A3 | All **Medium** findings resolved **or** explicitly accepted by Mateo with written rationale | Pending | List accepted Mediums below if any |

### Medium acceptances (Mateo only)

| Finding | Accepted? (Yes/No) | Rationale | Mateo initials / date |
|---------|-------------------|-----------|------------------------|
| F-004 | | | |
| F-009 | | | |
| F-010 | | | |
| F-011 | | | |
| F-012 | | | |
| F-014 | | | |
| F-017 | | | |
| F-018 | | | |
| F-019 | | | |
| _(other)_ | | | |

If a Medium is **resolved** rather than accepted, leave Accepted blank and point to the fixing commit in Evidence.

**Gate A signed:** No  
**Signer:** —  
**Date:** —

---

## Gate B — Independent reproductions

Three successful independent reproductions of the candidate artifact set.

| # | Operator | Environment | Commit / artifacts | Log | Pass? |
|---|----------|-------------|--------------------|-----|-------|
| 1 | TBD | TBD | TBD | TBD | No |
| 2 | TBD | TBD | TBD | TBD | No |
| 3 | TBD | TBD | TBD | TBD | No |

Packet: [drafts/release-packet-draft.md](drafts/release-packet-draft.md) / [templates/release-packet.md](templates/release-packet.md).

**Gate B signed:** No  
**Signer:** —  
**Date:** —

---

## Gate C — External formal + security approval

| Gate | Requirement | Reviewer handle | Packet | Approved? |
|------|-------------|-----------------|--------|-----------|
| C1 | External formal review approves theorem interpretation | TBD | Formal packet | **No** |
| C2 | Second Lean / formal reviewer (≠ implementer) | TBD | Formal packet | **No** |
| C3 | External security review approves input→Lean boundary | TBD | Generator packet | **No** |

At least three reviewers overall must be external to the implementation team (`docs/program/review-invitation.md`).

**Gate C signed:** No  
**Signer:** —  
**Date:** —

---

## Gate D — Artifacts after final code change

| Gate | Requirement | Status |
|------|-------------|--------|
| D1 | Final code change commit identified | TBD |
| D2 | RC artifacts regenerated **after** that commit (`package_rc_artifacts.sh`) | Not done |
| D3 | Checksums / SBOM / axiom report / evidence bundle refreshed | Not done |
| D4 | Reproductions (Gate B) re-run against the regenerated set | Not done |

**Gate D signed:** No  
**Signer:** —  
**Date:** —

---

## Gate E — Explicit RC prohibition until complete

| Statement | Affirmed? |
|-----------|-----------|
| No `v0.1.0-rc1` tag exists on the remote | Must remain true until final sign-off |
| No package publish has occurred for this RC | Must remain true |
| `STATUS.md` still says RC **not authorized** until final sign-off | Required |
| No LV-C* claim marked Supported until §14 / this checklist complete | Required |

**Gate E signed:** No (affirmation at cut time)  
**Signer:** —  
**Date:** —

---

## Final authorization (all gates A–E)

| Field | Value |
|-------|-------|
| Authorized to create `v0.1.0-rc1`? | **NO** |
| Program lead (Mateo) signature | — |
| Date | — |
| Candidate commit SHA | — |
| Notes | Authorization scaffolding only (R-08). Do not treat remediation branch landings as release approval. |

When (and only when) every gate is signed Yes/Pass, flip the authorized field to YES, update `STATUS.md`, then cut the tag using `docs/release-process.md`.
