# R-08 remediation campaign handoff (for Mateo)

**Date:** 2026-07-26  
**Branch:** `review/r-08-release-authorization` @ `8349595`  
**Local integration:** `recovery/post-audit-integrated` (points at R-08 tip + polish commits)  
**Purpose:** Summarize remediation branches/commits and how to merge; record that RC remains **not authorized**.

## Verdict

Post-audit remediation R-00–R-08 is integrated locally on `recovery/post-audit-integrated`. R-08 adds **review packet templates, drafts, and an explicit authorization checklist**. External reviewers are still TBD. **Do not** create `v0.1.0-rc1`, publish packages, or mark claims Supported.

## Remediation branch map

| PR | Short SHA | Branch | Notes |
|----|-----------|--------|-------|
| R-00 | `f09cc70` | `recovery/r-00-governance-ci` | Governance / CI / claims columns / findings tracker |
| R-01 | `c93639b` | `fix/r-01-kernel-only-generated-proofs` | Kernel-only proofs; concrete axiom audit; TCB |
| R-02 | `cf88065` | `feature/r-02-evidence-bundle` | Required `--output` evidence + inspect |
| R-03 | `8e6c846` | `fix/r-03-input-and-path-hardening` | Exact Fraction; limits; symlink hardening |
| R-04 | `6de2a42` | `fix/r-04-distribution-integrity` | Packaged resources; `uv.lock`; wheel smoke |
| R-05 | `1f1af08` | `test/r-05-assurance-matrix` | Integrates R-01 + R-02 + R-04; mutations; CI matrix |
| R-06 | `2a78730` | `chore/r-06-lean-4.32.1` | Lean/mathlib 4.32.1 pin |
| R-07 | `3377529` | `release/r-07-reproducible-rc` | Real reproduce; full axiom report; CycloneDX SBOM |
| R-08 | `8349595` | `review/r-08-release-authorization` | Review packets + authorization checklist only |

## Integration tip (how to merge / apply)

**Use R-08 as the linear integration tip** for the mainline working tree:

```text
R-00 → (parallel R-01, R-02, R-03→R-04) → R-05 (merge of R-01/R-02/R-04) → R-06 → R-07 → R-08
```

- Early parallel branches (R-01, R-02, R-03, R-04) were already merged into **R-05**; do not re-merge them as siblings onto master after R-07.
- **R-08** contains the full R-05 + R-06 + R-07 lineage.
- Local apply: `git checkout -B recovery/post-audit-integrated 8349595` (then polish commits on that branch).

Worktree cleanup: `/delete-worktree` when finished (see Cursor worktree commands).

## Required reviewers (roles; handles TBD)

| Role | Handle | Packet |
|------|--------|--------|
| Lean / formal methods (≥2) | TBD | `docs/program/review/` formal packet |
| Application security | TBD | generator packet |
| Python packaging | TBD | generator / release |
| Reproducible build | TBD | release packet |
| Program lead / release authority | Mateo (`@fraware`) | authorization checklist |

Do not invent GitHub usernames. Update `.github/CODEOWNERS` and `docs/program/admin-actions.md` when handles exist.

## Authorization state

See `docs/program/review/release-authorization-checklist.md`:

- Critical/High zero: pending Mateo confirmation against tracker on the integration tip
- Mediums: resolved on branches or pending explicit Mateo acceptance
- Three independent reproductions: **not run / not signed**
- External formal + security approval: **not signed**
- Artifacts regenerated after final code change: **not done**
- **NO `v0.1.0-rc1` until checklist signed**

## R-08 deliverables

- Templates: `docs/program/review/templates/{formal,generator,release}-packet.md`
- Drafts: `docs/program/review/drafts/*-packet-draft.md`
- Checklist: `docs/program/review/release-authorization-checklist.md`
- Index: `docs/program/review/README.md`
- Status / claims / findings / release-process cross-links updated
