# Independent review packets (R-08 / LV-11)

This directory holds the **review and release-authorization infrastructure** for LeanVerifier.
It does **not** constitute authorization of `v0.1.0-rc1`.

**RC status:** not authorized. Do not create tag `v0.1.0-rc1`, publish packages, or mark claims Supported.

## Contents

| Path | Purpose |
|------|---------|
| [templates/](templates/) | Blank packet templates (formal / generator / release) |
| [drafts/](drafts/) | Filled-in drafts pointing at the current tree (pointers + analysis; no fake sign-offs) |
| [release-authorization-checklist.md](release-authorization-checklist.md) | Explicit gates before any RC tag |

## Required reviewer roles

Handles are **TBD** until Mateo supplies them (`docs/program/admin-actions.md`). Roles only:

| Role | Count | Packet | Handle |
|------|-------|--------|--------|
| Lean / formal methods | ≥2 | Formal | TBD |
| Application security | ≥1 | Generator (input→Lean boundary) | TBD |
| Python packaging | ≥1 | Generator + Release | TBD |
| Reproducible build | ≥1 | Release | TBD |

At least three reviewers must be external to the implementation team.
See also `docs/program/review-invitation.md`.

## How to use

1. Assign one bounded packet per reviewer (do not ask anyone to review the whole repo).
2. Reviewer completes the packet checklist and files findings under `docs/program/findings/`.
3. Mateo tracks Medium acceptances in writing on the authorization checklist.
4. After any final code change, regenerate RC artifacts and re-run reproductions.
5. Only when the authorization checklist is fully signed may anyone cut `v0.1.0-rc1`.

## Related

- [release-authorization-checklist.md](release-authorization-checklist.md)
- [../findings-F001-F020.md](../findings-F001-F020.md)
- [../../release-process.md](../../release-process.md)
- [../../../STATUS.md](../../../STATUS.md)
- [../../../CLAIMS.md](../../../CLAIMS.md)
