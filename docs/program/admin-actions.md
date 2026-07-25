# Admin actions required (cannot complete from local recovery agent)

## GitHub repository settings (Mateo / org admin)

1. Rename default branch `master` → `main` on https://github.com/fraware/leanverifier
2. Update local remotes: `git fetch origin` and set upstream tracking to `origin/main`
3. Confirm `origin/HEAD` → `refs/remotes/origin/main`
4. After rename is confirmed and push CI is green on `main`, remove `master` from `on.push.branches` in `.github/workflows/ci.yml` (R-00 keeps both until then)
5. Activate a branch ruleset on `main` **only after** real multi-party CODEOWNERS handles exist:
   - Required PR reviews (implementer ≠ sole formal reviewer)
   - Require CODEOWNERS review
   - Disallow force-push
   - Require status checks from `.github/workflows/ci.yml` once green on `main`
6. Do **not** enable a bypass-resistant ruleset while every path still lists only `@fraware`

## CODEOWNERS pending handles (do not invent users)

Path-scoped roles are structured in `.github/CODEOWNERS`. Until Mateo supplies GitHub usernames, every role lists `@fraware` only.

| Role | Paths (summary) | Handle status |
|------|-----------------|---------------|
| Program lead / claim approver / release authority | `CLAIMS.md`, `STATUS.md`, release docs | `@fraware` (Mateo) |
| Formal Lead | `/LeanVerifier/`, lake/toolchain pins | **Pending** — placeholder `@fraware` |
| Formal Reviewer (≠ implementer) | same formal paths for second review | **Pending** — no second handle |
| Generator Lead | `/schemas/`, `/src/leanverifier/`, examples | **Pending** — placeholder `@fraware` |
| Security Reviewer | `SECURITY.md`, axiom audit, experimental, security/mutation tests | **Pending** — placeholder `@fraware` |
| Build/Release | `.github/workflows/`, release scripts, Dockerfile | **Pending** — placeholder `@fraware` |
| Docs | `/docs/`, README | **Pending** — placeholder `@fraware` |

When handles are supplied, replace the placeholders in `.github/CODEOWNERS` and only then turn on required CODEOWNERS review in the branch ruleset.

## External (LV-11)

Recruit reviewers using `docs/program/review-invitation.md`. v0.1.0 cannot be authorized without the completed §14 checklist in `docs/release-process.md`.

## Release attestations

`.github/workflows/release-rc.yml` is scaffolding only. **Do not cut an RC tag** until remediation R-08 authorization. Live GitHub Actions provenance runs only after that gate.
