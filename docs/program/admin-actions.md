# Admin actions required (cannot complete from local recovery agent)

## GitHub repository settings (Mateo / org admin)

1. Rename default branch `master` → `main` on https://github.com/fraware/leanverifier
2. Update local remotes: `git fetch origin` and set upstream tracking to `origin/main`
3. Confirm `origin/HEAD` → `refs/remotes/origin/main`
4. Activate a branch ruleset on `main`:
   - Required PR reviews (implementer ≠ sole formal reviewer)
   - Require CODEOWNERS review
   - Disallow force-push
   - Require status checks from `.github/workflows/ci.yml` once green on `main`
5. Replace CODEOWNERS placeholders with real GitHub usernames for Formal Lead, Formal Reviewer, Generator Lead, Security Reviewer, Build/Release, and Docs (implementer must not be the only reviewer on formal/schema/generator/workflow/release paths)

## External (LV-11)

Recruit reviewers using `docs/program/review-invitation.md`. v0.1.0 cannot be authorized without the completed §14 checklist in `docs/release-process.md`.

## Release attestations

`.github/workflows/release-rc.yml` is scaffolding only until an RC tag is cut and GitHub Actions provenance runs on the remote.
