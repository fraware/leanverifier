# Independent review invitation (LV-11)

LeanVerifier seeks bounded external reviews before authorizing `v0.1.0` of the reconstructed technical preview at https://github.com/fraware/leanverifier.

## Roles needed

| Role | Count | Scope |
|------|-------|-------|
| Lean / formal methods | ≥2 | Theorem statement + proof sketch + axiom surface |
| Supply chain | ≥1 | CI pins, attestations, Docker/elan bootstrap |
| Python validation | ≥1 | Schemas, canonicalize, generator path policy |
| Reproduction | ≥1 | Rebuild from RC artifacts; byte compare |

At least three reviewers must be outside the implementation team.
Assign one bounded task each; do not ask anyone to “review the whole repo.”

## Finding severity

| Severity | Policy |
|----------|--------|
| Critical | Blocks release |
| High | Blocks release |
| Medium | Resolve or Mateo-accepted with written rationale |
| Low | Track; need not block |

## Deliverable

Each reviewer files findings under `docs/program/findings/` (create as reviews arrive).
Regenerate RC artifacts after any final code change.
