# Independent review invitation (LV-11)

LeanVerifier seeks bounded external reviews before authorizing `v0.1.0-rc1` / `v0.1.0` of the reconstructed technical preview at https://github.com/fraware/leanverifier.

**Current state:** Review packet templates and drafts exist under [`docs/program/review/`](review/). Handles are **TBD**. RC remains **not authorized**.

## Roles needed

| Role | Count | Scope | Handle | Packet |
|------|-------|-------|--------|--------|
| Lean / formal methods | ≥2 | Theorem statement + proof sketch + axiom surface | TBD | [formal](review/templates/formal-packet.md) |
| Application security | ≥1 | Input→Lean boundary, path/injection limits | TBD | [generator](review/templates/generator-packet.md) |
| Python packaging / validation | ≥1 | Schemas, canonicalize, wheel surface | TBD | generator + [release](review/templates/release-packet.md) |
| Reproducible build | ≥1 | Rebuild from RC artifacts; byte compare | TBD | [release](review/templates/release-packet.md) |

At least three reviewers must be outside the implementation team.
Assign one bounded packet each; do not ask anyone to “review the whole repo.”
Do not invent GitHub usernames — Mateo supplies handles via `admin-actions.md`.

## Finding severity

| Severity | Policy |
|----------|--------|
| Critical | Blocks release |
| High | Blocks release |
| Medium | Resolve or Mateo-accepted with written rationale |
| Low | Track; need not block |

## Deliverable

1. Complete the assigned packet (start from `docs/program/review/drafts/` or a fresh template copy).
2. File findings under `docs/program/findings/` (create as reviews arrive).
3. Sign the relevant gate on [`review/release-authorization-checklist.md`](review/release-authorization-checklist.md) only if approving.
4. Regenerate RC artifacts after any final code change; re-run reproductions.

Implementers must not pre-sign packets or the authorization checklist.
