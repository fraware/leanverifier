# R-06 migration: Lean / mathlib 4.32.0 → 4.32.1

**PR:** `chore/r-06-lean-4.32.1`  
**Base:** R-05 tip (`test/r-05-assurance-matrix` @ `1f1af086188aaf8e320cfc1c4b2d7a79ea9319c8`)  
**Scope:** toolchain pin only — no theorem-statement changes.

## Pins

| Component | Previous | New |
|-----------|----------|-----|
| Lean (`lean-toolchain`) | `leanprover/lean4:v4.32.0` | `leanprover/lean4:v4.32.1` |
| mathlib (`lakefile.toml` / `lake-manifest.json`) | tag `v4.32.0` (`81a5d257…`) | tag `v4.32.1` (`520045ab14e26149ee970e2e617ca04b09bde5d6`) |

Packaged wheel copy under `src/leanverifier/resources/lean-project/` is kept in sync via `scripts/regen_resource_hashes.py`.

## Release notes (Lean 4.32.1)

Point release (2026-07-22) fixing a **kernel soundness bug** ([lean4#14484](https://github.com/leanprover/lean4/issues/14484), fix [lean4#14498](https://github.com/leanprover/lean4/pull/14498)). A malicious metaprogram running in-process with the kernel could previously trick acceptance of `False`. Comparator-based checking of possibly dishonest proofs is not affected. Official notes: <https://lean-lang.org/doc/reference/latest/releases/v4.32.1/>.

mathlib published a matching `v4.32.1` tag (2026-07-23) whose `lean-toolchain` is `leanprover/lean4:v4.32.1`.

## Explicit non-goals

- No upgrade to Lean / mathlib `4.33-rc*` (deferred; R-07/R-08 remain separate).
- No public RC tag or package publish.
- No formal theorem-statement edits; any proof breakage must be treated as needing formal review before merge.

## Verification performed in this PR

Recorded in the R-06 audit §8 handoff: `lake build`, axiom audit, generated-proof check, and `lean4checker` when available on the runner.
