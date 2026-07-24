# LeanVerifier Status

**Version target:** v0.1.0 technical preview  
**Status date:** 2026-07-24  
**Program lead:** Mateo Petel  
**Repository:** https://github.com/fraware/leanverifier  
**Working branch:** `recovery/v0.1.0`

## Current claim posture

LeanVerifier is under reconstruction. The **only** supported formal claim for v0.1.0 is:

> Given a dimensionally valid affine binary classifier over exact rationals, a reference input, and a nonnegative L∞ radius ε, if the absolute score margin strictly exceeds ε · ‖w‖₁, then the classification sign is invariant on the closed L∞ ball.

Everything else formerly advertised under FormalVerifML (neural nets, transformers, vision, SMT, distributed/enterprise web verification, PyTorch equivalence, floating-point correctness) is **unsupported** and lives under `experimental/legacy-formalverifml/` for historical reference only.

## Lifecycle

| Stage | State |
|-------|-------|
| Truth reset (LV-00) | Complete in working tree |
| Governance templates | Complete in tree; GitHub `master`→`main` rename + remote ruleset **blocked** (admin) |
| Build baseline (LV-01) | Lean 4.32.0 + mathlib `v4.32.0` pinned; SHA-pinned CI; Docker elan checksummed |
| Affine kernel (LV-02) | Complete |
| L∞ theorem + axiom audit (LV-03) | Complete; `#print axioms` = `propext`, `Classical.choice`, `Quot.sound` only |
| Schemas (LV-04) | Complete |
| Generator (LV-05) | Complete; Fin-pattern vectors (no bang indexing) |
| CLI (LV-06) | Complete; example `leanverifier verify` → `verified` / exit 0 |
| Adversarial tests (LV-07) | Complete (`pytest tests`) |
| Legacy isolation (LV-08) | Complete under `experimental/` |
| Assurance docs (LV-09) | Complete |
| RC release pipeline (LV-10) | Scaffolding in tree; **live attestations blocked** until RC tag on GitHub Actions |
| Independent review (LV-11) | **Blocked** on external reviewers |
| v0.1.0 authorization | **Not authorized** until §14 checklist in `docs/release-process.md` passes |

## Security posture

- The legacy web upload → translator → `lake build` path is **quarantined** and must not be exposed.
- Supported verification uses an isolated temp workspace and a deterministic generator with path containment.
- See [SECURITY.md](SECURITY.md).

## What “verified” means

Exit code `0` / status `verified` means only that the generated Lean module compiled under the pinned toolchain and the axiom audit passed for the concrete margin instance of the supported theorem. It does **not** mean model correctness in floating point, deployment readiness, or any property outside the L∞ margin statement.

## Blockers (do not fabricate completion)

1. GitHub default branch rename `master` → `main` (admin)
2. Remote branch ruleset + real multi-party CODEOWNERS reviewers
3. External LV-11 reviewer recruitment and findings resolution
4. Live GitHub Actions provenance/SBOM attestations for an RC tag
