# LV-00 handoff evidence

**Branch:** `recovery/v0.1.0` (combined recovery tree; plan originally named `recovery/lv-00-truth-reset`)  
**Date:** 2026-07-24  
**Repository:** https://github.com/fraware/leanverifier

## Delivered (LV-00)

- Created `STATUS.md`, `CLAIMS.md`, `SECURITY.md`, `docs/history/README.md`
- Replaced active docs: `README.md`, `CONTRIBUTING.md`, `docs/user_guide.md`, `docs/developer_guide.md`
- Moved historical/overclaimed materials under `docs/history/` with unsupported headers
- Quarantined legacy FormalVerifML under `experimental/legacy-formalverifml/` with warnings
- Prohibited-claim scan enforced by `scripts/verify_docs.py` (passes on active docs)

## Claim posture after reset

Post-LV-00 target surface is exact affine binary + L∞ margin robustness only. Former production/enterprise/web/distributed claims withdrawn. **No LV-C* claim is Supported** until the signed release-authorization checklist (see current `CLAIMS.md` / `STATUS.md`); LV-00 vocabulary used Implementation / Experimental / Planned — not premature Supported.

## Follow-on delivered in the same working tree

Because sequential PR merges are not available locally, LV-01–LV-10 implementation and governance templates continue on `recovery/v0.1.0` so the repository reaches the target architecture. This is an intentional recovery deviation from the one-PR-at-a-time merge rule; remote review should still treat formal/schema/generator/workflow surfaces with implementer ≠ reviewer.

## Admin actions still required (blocked)

See `docs/program/admin-actions.md`:

1. GitHub default branch rename `master` → `main`
2. Branch ruleset + real multi-party CODEOWNERS reviewer usernames
3. External LV-11 reviews (`docs/program/review-invitation.md`)
4. Live RC attestations on GitHub Actions (`release-rc.yml`)

## Acceptance evidence

- Active docs link to STATUS / CLAIMS / SECURITY
- `CLAIMS.md` covers README capabilities with Implementation/Assurance columns (no Supported until checklist)
- `python scripts/verify_docs.py` → DOC VERIFY PASSED
- Legacy web path carries SECURITY / WARNING headers
