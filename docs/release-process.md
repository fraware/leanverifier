# Release process

## Versioning

- `v0.1.0-rcN` — technical preview candidate (**blocked until R-08 authorization**)
- `v0.1.0` — only after the §14 checklist below is fully satisfied (LV-11)

Repository: https://github.com/fraware/leanverifier

**Hard constraints (current tree):** do not create tag `v0.1.0-rc1`, do not publish to PyPI, and do not describe the project as released. `STATUS.md` remains “RC not authorized” until the signed checklist in [`docs/program/review/release-authorization-checklist.md`](program/review/release-authorization-checklist.md) (R-08 scaffolding is not approval).

## RC artifact list

- Source tarball / Python wheel (built under `SOURCE_DATE_EPOCH`)
- Canonical example model + request
- Generated Lean module
- `verification-result.json` (positive `leanverifier verify`)
- Full axiom audit report (`#print axioms`; **no** `--skip-lean`)
- `checksums.sha256`
- SBOM (CycloneDX JSON from pinned `cyclonedx-bom`, enriched with Lean/mathlib + artifact hashes)
- `reproduction-log.txt` (from `scripts/reproduce_release.sh`)

Packaging helpers: `scripts/package_rc_artifacts.sh`, `scripts/generate_sbom.py`, `scripts/reproduce_release.sh`.

## Determinism

- `SOURCE_DATE_EPOCH` is set from `git log -1 --format=%ct` of the release commit.
- `PYTHONHASHSEED=0` and `TZ=UTC` during package/reproduce.
- SBOM generation uses `cyclonedx-py --output-reproducible` plus sorted JSON enrichment.
- Checksums are written over a sorted file list.

## Attestation + reproduce workflow

See `.github/workflows/release-rc.yml`:

1. `build-artifacts` — frozen `uv sync --all-extras`, `lake build`, `uv build`, full package script, SBOM/axiom guards
2. `attest` — `actions/attest-build-provenance` over wheel/sdist/SBOM/axiom report/checksums/result
3. `reproduce` — end-to-end `scripts/reproduce_release.sh` against the uploaded artifact bundle

Live provenance on a **public RC tag** remains blocked until the R-08 authorization checklist is signed. Workflow dispatch may exercise scaffolding without authorizing release.

## §14 authorization checklist (v0.1.0)

Operational gate (must be fully signed before any RC tag):  
[`docs/program/review/release-authorization-checklist.md`](program/review/release-authorization-checklist.md)

Summary (unchecked here on purpose — use the signed checklist file):

- [ ] Formal: central theorem reviewed; axiom audit green on pinned toolchain
- [ ] Integrity: schemas + digests; mutation tests fail closed
- [ ] Generator security: path/injection/timeout tests green; security reviewer sign-off
- [ ] Verification behavior: exit codes 0/2/3/4/5/6/8 demonstrated
- [ ] Reproducibility: three independent `scripts/reproduce_release.sh` byte-compares
- [ ] Supply chain: SHA-pinned Actions; provenance attestation verified
- [ ] Docs: CLAIMS/STATUS/SECURITY/assurance set match the product
- [ ] Findings: no open Critical/High; Medium accepted only with Mateo rationale
- [ ] Artifacts regenerated after the final code change
- [ ] **NO `v0.1.0-rc1` until the authorization checklist is signed**

## Independent review invitations

See `docs/program/review-invitation.md` and review packets under `docs/program/review/`.
External reviewers are required; this checklist cannot be self-certified by implementers alone.
