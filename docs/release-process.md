# Release process

## Versioning

- `v0.1.0-rcN` — technical preview candidate (LV-10 scaffolding)
- `v0.1.0` — only after the §14 checklist below is fully satisfied (LV-11)

Repository: https://github.com/fraware/leanverifier

## RC artifact list

- Source tarball
- Python wheel
- Canonical example model + request
- Generated Lean module
- `verification-result.json`
- Axiom audit report
- `checksums.sha256`
- SBOM (CycloneDX or SPDX JSON)
- Reproduction log

Packaging helpers: `scripts/package_rc_artifacts.sh`, `scripts/reproduce_release.sh`.

## Attestation job skeleton

See `.github/workflows/release-rc.yml`. Default permissions remain `contents: read`; the attestation job alone requests `id-token: write` and `attestations: write`.

Live provenance/SBOM attestations remain blocked until an RC tag is cut on GitHub Actions.

## §14 authorization checklist (v0.1.0)

- [ ] Formal: central theorem reviewed; axiom audit green on pinned toolchain
- [ ] Integrity: schemas + digests; mutation tests fail closed
- [ ] Generator security: path/injection/timeout tests green; security reviewer sign-off
- [ ] Verification behavior: exit codes 0/2/3/4/5/6/8 demonstrated
- [ ] Reproducibility: `scripts/reproduce_release.sh` byte-compares artifacts
- [ ] Supply chain: SHA-pinned Actions; provenance attestation verified
- [ ] Docs: CLAIMS/STATUS/SECURITY/assurance set match the product
- [ ] Findings: no open Critical/High; Medium accepted only with Mateo rationale

## Independent review invitations

See `docs/program/review-invitation.md`. External reviewers are required; this checklist cannot be self-certified by implementers alone.
