# Release / reproducibility review packet (template)

**Reviewer:** TBD  
**Role:** Reproducible-build and/or packaging  
**Review date:** YYYY-MM-DD  
**Tree / commit under review:** `<sha>`  
**Artifact set digest / path:** TBD  
**Finding file (if any):** `docs/program/findings/<id>.md`

Status vocabulary: `Not started` | `In progress` | `Pass` | `Fail` | `N/A`.

**Hard constraint:** Completing this packet does not authorize `v0.1.0-rc1`. Tagging remains blocked until `release-authorization-checklist.md` is fully signed.

---

## 1. Artifact list

Expected RC bundle contents (see `docs/release-process.md`):

| Artifact | Present? | Path / notes |
|----------|----------|--------------|
| Source tarball / wheel (`SOURCE_DATE_EPOCH`) | | |
| Canonical example model + request | | |
| Generated Lean module | | |
| `verification-result.json` (positive verify) | | |
| Full axiom audit report (no `--skip-lean`) | | |
| `checksums.sha256` | | |
| SBOM (CycloneDX JSON) | | |
| `reproduction-log.txt` | | |

Packaging: `scripts/package_rc_artifacts.sh`

---

## 2. SBOM

- Generator: `scripts/generate_sbom.py` (pinned `cyclonedx-bom`)
- Artifact: `sbom.json`
- Confirm: tool-generated (not handwritten inventory); Lean/mathlib enrichment present

**Status:** Not started

---

## 3. Checksums

- File: `checksums.sha256`
- Confirm sorted file list; excludes self; verifies cleanly

**Status:** Not started

---

## 4. Reproduction commands

```bash
# From a clean checkout at the release commit:
bash scripts/package_rc_artifacts.sh /path/to/artifacts
bash scripts/reproduce_release.sh /path/to/reference-artifacts /path/to/reproduction-log.txt
```

Optional CI: `.github/workflows/release-rc.yml` (`build-artifacts` / `attest` / `reproduce`).
Live public-tag provenance remains blocked until authorization.

**Status:** Not started

---

## 5. Clean-environment logs checklist

| Step | Log evidence | Status |
|------|--------------|--------|
| Clean runner / worktree | | |
| Frozen deps (`uv sync --frozen --all-extras`) | | |
| `lake build` | | |
| `lean4checker --fresh` | | |
| Wheel / sdist build | | |
| Full axiom report | | |
| Byte-compare vs reference artifacts | | |
| Attestation verify (when exercised) | | |

---

## 6. Installed-wheel run

| Step | Evidence | Status |
|------|----------|--------|
| Install wheel into clean venv / temp env | | |
| `leanverifier verify` with `--output` on positive example | | |
| Exit 0 / status `verified` | | |
| Inspect / checksum consistency | | |

CI reference: `wheel-smoke` job with `--require-verify`.

---

## Sign-off (do not forge)

| Field | Value |
|-------|-------|
| Reproduction approved? | **No / Yes** |
| Independent reproduction # (1–3) | |
| Blocking findings filed? | |
| Reviewer signature / handle | TBD |
| Date | |
