# Release review packet (draft)

**Draft status:** Scaffolding for external review — **not signed**  
**Prepared for:** R-08 / LV-11  
**Tree tip this draft describes:** `release/r-07-reproducible-rc` @ `337752957fa46646f46f791ebbbe23170ce11d74` (plus this R-08 docs commit)  
**Reviewer:** TBD  
**Reproduction approved?** **No**  
**RC tag authorized?** **No** — see `../release-authorization-checklist.md`

This draft lists how to assemble and review a release packet. It does **not** attach a live public RC artifact set (none may be tagged yet).

---

## 1. Artifact list (expected)

Produced by `scripts/package_rc_artifacts.sh <out-dir>`:

| Artifact | Purpose |
|----------|---------|
| Wheel / sdist | Installable distribution under `SOURCE_DATE_EPOCH` |
| Example model + request | Canonical positive inputs |
| Generated Lean + `verification-result.json` | Positive verify evidence |
| Axiom audit report | Full library + generated (`#print axioms`) |
| `sbom.json` | CycloneDX from pinned tool + enrichment |
| `checksums.sha256` | Sorted hashes over packaged files |
| `reproduction-log.txt` | From `scripts/reproduce_release.sh` |

Human index: `docs/release-process.md`.

---

## 2. SBOM

| Item | Location |
|------|----------|
| Generator | `scripts/generate_sbom.py` |
| Tool pin | `cyclonedx-bom==7.3.0` (via project extras / script) |
| Output | `sbom.json` in artifact dir |

Must not be a handwritten inventory scaffold (F-017 addressed in R-07).

---

## 3. Checksums

| Item | Location |
|------|----------|
| Aggregate | `checksums.sha256` in artifact dir |
| Verify evidence bundle | `verification-bundle-checksums.sha256` (from package script) |

---

## 4. Reproduction commands

```bash
# Package at the candidate commit (clean tree):
bash scripts/package_rc_artifacts.sh ./artifacts-ref

# Independent reproduction:
bash scripts/reproduce_release.sh ./artifacts-ref ./reproduction-log.txt
```

CI scaffolding: `.github/workflows/release-rc.yml` (`build-artifacts`, `attest`, `reproduce`).
Workflow dispatch may exercise scaffolding; **public RC tag provenance is blocked** until the authorization checklist is signed.

Dry-run: `bash scripts/reproduce_release.sh --dry-run`.

---

## 5. Clean-environment logs checklist

Reviewer should collect logs demonstrating:

1. Clean worktree / clean runner
2. `uv sync --frozen --all-extras`
3. `lake build`
4. `lean4checker --fresh` (or CI equivalent)
5. `uv build` / package script
6. Full axiom audit (no `--skip-lean`)
7. Byte-compare of deterministic artifacts vs reference
8. Attestation verification when the attest job is exercised

Store paths or CI run URLs in the dated reviewer copy of this packet.

---

## 6. Installed-wheel run

| Step | How |
|------|-----|
| Build wheel | `uv build` or package script |
| Clean install | temp venv / CI `wheel-smoke` |
| Positive verify | `leanverifier verify` + required `--output` |
| Inspect | `leanverifier inspect` on the evidence bundle |

---

## Independent reproductions (gate)

Three independent reproductions are required before RC authorization. Record them here when performed:

| # | Operator (≠ implementer where required) | Environment | Log path / URL | Result |
|---|------------------------------------------|-------------|----------------|--------|
| 1 | TBD | TBD | TBD | Not run |
| 2 | TBD | TBD | TBD | Not run |
| 3 | TBD | TBD | TBD | Not run |

---

## Sign-off

| Field | Value |
|-------|-------|
| Reproduction approved? | **No** |
| Reviewer handle | TBD |
| Date | — |
