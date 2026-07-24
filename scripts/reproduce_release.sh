#!/usr/bin/env bash
# Reproduce a release artifact directory (skeleton).
set -euo pipefail
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "LeanVerifier reproduce_release"
echo "repo=$ROOT"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry-run: would download attested archive, verify attestation, rebuild, byte-compare"
  exit 0
fi
ART="${1:-artifacts}"
if [[ ! -d "$ART" ]]; then
  echo "missing artifacts dir: $ART" >&2
  exit 1
fi
if [[ -f "$ART/checksums.sha256" ]]; then
  (cd "$ART" && sha256sum -c checksums.sha256)
fi
echo "Reproduction checks completed for present checksums."
