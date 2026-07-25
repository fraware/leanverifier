#!/usr/bin/env bash
# Package RC artifacts into $1 (default: artifacts/).
#
# Requirements (F-015/F-016/F-017):
#   - Full axiom report via real #print axioms (NO --skip-lean)
#   - Tool-generated CycloneDX SBOM (pinned cyclonedx-bom) + Lean/mathlib enrichment
#   - SOURCE_DATE_EPOCH from the release commit for deterministic stamps
#
# Does NOT create tags, publish to PyPI, or authorize an RC (R-08 gate).
# Live provenance attestations remain a GitHub Actions concern (release-rc.yml).
set -euo pipefail

OUT="${1:-artifacts}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TZ=UTC
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}"
COMMIT="$(git rev-parse HEAD)"

if ! command -v lake >/dev/null 2>&1; then
  echo "error: lake is required for RC packaging (full axiom report; no --skip-lean)" >&2
  exit 1
fi

mkdir -p "$OUT"
# Fresh package directory contents (keep directory node).
find "$OUT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

echo "package_rc_artifacts: SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"
echo "package_rc_artifacts: HEAD=${COMMIT}"

# Ensure oleans exist for #print axioms / PublicAxioms.
if [[ "${SKIP_LAKE_BUILD:-0}" != "1" ]]; then
  lake build
fi

# Core provenance copies.
cp "$ROOT/examples/affine_binary/model.json" "$OUT/"
cp "$ROOT/examples/affine_binary/request.json" "$OUT/"
cp "$ROOT/lean-toolchain" "$OUT/"
cp "$ROOT/CLAIMS.md" "$OUT/"
cp "$ROOT/STATUS.md" "$OUT/"
cp "$ROOT/SECURITY.md" "$OUT/"
cp "$ROOT/lakefile.toml" "$OUT/"
cp "$ROOT/lake-manifest.json" "$OUT/"

# Optional wheel/sdist from python -m build / uv build
shopt -s nullglob
for f in "$ROOT"/dist/*.whl "$ROOT"/dist/*.tar.gz; do
  cp "$f" "$OUT/"
done
shopt -u nullglob

# Deterministic generated Lean module for artifact inspection.
GEN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lv-rc-gen.XXXXXX")"
VERIFY_OUT=""
cleanup() {
  rm -rf "$GEN_DIR"
  if [[ -n "${VERIFY_OUT}" && -d "${VERIFY_OUT}" ]]; then
    rm -rf "$VERIFY_OUT"
  fi
}
trap cleanup EXIT

run_py() {
  if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/uv.lock" ]]; then
    uv run "$@"
  else
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$@"
  fi
}

run_py leanverifier generate \
  --model "$ROOT/examples/affine_binary/model.json" \
  --request "$ROOT/examples/affine_binary/request.json" \
  --out-dir "$GEN_DIR"

GENERATED="$(ls "$GEN_DIR"/GeneratedModel_*.lean | head -n 1)"
cp "$GENERATED" "$OUT/"
GENERATED_NAME="$(basename "$GENERATED")"

# Full axiom audit — library + generated (F-016: no --skip-lean).
run_py python "$ROOT/scripts/audit_axioms.py" \
  --repo "$ROOT" \
  --generated "$GENERATED" \
  --report "$OUT/axiom-report.txt"

# Positive verify evidence (verification-result.json).
VERIFY_OUT="$(mktemp -d "${TMPDIR:-/tmp}/lv-rc-verify.XXXXXX")"
run_py leanverifier verify \
  --model "$ROOT/examples/affine_binary/model.json" \
  --request "$ROOT/examples/affine_binary/request.json" \
  --output "$VERIFY_OUT"
cp "$VERIFY_OUT/verification-result.json" "$OUT/"
cp "$VERIFY_OUT/checksums.sha256" "$OUT/verification-bundle-checksums.sha256"

{
  echo "# RC artifact list"
  echo "- model.json / request.json"
  echo "- GeneratedModel_*.lean"
  echo "- lean-toolchain / lakefile.toml / lake-manifest.json"
  echo "- CLAIMS.md / STATUS.md / SECURITY.md"
  echo "- wheel / sdist (if built under dist/)"
  echo "- axiom-report.txt (full #print axioms; no --skip-lean)"
  echo "- verification-result.json (positive verify)"
  echo "- sbom.json (cyclonedx-bom pinned tool + Lean/mathlib enrichment)"
  echo "- checksums.sha256"
  echo "- LIVE ATTESTATIONS: GitHub Actions attest job only; RC tag blocked until R-08"
  echo "- SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"
  echo "- commit=${COMMIT}"
  echo "- generated=${GENERATED_NAME}"
} > "$OUT/ARTIFACTS.md"

# Authoritative SBOM via pinned cyclonedx-bom (F-017). Written before checksums
# so component file hashes cover the packaged evidence set (excluding sbom/checksums).
run_py python "$ROOT/scripts/generate_sbom.py" \
  --repo "$ROOT" \
  --artifacts "$OUT" \
  --out "$OUT/sbom.json"

# Deterministic checksums: sorted file list, exclude the checksum file itself.
python3 - <<'PY' "$OUT"
import hashlib
import sys
from pathlib import Path

out = Path(sys.argv[1])
lines: list[str] = []
for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "checksums.sha256"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.name}")
(out / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out / 'checksums.sha256'} ({len(lines)} files)")
PY

echo "Wrote RC artifacts to $OUT"
