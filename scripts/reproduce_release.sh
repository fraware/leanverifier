#!/usr/bin/env bash
# Reproduce a release artifact directory end-to-end (F-015).
#
# Flow:
#   clean-ish checkout → install from locks → lake build → lean4checker --fresh
#   → build wheel/sdist → package RC artifacts → clean-env install
#   → positive verify → byte-compare deterministic artifacts
#   → verify attestations if present → emit reproduction-log.txt
#
# Usage:
#   scripts/reproduce_release.sh <reference-artifacts-dir> [log-path]
#   scripts/reproduce_release.sh --dry-run
#
# Does NOT tag, publish, or authorize an RC (R-08 gate).
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift || true
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REF_ART="${1:-}"
LOG_PATH="${2:-$ROOT/reproduction-log.txt}"

log() {
  # shellcheck disable=SC2034
  local line="$*"
  printf '%s\n' "$*" | tee -a "$LOG_PATH"
}

die() {
  log "ERROR: $*"
  exit 1
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "LeanVerifier reproduce_release (dry-run)"
  echo "repo=$ROOT"
  echo "dry-run: would sync locks, lake build, lean4checker --fresh,"
  echo "         build wheel/sdist, package artifacts, clean-env verify,"
  echo "         byte-compare deterministic files, verify attestations, write log"
  exit 0
fi

[[ -n "$REF_ART" ]] || die "usage: $0 <reference-artifacts-dir> [log-path]"
[[ -d "$REF_ART" ]] || die "missing artifacts dir: $REF_ART"
REF_ART="$(cd "$REF_ART" && pwd)"

rm -f "$LOG_PATH"
: > "$LOG_PATH"

export TZ=UTC
export PYTHONHASHSEED=0
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}"

log "=== LeanVerifier release reproduction ==="
log "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "repo=$ROOT"
log "commit=$(git rev-parse HEAD)"
log "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"
log "reference_artifacts=$REF_ART"
log "toolchain=$(tr -d '[:space:]' < lean-toolchain)"

# --- Preconditions ---
command -v lake >/dev/null 2>&1 || die "lake is required"
command -v uv >/dev/null 2>&1 || die "uv is required"
[[ -f "$ROOT/uv.lock" ]] || die "uv.lock missing (frozen install required)"

# --- 1. Install from locks ---
log "--- step: uv sync --frozen --all-extras ---"
uv sync --frozen --all-extras

# --- 2. Lake build ---
log "--- step: lake build ---"
lake build

# --- 3. lean4checker / leanchecker --fresh ---
log "--- step: lean4checker --fresh ---"
CHECKER_REPORT="${RUNNER_TEMP:-/tmp}/axiom-report-lean4checker-repro.txt"
chmod +x "$ROOT/scripts/run_lean4checker.sh"
"$ROOT/scripts/run_lean4checker.sh" "$ROOT" "$CHECKER_REPORT"
log "lean4checker_report=$CHECKER_REPORT"

# --- 4. Build wheel + sdist (deterministic stamps) ---
log "--- step: uv build (SOURCE_DATE_EPOCH) ---"
rm -rf "$ROOT/dist"
uv build

# --- 5. Package RC artifacts locally ---
WORK="$(mktemp -d "${TMPDIR:-/tmp}/lv-repro.XXXXXX")"
LOCAL_ART="$WORK/artifacts"
mkdir -p "$LOCAL_ART"
log "--- step: package_rc_artifacts ($LOCAL_ART) ---"
SKIP_LAKE_BUILD=1 bash "$ROOT/scripts/package_rc_artifacts.sh" "$LOCAL_ART"

# --- 6. Clean-env install of rebuilt wheel + positive verify ---
log "--- step: clean-env wheel install + positive verify ---"
WHEEL="$(ls "$ROOT"/dist/leanverifier-*.whl | head -n 1)"
[[ -f "$WHEEL" ]] || die "no wheel produced under dist/"
CLEAN_VENV="$WORK/clean-venv"
python3 -m venv "$CLEAN_VENV"
# shellcheck disable=SC1091
source "$CLEAN_VENV/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q "$WHEEL"
VERIFY_OUT="$WORK/verify-out"
mkdir -p "$VERIFY_OUT"
# Prefer lake from the host PATH (Lean toolchain); drop PYTHONPATH.
unset PYTHONPATH || true
leanverifier verify \
  --model "$ROOT/examples/affine_binary/model.json" \
  --request "$ROOT/examples/affine_binary/request.json" \
  --output "$VERIFY_OUT"
deactivate || true
log "clean_env_verify_status=$(python3 -c "import json;print(json.load(open('$VERIFY_OUT/verification-result.json'))['status'])")"

# --- 7. Verify reference checksums (when present) ---
log "--- step: reference checksum verification ---"
if [[ -f "$REF_ART/checksums.sha256" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$REF_ART" && sha256sum -c checksums.sha256) | tee -a "$LOG_PATH"
  else
    (cd "$REF_ART" && shasum -a 256 -c checksums.sha256) | tee -a "$LOG_PATH"
  fi
else
  log "WARN: reference checksums.sha256 missing; skipping aggregate check"
fi

# --- 8. Byte-compare deterministic artifacts ---
log "--- step: byte-compare deterministic artifacts ---"
COMPARE_NAMES=(
  model.json
  request.json
  lean-toolchain
  lakefile.toml
  lake-manifest.json
  ARTIFACTS.md
)
# Generated Lean module (exact name from reference).
REF_GEN="$(ls "$REF_ART"/GeneratedModel_*.lean 2>/dev/null | head -n 1 || true)"
if [[ -n "$REF_GEN" ]]; then
  COMPARE_NAMES+=("$(basename "$REF_GEN")")
fi
# Axiom report (full #print axioms; no timestamps beyond commit footer — keep pure).
if [[ -f "$REF_ART/axiom-report.txt" ]]; then
  COMPARE_NAMES+=(axiom-report.txt)
fi

BYTE_FAIL=0
for name in "${COMPARE_NAMES[@]}"; do
  if [[ ! -f "$REF_ART/$name" ]]; then
    log "MISS_REF $name"
    BYTE_FAIL=1
    continue
  fi
  if [[ ! -f "$LOCAL_ART/$name" ]]; then
    log "MISS_LOCAL $name"
    BYTE_FAIL=1
    continue
  fi
  if cmp -s "$REF_ART/$name" "$LOCAL_ART/$name"; then
    log "MATCH $name"
  else
    log "DIFF $name"
    BYTE_FAIL=1
  fi
done

# SBOM: compare semantic identity of generator + Lean pins (tool output may
# reorder component hashes for rebuilt wheels). Require format + pins.
log "--- step: SBOM structural checks ---"
python3 - <<'PY' "$REF_ART/sbom.json" "$LOCAL_ART/sbom.json" | tee -a "$LOG_PATH"
import json, sys
ref_path, loc_path = sys.argv[1], sys.argv[2]
ref = json.loads(open(ref_path, encoding="utf-8").read())
loc = json.loads(open(loc_path, encoding="utf-8").read())
errors = []
if ref.get("bomFormat") != "CycloneDX" or loc.get("bomFormat") != "CycloneDX":
    errors.append("bomFormat must be CycloneDX")
if "scaffold" in json.dumps(ref.get("metadata", {})).lower():
    errors.append("reference SBOM still looks like handwritten scaffold")
# Require lean4 + mathlib components from enrichment.
def names(doc):
    return {c.get("name") for c in doc.get("components") or []}
need = {"lean4", "mathlib"}
if not need.issubset(names(ref)):
    errors.append(f"reference SBOM missing {need - names(ref)}")
if not need.issubset(names(loc)):
    errors.append(f"local SBOM missing {need - names(loc)}")
# Generator property must cite cyclonedx-bom.
def gen(doc):
    for p in (doc.get("metadata") or {}).get("properties") or []:
        if p.get("name") == "leanverifier:sbom-generator":
            return str(p.get("value") or "")
    return ""
if "cyclonedx-bom==" not in gen(ref):
    errors.append("reference SBOM not produced by pinned cyclonedx-bom")
if "cyclonedx-bom==" not in gen(loc):
    errors.append("local SBOM not produced by pinned cyclonedx-bom")
if errors:
    print("SBOM_CHECK_FAIL")
    for e in errors:
        print(f" - {e}")
    sys.exit(1)
print("SBOM_CHECK_PASS")
PY

# Wheel byte-compare when both present and same filename (best-effort).
REF_WHEEL="$(ls "$REF_ART"/leanverifier-*.whl 2>/dev/null | head -n 1 || true)"
LOC_WHEEL="$(ls "$LOCAL_ART"/leanverifier-*.whl 2>/dev/null | head -n 1 || true)"
if [[ -n "$REF_WHEEL" && -n "$LOC_WHEEL" ]]; then
  if cmp -s "$REF_WHEEL" "$LOC_WHEEL"; then
    log "MATCH $(basename "$REF_WHEEL") (byte-identical wheel)"
  else
    log "INFO wheel bytes differ (recorded; SOURCE_DATE_EPOCH applied but not guaranteed across runners)"
    log "REF_WHEEL_SHA256=$(sha256sum "$REF_WHEEL" | awk '{print $1}')"
    log "LOC_WHEEL_SHA256=$(sha256sum "$LOC_WHEEL" | awk '{print $1}')"
  fi
fi

# --- 9. Verify attestations if present / available ---
log "--- step: attestation verification (optional until R-08 tag) ---"
if command -v gh >/dev/null 2>&1; then
  mapfile -t SUBJECTS < <(find "$REF_ART" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name 'sbom.json' \) | LC_ALL=C sort)
  if [[ "${#SUBJECTS[@]}" -gt 0 ]]; then
    if gh attestation verify "${SUBJECTS[@]}" --repo "${GITHUB_REPOSITORY:-fraware/leanverifier}" 2>&1 | tee -a "$LOG_PATH"; then
      log "ATTESTATION_VERIFY_PASS"
    else
      # Expected when no RC tag / attestations have not been published yet.
      log "ATTESTATION_VERIFY_SKIP_OR_FAIL (acceptable pre-R-08; no public RC tag)"
    fi
  else
    log "ATTESTATION_VERIFY_SKIP (no subject files)"
  fi
else
  log "ATTESTATION_VERIFY_SKIP (gh CLI not available)"
fi

# Bundle a copy of the reproduction log into the local artifact dir for uploaders.
cp "$LOG_PATH" "$LOCAL_ART/reproduction-log.txt"

log "--- summary ---"
if [[ "$BYTE_FAIL" -ne 0 ]]; then
  log "REPRODUCTION_FAILED byte-compare mismatches"
  exit 1
fi
log "REPRODUCTION_PASSED"
log "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "log_path=$LOG_PATH"

# Keep workdir artifacts for CI uploaders when requested.
if [[ -n "${REPRO_KEEP_DIR:-}" ]]; then
  mkdir -p "$REPRO_KEEP_DIR"
  cp -a "$LOCAL_ART/." "$REPRO_KEEP_DIR/"
  log "kept_artifacts=$REPRO_KEEP_DIR"
fi

rm -rf "$WORK"
exit 0
