#!/usr/bin/env bash
# Package RC artifacts into $1 (default: artifacts/)
# Live provenance attestations remain a GitHub Actions concern (see release-rc.yml).
set -euo pipefail
OUT="${1:-artifacts}"
mkdir -p "$OUT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cp "$ROOT/examples/affine_binary/model.json" "$OUT/"
cp "$ROOT/examples/affine_binary/request.json" "$OUT/"
cp "$ROOT/lean-toolchain" "$OUT/"
cp "$ROOT/CLAIMS.md" "$OUT/"
cp "$ROOT/STATUS.md" "$OUT/"
cp "$ROOT/SECURITY.md" "$OUT/"
cp "$ROOT/lakefile.toml" "$OUT/"
cp "$ROOT/lake-manifest.json" "$OUT/"

# Optional wheel from python -m build
if compgen -G "$ROOT/dist/*.whl" > /dev/null; then
  cp "$ROOT"/dist/*.whl "$OUT/"
fi

# Generate Lean module bytes (no compile) for artifact inspection
python - <<'PY' "$ROOT" "$OUT"
import sys
from pathlib import Path
root, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "src"))
from leanverifier.generator import write_module_atomic
from leanverifier.manifest import load_model_path, load_request_path
model = load_model_path(root / "examples/affine_binary/model.json")
request = load_request_path(root / "examples/affine_binary/request.json")
path = write_module_atomic(model, request, out.resolve())
print(path)
PY

# Source axiom audit (skip-lean) for offline packaging; full audit runs after lake build
python "$ROOT/scripts/audit_axioms.py" --repo "$ROOT" --skip-lean | tee "$OUT/axiom-report.txt"

# Minimal CycloneDX-ish SBOM scaffold (component inventory; live attestation is remote)
python - <<'PY' "$ROOT" "$OUT"
import json, hashlib, sys
from pathlib import Path
from datetime import datetime, timezone
root, out = map(Path, sys.argv[1:])
toolchain = (root / "lean-toolchain").read_text(encoding="utf-8").strip()
components = [
    {"type": "library", "name": "leanverifier", "version": "0.1.0",
     "purl": "pkg:generic/leanverifier@0.1.0"},
    {"type": "library", "name": "lean4", "version": toolchain,
     "purl": f"pkg:generic/lean4@{toolchain}"},
    {"type": "library", "name": "mathlib4", "version": "v4.32.0",
     "purl": "pkg:github/leanprover-community/mathlib4@v4.32.0"},
    {"type": "library", "name": "jsonschema", "version": "4.23.0",
     "purl": "pkg:pypi/jsonschema@4.23.0"},
]
sbom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "version": 1,
    "metadata": {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "component": {"type": "application", "name": "leanverifier", "version": "0.1.0-rc1"},
        "note": "Local scaffold only; GitHub Actions provenance attestation is required for release trust.",
    },
    "components": components,
}
(out / "sbom.json").write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
print("wrote sbom.json")
PY

{
  echo "# RC artifact list"
  echo "- model.json / request.json"
  echo "- GeneratedModel_*.lean"
  echo "- lean-toolchain / lakefile.toml / lake-manifest.json"
  echo "- CLAIMS.md / STATUS.md / SECURITY.md"
  echo "- wheel (if built)"
  echo "- axiom-report.txt"
  echo "- sbom.json (local scaffold; attest remotely)"
  echo "- checksums.sha256"
  echo "- verification-result.json (produced by full verify job with Lean)"
  echo "- LIVE ATTESTATIONS: blocked until RC tag runs on GitHub Actions"
} > "$OUT/ARTIFACTS.md"

(
  cd "$OUT"
  if command -v sha256sum >/dev/null; then
    sha256sum * > checksums.sha256
  elif command -v shasum >/dev/null; then
    shasum -a 256 * > checksums.sha256
  fi
)
echo "Wrote RC artifacts to $OUT"
