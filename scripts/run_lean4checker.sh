#!/usr/bin/env bash
# Run leanchecker / lean4checker --fresh on supported LeanVerifier modules.
# Usage: scripts/run_lean4checker.sh [REPO_ROOT] [REPORT_PATH]
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
REPORT_PATH="${2:-${REPO_ROOT}/axiom-report-lean4checker.txt}"
cd "${REPO_ROOT}"

TOOLCHAIN="$(tr -d '[:space:]' < lean-toolchain)"

run_checker() {
  local module="${LEANCHECKER_MODULE:-LeanVerifier}"
  if command -v leanchecker >/dev/null 2>&1 || lake env which leanchecker >/dev/null 2>&1; then
    lake env leanchecker --fresh "${module}"
    return
  fi
  if command -v lean4checker >/dev/null 2>&1 || lake env which lean4checker >/dev/null 2>&1; then
    lake env lean4checker --fresh "${module}"
    return
  fi

  # Fallback: clone external lean4checker matching the toolchain tag (pre-bundled eras).
  TOOL_DIR="${RUNNER_TEMP:-/tmp}/lean4checker-src"
  TAG="${TOOLCHAIN##*:}"
  rm -rf "${TOOL_DIR}"
  git clone --depth 1 --branch "${TAG}" \
    https://github.com/leanprover/lean4checker.git "${TOOL_DIR}"
  (
    cd "${TOOL_DIR}"
    echo "${TOOLCHAIN}" > lean-toolchain
    lake update
    lake build
  )
  CHECKER="${TOOL_DIR}/.lake/build/bin/lean4checker"
  if [[ ! -x "${CHECKER}" ]]; then
    CHECKER="${TOOL_DIR}/.lake/build/bin/lean4checker.exe"
  fi
  lake env "${CHECKER}" --fresh "${module}"
}

{
  echo "lean4checker/leanchecker --fresh report"
  echo "toolchain=${TOOLCHAIN}"
  echo "repo=${REPO_ROOT}"
  echo "---"
  run_checker
} | tee "${REPORT_PATH}"

echo "Wrote ${REPORT_PATH}"
