"""Immutable verification evidence bundles."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from leanverifier.canonicalize import (
    canonical_json_bytes,
    digest_bytes,
    digest_obj,
    loads_strict_json,
)
from leanverifier.errors import InvalidInputError
from leanverifier.manifest import validate_result_obj
from leanverifier.result import STATUS_TO_EXIT, VerificationResult

RESULT_NAME = "verification-result.json"
CHECKSUMS_NAME = "checksums.sha256"
GENERATED_PREFIX = "GeneratedModel_"
GENERATED_SUFFIX = ".lean"


def read_mathlib_revision(repo: Path) -> str:
    manifest_path = repo / "lake-manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    packages = data.get("packages")
    if not isinstance(packages, list):
        return "unknown"
    for pkg in packages:
        if isinstance(pkg, dict) and pkg.get("name") == "mathlib":
            rev = pkg.get("rev")
            if isinstance(rev, str) and rev:
                return rev
    return "unknown"


def _file_digest_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum_lines(named_bytes: dict[str, bytes]) -> str:
    lines = [f"{_file_digest_hex(named_bytes[name])}  {name}" for name in sorted(named_bytes)]
    return ("\n".join(lines) + "\n") if lines else ""


def compute_evidence_bundle_digest(named_bytes: dict[str, bytes]) -> str:
    """Digest of sorted checksum lines for content artifacts (excludes result/checksums)."""
    return digest_bytes(checksum_lines(named_bytes).encode("utf-8"))


def parse_checksums(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise InvalidInputError(f"malformed checksums line {line_no}")
        digest_hex, name = parts
        if len(digest_hex) != 64 or any(c not in "0123456789abcdef" for c in digest_hex):
            raise InvalidInputError(f"malformed checksum hex at line {line_no}")
        if name in out:
            raise InvalidInputError(f"duplicate checksum entry: {name}")
        out[name] = digest_hex
    return out


def _assert_bundle_dir(output_dir: Path) -> Path:
    root = output_dir.expanduser()
    if not root.is_absolute():
        raise InvalidInputError("--output must be an absolute path")
    if root.exists() and root.is_symlink():
        raise InvalidInputError("--output must not be a symlink")
    if root.exists() and not root.is_dir():
        raise InvalidInputError("--output must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve(strict=False)
    if resolved.is_symlink():
        raise InvalidInputError("--output must not be a symlink")
    return resolved


def write_text_atomic(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def build_environment(
    *,
    repo: Path,
    lean_toolchain: str,
    mathlib_revision: str,
    generator_version: str,
    argv: list[str],
    duration_ms: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "generator_version": generator_version,
        "lean_toolchain": lean_toolchain,
        "mathlib_revision": mathlib_revision,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "offline": True,
        "command": argv,
        "repo_markers": {
            "lean_toolchain_file": lean_toolchain,
            "lake_manifest_present": (repo / "lake-manifest.json").is_file(),
        },
    }
    if duration_ms is not None:
        env["duration_ms"] = duration_ms
    if extra:
        env.update(extra)
    return env


def persist_evidence_bundle(
    output_dir: Path,
    *,
    result: VerificationResult,
    model_obj: dict[str, Any] | None,
    request_obj: dict[str, Any] | None,
    generated_lean: bytes | None,
    generated_module_name: str | None,
    lean_build_log: str,
    generated_module_log: str,
    axiom_report: str,
    environment: dict[str, Any],
) -> VerificationResult:
    """Write the immutable evidence bundle and return the finalized result."""
    root = _assert_bundle_dir(output_dir)
    artifacts: dict[str, bytes] = {
        "lean-build.log": lean_build_log.encode("utf-8"),
        "generated-module.log": generated_module_log.encode("utf-8"),
        "axiom-report.txt": axiom_report.encode("utf-8"),
        "environment.json": canonical_json_bytes(environment),
    }
    if model_obj is not None:
        artifacts["model.canonical.json"] = canonical_json_bytes(model_obj)
    if request_obj is not None:
        artifacts["request.canonical.json"] = canonical_json_bytes(request_obj)
    if generated_lean is not None and generated_module_name is not None:
        lean_name = f"{generated_module_name}.lean"
        if not lean_name.startswith(GENERATED_PREFIX) or not lean_name.endswith(GENERATED_SUFFIX):
            raise InvalidInputError(f"illegal generated module filename: {lean_name}")
        artifacts[lean_name] = generated_lean
        result.generated_module = generated_module_name
        result.generated_module_digest = digest_bytes(generated_lean)

    result.evidence_bundle_digest = compute_evidence_bundle_digest(artifacts)
    result_dict = result.to_dict()
    result_bytes = canonical_json_bytes(result_dict)
    all_files = dict(artifacts)
    all_files[RESULT_NAME] = result_bytes

    for name, data in all_files.items():
        write_bytes_atomic(root / name, data)
    write_text_atomic(root / CHECKSUMS_NAME, checksum_lines(all_files))
    return result


def load_bundle_files(bundle_dir: Path) -> dict[str, bytes]:
    root = bundle_dir.resolve(strict=True)
    if not root.is_dir():
        raise InvalidInputError("inspect target must be a directory")
    files: dict[str, bytes] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == CHECKSUMS_NAME:
            continue
        if path.is_symlink():
            raise InvalidInputError(f"symlink not allowed in bundle: {path.name}")
        files[path.name] = path.read_bytes()
    return files


def inspect_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate schemas, checksums, digests, and status consistency.

    Raises InvalidInputError on tampering or inconsistency.
    """
    root = bundle_dir.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise InvalidInputError("inspect target must be an existing directory")
    checksums_path = root / CHECKSUMS_NAME
    if not checksums_path.is_file():
        raise InvalidInputError("missing checksums.sha256")
    recorded = parse_checksums(checksums_path.read_text(encoding="utf-8"))
    files = load_bundle_files(root)

    if RESULT_NAME not in files:
        raise InvalidInputError("missing verification-result.json")
    if set(recorded) != set(files):
        missing = sorted(set(recorded) - set(files))
        extra = sorted(set(files) - set(recorded))
        raise InvalidInputError(f"checksum inventory mismatch; missing={missing} extra={extra}")

    for name, data in files.items():
        actual = _file_digest_hex(data)
        if actual != recorded[name]:
            raise InvalidInputError(f"checksum mismatch for {name}")

    result_obj = loads_strict_json(files[RESULT_NAME])
    if not isinstance(result_obj, dict):
        raise InvalidInputError("verification-result.json must be an object")
    validate_result_obj(result_obj)
    expected_exit = STATUS_TO_EXIT.get(str(result_obj.get("status")))
    if expected_exit is None or result_obj.get("exit_code") != expected_exit:
        raise InvalidInputError("status/exit_code inconsistency in result")

    content_for_digest = {name: data for name, data in files.items() if name != RESULT_NAME}
    expected_bundle_digest = compute_evidence_bundle_digest(content_for_digest)
    if result_obj.get("evidence_bundle_digest") != expected_bundle_digest:
        raise InvalidInputError("evidence_bundle_digest mismatch")

    if "model.canonical.json" in files:
        model_obj = loads_strict_json(files["model.canonical.json"])
        if not isinstance(model_obj, dict):
            raise InvalidInputError("model.canonical.json must be an object")
        if digest_obj(model_obj) != result_obj.get("model_digest"):
            raise InvalidInputError("model digest does not match canonical file")
    if "request.canonical.json" in files:
        request_obj = loads_strict_json(files["request.canonical.json"])
        if not isinstance(request_obj, dict):
            raise InvalidInputError("request.canonical.json must be an object")
        if digest_obj(request_obj) != result_obj.get("request_digest"):
            raise InvalidInputError("request digest does not match canonical file")

    lean_files = [
        name
        for name in files
        if name.startswith(GENERATED_PREFIX) and name.endswith(GENERATED_SUFFIX)
    ]
    if result_obj.get("status") == "verified":
        if len(lean_files) != 1:
            raise InvalidInputError("verified bundle must contain exactly one generated module")
        lean_name = lean_files[0]
        module = lean_name[: -len(GENERATED_SUFFIX)]
        if result_obj.get("generated_module") != module:
            raise InvalidInputError("generated_module name mismatch")
        if result_obj.get("generated_module_digest") != digest_bytes(files[lean_name]):
            raise InvalidInputError("generated_module_digest mismatch")
        for required in (
            "lean-build.log",
            "generated-module.log",
            "axiom-report.txt",
            "environment.json",
        ):
            if required not in files:
                raise InvalidInputError(f"missing required evidence file: {required}")
        if result_obj.get("axiom_audit") != "passed":
            raise InvalidInputError("verified status requires axiom_audit=passed")
        if result_obj.get("build_exit_code") != 0:
            raise InvalidInputError("verified status requires build_exit_code=0")
        if result_obj.get("axiom_audit_exit_code") != 0:
            raise InvalidInputError("verified status requires axiom_audit_exit_code=0")
    elif lean_files:
        if len(lean_files) != 1:
            raise InvalidInputError("bundle must contain at most one generated module")
        lean_name = lean_files[0]
        module = lean_name[: -len(GENERATED_SUFFIX)]
        if result_obj.get("generated_module") not in (None, module):
            raise InvalidInputError("generated_module name mismatch")
        digest = result_obj.get("generated_module_digest")
        if digest is not None and digest != digest_bytes(files[lean_name]):
            raise InvalidInputError("generated_module_digest mismatch")

    return result_obj
