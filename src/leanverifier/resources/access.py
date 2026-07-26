"""importlib.resources accessors, hash verification, and Lean project materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from functools import lru_cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final

from leanverifier.errors import InvalidInputError

_PACKAGE: Final = "leanverifier.resources"
_HASHES_NAME: Final = "hashes.json"
_LEAN_PREFIX: Final = "lean-project/"
_SCHEMA_PREFIX: Final = "schemas/"
_AUDIT_SCRIPT: Final = "scripts/generated_axiom_audit.py"


def resource_root() -> Traversable:
    """Traversable root of packaged resources."""
    return files(_PACKAGE)


def _read_bytes(relative: str) -> bytes:
    node = resource_root().joinpath(*relative.split("/"))
    return node.read_bytes()


def _read_text(relative: str) -> str:
    return _read_bytes(relative).decode("utf-8")


def _sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def recorded_hashes() -> dict[str, str]:
    """Return the committed resource digest map."""
    raw = json.loads(_read_text(_HASHES_NAME))
    if not isinstance(raw, dict):
        raise InvalidInputError("resource hashes.json root must be an object")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise InvalidInputError("resource hashes.json entries must be strings")
        out[key] = value
    return out


@lru_cache(maxsize=1)
def verify_resource_hashes() -> dict[str, str]:
    """Verify every recorded packaged file matches its digest; return the map."""
    expected = recorded_hashes()
    if not expected:
        raise InvalidInputError("resource hashes.json is empty")
    for relative, digest in expected.items():
        actual = _sha256_digest(_read_bytes(relative))
        if actual != digest:
            raise InvalidInputError(
                f"packaged resource digest mismatch for {relative}: expected {digest}, got {actual}"
            )
    return expected


def load_schema_json(name: str) -> dict[str, Any]:
    """Load a JSON schema object from packaged resources."""
    hashes = verify_resource_hashes()
    relative = f"{_SCHEMA_PREFIX}{name}"
    if relative not in hashes:
        raise InvalidInputError(f"schema not in resource hashes: {name}")
    data = json.loads(_read_text(relative))
    if not isinstance(data, dict):
        raise InvalidInputError(f"schema root must be object: {name}")
    return data


def read_toolchain() -> str:
    """Return the packaged lean-toolchain pin string."""
    verify_resource_hashes()
    return _read_text(f"{_LEAN_PREFIX}lean-toolchain").strip()


def _default_cache_root() -> Path:
    override = os.environ.get("LEANVERIFIER_CACHE")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "leanverifier"


def _aggregate_digest(hashes: dict[str, str]) -> str:
    payload = "\n".join(f"{k}:{hashes[k]}" for k in sorted(hashes))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def materialize_lean_project(*, cache_root: Path | None = None) -> Path:
    """Copy the packaged Lake project into an isolated cache after hash checks.

    Returns the materialized project root. Subsequent calls with the same
    resource digests reuse the cache directory.
    """
    hashes = verify_resource_hashes()
    lean_hashes = {k: v for k, v in hashes.items() if k.startswith(_LEAN_PREFIX)}
    if not lean_hashes:
        raise InvalidInputError("no lean-project resources recorded in hashes.json")

    root = cache_root if cache_root is not None else _default_cache_root()
    digest = _aggregate_digest(lean_hashes)
    dest = root / "lean-project" / digest
    marker = dest / ".leanverifier-resources-ok"
    if dest.is_dir() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="lv-lean-mat-", dir=str(dest.parent)))
    try:
        for relative, expected in lean_hashes.items():
            data = _read_bytes(relative)
            actual = _sha256_digest(data)
            if actual != expected:
                raise InvalidInputError(
                    f"packaged resource digest mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )
            rel_path = relative[len(_LEAN_PREFIX) :]
            target = tmp / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (tmp / ".leanverifier-resources-ok").write_text(digest + "\n", encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def write_axiom_audit_script(dest_dir: Path) -> Path:
    """Write the packaged axiom-audit entrypoint into *dest_dir* after hash checks."""
    hashes = verify_resource_hashes()
    expected = hashes.get(_AUDIT_SCRIPT)
    if expected is None:
        raise InvalidInputError(f"missing recorded hash for {_AUDIT_SCRIPT}")
    data = _read_bytes(_AUDIT_SCRIPT)
    actual = _sha256_digest(data)
    if actual != expected:
        raise InvalidInputError(
            f"packaged resource digest mismatch for {_AUDIT_SCRIPT}: "
            f"expected {expected}, got {actual}"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "generated_axiom_audit.py"
    path.write_bytes(data)
    return path
