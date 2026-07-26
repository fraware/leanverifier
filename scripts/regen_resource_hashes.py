#!/usr/bin/env python3
"""Regenerate leanverifier.resources hashes.json after editing packaged files.

Usage (from repo root)::

    python scripts/regen_resource_hashes.py

Also copies checkout ``schemas/`` mirrors into the packaged tree when present,
so docs/checkout copies stay aligned with runtime resources.

Digests are always computed over LF-normalized bytes so Windows checkouts with
``core.autocrlf`` cannot poison ``hashes.json`` (CI verifies LF blobs).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "src" / "leanverifier" / "resources"
SKIP_NAMES = frozenset({"hashes.json", "__init__.py", "access.py"})


def _canonical_bytes(data: bytes) -> bytes:
    """Normalize newlines to LF for stable cross-platform digests."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(data)).hexdigest()


def _write_lf(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(data))


def sync_schema_mirrors() -> None:
    src = ROOT / "schemas"
    dest = RES / "schemas"
    if not src.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.glob("*.json"):
        _write_lf(dest / path.name, path.read_bytes())


def sync_lean_mirrors() -> None:
    """Refresh packaged lean-project from checkout Lake layout when present."""
    dest = RES / "lean-project"
    mapping = [
        (ROOT / "LeanVerifier.lean", dest / "LeanVerifier.lean"),
        (ROOT / "lakefile.toml", dest / "lakefile.toml"),
        (ROOT / "lake-manifest.json", dest / "lake-manifest.json"),
        (ROOT / "lean-toolchain", dest / "lean-toolchain"),
    ]
    for src, target in mapping:
        if src.is_file():
            _write_lf(target, src.read_bytes())
    lean_src = ROOT / "LeanVerifier"
    lean_dest = dest / "LeanVerifier"
    if lean_src.is_dir():
        if lean_dest.exists():
            shutil.rmtree(lean_dest)
        for path in sorted(lean_src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(lean_src)
            _write_lf(lean_dest / rel, path.read_bytes())


def main() -> int:
    if not RES.is_dir():
        print(f"missing resources tree: {RES}", file=sys.stderr)
        return 1
    sync_schema_mirrors()
    sync_lean_mirrors()
    entries: dict[str, str] = {}
    for path in sorted(RES.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.suffix == ".pyc":
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(RES).as_posix()
        if rel.endswith(".py") and not rel.startswith("scripts/"):
            continue
        entries[rel] = _sha(path.read_bytes())
    out = RES / "hashes.json"
    out.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(entries)} hashes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
