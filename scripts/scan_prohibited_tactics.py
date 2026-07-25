#!/usr/bin/env python3
"""Fail if supported sources use prohibited native/bitvector decide tactics.

Scans LeanVerifier, src, tests, and scripts. Supported paths must have zero hits.
Allowlist entries use relative POSIX-style paths.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Built from fragments so this file does not match its own pattern.
PATTERN = re.compile(
    r"\b" + "native" + r"_decide\b|"
    r"decide\s+\+" + "native" + r"|"
    r"\b" + "bv" + r"_decide\b"
)

# Relative paths (forward slashes) explicitly permitted. Keep empty for supported tree.
ALLOWLIST: frozenset[str] = frozenset()

SCAN_ROOTS = ("LeanVerifier", "src", "tests", "scripts")
TEXT_SUFFIXES = {".lean", ".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json"}


def _rel_posix(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def scan(repo: Path) -> list[str]:
    hits: list[str] = []
    for root_name in SCAN_ROOTS:
        root = repo / root_name
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix != "":
                continue
            rel = _rel_posix(repo, path)
            if rel in ALLOWLIST:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if PATTERN.search(line):
                    hits.append(f"{rel}:{i}:{line.strip()}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    args = parser.parse_args()
    repo: Path = args.repo.resolve()
    hits = scan(repo)
    if hits:
        print("PROHIBITED TACTIC SCAN FAILED", file=sys.stderr)
        for h in hits:
            print(f" - {h}", file=sys.stderr)
        return 1
    print("PROHIBITED TACTIC SCAN PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
