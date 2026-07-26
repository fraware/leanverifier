#!/usr/bin/env python3
"""Materialize the packaged Lake project and warm ``lake build`` for CI.

Verify always builds against the *packaged* Lake tree under ``LEANVERIFIER_CACHE``
(not the checkout). After ``lean-action`` builds the checkout, seed ``.lake/packages``
from that tree so Mathlib is not recompiled from scratch within the 600s default
verify budget.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from leanverifier.resources import materialize_lean_project


def _seed_packages(dest_root: Path, seed_from: Path) -> None:
    src = seed_from / ".lake" / "packages"
    if not src.is_dir():
        print(f"no package seed at {src}", flush=True)
        return
    dest_lake = dest_root / ".lake"
    dest = dest_lake / "packages"
    dest_lake.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, symlinks=True)
    print(f"seeded packages from {src} -> {dest}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-from",
        type=Path,
        default=None,
        help="Checkout root with a warm .lake/packages tree (after lean-action)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Seconds allowed for lake build in the materialized tree",
    )
    args = parser.parse_args(argv)

    root = materialize_lean_project()
    print(f"materialized: {root}", flush=True)
    if args.seed_from is not None:
        _seed_packages(root, args.seed_from.resolve())

    if shutil.which("lake") is None:
        print("lake not on PATH", file=sys.stderr)
        return 1

    print(f"+ lake build (cwd={root}, timeout={args.timeout})", flush=True)
    try:
        subprocess.run(
            ["lake", "build"],
            cwd=root,
            check=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        print("lake build timed out while warming packaged cache", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"lake build failed with exit {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    print("warm packaged lake build ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
