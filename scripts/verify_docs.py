#!/usr/bin/env python3
"""Verify documentation claims, links, and executable commands for CI (F-018).

Modes:
  claims   — prohibited language + required assurance docs present
  links    — README/user-guide cross-links resolve
  commands — execute documented CLI commands from README / user guide
  all      — claims + links + commands (default)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROHIBITED = [
    r"production[- ]ready",
    r"enterprise-grade",
    r"100M\+",
    r"distributed verification",
    r"encryption",
]

ALLOW_NEAR = re.compile(
    r"withdrawn|unsupported|must not|quarantine|rejected|not claim|do not claim|"
    r"prohibited|historical|relocated|not release evidence",
    flags=re.I,
)

ACTIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "STATUS.md",
    ROOT / "CLAIMS.md",
    ROOT / "SECURITY.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "user_guide.md",
    ROOT / "docs" / "developer_guide.md",
    ROOT / "docs" / "assurance-boundary.md",
    ROOT / "docs" / "limitations.md",
    ROOT / "docs" / "threat-model.md",
    ROOT / "docs" / "trusted-computing-base.md",
    ROOT / "docs" / "model-format.md",
    ROOT / "docs" / "release-process.md",
    ROOT / "docs" / "program" / "admin-actions.md",
    ROOT / "docs" / "program" / "review-invitation.md",
    ROOT / "docs" / "program" / "weekly-review-template.md",
    ROOT / "experimental" / "README.md",
    ROOT / "experimental" / "legacy-formalverifml" / "WARNING.md",
]

REQUIRED_LINKS = [
    (ROOT / "README.md", ["STATUS.md", "CLAIMS.md", "SECURITY.md", "docs/user_guide.md"]),
    (ROOT / "docs" / "user_guide.md", ["model-format.md", "assurance-boundary.md"]),
]

REQUIRED_DOCS = [
    ROOT / "docs" / "assurance-boundary.md",
    ROOT / "docs" / "trusted-computing-base.md",
    ROOT / "docs" / "threat-model.md",
    ROOT / "docs" / "limitations.md",
    ROOT / "docs" / "model-format.md",
    ROOT / "docs" / "release-process.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CITATION.cff",
    ROOT / "experimental" / "legacy-formalverifml" / "WARNING.md",
]


def check_prohibited(path: Path, text: str, errors: list[str]) -> None:
    for pat in PROHIBITED:
        for m in re.finditer(pat, text, flags=re.I):
            window = text[max(0, m.start() - 100) : m.end() + 100]
            if not ALLOW_NEAR.search(window):
                errors.append(f"{path}: prohibited claim language near: {m.group(0)!r}")


def check_stale_install_paths(path: Path, text: str, errors: list[str]) -> None:
    if re.search(r"(?i)(?:cd|pip install|PYTHONPATH).{0,40}formal_verif_ml", text):
        errors.append(f"{path}: stale formal_verif_ml install instruction")
    if re.search(r"(?i)lake\s+build.{0,40}lean/FormalVerifML", text):
        errors.append(f"{path}: stale lean/FormalVerifML build path")


def verify_claims(errors: list[str]) -> None:
    for path in ACTIVE_DOCS:
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        check_prohibited(path, text, errors)
        check_stale_install_paths(path, text, errors)

    for required in REQUIRED_DOCS:
        if not required.exists():
            errors.append(f"missing {required}")

    import_experimental = re.compile(r"(?m)^\s*import\s+.*experimental")
    lean_files = (
        list((ROOT / "LeanVerifier").rglob("*.lean")) if (ROOT / "LeanVerifier").is_dir() else []
    )
    root_lean = ROOT / "LeanVerifier.lean"
    if root_lean.exists():
        lean_files.append(root_lean)
    for lean in lean_files:
        text = lean.read_text(encoding="utf-8")
        if import_experimental.search(text):
            errors.append(f"{lean}: imports experimental")

    claims = ROOT / "CLAIMS.md"
    if claims.is_file():
        text = claims.read_text(encoding="utf-8")
        if re.search(r"(?m)^\| LV-C00[1-6].*\|\s*Supported\s*\|", text):
            errors.append("CLAIMS.md: LV-C* must not use Supported until release checklist")


def verify_links(errors: list[str]) -> None:
    for doc, needles in REQUIRED_LINKS:
        if not doc.exists():
            errors.append(f"missing {doc}")
            continue
        text = doc.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"{doc}: missing expected link/text {needle!r}")
            else:
                # Resolve relative markdown links when needle looks like a path.
                candidate = (doc.parent / needle).resolve()
                if "/" in needle or needle.endswith(".md"):
                    if not candidate.exists() and not (ROOT / needle).exists():
                        errors.append(f"{doc}: broken link target {needle!r}")


def _cli() -> list[str]:
    """Always drive the CLI via the active interpreter (avoid stale PATH scripts)."""
    return [sys.executable, "-m", "leanverifier.cli"]


def verify_commands(errors: list[str]) -> None:
    """Execute documented validate / generate / verify / inspect commands (F-018)."""
    model = ROOT / "examples" / "affine_binary" / "model.json"
    request = ROOT / "examples" / "affine_binary" / "request.json"
    if not model.is_file() or not request.is_file():
        errors.append("missing examples/affine_binary fixtures for docs-commands")
        return

    cli = _cli()
    with tempfile.TemporaryDirectory(prefix="lv-docs-commands-") as tmp:
        tmp_path = Path(tmp)
        out_gen = tmp_path / "gen"
        out_gen.mkdir()
        evidence = tmp_path / "evidence"
        commands: list[tuple[str, list[str]]] = [
            ("validate-model", [*cli, "validate-model", str(model)]),
            ("validate-request", [*cli, "validate-request", str(request)]),
            (
                "generate",
                [
                    *cli,
                    "generate",
                    "--model",
                    str(model),
                    "--request",
                    str(request),
                    "--out-dir",
                    str(out_gen),
                ],
            ),
        ]
        for label, argv in commands:
            proc = subprocess.run(
                argv,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                shell=False,
                check=False,
            )
            if proc.returncode != 0:
                errors.append(
                    f"docs-commands {label} failed ({proc.returncode}): "
                    f"{(proc.stdout or '') + (proc.stderr or '')}"
                )

        # Full verify+inspect only in CI (or when explicitly opted in): materializing
        # the packaged Lake project and fetching mathlib is too heavy for default local docs.
        run_verify = shutil.which("lake") is not None and (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("LEANVERIFIER_DOCS_VERIFY") == "1"
        )
        if run_verify:
            proc = subprocess.run(
                [
                    *cli,
                    "verify",
                    "--model",
                    str(model),
                    "--request",
                    str(request),
                    "--output",
                    str(evidence),
                    "--timeout",
                    "1800",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=2400,
                shell=False,
                check=False,
            )
            if proc.returncode != 0:
                errors.append(
                    "docs-commands verify failed "
                    f"({proc.returncode}): {(proc.stdout or '') + (proc.stderr or '')}"
                )
            else:
                insp = subprocess.run(
                    [*cli, "inspect", str(evidence)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    shell=False,
                    check=False,
                )
                if insp.returncode != 0:
                    errors.append(
                        "docs-commands inspect failed "
                        f"({insp.returncode}): {(insp.stdout or '') + (insp.stderr or '')}"
                    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("claims", "links", "commands", "all"),
        default="all",
        help="Which documentation gate to run",
    )
    args = parser.parse_args(argv)
    errors: list[str] = []
    if args.mode in ("claims", "all"):
        verify_claims(errors)
    if args.mode in ("links", "all"):
        verify_links(errors)
    if args.mode in ("commands", "all"):
        verify_commands(errors)

    if errors:
        print(f"DOC VERIFY FAILED ({args.mode})", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        return 1
    print(f"DOC VERIFY PASSED ({args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
