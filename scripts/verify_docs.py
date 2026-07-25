#!/usr/bin/env python3
"""Verify documentation commands and claim hygiene for CI."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROHIBITED = [
    r"production[- ]ready",
    r"enterprise-grade",
    r"100M\+",
    r"distributed verification",
    r"encryption",
]

# Mentions that withdraw / quarantine / reject the phrase are allowed.
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


def check_prohibited(path: Path, text: str, errors: list[str]) -> None:
    for pat in PROHIBITED:
        for m in re.finditer(pat, text, flags=re.I):
            window = text[max(0, m.start() - 100) : m.end() + 100]
            if not ALLOW_NEAR.search(window):
                errors.append(f"{path}: prohibited claim language near: {m.group(0)!r}")


def check_stale_install_paths(path: Path, text: str, errors: list[str]) -> None:
    # Active docs must not instruct installing FormalVerifML at lean/FormalVerifML.
    if re.search(r"(?i)(?:cd|pip install|PYTHONPATH).{0,40}formal_verif_ml", text):
        errors.append(f"{path}: stale formal_verif_ml install instruction")
    if re.search(r"(?i)lake\s+build.{0,40}lean/FormalVerifML", text):
        errors.append(f"{path}: stale lean/FormalVerifML build path")


def main() -> int:
    errors: list[str] = []
    for path in ACTIVE_DOCS:
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        check_prohibited(path, text, errors)
        check_stale_install_paths(path, text, errors)

    for required in [
        ROOT / "docs" / "assurance-boundary.md",
        ROOT / "docs" / "trusted-computing-base.md",
        ROOT / "docs" / "threat-model.md",
        ROOT / "docs" / "limitations.md",
        ROOT / "docs" / "model-format.md",
        ROOT / "docs" / "release-process.md",
        ROOT / "CHANGELOG.md",
        ROOT / "CITATION.cff",
        ROOT / "experimental" / "legacy-formalverifml" / "WARNING.md",
    ]:
        if not required.exists():
            errors.append(f"missing {required}")

    for doc, needles in REQUIRED_LINKS:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"{doc}: missing expected link/text {needle!r}")

    # Supported Lean must not import experimental (real import lines only)
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

    if errors:
        print("DOC VERIFY FAILED", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        return 1
    print("DOC VERIFY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
