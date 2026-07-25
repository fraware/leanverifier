#!/usr/bin/env python3
"""Audit LeanVerifier public axioms.

Allowed: propext, Classical.choice, Quot.sound
Forbidden: sorryAx, Lean.trustCompiler, any other axiom name.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ALLOWED = frozenset({"propext", "Classical.choice", "Quot.sound"})
FORBIDDEN_SUBSTRINGS = ("sorryAx", "Lean.trustCompiler", "sorry")

THEOREM_NAMES = [
    "LeanVerifier.AffineBinaryClassifier.affine_margin_robustness",
    "LeanVerifier.AffineBinaryClassifier.abs_score_sub_score_le",
    "LeanVerifier.AffineBinaryClassifier.classify_true_of_margin",
    "LeanVerifier.AffineBinaryClassifier.classify_false_of_margin",
]


def run_print_axioms(repo: Path, theorem: str) -> str:
    cmd = [
        "lake",
        "env",
        "lean",
        "-e",
        f"#print axioms {theorem}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=600,
        shell=False,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "axioms" not in out.lower():
        raise RuntimeError(f"Failed to print axioms for {theorem}:\n{out}")
    return out


def parse_axioms(output: str) -> set[str]:
    # Typical: "'thm' depends on axioms: [propext, Classical.choice, Quot.sound]"
    # or "'thm' does not depend on any axioms"
    if re.search(r"does not depend on any axioms", output):
        return set()
    m = re.search(r"depends on axioms:\s*\[(.*?)\]", output, re.S)
    if not m:
        # Fallback: collect known axiom identifiers present after the phrase
        m2 = re.search(r"depends on axioms:\s*(.*)", output)
        if not m2:
            return set()
        blob = m2.group(1)
    else:
        blob = m.group(1)
    parts = [p.strip().strip("'\"") for p in re.split(r"[,\s]+", blob) if p.strip()]
    return {p for p in parts if p and p not in {"]", "["}}


def audit_source_tree(repo: Path) -> list[str]:
    errors: list[str] = []
    roots = [repo / "LeanVerifier", repo / "LeanVerifier.lean"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.lean"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?m)^\s*axiom\s+", text):
            errors.append(f"custom axiom declaration in {path}")
        # Strip block/line comments before scanning for sorry/admit/trust
        stripped = re.sub(r"/-.*?-/", "", text, flags=re.S)
        stripped = re.sub(r"--[^\n]*", "", stripped)
        if re.search(r"\bsorry\b", stripped):
            errors.append(f"sorry in {path}")
        if re.search(r"\badmit\b", stripped):
            errors.append(f"admit in {path}")
        if "Lean.trustCompiler" in stripped:
            errors.append(f"Lean.trustCompiler in {path}")
        if re.search(r"import\s+.*experimental", stripped):
            errors.append(f"experimental import in {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    parser.add_argument(
        "--skip-lean",
        action="store_true",
        help="Only scan sources; skip lake env lean #print axioms",
    )
    args = parser.parse_args()
    repo: Path = args.repo.resolve()

    errors = audit_source_tree(repo)
    if not args.skip_lean:
        for thm in THEOREM_NAMES:
            try:
                out = run_print_axioms(repo, thm)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                continue
            for bad in FORBIDDEN_SUBSTRINGS:
                if bad in out and "forbidden" not in out:
                    # 'sorry' may appear in help text; check axiom lists specifically
                    axioms = parse_axioms(out)
                    if any(bad in a for a in axioms) or (bad in out and "depends on axioms" in out):
                        errors.append(f"{thm}: forbidden axiom material ({bad})")
            axioms = parse_axioms(out)
            extra = axioms - ALLOWED
            if extra:
                errors.append(f"{thm}: disallowed axioms {sorted(extra)}")

    if errors:
        print("AXIOM AUDIT FAILED", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        return 5
    print("AXIOM AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
