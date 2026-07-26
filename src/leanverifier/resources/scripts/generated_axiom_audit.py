#!/usr/bin/env python3
"""Audit LeanVerifier public and generated axioms.

Allowed: propext, Classical.choice, Quot.sound
Forbidden: sorryAx, Lean.trustCompiler, dedicated native axioms, any other name.

Library theorems cover every CLAIMS-referenced public declaration.
Pass --generated PATH.lean to audit eps_nonneg / margin_strict / certified on a
concrete generated module (required before status verified).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED = frozenset({"propext", "Classical.choice", "Quot.sound"})
FORBIDDEN_SUBSTRINGS = ("sorryAx", "Lean.trustCompiler", "sorry")

# CLAIMS-referenced public theorems (library).
LIBRARY_THEOREM_NAMES = [
    "LeanVerifier.AffineBinaryClassifier.affine_margin_robustness",
    "LeanVerifier.AffineBinaryClassifier.abs_score_sub_score_le",
    "LeanVerifier.AffineBinaryClassifier.classify_true_of_margin",
    "LeanVerifier.AffineBinaryClassifier.classify_false_of_margin",
    "LeanVerifier.AffineBinaryClassifier.repeated_weight_score_ones",
    "LeanVerifier.AffineBinaryClassifier.repeated_weight_classify_ones",
    "LeanVerifier.AffineBinaryClassifier.repeated_weight_score_focus",
    "LeanVerifier.AffineBinaryClassifier.repeated_weight_classify_focus",
]

GENERATED_LOCAL_NAMES = ("eps_nonneg", "margin_strict", "certified")

# Backward-compatible alias used by older call sites / docs.
THEOREM_NAMES = LIBRARY_THEOREM_NAMES


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


def run_library_axiom_prints(repo: Path) -> str:
    """Elaborate PublicAxioms.lean and capture all `#print axioms` info lines."""
    audit_lean = repo / "LeanVerifier" / "Audit" / "PublicAxioms.lean"
    if not audit_lean.is_file():
        raise RuntimeError(f"missing public axiom inventory: {audit_lean}")
    proc = subprocess.run(
        ["lake", "env", "lean", str(audit_lean.resolve())],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=600,
        shell=False,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "depends on axioms" not in out and "does not depend" not in out:
        raise RuntimeError(f"Failed to elaborate PublicAxioms.lean:\n{out}")
    return out


def run_generated_axiom_prints(repo: Path, generated_lean: Path) -> str:
    """Elaborate a generated module copy that also `#print axioms` local decls."""
    src = generated_lean.read_text(encoding="utf-8")
    if not src.rstrip().endswith(f"end {generated_lean.stem}"):
        # Still append prints; namespace may already be closed.
        pass
    prints = "\n".join(f"#print axioms {name}" for name in GENERATED_LOCAL_NAMES)
    # Insert prints before the final `end ModuleName` so names resolve in-namespace.
    end_marker = f"end {generated_lean.stem}"
    if end_marker in src:
        body = src.replace(end_marker, prints + "\n\n" + end_marker, 1)
    else:
        body = src + "\n" + prints + "\n"

    with tempfile.TemporaryDirectory(prefix="lv-axiom-gen-") as tmp:
        tmp_path = Path(tmp)
        audit_file = tmp_path / generated_lean.name
        audit_file.write_text(body, encoding="utf-8")
        proc = subprocess.run(
            ["lake", "env", "lean", str(audit_file.resolve())],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=600,
            shell=False,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to elaborate generated axiom audit for {generated_lean}:\n{out}"
            )
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


def parse_axiom_blocks(output: str) -> dict[str, set[str]]:
    """Map declaration name → axiom set from one or more `#print axioms` blocks."""
    results: dict[str, set[str]] = {}
    # Match both "foo depends on axioms: [...]" and "foo does not depend on any axioms"
    for m in re.finditer(
        r"'([^']+)'\s+(depends on axioms:\s*\[[^\]]*\]|does not depend on any axioms)",
        output,
    ):
        name = m.group(1)
        results[name] = parse_axioms(m.group(0))
    # Unquoted name form (some Lean builds)
    for m in re.finditer(
        r"(?m)^([A-Za-z0-9_'.]+)\s+(depends on axioms:\s*\[[^\]]*\]|does not depend on any axioms)",
        output,
    ):
        name = m.group(1)
        if name not in results:
            results[name] = parse_axioms(m.group(0))
    return results


def check_axiom_set(label: str, axioms: set[str], errors: list[str]) -> None:
    for bad in FORBIDDEN_SUBSTRINGS:
        if any(bad in a for a in axioms):
            errors.append(f"{label}: forbidden axiom material ({bad})")
    extra = axioms - ALLOWED
    if extra:
        errors.append(f"{label}: disallowed axioms {sorted(extra)}")


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
        prohibited = re.compile(
            r"\b" + "native" + r"_decide\b|"
            r"decide\s+\+" + "native" + r"|"
            r"\b" + "bv" + r"_decide\b"
        )
        if prohibited.search(stripped):
            errors.append(f"prohibited decide tactic in {path}")
    return errors


def audit_library_theorems(repo: Path, errors: list[str]) -> str:
    """Run `#print axioms` via PublicAxioms.lean; fall back to per-decl prints."""
    try:
        out = run_library_axiom_prints(repo)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return ""
    blocks = parse_axiom_blocks(out)
    for thm in LIBRARY_THEOREM_NAMES:
        short = thm.rsplit(".", 1)[-1]
        found = None
        for key, axioms in blocks.items():
            if key == thm or key.endswith("." + short) or key == short:
                found = (key, axioms)
                break
        if found is None:
            # Fallback: individual lake env lean -e (may be flaky on some hosts)
            try:
                single = run_print_axioms(repo, thm)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                continue
            axioms = parse_axioms(single)
            check_axiom_set(thm, axioms, errors)
            continue
        check_axiom_set(thm, found[1], errors)
    return out


def audit_generated_file(repo: Path, generated: Path, errors: list[str]) -> str:
    # Source-scan the concrete module (custom axiom / sorry / prohibited tactics).
    single_root_errors: list[str] = []
    text = generated.read_text(encoding="utf-8")
    if re.search(r"(?m)^\s*axiom\s+", text):
        single_root_errors.append(f"custom axiom declaration in {generated}")
    stripped = re.sub(r"/-.*?-/", "", text, flags=re.S)
    stripped = re.sub(r"--[^\n]*", "", stripped)
    if re.search(r"\bsorry\b", stripped):
        single_root_errors.append(f"sorry in {generated}")
    if re.search(r"\badmit\b", stripped):
        single_root_errors.append(f"admit in {generated}")
    if "Lean.trustCompiler" in stripped:
        single_root_errors.append(f"Lean.trustCompiler in {generated}")
    prohibited = re.compile(
        r"\b" + "native" + r"_decide\b|"
        r"decide\s+\+" + "native" + r"|"
        r"\b" + "bv" + r"_decide\b"
    )
    if prohibited.search(stripped):
        single_root_errors.append(f"prohibited decide tactic in {generated}")
    errors.extend(single_root_errors)

    try:
        out = run_generated_axiom_prints(repo, generated)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
        return ""
    blocks = parse_axiom_blocks(out)
    # Match local names and optionally Module.local
    stem = generated.stem
    for local in GENERATED_LOCAL_NAMES:
        candidates = [local, f"{stem}.{local}"]
        found = None
        for key, axioms in blocks.items():
            short = key.rsplit(".", 1)[-1]
            if key in candidates or short == local:
                found = (key, axioms)
                break
        if found is None:
            # If source already failed, missing prints are secondary.
            if not single_root_errors:
                errors.append(
                    f"generated {generated.name}: missing #print axioms output for {local}"
                )
            continue
        check_axiom_set(f"{generated.name}:{found[0]}", found[1], errors)
    return out


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
    parser.add_argument(
        "--generated",
        type=Path,
        action="append",
        default=[],
        help="Generated .lean module to audit (eps_nonneg/margin_strict/certified). Repeatable.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write full axiom print output",
    )
    args = parser.parse_args()
    repo: Path = args.repo.resolve()

    errors = audit_source_tree(repo)
    report_chunks: list[str] = []

    if not args.skip_lean:
        report_chunks.append(audit_library_theorems(repo, errors))
        for gen in args.generated:
            gen_path = gen.resolve()
            if not gen_path.is_file():
                errors.append(f"generated module not found: {gen_path}")
                continue
            report_chunks.append(
                f"=== GENERATED {gen_path.name} ===\n"
                + audit_generated_file(repo, gen_path, errors)
            )

    report_text = "\n".join(report_chunks)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")

    if errors:
        print("AXIOM AUDIT FAILED", file=sys.stderr)
        for e in errors:
            print(f" - {e}", file=sys.stderr)
        return 5
    print("AXIOM AUDIT PASSED")
    if args.generated:
        print(f"Generated modules audited: {len(args.generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
