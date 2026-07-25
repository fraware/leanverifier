"""Audit mutation table — real verify / audit / inspect failure paths (F-013)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from leanverifier.cli import main
from leanverifier.errors import (
    EXIT_AXIOM,
    EXIT_BUILD,
    EXIT_DIGEST,
    EXIT_INVALID,
    EXIT_NOT_CERTIFIED,
    EXIT_VERIFIED,
    InvalidInputError,
)
from leanverifier.evidence import inspect_bundle
from leanverifier.generator import render_module, write_module_atomic
from leanverifier.manifest import load_model_path, load_request_path, parse_model, parse_request
from tests.fixtures.builders import make_model, make_request

pytestmark = pytest.mark.mutation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _lake_ready(repo: Path) -> bool:
    if shutil.which("lake") is None:
        return False
    build_lib = repo / ".lake" / "build" / "lib" / "lean"
    if not build_lib.is_dir():
        return False
    return (
        any(build_lib.rglob("LeanVerifier*.olean")) or (build_lib / "LeanVerifier.olean").is_file()
    )


def _write_pair(tmp_path: Path, model: dict, req: dict) -> tuple[Path, Path, Path]:
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    out = tmp_path / "evidence"
    mp.write_text(json.dumps(model), encoding="utf-8")
    rp.write_text(json.dumps(req), encoding="utf-8")
    return mp, rp, out


def test_mutation_weight_breaks_digest_via_verify(tmp_path: Path) -> None:
    model = make_model([1, 1, -2], 0)
    req = make_request(model, [2, 0, 0], "1/10")
    model["weights"][0] = "2"
    mp, rp, out = _write_pair(tmp_path, model, req)
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_DIGEST
    )
    result = json.loads((out / "verification-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "digest-mismatch"
    assert inspect_bundle(out)["status"] == "digest-mismatch"


def test_mutation_bias_breaks_digest_via_verify(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [2], "1/10")
    model["bias"] = "1"
    mp, rp, out = _write_pair(tmp_path, model, req)
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_DIGEST
    )


def test_mutation_epsilon_uncertifies_via_verify(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [1], "2")
    mp, rp, out = _write_pair(tmp_path, model, req)
    code = main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
    assert code == EXIT_NOT_CERTIFIED
    assert inspect_bundle(out)["status"] == "not-certified"


def test_mutation_comment_not_injected_as_lean() -> None:
    model = parse_model(make_model([1], 0, comment="axiom Foo : True; sorry"))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    _name, source = render_module(model, req)
    assert "axiom Foo" not in source
    assert "sorry" not in source


@pytest.mark.lean
@pytest.mark.skipif(shutil.which("lake") is None, reason="lake not on PATH")
def test_mutation_import_removal_build_failure(tmp_path: Path) -> None:
    """Removing the robustness import must fail Lean elaboration (build path)."""
    root = _repo_root()
    if not _lake_ready(root):
        pytest.skip("LeanVerifier Lake build artifacts missing; run `lake build` first")
    model = load_model_path(root / "examples" / "affine_binary" / "model.json")
    request = load_request_path(root / "examples" / "affine_binary" / "request.json")
    out = tmp_path.resolve()
    lean_path = write_module_atomic(model, request, out)
    text = lean_path.read_text(encoding="utf-8")
    mutated = text.replace("import LeanVerifier.Theorem.AffineMarginRobustness", "")
    assert "import LeanVerifier.Theorem.AffineMarginRobustness" not in mutated
    lean_path.write_text(mutated, encoding="utf-8")
    proc = subprocess.run(
        ["lake", "env", "lean", str(lean_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        shell=False,
        check=False,
    )
    assert proc.returncode != 0, "import removal must fail elaboration"
    # Stable CLI exit code family for build failures.
    assert EXIT_BUILD == 4


@pytest.mark.lean
@pytest.mark.skipif(shutil.which("lake") is None, reason="lake not on PATH")
@pytest.mark.parametrize(
    "mutation,needle",
    [
        ("sorry", "sorry"),
        ("axiom", "axiom MutBad : True"),
        ("nd", "native" + "_decide"),
    ],
)
def test_mutation_axiom_insert_fails_audit(tmp_path: Path, mutation: str, needle: str) -> None:
    """Inserting sorry / axiom / prohibited decide tactic into a generated module fails audit."""
    root = _repo_root()
    if not _lake_ready(root):
        pytest.skip("LeanVerifier Lake build artifacts missing; run `lake build` first")
    model = load_model_path(root / "examples" / "affine_binary" / "model.json")
    request = load_request_path(root / "examples" / "affine_binary" / "request.json")
    out = tmp_path.resolve()
    lean_path = write_module_atomic(model, request, out)
    text = lean_path.read_text(encoding="utf-8")
    if mutation == "sorry":
        # Replace a proof body with sorry so #print axioms sees sorryAx.
        mutated = text.replace("try norm_num", "sorry", 1)
    elif mutation == "axiom":
        mutated = text.replace(
            f"end {lean_path.stem}",
            f"axiom MutBad : True\n\nend {lean_path.stem}",
            1,
        )
    else:
        mutated = text.replace("try norm_num", "native" + "_decide", 1)
    assert needle.split()[0] in mutated or needle in mutated
    lean_path.write_text(mutated, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_axioms.py"),
            "--repo",
            str(root),
            "--generated",
            str(lean_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        shell=False,
        check=False,
    )
    assert proc.returncode == EXIT_AXIOM, proc.stdout + proc.stderr


@pytest.mark.lean
@pytest.mark.skipif(shutil.which("lake") is None, reason="lake not on PATH")
def test_mutation_evidence_tamper_fails_inspect(tmp_path: Path) -> None:
    """Tampering with a verified evidence artifact must fail inspect (F-013 digest path)."""
    root = _repo_root()
    if not _lake_ready(root):
        pytest.skip("LeanVerifier Lake build artifacts missing; run `lake build` first")
    model_path = root / "examples" / "affine_binary" / "model.json"
    request_path = root / "examples" / "affine_binary" / "request.json"
    out = (tmp_path / "evidence").resolve()
    code = main(
        [
            "verify",
            "--model",
            str(model_path),
            "--request",
            str(request_path),
            "--output",
            str(out),
        ]
    )
    assert code == EXIT_VERIFIED
    report = out / "axiom-report.txt"
    report.write_text(report.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        inspect_bundle(out)
    assert main(["inspect", str(out)]) == EXIT_INVALID


def test_mutation_unit_evidence_checksum_tamper(tmp_path: Path) -> None:
    """Checksum / digest tamper without lake: forge a minimal failed bundle then corrupt it."""
    from leanverifier import __version__
    from leanverifier.evidence import persist_evidence_bundle
    from leanverifier.result import VerificationResult

    out = (tmp_path / "bundle").resolve()
    result = VerificationResult(
        status="not-certified",
        exit_code=EXIT_NOT_CERTIFIED,
        model_digest="sha256:" + "a" * 64,
        request_digest="sha256:" + "b" * 64,
        lean_toolchain="leanprover/lean4:v4.32.1",
        mathlib_revision="unknown",
        generator_version=__version__,
        evidence_bundle_digest="sha256:" + "0" * 64,
        axiom_audit="not-run",
        message="fixture",
    )
    persist_evidence_bundle(
        out,
        result=result,
        model_obj=None,
        request_obj=None,
        generated_lean=None,
        generated_module_name=None,
        lean_build_log="",
        generated_module_log="",
        axiom_report="",
        environment={"offline": True},
    )
    assert inspect_bundle(out)["status"] == "not-certified"
    (out / "lean-build.log").write_text("tampered-build-log\n", encoding="utf-8")
    with pytest.raises(InvalidInputError):
        inspect_bundle(out)
