from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from leanverifier.cli import main
from leanverifier.errors import EXIT_INVALID, EXIT_NOT_CERTIFIED, EXIT_VERIFIED
from leanverifier.evidence import inspect_bundle
from leanverifier.generator import write_module_atomic
from leanverifier.manifest import load_model_path, load_request_path, parse_model
from tests.fixtures.builders import make_model, make_request


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _lake_project_ready(repo: Path) -> bool:
    """True when lake is on PATH and LeanVerifier oleans exist from a prior build."""
    if shutil.which("lake") is None:
        return False
    build_lib = repo / ".lake" / "build" / "lib" / "lean"
    if not build_lib.is_dir():
        return False
    return (
        any(build_lib.rglob("LeanVerifier*.olean")) or (build_lib / "LeanVerifier.olean").is_file()
    )


def test_positive_validate_and_generate(tmp_path: Path) -> None:
    model = make_model([1, 0], 0)
    # score at [2,0] = 2; eps=1/10; l1=1 → margin ok
    req = make_request(model, [2, 0], "1/10")
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    mp.write_text(json.dumps(model), encoding="utf-8")
    rp.write_text(json.dumps(req), encoding="utf-8")
    assert main(["validate-model", str(mp)]) == EXIT_VERIFIED
    assert main(["validate-request", str(rp)]) == EXIT_VERIFIED
    out = tmp_path / "out"
    out.mkdir()
    assert (
        main(["generate", "--model", str(mp), "--request", str(rp), "--out-dir", str(out)])
        == EXIT_VERIFIED
    )
    lean_files = list(out.glob("GeneratedModel_*.lean"))
    assert len(lean_files) == 1
    text = lean_files[0].read_text(encoding="utf-8")
    assert "affine_margin_robustness" in text
    assert "experimental" not in text


def test_negative_insufficient_margin(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [1], "2")  # |1| > 2*1 ? no
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    out = tmp_path / "evidence"
    mp.write_text(json.dumps(model), encoding="utf-8")
    rp.write_text(json.dumps(req), encoding="utf-8")
    code = main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
    assert code == EXIT_NOT_CERTIFIED
    result = json.loads((out / "verification-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "not-certified"
    assert result["exit_code"] == EXIT_NOT_CERTIFIED
    assert (out / "checksums.sha256").is_file()
    assert (out / "model.canonical.json").is_file()
    assert inspect_bundle(out)["status"] == "not-certified"


def test_negative_invalid_model(tmp_path: Path) -> None:
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    out = tmp_path / "evidence"
    mp.write_text('{"schema_version":"leanverifier.model.v1"}', encoding="utf-8")
    rp.write_text("{}", encoding="utf-8")
    code = main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
    assert code == EXIT_INVALID
    result = json.loads((out / "verification-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "invalid-input"
    assert result["axiom_audit"] == "not-run"
    assert inspect_bundle(out)["status"] == "invalid-input"


def test_positive_fixture_files() -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = root / "examples" / "affine_binary" / "model.json"
    request_path = root / "examples" / "affine_binary" / "request.json"
    assert model_path.exists()
    m = parse_model(json.loads(model_path.read_text(encoding="utf-8")))
    assert m.dimension == 3
    assert request_path.exists()


@pytest.mark.skipif(shutil.which("lake") is None, reason="lake not on PATH")
def test_generated_example_typechecks(tmp_path: Path) -> None:
    """Generated Fin-pattern vectors must elaborate under the pinned toolchain."""
    root = _repo_root()
    if not _lake_project_ready(root):
        pytest.skip("LeanVerifier Lake build artifacts missing; run `lake build` first")
    model = load_model_path(root / "examples" / "affine_binary" / "model.json")
    request = load_request_path(root / "examples" / "affine_binary" / "request.json")
    out = tmp_path.resolve()
    lean_path = write_module_atomic(model, request, out)
    proc = subprocess.run(
        ["lake", "env", "lean", str(lean_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        shell=False,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("lake") is None, reason="lake not on PATH")
def test_positive_verify_evidence_bundle(tmp_path: Path) -> None:
    """Full positive verify path with required evidence bundle (F-005 / F-006)."""
    root = _repo_root()
    if not _lake_project_ready(root):
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
    result = json.loads((out / "verification-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "verified"
    assert result["exit_code"] == 0
    assert result["claim"] == "local_linf_binary_classification_robustness"
    assert result["axiom_audit"] == "passed"
    assert result["build_exit_code"] == 0
    assert result["axiom_audit_exit_code"] == 0
    lean_files = list(out.glob("GeneratedModel_*.lean"))
    assert len(lean_files) == 1
    assert (out / "lean-build.log").is_file()
    assert (out / "generated-module.log").is_file()
    assert (out / "axiom-report.txt").is_file()
    assert (out / "environment.json").is_file()
    assert (out / "checksums.sha256").is_file()
    assert (out / "model.canonical.json").is_file()
    assert (out / "request.canonical.json").is_file()
    inspected = inspect_bundle(out)
    assert inspected["generated_module_digest"] == result["generated_module_digest"]
    assert main(["inspect", str(out)]) == EXIT_VERIFIED
