"""Evidence bundle and inspect command tests (F-005 / F-006)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leanverifier.cli import main
from leanverifier.errors import EXIT_DIGEST, EXIT_INVALID, EXIT_NOT_CERTIFIED, EXIT_VERIFIED
from leanverifier.evidence import inspect_bundle
from tests.fixtures.builders import make_model, make_request


def _write_pair(tmp_path: Path, model: dict, req: dict) -> tuple[Path, Path, Path]:
    mp = tmp_path / "model.json"
    rp = tmp_path / "request.json"
    out = tmp_path / "evidence"
    mp.write_text(json.dumps(model), encoding="utf-8")
    rp.write_text(json.dumps(req), encoding="utf-8")
    return mp, rp, out


def test_verify_requires_output(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [2], "1/10")
    mp, rp, _out = _write_pair(tmp_path, model, req)
    with pytest.raises(SystemExit):
        main(["verify", "--model", str(mp), "--request", str(rp)])


def test_failure_bundle_valid_and_inspectable(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [1], "2")
    mp, rp, out = _write_pair(tmp_path, model, req)
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_NOT_CERTIFIED
    )
    result = inspect_bundle(out)
    assert result["status"] == "not-certified"
    assert result["evidence_bundle_digest"].startswith("sha256:")
    assert result["generator_version"]
    assert result["lean_toolchain"]
    assert result["mathlib_revision"]
    assert result["axiom_audit"] == "not-run"
    assert result["build_exit_code"] is None


def test_inspect_rejects_checksum_tampering(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [1], "2")
    mp, rp, out = _write_pair(tmp_path, model, req)
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_NOT_CERTIFIED
    )
    log = out / "lean-build.log"
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(Exception, match="checksum mismatch"):
        inspect_bundle(out)
    assert main(["inspect", str(out)]) == EXIT_INVALID


def test_inspect_rejects_missing_checksums(tmp_path: Path) -> None:
    model = make_model([1], 0)
    req = make_request(model, [1], "2")
    mp, rp, out = _write_pair(tmp_path, model, req)
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_NOT_CERTIFIED
    )
    (out / "checksums.sha256").unlink()
    with pytest.raises(Exception, match="missing checksums"):
        inspect_bundle(out)


def test_digest_mismatch_writes_bundle(tmp_path: Path) -> None:
    model = make_model([1, 1, -2], 0)
    req = make_request(model, [2, 0, 0], "1/10")
    model["weights"][0] = "2"
    mp, rp, out = _write_pair(tmp_path, model, req)
    assert (
        main(["verify", "--model", str(mp), "--request", str(rp), "--output", str(out)])
        == EXIT_DIGEST
    )
    result = inspect_bundle(out)
    assert result["status"] == "digest-mismatch"
    assert main(["inspect", str(out)]) == EXIT_VERIFIED
