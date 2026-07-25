"""Packaged resource integrity and materialization (F-007)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from leanverifier.errors import InvalidInputError
from leanverifier.resources import (
    load_schema_json,
    materialize_lean_project,
    read_toolchain,
    verify_resource_hashes,
    write_axiom_audit_script,
)
from leanverifier.resources.access import recorded_hashes


def test_resource_hashes_verify() -> None:
    hashes = verify_resource_hashes()
    assert "schemas/model.schema.json" in hashes
    assert "lean-project/lean-toolchain" in hashes
    assert "scripts/generated_axiom_audit.py" in hashes


def test_load_schema_via_resources() -> None:
    schema = load_schema_json("model.schema.json")
    assert schema["title"] == "LeanVerifierAffineBinaryModel"


def test_read_toolchain_via_resources() -> None:
    pin = read_toolchain()
    assert pin.startswith("leanprover/lean4:")


def test_materialize_lean_project(tmp_path: Path) -> None:
    root = materialize_lean_project(cache_root=tmp_path)
    assert (root / "lean-toolchain").is_file()
    assert (root / "lakefile.toml").is_file()
    assert (root / "LeanVerifier.lean").is_file()
    assert (root / "LeanVerifier" / "Model" / "AffineBinary.lean").is_file()
    assert (root / ".leanverifier-resources-ok").is_file()
    # Second call reuses the same digest directory.
    again = materialize_lean_project(cache_root=tmp_path)
    assert again == root


def test_write_axiom_audit_script(tmp_path: Path) -> None:
    path = write_axiom_audit_script(tmp_path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "AXIOM AUDIT" in text or "Allowed" in text


def test_checkout_schema_mirrors_match_packaged() -> None:
    """When running from a source checkout, top-level schemas/ must match package."""
    checkout = Path(__file__).resolve().parents[2]
    schemas = checkout / "schemas"
    if not schemas.is_dir():
        pytest.skip("not a full source checkout")
    hashes = recorded_hashes()
    for name in ("model.schema.json", "request.schema.json", "result.schema.json"):
        data = (schemas / name).read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        assert hashes[f"schemas/{name}"] == digest


def test_checkout_lean_mirrors_match_packaged() -> None:
    checkout = Path(__file__).resolve().parents[2]
    if not (checkout / "lean-toolchain").is_file():
        pytest.skip("not a full source checkout")
    hashes = recorded_hashes()
    mapping = {
        "lean-project/lean-toolchain": checkout / "lean-toolchain",
        "lean-project/lakefile.toml": checkout / "lakefile.toml",
        "lean-project/lake-manifest.json": checkout / "lake-manifest.json",
        "lean-project/LeanVerifier.lean": checkout / "LeanVerifier.lean",
    }
    for key, path in mapping.items():
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[key] == digest
    for lean in (checkout / "LeanVerifier").rglob("*.lean"):
        rel = lean.relative_to(checkout).as_posix()
        key = f"lean-project/{rel}"
        digest = "sha256:" + hashlib.sha256(lean.read_bytes()).hexdigest()
        assert hashes[key] == digest


def test_hashes_json_is_sorted_object() -> None:
    raw = json.loads(
        Path(__file__)
        .resolve()
        .parents[2]
        .joinpath("src/leanverifier/resources/hashes.json")
        .read_text(encoding="utf-8")
    )
    assert list(raw) == sorted(raw)


def test_missing_schema_rejected() -> None:
    with pytest.raises(InvalidInputError):
        load_schema_json("does-not-exist.schema.json")
