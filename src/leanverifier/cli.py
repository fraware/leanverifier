"""CLI entrypoints for LeanVerifier."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from leanverifier.canonicalize import margin_ok
from leanverifier.errors import (
    EXIT_AXIOM,
    EXIT_BUILD,
    EXIT_DIGEST,
    EXIT_INTERNAL,
    EXIT_INVALID,
    EXIT_NOT_CERTIFIED,
    EXIT_TIMEOUT,
    EXIT_VERIFIED,
    AxiomFailureError,
    BuildFailureError,
    DigestMismatchError,
    InvalidInputError,
    LeanVerifierError,
    NotCertifiedError,
    TimeoutFailureError,
)
from leanverifier.generator import write_module_atomic
from leanverifier.limits import CODE_TIMEOUT, MAX_BUILD_LOG_BYTES
from leanverifier.manifest import load_model_path, load_request_path
from leanverifier.result import VerificationResult

DEFAULT_TIMEOUT_S = 600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_toolchain(repo: Path) -> str:
    return (repo / "lean-toolchain").read_text(encoding="utf-8").strip()


def _strip_env() -> dict[str, str]:
    """Minimal subprocess environment — secrets and unrelated vars are dropped."""
    keep = ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMP", "TEMP", "LEAN_PATH"]
    env = {k: v for k, v in os.environ.items() if k in keep or k.startswith("ELAN")}
    env["LEANVERIFIER_OFFLINE"] = "1"
    return env


def _truncate_log(text: str, *, limit: int = MAX_BUILD_LOG_BYTES) -> str:
    """Retain at most *limit* UTF-8 bytes of log text (F-011 build-log bound)."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text
    omitted = len(raw) - limit
    prefix = f"[truncated {omitted} bytes]\n"
    prefix_b = prefix.encode("utf-8")
    budget = max(0, limit - len(prefix_b))
    truncated = raw[-budget:] if budget else b""
    # Avoid splitting a multibyte character at the front of the retained tail.
    decoded = truncated.decode("utf-8", errors="ignore")
    out = prefix + decoded
    # Hard cap even if prefix accounting races with replacement chars.
    out_b = out.encode("utf-8")
    if len(out_b) > limit:
        out = out_b[:limit].decode("utf-8", errors="ignore")
    return out


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """Best-effort kill of the process and its children (F-014 timeout)."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            # /T kills the tree rooted at this PID.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
                timeout=30,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


def _run_checked(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
    failure: type[LeanVerifierError],
) -> None:
    """Run a subprocess with timeout and process-group kill on expiry."""
    if sys.platform == "win32":
        proc: subprocess.Popen[str] = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=_strip_env(),
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=_strip_env(),
            text=True,
            start_new_session=True,
        )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        detail = _truncate_log((stdout or "") + (stderr or "") + f"\n{exc}")
        raise TimeoutFailureError(detail, code=CODE_TIMEOUT) from exc

    combined = _truncate_log((stdout or "") + (stderr or ""))
    if proc.returncode != 0:
        raise failure(combined)


def cmd_validate_model(path: Path) -> int:
    model = load_model_path(path)
    print(json.dumps({"ok": True, "digest": model.digest}, indent=2))
    return EXIT_VERIFIED


def cmd_validate_request(path: Path) -> int:
    req = load_request_path(path)
    print(json.dumps({"ok": True, "digest": req.digest}, indent=2))
    return EXIT_VERIFIED


def cmd_generate(model_path: Path, request_path: Path, out_dir: Path) -> int:
    model = load_model_path(model_path)
    request = load_request_path(request_path)
    if request.model_digest != model.digest:
        raise DigestMismatchError("model digest mismatch")
    if not margin_ok(model.weights, model.bias, request.reference_input, request.epsilon):
        raise NotCertifiedError("insufficient margin")
    # Pass the user path without resolve() so symlink components remain visible.
    path = write_module_atomic(model, request, out_dir)
    print(json.dumps({"ok": True, "path": str(path)}, indent=2))
    return EXIT_VERIFIED


def _run_lake_build(repo: Path, timeout: int) -> None:
    _run_checked(
        ["lake", "build"],
        cwd=repo,
        timeout=timeout,
        failure=BuildFailureError,
    )


def _run_lean_file(repo: Path, lean_file: Path, timeout: int) -> None:
    _run_checked(
        ["lake", "env", "lean", str(lean_file.resolve())],
        cwd=repo,
        timeout=timeout,
        failure=BuildFailureError,
    )


def _run_axiom_audit(repo: Path, timeout: int) -> None:
    script = repo / "scripts" / "audit_axioms.py"
    _run_checked(
        [sys.executable, str(script), "--repo", str(repo)],
        cwd=repo,
        timeout=timeout,
        failure=AxiomFailureError,
    )


def cmd_verify(
    model_path: Path,
    request_path: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> int:
    """Verify only after margin, compile, and axiom audit succeed.

    Status ``verified`` / exit 0 is never returned without a successful Lean
    compile of the generated module and a passing axiom audit.
    """
    repo = _repo_root()
    toolchain = _read_toolchain(repo)
    try:
        model = load_model_path(model_path)
        request = load_request_path(request_path)
    except InvalidInputError as exc:
        result = VerificationResult(
            status="invalid-input",
            exit_code=EXIT_INVALID,
            model_digest="sha256:" + "0" * 64,
            request_digest="sha256:" + "0" * 64,
            toolchain=toolchain,
            message=f"{exc.code}: {exc}",
        )
        print(json.dumps(result.to_dict(), indent=2))
        return EXIT_INVALID

    if request.model_digest != model.digest:
        result = VerificationResult(
            status="digest-mismatch",
            exit_code=EXIT_DIGEST,
            model_digest=model.digest,
            request_digest=request.digest,
            toolchain=toolchain,
            message="request.model_digest does not match model",
        )
        print(json.dumps(result.to_dict(), indent=2))
        return EXIT_DIGEST

    if not margin_ok(model.weights, model.bias, request.reference_input, request.epsilon):
        result = VerificationResult(
            status="not-certified",
            exit_code=EXIT_NOT_CERTIFIED,
            model_digest=model.digest,
            request_digest=request.digest,
            toolchain=toolchain,
            message="insufficient strict margin",
        )
        print(json.dumps(result.to_dict(), indent=2))
        return EXIT_NOT_CERTIFIED

    workspace = Path(tempfile.mkdtemp(prefix="leanverifier-verify-"))
    try:
        lean_path = write_module_atomic(model, request, workspace)
        module_name = lean_path.stem
        started = time.monotonic()
        remaining = timeout
        _run_lake_build(repo, remaining)
        remaining = max(1, timeout - int(time.monotonic() - started))
        _run_lean_file(repo, lean_path, remaining)
        remaining = max(1, timeout - int(time.monotonic() - started))
        _run_axiom_audit(repo, remaining)
        result = VerificationResult(
            status="verified",
            exit_code=EXIT_VERIFIED,
            model_digest=model.digest,
            request_digest=request.digest,
            toolchain=toolchain,
            generated_module=module_name,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return EXIT_VERIFIED
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def cmd_inspect(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    from leanverifier.manifest import validate_result_obj

    validate_result_obj(data)
    print(json.dumps(data, indent=2))
    return EXIT_VERIFIED


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leanverifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate-model")
    v.add_argument("path", type=Path)

    r = sub.add_parser("validate-request")
    r.add_argument("path", type=Path)

    g = sub.add_parser("generate")
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--request", type=Path, required=True)
    g.add_argument("--out-dir", type=Path, required=True)

    ver = sub.add_parser("verify")
    ver.add_argument("--model", type=Path, required=True)
    ver.add_argument("--request", type=Path, required=True)
    ver.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)

    i = sub.add_parser("inspect")
    i.add_argument("path", type=Path)
    return p


def _format_error(exc: LeanVerifierError) -> str:
    return f"{exc.code}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "validate-model":
            return cmd_validate_model(args.path)
        if args.cmd == "validate-request":
            return cmd_validate_request(args.path)
        if args.cmd == "generate":
            return cmd_generate(args.model, args.request, args.out_dir)
        if args.cmd == "verify":
            return cmd_verify(args.model, args.request, timeout=args.timeout)
        if args.cmd == "inspect":
            return cmd_inspect(args.path)
        return EXIT_INTERNAL
    except NotCertifiedError as exc:
        print(_format_error(exc), file=sys.stderr)
        return EXIT_NOT_CERTIFIED
    except InvalidInputError as exc:
        print(_format_error(exc), file=sys.stderr)
        return EXIT_INVALID
    except BuildFailureError as exc:
        print(_format_error(exc), file=sys.stderr)
        return EXIT_BUILD
    except AxiomFailureError as exc:
        print(_format_error(exc), file=sys.stderr)
        return EXIT_AXIOM
    except DigestMismatchError as exc:
        print(_format_error(exc), file=sys.stderr)
        return EXIT_DIGEST
    except TimeoutFailureError as exc:
        print(_format_error(exc), file=sys.stderr)
        return EXIT_TIMEOUT
    except LeanVerifierError as exc:
        print(_format_error(exc), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
