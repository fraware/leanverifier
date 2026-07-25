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
from typing import Any

from leanverifier import __version__
from leanverifier.canonicalize import digest_bytes, margin_ok
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
from leanverifier.evidence import (
    build_environment,
    inspect_bundle,
    persist_evidence_bundle,
    read_mathlib_revision,
)
from leanverifier.generator import write_module_atomic
from leanverifier.limits import CODE_TIMEOUT, MAX_BUILD_LOG_BYTES
from leanverifier.manifest import load_model_path, load_request_path
from leanverifier.resources import (
    materialize_lean_project,
    read_toolchain,
    write_axiom_audit_script,
)
from leanverifier.result import AxiomAuditStatus, VerificationResult

DEFAULT_TIMEOUT_S = 600
ZERO_DIGEST = "sha256:" + "0" * 64


def _strip_env() -> dict[str, str]:
    """Minimal subprocess environment — secrets and unrelated vars are dropped."""
    keep = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "USERNAME",
        "USER",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LEAN_PATH",
        "LAKE_CACHE",
        "ELAN_HOME",
        "APPDATA",
        "LOCALAPPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if k in keep or k.startswith("ELAN") or k.startswith("LEAN")
    }
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
    decoded = truncated.decode("utf-8", errors="ignore")
    out = prefix + decoded
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


def _run_captured(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> tuple[int, str]:
    """Run a subprocess with timeout, process-group kill, and truncated logs."""
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

    return proc.returncode, _truncate_log((stdout or "") + (stderr or ""))


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


def _emit_and_persist(
    *,
    output_dir: Path,
    result: VerificationResult,
    model_obj: dict[str, Any] | None,
    request_obj: dict[str, Any] | None,
    generated_lean: bytes | None,
    generated_module_name: str | None,
    lean_build_log: str,
    generated_module_log: str,
    axiom_report: str,
    environment: dict[str, Any],
) -> int:
    finalized = persist_evidence_bundle(
        output_dir,
        result=result,
        model_obj=model_obj,
        request_obj=request_obj,
        generated_lean=generated_lean,
        generated_module_name=generated_module_name,
        lean_build_log=lean_build_log,
        generated_module_log=generated_module_log,
        axiom_report=axiom_report,
        environment=environment,
    )
    print(json.dumps(finalized.to_dict(), indent=2))
    return finalized.exit_code


def cmd_verify(
    model_path: Path,
    request_path: Path,
    output_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
    argv: list[str] | None = None,
) -> int:
    """Verify only after margin, compile, and axiom audit succeed.

    Status ``verified`` / exit 0 is never returned without a successful Lean
    compile of the generated module and a passing axiom audit. An evidence
    bundle is always written under ``--output`` when that directory is writable.

    The Lake project is materialized from packaged resources into an isolated
    cache (F-007); the source checkout is not required on PYTHONPATH.
    """
    try:
        lean_toolchain = read_toolchain()
    except InvalidInputError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return EXIT_INVALID

    command_argv = (
        argv
        if argv is not None
        else [
            "leanverifier",
            "verify",
            "--model",
            str(model_path),
            "--request",
            str(request_path),
            "--output",
            str(output_dir),
            "--timeout",
            str(timeout),
        ]
    )
    started = time.monotonic()
    model_obj: dict[str, Any] | None = None
    request_obj: dict[str, Any] | None = None
    generated_lean: bytes | None = None
    generated_module_name: str | None = None
    lean_build_log = ""
    generated_module_log = ""
    axiom_report = ""
    build_exit: int | None = None
    axiom_exit: int | None = None
    axiom_status: AxiomAuditStatus = "not-run"
    mathlib_revision = "unknown"
    lean_root: Path | None = None

    def environment(*, duration_ms: int | None = None) -> dict[str, Any]:
        repo_for_env = lean_root if lean_root is not None else Path.cwd()
        return build_environment(
            repo=repo_for_env,
            lean_toolchain=lean_toolchain,
            mathlib_revision=mathlib_revision,
            generator_version=__version__,
            argv=command_argv,
            duration_ms=duration_ms,
        )

    def fail(
        *,
        status: str,
        exit_code: int,
        message: str,
        model_digest: str = ZERO_DIGEST,
        request_digest: str = ZERO_DIGEST,
    ) -> int:
        duration_ms = int((time.monotonic() - started) * 1000)
        result = VerificationResult(
            status=status,
            exit_code=exit_code,
            model_digest=model_digest,
            request_digest=request_digest,
            lean_toolchain=lean_toolchain,
            mathlib_revision=mathlib_revision,
            generator_version=__version__,
            evidence_bundle_digest=ZERO_DIGEST,
            build_exit_code=build_exit,
            axiom_audit_exit_code=axiom_exit,
            axiom_audit=axiom_status,
            message=message,
            generated_module=generated_module_name,
            generated_module_digest=(
                digest_bytes(generated_lean) if generated_lean is not None else None
            ),
            duration_ms=duration_ms,
        )
        return _emit_and_persist(
            output_dir=output_dir,
            result=result,
            model_obj=model_obj,
            request_obj=request_obj,
            generated_lean=generated_lean,
            generated_module_name=generated_module_name,
            lean_build_log=lean_build_log,
            generated_module_log=generated_module_log,
            axiom_report=axiom_report,
            environment=environment(duration_ms=duration_ms),
        )

    try:
        model = load_model_path(model_path)
        request = load_request_path(request_path)
    except InvalidInputError as exc:
        return fail(status="invalid-input", exit_code=EXIT_INVALID, message=str(exc))

    model_obj = model.raw
    request_obj = request.raw

    if request.model_digest != model.digest:
        return fail(
            status="digest-mismatch",
            exit_code=EXIT_DIGEST,
            message="request.model_digest does not match model",
            model_digest=model.digest,
            request_digest=request.digest,
        )

    if not margin_ok(model.weights, model.bias, request.reference_input, request.epsilon):
        return fail(
            status="not-certified",
            exit_code=EXIT_NOT_CERTIFIED,
            message="insufficient strict margin",
            model_digest=model.digest,
            request_digest=request.digest,
        )

    workspace = Path(tempfile.mkdtemp(prefix="leanverifier-verify-"))
    try:
        lean_root = materialize_lean_project()
        mathlib_revision = read_mathlib_revision(lean_root)
        audit_script = write_axiom_audit_script(workspace / "_packaged_scripts")
        lean_path = write_module_atomic(model, request, workspace)
        generated_module_name = lean_path.stem
        generated_lean = lean_path.read_bytes()

        remaining = max(1, timeout - int(time.monotonic() - started))
        build_exit, lean_build_log = _run_captured(
            ["lake", "build"],
            cwd=lean_root,
            timeout=remaining,
        )
        if build_exit != 0:
            return fail(
                status="build-failure",
                exit_code=EXIT_BUILD,
                message="lake build failed",
                model_digest=model.digest,
                request_digest=request.digest,
            )

        remaining = max(1, timeout - int(time.monotonic() - started))
        module_exit, generated_module_log = _run_captured(
            ["lake", "env", "lean", str(lean_path.resolve())],
            cwd=lean_root,
            timeout=remaining,
        )
        if module_exit != 0:
            build_exit = module_exit
            return fail(
                status="build-failure",
                exit_code=EXIT_BUILD,
                message="generated module failed to elaborate",
                model_digest=model.digest,
                request_digest=request.digest,
            )

        remaining = max(1, timeout - int(time.monotonic() - started))
        axiom_cmd = [
            sys.executable,
            str(audit_script),
            "--repo",
            str(lean_root),
            "--generated",
            str(lean_path.resolve()),
        ]
        axiom_exit, axiom_report = _run_captured(
            axiom_cmd,
            cwd=lean_root,
            timeout=remaining,
        )
        if axiom_exit != 0:
            axiom_status = "failed"
            return fail(
                status="axiom-failure",
                exit_code=EXIT_AXIOM,
                message="axiom audit failed",
                model_digest=model.digest,
                request_digest=request.digest,
            )
        axiom_status = "passed"

        duration_ms = int((time.monotonic() - started) * 1000)
        result = VerificationResult(
            status="verified",
            exit_code=EXIT_VERIFIED,
            model_digest=model.digest,
            request_digest=request.digest,
            lean_toolchain=lean_toolchain,
            mathlib_revision=mathlib_revision,
            generator_version=__version__,
            evidence_bundle_digest=ZERO_DIGEST,
            build_exit_code=0,
            axiom_audit_exit_code=0,
            axiom_audit="passed",
            generated_module=generated_module_name,
            generated_module_digest=digest_bytes(generated_lean),
            duration_ms=duration_ms,
        )
        return _emit_and_persist(
            output_dir=output_dir,
            result=result,
            model_obj=model_obj,
            request_obj=request_obj,
            generated_lean=generated_lean,
            generated_module_name=generated_module_name,
            lean_build_log=lean_build_log,
            generated_module_log=generated_module_log,
            axiom_report=axiom_report,
            environment=environment(duration_ms=duration_ms),
        )
    except TimeoutFailureError as exc:
        return fail(
            status="timeout",
            exit_code=EXIT_TIMEOUT,
            message=str(exc)[:2048],
            model_digest=model.digest,
            request_digest=request.digest,
        )
    except Exception as exc:  # noqa: BLE001
        return fail(
            status="internal-error",
            exit_code=EXIT_INTERNAL,
            message=str(exc)[:2048],
            model_digest=model.digest if model_obj is not None else ZERO_DIGEST,
            request_digest=request.digest if request_obj is not None else ZERO_DIGEST,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def cmd_inspect(path: Path, *, rerun: bool = False) -> int:
    result_obj = inspect_bundle(path)
    report: dict[str, Any] = {"ok": True, "result": result_obj, "path": str(path)}

    if rerun:
        if result_obj.get("status") != "verified":
            raise InvalidInputError("--rerun requires a verified evidence bundle")
        model_path = path / "model.canonical.json"
        request_path = path / "request.canonical.json"
        if not model_path.is_file() or not request_path.is_file():
            raise InvalidInputError("--rerun requires model and request canonical files")
        rerun_dir = Path(tempfile.mkdtemp(prefix="leanverifier-inspect-rerun-"))
        try:
            code = cmd_verify(
                model_path,
                request_path,
                rerun_dir,
                timeout=DEFAULT_TIMEOUT_S,
                argv=[
                    "leanverifier",
                    "verify",
                    "--model",
                    str(model_path),
                    "--request",
                    str(request_path),
                    "--output",
                    str(rerun_dir),
                ],
            )
            rerun_result = inspect_bundle(rerun_dir)
            report["rerun"] = {
                "exit_code": code,
                "status": rerun_result.get("status"),
                "model_digest_match": rerun_result.get("model_digest")
                == result_obj.get("model_digest"),
                "request_digest_match": rerun_result.get("request_digest")
                == result_obj.get("request_digest"),
                "generated_module_digest_match": rerun_result.get("generated_module_digest")
                == result_obj.get("generated_module_digest"),
                "status_match": rerun_result.get("status") == result_obj.get("status"),
            }
            if (
                code != EXIT_VERIFIED
                or not report["rerun"]["status_match"]
                or not report["rerun"]["generated_module_digest_match"]
            ):
                report["ok"] = False
                print(json.dumps(report, indent=2))
                return EXIT_DIGEST
        finally:
            shutil.rmtree(rerun_dir, ignore_errors=True)

    print(json.dumps(report, indent=2))
    return EXIT_VERIFIED if report["ok"] else EXIT_DIGEST


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
    ver.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the immutable verification evidence bundle",
    )
    ver.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)

    i = sub.add_parser("inspect")
    i.add_argument("path", type=Path, help="Evidence bundle directory")
    i.add_argument(
        "--rerun",
        action="store_true",
        help="Re-execute verify from canonical inputs and compare digests",
    )
    return p


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
            output = args.output
            if not output.is_absolute():
                output = (Path.cwd() / output).resolve()
            full_argv = ["leanverifier", *(argv if argv is not None else sys.argv[1:])]
            return cmd_verify(
                args.model,
                args.request,
                output,
                timeout=args.timeout,
                argv=full_argv,
            )
        if args.cmd == "inspect":
            return cmd_inspect(args.path, rerun=args.rerun)
        return EXIT_INTERNAL
    except NotCertifiedError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NOT_CERTIFIED
    except InvalidInputError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID
    except BuildFailureError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_BUILD
    except AxiomFailureError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_AXIOM
    except DigestMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DIGEST
    except TimeoutFailureError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_TIMEOUT
    except LeanVerifierError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
