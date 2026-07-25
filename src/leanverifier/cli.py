"""CLI entrypoints for LeanVerifier."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from leanverifier.manifest import load_model_path, load_request_path
from leanverifier.result import VerificationResult

DEFAULT_TIMEOUT_S = 600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_toolchain(repo: Path) -> str:
    return (repo / "lean-toolchain").read_text(encoding="utf-8").strip()


def _strip_env() -> dict[str, str]:
    keep = ["PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TMP", "TEMP", "LEAN_PATH"]
    env = {k: v for k, v in os.environ.items() if k in keep or k.startswith("ELAN")}
    env["LEANVERIFIER_OFFLINE"] = "1"
    return env


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
    path = write_module_atomic(model, request, out_dir.resolve())
    print(json.dumps({"ok": True, "path": str(path)}, indent=2))
    return EXIT_VERIFIED


def _run_lake_build(repo: Path, timeout: int) -> None:
    proc = subprocess.run(
        ["lake", "build"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=_strip_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise BuildFailureError(proc.stdout + proc.stderr)


def _run_lean_file(repo: Path, lean_file: Path, timeout: int) -> None:
    proc = subprocess.run(
        ["lake", "env", "lean", str(lean_file.resolve())],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=_strip_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise BuildFailureError(proc.stdout + proc.stderr)


def _run_axiom_audit(
    repo: Path,
    timeout: int,
    *,
    generated: Path | None = None,
) -> None:
    """Run library axiom audit; when `generated` is set, also audit concrete decls.

    Status ``verified`` requires the concrete generated module's ``eps_nonneg``,
    ``margin_strict``, and ``certified`` axioms to pass the allowlist.
    """
    script = repo / "scripts" / "audit_axioms.py"
    cmd = [sys.executable, str(script), "--repo", str(repo)]
    if generated is not None:
        cmd.extend(["--generated", str(generated.resolve())])
    proc = subprocess.run(
        cmd,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=_strip_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise AxiomFailureError(proc.stdout + proc.stderr)


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
            message=str(exc),
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
        _run_axiom_audit(repo, remaining, generated=lean_path)
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
    except subprocess.TimeoutExpired as exc:
        raise TimeoutFailureError(str(exc)) from exc
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
