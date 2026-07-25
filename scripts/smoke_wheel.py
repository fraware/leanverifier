#!/usr/bin/env python3
"""Wheel/sdist smoke: install into a clean venv and run CLI outside the repo (F-007).

Builds the distribution unless ``--dist`` points at an existing artifact, installs
it into a temporary virtualenv, clears PYTHONPATH, runs from outside the checkout,
and always exercises ``validate-model`` / ``validate-request``.

Pass ``--require-verify`` to also run ``verify`` (needs lake + mathlib). CI job
``wheel-smoke`` is the packaging stub; R-05 may enable ``--require-verify`` with
lean-action.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_leanverifier(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "leanverifier.exe"
    return venv_dir / "bin" / "leanverifier"


def _build_artifacts(root: Path, out_dir: Path) -> list[Path]:
    _run([sys.executable, "-m", "pip", "install", "-q", "build==1.2.2.post1"], cwd=root)
    _run([sys.executable, "-m", "build", "--outdir", str(out_dir)], cwd=root)
    wheels = sorted(out_dir.glob("leanverifier-*.whl"))
    sdists = sorted(out_dir.glob("leanverifier-*.tar.gz"))
    if not wheels and not sdists:
        raise SystemExit("build produced no leanverifier artifacts")
    return wheels + sdists


def _clean_env(venv_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Prefer the venv binaries.
    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv_dir)
    return env


def smoke_one(
    artifact: Path,
    examples: Path,
    work: Path,
    *,
    require_verify: bool,
) -> None:
    venv_dir = work / f"venv-{artifact.suffixes[0].lstrip('.')}-{artifact.stem}"
    venv.create(venv_dir, with_pip=True, clear=True)
    py = _venv_python(venv_dir)
    env = _clean_env(venv_dir)
    _run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)
    _run([str(py), "-m", "pip", "install", "-q", str(artifact)], env=env)

    # Prove import works without the checkout on PYTHONPATH.
    _run(
        [
            str(py),
            "-c",
            "import leanverifier, leanverifier.resources; print(leanverifier.__file__)",
        ],
        cwd=work,
        env=env,
    )

    cli = _venv_leanverifier(venv_dir)
    model = examples / "affine_binary" / "model.json"
    request = examples / "affine_binary" / "request.json"
    # Copy fixtures into the work dir so the CLI never needs the repo path.
    local_model = work / "model.json"
    local_request = work / "request.json"
    shutil.copy2(model, local_model)
    shutil.copy2(request, local_request)

    _run([str(cli), "validate-model", str(local_model)], cwd=work, env=env)
    _run([str(cli), "validate-request", str(local_request)], cwd=work, env=env)

    if not require_verify:
        print(
            "skipping verify smoke (pass --require-verify when Lake/mathlib are available; R-05).",
            flush=True,
        )
        return

    if shutil.which("lake", path=env.get("PATH")) is None:
        raise SystemExit(
            "lake not on PATH but --require-verify was set "
            "(install elan/lake or omit --require-verify)."
        )

    cache = work / "lv-cache"
    evidence = work / "evidence"
    env = dict(env)
    env["LEANVERIFIER_CACHE"] = str(cache)
    _run(
        [
            str(cli),
            "verify",
            "--model",
            str(local_model),
            "--request",
            str(local_request),
            "--output",
            str(evidence),
            "--timeout",
            "600",
        ],
        cwd=work,
        env=env,
    )
    _run([str(cli), "inspect", str(evidence)], cwd=work, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (used only to locate examples / build)",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=None,
        help="Existing dist directory; build into a temp dir when omitted",
    )
    parser.add_argument(
        "--artifact",
        choices=("wheel", "sdist", "both"),
        default="both",
        help="Which artifact kinds to smoke",
    )
    parser.add_argument(
        "--require-verify",
        action="store_true",
        help=(
            "Also run leanverifier verify from the installed artifact "
            "(needs lake + mathlib; default skips — R-05 CI stub)"
        ),
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()
    examples = root / "examples"
    if not (examples / "affine_binary" / "model.json").is_file():
        print(f"missing examples under {examples}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="lv-wheel-smoke-") as tmp:
        work = Path(tmp)
        if args.dist is not None:
            dist_dir = args.dist.resolve()
            artifacts = sorted(dist_dir.glob("leanverifier-*.whl")) + sorted(
                dist_dir.glob("leanverifier-*.tar.gz")
            )
        else:
            dist_dir = work / "dist"
            artifacts = _build_artifacts(root, dist_dir)

        selected: list[Path] = []
        for art in artifacts:
            is_wheel = art.suffix == ".whl"
            is_sdist = art.name.endswith(".tar.gz")
            if args.artifact == "wheel" and not is_wheel:
                continue
            if args.artifact == "sdist" and not is_sdist:
                continue
            if args.artifact == "both" and not (is_wheel or is_sdist):
                continue
            selected.append(art)
        if not selected:
            print("no matching artifacts to smoke", file=sys.stderr)
            return 1

        # Run smokes from a directory that is not the repo root.
        outside = work / "outside"
        outside.mkdir()
        for art in selected:
            print(f"=== smoking {art.name} ===", flush=True)
            smoke_one(
                art,
                examples,
                outside / art.name,
                require_verify=args.require_verify,
            )

    print("WHEEL/SDIST SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
