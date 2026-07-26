#!/usr/bin/env python3
"""Generate an authoritative CycloneDX SBOM via the pinned cyclonedx-bom tool.

Python components come from ``cyclonedx-py environment`` over the resolved venv.
Lean toolchain, mathlib (and other lake-manifest pins), and artifact file hashes
are merged deterministically. Handwritten component inventories are not accepted
as the authoritative SBOM (F-017).

Pinned tool: cyclonedx-bom==CYCLONEDX_BOM_VERSION
(see pyproject optional-dependencies.release).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Must match pyproject.toml optional-dependencies.release pin.
CYCLONEDX_BOM_VERSION = "7.3.0"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sorted_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _load_lake_packages(repo: Path) -> list[dict[str, str]]:
    manifest = repo / "lake-manifest.json"
    if not manifest.is_file():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for pkg in data.get("packages") or []:
        name = str(pkg.get("name") or "")
        rev = str(pkg.get("rev") or "")
        url = str(pkg.get("url") or "")
        input_rev = str(pkg.get("inputRev") or "")
        if not name or not rev:
            continue
        out.append(
            {
                "name": name,
                "rev": rev,
                "url": url,
                "inputRev": input_rev,
            }
        )
    out.sort(key=lambda p: p["name"])
    return out


def _python_exe_for_env(repo: Path) -> Path:
    """Prefer a Python that can import cyclonedx_py; favor the project venv."""
    candidates: list[Path] = []
    venv = repo / ".venv"
    if sys.platform == "win32":
        candidates.append(venv / "Scripts" / "python.exe")
    else:
        candidates.append(venv / "bin" / "python")
    candidates.append(Path(sys.executable))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import cyclonedx_py"],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return candidate
    # Last resort: current interpreter (caller must have installed the pin).
    return Path(sys.executable)


def _generate_python_sbom(repo: Path, raw_path: Path) -> dict[str, Any]:
    py = _python_exe_for_env(repo)
    env = os.environ.copy()
    # Deterministic timestamps in tool output when supported.
    env.setdefault("SOURCE_DATE_EPOCH", env.get("SOURCE_DATE_EPOCH", "0"))
    env["TZ"] = "UTC"
    env["PYTHONHASHSEED"] = "0"

    cmd = [
        str(py),
        "-m",
        "cyclonedx_py",
        "environment",
        str(py),
        "--of",
        "JSON",
        "--sv",
        "1.5",
        "--output-reproducible",
        "-o",
        str(raw_path),
    ]
    try:
        _run(cmd, cwd=repo, env=env)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "cyclonedx-py failed. Install the pinned tool via "
            f"`uv sync --frozen --extra release` (cyclonedx-bom=={CYCLONEDX_BOM_VERSION}). "
            f"Detail: {exc}"
        ) from exc

    return json.loads(raw_path.read_text(encoding="utf-8"))


def _component(
    *,
    ctype: str,
    name: str,
    version: str,
    purl: str | None = None,
    hashes: list[dict[str, str]] | None = None,
    properties: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    comp: dict[str, Any] = {
        "type": ctype,
        "name": name,
        "version": version,
    }
    if purl:
        comp["purl"] = purl
    if hashes:
        comp["hashes"] = hashes
    if properties:
        comp["properties"] = properties
    return comp


def _enrich(
    sbom: dict[str, Any],
    *,
    repo: Path,
    artifacts_dir: Path,
    commit: str,
    source_date_epoch: str,
) -> dict[str, Any]:
    toolchain = (repo / "lean-toolchain").read_text(encoding="utf-8").strip()
    components: list[dict[str, Any]] = list(sbom.get("components") or [])

    components.append(
        _component(
            ctype="library",
            name="lean4",
            version=toolchain,
            purl=f"pkg:generic/lean4@{toolchain}",
            properties=[{"name": "leanverifier:role", "value": "toolchain"}],
        )
    )

    for pkg in _load_lake_packages(repo):
        version = pkg["inputRev"] or pkg["rev"]
        purl = None
        if "github.com/" in pkg["url"]:
            # pkg:github/owner/repo@version
            path = pkg["url"].rstrip("/").split("github.com/", 1)[-1]
            if path.endswith(".git"):
                path = path[: -len(".git")]
            purl = f"pkg:github/{path}@{pkg['rev']}"
        components.append(
            _component(
                ctype="library",
                name=pkg["name"],
                version=version,
                purl=purl,
                properties=[
                    {"name": "leanverifier:git-rev", "value": pkg["rev"]},
                    {"name": "leanverifier:input-rev", "value": pkg["inputRev"]},
                    {"name": "leanverifier:url", "value": pkg["url"]},
                ],
            )
        )

    if artifacts_dir.is_dir():
        for path in sorted(artifacts_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name in {"sbom.json", "sbom.raw.json", "checksums.sha256"}:
                continue
            digest = _sha256_file(path)
            components.append(
                _component(
                    ctype="file",
                    name=path.name,
                    version=commit[:12] or "unknown",
                    hashes=[{"alg": "SHA-256", "content": digest}],
                    properties=[{"name": "leanverifier:artifact", "value": "true"}],
                )
            )

    # Deterministic ordering.
    components.sort(
        key=lambda c: (
            str(c.get("type", "")),
            str(c.get("name", "")),
            str(c.get("version", "")),
        )
    )
    sbom["components"] = components

    metadata = dict(sbom.get("metadata") or {})
    # Force reproducible metadata stamp from SOURCE_DATE_EPOCH.
    epoch = int(source_date_epoch)
    metadata["timestamp"] = datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    props = list(metadata.get("properties") or [])
    props = [
        p
        for p in props
        if p.get("name")
        not in {
            "leanverifier:source-date-epoch",
            "leanverifier:git-commit",
            "leanverifier:sbom-generator",
        }
    ]
    props.extend(
        [
            {"name": "leanverifier:source-date-epoch", "value": source_date_epoch},
            {"name": "leanverifier:git-commit", "value": commit},
            {
                "name": "leanverifier:sbom-generator",
                "value": f"cyclonedx-bom=={CYCLONEDX_BOM_VERSION}+leanverifier-enrich",
            },
        ]
    )
    props.sort(key=lambda p: (str(p.get("name", "")), str(p.get("value", ""))))
    metadata["properties"] = props
    sbom["metadata"] = metadata

    # Stable top-level key order via rewrite through sorted dump/load is applied by writer.
    return sbom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True, help="Output CycloneDX JSON path")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifact directory whose file hashes are recorded in the SBOM",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    artifacts = (args.artifacts or out.parent).resolve()

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if not source_date_epoch:
        proc = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%ct"],
            capture_output=True,
            text=True,
            check=False,
        )
        source_date_epoch = (proc.stdout or "").strip() or "0"
        os.environ["SOURCE_DATE_EPOCH"] = source_date_epoch

    commit_proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = (commit_proc.stdout or "").strip() or "unknown"

    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix(".raw.json")
    sbom = _generate_python_sbom(repo, raw_path)
    enriched = _enrich(
        sbom,
        repo=repo,
        artifacts_dir=artifacts,
        commit=commit,
        source_date_epoch=source_date_epoch,
    )
    out.write_text(_sorted_json(enriched), encoding="utf-8")
    # Drop intermediate tool dump from the artifact set when co-located.
    if raw_path.parent == out.parent:
        try:
            raw_path.unlink()
        except OSError:
            pass
    print(f"wrote {out} (cyclonedx-bom=={CYCLONEDX_BOM_VERSION})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
