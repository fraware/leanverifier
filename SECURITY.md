# Security Policy

## Supported surface

Security review applies only to the supported v0.1.0 path:

- Canonical schemas and rational grammar (`schemas/`, `leanverifier.canonicalize`)
- Deterministic Lean generator (`src/leanverifier/`)
- CLI verification in an isolated temporary workspace
- CI workflows with least privilege and SHA-pinned Actions

## Critical warning: legacy web path

The FormalVerifML web application under `experimental/legacy-formalverifml/webapp/` accepted model JSON uploads, ran a translator, and invoked `lake build`. That path is **quarantined**.

**Do not:**

- Expose the legacy Flask app on any network
- Wire upload handlers into the supported CLI
- Treat legacy translator output as trusted Lean

Untrusted input reaching a Lean build is a supply-chain and code-injection risk.

## Reporting vulnerabilities

Prefer [GitHub Security Advisories](https://github.com/fraware/leanverifier/security/advisories) for the `fraware/leanverifier` repository. Contact the program lead (Mateo Petel) for issues that must stay private.

Do not open public issues for exploitable generator, path-escape, or CI privilege flaws until a fix or mitigation is ready.

## Generator / CLI hardening goals

- No `eval` / `exec` / shell interpolation of user data
- Subprocess: `shell=False`, argv list, timeout with process-group kill, minimal environment (`_strip_env`)
- Output path containment: `lstat` every path component **before** `resolve`; reject any symlink component; CLI does not resolve the user path away before the check
- Resource limits (1 MiB manifests, dim ≤ 4096, rational ≤ 256 chars, comment ≤ 512, generated Lean ≤ 8 MiB, retained logs ≤ 16 MiB) with stable failure codes — see [docs/model-format.md](docs/model-format.md)
- Module names match `GeneratedModel_<16 hex>` only
- Digest mismatch fails closed before compile
- Insufficient margin returns `not-certified` (exit 2), never `verified`
- Malformed UTF-8 input fails with `utf8-invalid` (exit 3)

## Trusted computing base

See [docs/trusted-computing-base.md](docs/trusted-computing-base.md).
