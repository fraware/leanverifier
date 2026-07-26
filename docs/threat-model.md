# Threat model

## Assets

- Integrity of `verified` results
- Integrity of generated Lean sources and digests
- Confidentiality of local secrets in the environment (must not leak into the Lean subprocess environment)

## Adversaries

- Malicious model/request JSON authors
- Supply-chain attackers swapping Actions or dependency tags
- Operators mistakenly exposing the legacy web upload path

## Threats and controls

| Threat | Control |
|--------|---------|
| Lean injection via crafted strings | Rational grammar; comments-only free text; fixed template |
| Path escape / symlink race | Absolute output root; lstat components before resolve; reject symlink roots/intermediates/destinations; parent check; atomic replace |
| Resource exhaustion (huge JSON/rationals) | Pre-decode 1 MiB cap; dim/rational/comment/Lean/log bounds; stable codes |
| Env secret leakage into Lean | `_strip_env` allowlist only; tested |
| Hung Lean/lake subprocess | Timeout + process-group kill (`timeout` exit 8) |
| False `verified` on weak margin | Executable margin check before generate/verify; exit 2 |
| Axiom smuggling / native axioms | Prohibited-tactic scan; source scan; `#print axioms` allowlist on library **and** concrete generated decls (`eps_nonneg`, `margin_strict`, `certified`) before `verified` |
| CI privilege escalation | No `pull_request_target`; least privilege; SHA pins |
| Legacy upload → build | Quarantine under `experimental/`; [SECURITY.md](../SECURITY.md) warnings |

Supported proofs must not use native evaluation decide tactics, `decide +native`,
or bitvector decide. CI fails on those tactics under `LeanVerifier`, `src`,
`tests`, and `scripts`.

## Non-goals

Defending against a compromised Lean kernel or a malicious mathlib pin accepted by maintainers.
