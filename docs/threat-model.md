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
| Path escape / symlink race | Absolute output root; resolve; parent check; atomic replace |
| False `verified` on weak margin | Executable margin check before generate/verify; exit 2 |
| Axiom smuggling | Source scan + `#print axioms` allowlist |
| CI privilege escalation | No `pull_request_target`; least privilege; SHA pins |
| Legacy upload → build | Quarantine under `experimental/`; [SECURITY.md](../SECURITY.md) warnings |

## Non-goals

Defending against a compromised Lean kernel or a malicious mathlib pin accepted by maintainers.
