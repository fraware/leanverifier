# Assurance boundary

## In scope

- Exact ℚ affine binary classifiers with dimensionally matched weights and input
- Closed L∞ balls of nonnegative radius ε
- Strict absolute margin condition `|score(x)| > ε · ‖w‖₁`
- Deterministic generation of a Lean instance of `affine_margin_robustness`
- Axiom audit limited to approved builtins (`propext`, `Classical.choice`, `Quot.sound`)
- Isolated CLI verify path under the pinned Lean / mathlib toolchain

## Out of scope

- Floating-point semantics and hardware execution
- PyTorch / ONNX / HuggingFace model equivalence
- Neural networks, transformers, vision, decision trees beyond legacy experimental code
- Network-exposed verification services
- Cryptographic product claims
- Any property outside the concrete L∞ margin instance that was certified

## Trust boundary

| Side | Contents |
|------|----------|
| Untrusted | User-supplied model/request JSON |
| Trusted after validation | Canonical digests, generated Lean module under path policy, pinned Lean/mathlib, Lean kernel |

Insufficient margin is a deliberate non-certification outcome (exit 2), not a proof of attack success.

## Related

- [CLAIMS.md](../CLAIMS.md) — claim register
- [trusted-computing-base.md](trusted-computing-base.md) — TCB inventory
- [threat-model.md](threat-model.md) — adversaries and controls
- [limitations.md](limitations.md) — known limits
