# Limitations

- Single classifier family: affine binary over ℚ.
- Single robustness notion: L∞ margin with ‖w‖₁ Lipschitz factor.
- Strict inequality required; boundary equality is not certified.
- Classification uses strict positivity (`score > 0`); score `0` is class `false`.
- No floating-point soundness claim.
- Docker/CI Lean builds download mathlib on first run (large).
- v0.1.0 is unauthorized until independent review (LV-11) and the §14 checklist in [release-process.md](release-process.md) complete.
- Legacy FormalVerifML breadth is intentionally excluded from the supported surface.
