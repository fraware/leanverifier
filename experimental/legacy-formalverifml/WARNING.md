# WARNING -- Legacy FormalVerifML

This tree is quarantined experimental history from the pre-recovery FormalVerifML packaging.

- **Not** imported by `LeanVerifier` or `LeanVerifier.lean`
- **Not** listed in `lakefile.toml` and **not** installed by the `leanverifier` Python package
- Web upload -> translator -> `lake build` is a security hazard; keep offline and unused
- Known formal defects (custom axioms, empty generated models, True-only theorems, etc.) are why this surface was removed from supported claims
- Paths and install instructions that mention `lean/FormalVerifML` or `formal_verif_ml` refer to this quarantine only -- they are not the supported product

Do not cite this tree as evidence for LeanVerifier v0.1.0.
