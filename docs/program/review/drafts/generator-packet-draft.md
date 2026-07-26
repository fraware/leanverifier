# Generator review packet (draft)

**Draft status:** Scaffolding for external review — **not signed**  
**Prepared for:** R-08 / LV-11  
**Tree tip this draft describes:** `release/r-07-reproducible-rc` @ `337752957fa46646f46f791ebbbe23170ce11d74` (plus this R-08 docs commit)  
**Reviewer:** TBD  
**Input→Lean boundary approved?** **No**  
**Packaging surface approved?** **No**

---

## 1. Schema

| Schema | Checkout path | Packaged twin |
|--------|---------------|---------------|
| Model `leanverifier.model.v1` | `schemas/model.schema.json` | `src/leanverifier/resources/schemas/` |
| Request `leanverifier.request.v1` | `schemas/request.schema.json` | same |
| Result `leanverifier.result.v1` | `schemas/result.schema.json` | same |

Constraints to verify: `additionalProperties: false`; dimension and vector `minItems` ≥ 1; rationals as strings (no JSON numbers). Resource hash sync: `python scripts/regen_resource_hashes.py`.

---

## 2. Canonicalization specification

| Item | Location |
|------|----------|
| Human spec | `docs/model-format.md` |
| Implementation | `src/leanverifier/canonicalize.py` |
| Limits / codes | `src/leanverifier/limits.py` |

Key properties (R-03 / F-009–F-012): exact `Fraction` reduction only; documented rational grammar; pre-resolve symlink rejection; UTF-8 / size / dimension bounds.

---

## 3. Fixed source template

| Item | Location |
|------|----------|
| Generator | `src/leanverifier/generator.py` |
| Path policy | `_assert_safe_output_root` (lstat before resolve) |
| Instance proofs | `norm_num` / kernel-checkable tactics (R-01 / F-001) |

Reviewer should read the emitted Lean for `examples/affine_binary/` after a fresh `leanverifier verify --output …` run.

---

## 4. Input grammar

Documented in `docs/model-format.md`:

- Rational strings: `0`, integers, or lowest-terms fractions
- Canonical JSON: UTF-8, sorted keys, no insignificant whitespace
- Digest: `sha256:` + hex of canonical model bytes; request `model_digest` must match

---

## 5. Positive example

| File | Role |
|------|------|
| `examples/affine_binary/model.json` | Repeated-weight unit example (`dimension` 3) |
| `examples/affine_binary/request.json` | Matching digest + `epsilon` `1/10` |

Positive path: `leanverifier verify` with required `--output` evidence bundle; `leanverifier inspect` validates checksums (R-02 / F-005–F-006). CI: `verify-positive` job.

---

## 6. Mutation corpus

Primary table: `tests/mutation/test_mutations.py` (R-05 / F-013), including:

- Weight / bias mutations breaking digest via verify
- Epsilon mutation that uncertifies
- Comment not injected as Lean; import removal build failure
- `sorry` / custom axiom insert fails audit
- Evidence / checksum tamper fails inspect

Additional: security matrix jobs; Hypothesis L∞ classification invariance when margin premise holds.

---

## 7. Generated axiom audit

| Item | Location |
|------|----------|
| Auditor | `scripts/audit_axioms.py` |
| Packaged helper | `src/leanverifier/resources/scripts/generated_axiom_audit.py` |
| Concrete decls | `eps_nonneg`, `margin_strict`, `certified` |
| Gate | Audit must pass before CLI status `verified` |

Full release packaging must run the audit without `--skip-lean` (R-07 / F-016).

---

## Sign-off

| Field | Value |
|-------|-------|
| Input→Lean boundary approved? | **No** |
| Packaging surface approved? | **No** |
| Reviewer handle | TBD |
| Date | — |
