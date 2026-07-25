# Generator / input-boundary review packet (template)

**Reviewer:** TBD  
**Role:** Application security and/or Python packaging  
**Review date:** YYYY-MM-DD  
**Tree / commit under review:** `<sha>`  
**Finding file (if any):** `docs/program/findings/<id>.md`

Status vocabulary: `Not started` | `In progress` | `Pass` | `Fail` | `N/A`.

---

## 1. Schema

| Schema | Path | Status |
|--------|------|--------|
| Model | `schemas/model.schema.json` (+ packaged twin) | |
| Request | `schemas/request.schema.json` | |
| Result | `schemas/result.schema.json` | |

Confirm: `additionalProperties: false`; dimension / `minItems` ≥ 1; no JSON floats.

---

## 2. Canonicalization specification

Primary doc: `docs/model-format.md`  
Implementation: `src/leanverifier/canonicalize.py`

| Rule | Status |
|------|--------|
| Exact `Fraction` reduction (no `limit_denominator`) | |
| Rational grammar as documented | |
| Canonical JSON bytes + `sha256:` digests | |
| Resource limits / stable failure codes | |

---

## 3. Fixed source template

Generator: `src/leanverifier/generator.py`  
Confirm deterministic emission of instance proofs (`norm_num` / kernel-checkable tactics only).

**Template / emission review notes:**

**Status:** Not started

---

## 4. Input grammar

Documented grammar + limits in `docs/model-format.md` and `src/leanverifier/limits.py`.

| Check | Status |
|-------|--------|
| UTF-8 rejection | |
| Symlink rejection (pre-resolve) | |
| File size / dimension / rational length bounds | |
| Duplicate JSON keys rejected | |

---

## 5. Generated positive example

| Artifact | Path |
|----------|------|
| Model | `examples/affine_binary/model.json` |
| Request | `examples/affine_binary/request.json` |
| Positive verify evidence | (attach `--output` bundle path after regeneration) |

**Status:** Not started

---

## 6. Mutation corpus

Primary suite: `tests/mutation/test_mutations.py`  
Also: security matrix / Hypothesis L∞ tests as applicable.

| Mutation class | Expected fail-closed behavior | Status |
|----------------|-------------------------------|--------|
| Weight / bias / digest break | verify fail | |
| Epsilon uncertifies | exit 2 | |
| Comment / import injection | fail | |
| Axiom / sorry insert | audit fail | |
| Evidence tamper | inspect fail | |

---

## 7. Generated axiom audit

- Concrete decls: `eps_nonneg`, `margin_strict`, `certified`
- Tooling: `scripts/audit_axioms.py` / packaged `generated_axiom_audit`
- Confirm allowlist-only axioms before status `verified`

**Status:** Not started  
**Notes:**

---

## Sign-off (do not forge)

| Field | Value |
|-------|-------|
| Input→Lean boundary approved? | **No / Yes** |
| Packaging surface approved? | **No / Yes / N/A** |
| Blocking findings filed? | |
| Reviewer signature / handle | TBD |
| Date | |
