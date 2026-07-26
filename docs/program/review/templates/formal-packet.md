# Formal review packet (template)

**Reviewer:** TBD  
**Affiliation / independence:** TBD (must be ≠ implementer for at least one of ≥2 Lean reviewers)  
**Review date:** YYYY-MM-DD  
**Tree / commit under review:** `<sha>`  
**Finding file (if any):** `docs/program/findings/<id>.md`

Status vocabulary for each item: `Not started` | `In progress` | `Pass` | `Fail` | `N/A`.

---

## 1. Informal claim

State the English claim under review (must match `CLAIMS.md` LV-C002 / LV-C003 language):

> …

Reviewer judgment:

- [ ] Claim matches theorem intent
- [ ] Claim does not overclaim FP / other norms / equality case

**Status:** Not started  
**Notes:**

---

## 2. Exact theorem statements (pointers)

List theorem names and file paths reviewed:

| Theorem | File | Lines / anchor | Status |
|---------|------|----------------|--------|
| | | | |

---

## 3. Definitions

| Definition | File | Status |
|------------|------|--------|
| `AffineBinaryClassifier` / `score` / `classify` | | |
| `InLinfBall` / L∞ ball | | |
| `weightL1` / `LocallyRobustAt` | | |
| Decision boundary (`score > 0`) | | |

---

## 4. Proof files

| Module | Path | Reviewed | Status |
|--------|------|----------|--------|
| Model | `LeanVerifier/Model/AffineBinary.lean` | | |
| Property | `LeanVerifier/Property/*.lean` | | |
| Central theorem | `LeanVerifier/Theorem/AffineMarginRobustness.lean` | | |
| Public axiom anchors | `LeanVerifier/Audit/PublicAxioms.lean` | | |

Confirm: no `sorry`, no custom axioms, no native-evaluation decide tactics on supported decls.

---

## 5. Axiom report

- Command: `python scripts/audit_axioms.py` (library + generated; no `--skip-lean`)
- Artifact location (if packaged): `…/axiom-report.txt` or CI log path
- Allowlist expected: `{propext, Classical.choice, Quot.sound}`

**Status:** Not started  
**Notes:**

---

## 6. Checker report

- Command: `scripts/run_lean4checker.sh` / CI `lean4checker` job
- Artifact / log location:

**Status:** Not started  
**Notes:**

---

## 7. Decision-boundary analysis

Document treatment of:

- Strict half-space `score > 0`
- Equality case `|score(x)| = ε · ‖w‖₁` (must remain uncertified)
- Zero score at the reference point

**Status:** Not started  
**Notes:**

---

## 8. Negative examples

List cases that must **not** certify (or must fail closed):

| Case | Expected outcome | Evidence pointer | Status |
|------|------------------|------------------|--------|
| Insufficient margin | exit 2 / non-certify | | |
| Equality margin | not certified | | |
| Wrong digest / mutated weights | verify fail | | |
| Other norms / FP models | out of scope | | |

---

## Sign-off (do not forge)

| Field | Value |
|-------|-------|
| Formal interpretation approved? | **No / Yes** (circle one; default No until filled by reviewer) |
| Blocking findings filed? | |
| Reviewer signature / handle | TBD |
| Date | |

This template signature block is blank by design. Implementers must not pre-fill Pass or Yes.
