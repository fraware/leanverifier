# Formal review packet (draft)

**Draft status:** Scaffolding for external review — **not signed**  
**Prepared for:** R-08 / LV-11  
**Tree tip this draft describes:** `release/r-07-reproducible-rc` @ `337752957fa46646f46f791ebbbe23170ce11d74` (plus this R-08 docs commit)  
**Reviewer:** TBD  
**Formal interpretation approved?** **No** (awaiting external Lean reviewers)

Copy this draft into a dated finding folder when a reviewer is assigned; do not treat unchecked boxes as Pass.

---

## 1. Informal claim

LeanVerifier claims that for an exact rational affine binary classifier, if the absolute score margin at a reference input strictly exceeds `ε · ‖w‖₁` with `ε ≥ 0`, then classification is invariant for every point in the closed L∞ ball of radius `ε` about that reference (CLAIMS LV-C002 / LV-C003). Classification uses the strict decision boundary `score > 0`. The equality-margin case is intentionally not certified.

---

## 2. Exact theorem statements (pointers)

| Theorem | File |
|---------|------|
| `abs_score_sub_score_le` | `LeanVerifier/Property/ClassificationRobustness.lean` |
| `classify_true_of_margin` / `classify_false_of_margin` | same |
| `affine_margin_robustness` | `LeanVerifier/Theorem/AffineMarginRobustness.lean` |
| `classify_eq_of_strict_margin` | same |
| `LocallyRobustAt` | `LeanVerifier/Theorem/AffineMarginRobustness.lean` |
| Unit / repeated-weight theorems | `LeanVerifier/Model/AffineBinary.lean` |

Central statement (paraphrase; review the Lean source as authoritative):

```lean
theorem affine_margin_robustness (M : AffineBinaryClassifier n) (x : Vec n) (ε : Rat)
    (hε : 0 ≤ ε) (hmargin : ε * weightL1 M < |score M x|) :
    LocallyRobustAt M x ε
```

where `LocallyRobustAt` means `∀ y, InLinfBall x ε y → classify M y = classify M x`.

---

## 3. Definitions

| Definition | Location |
|------------|----------|
| `AffineBinaryClassifier`, `score`, `classify`, `weightL1` | `LeanVerifier/Model/AffineBinary.lean` |
| `Vec`, `l1Norm` | `LeanVerifier/Data/Vector.lean` |
| `InLinfBall` | `LeanVerifier/Property/LinfBall.lean` |
| Decision boundary | Module docstring + `classify := decide (0 < score …)` |

---

## 4. Proof files

| Module | Path | Notes for reviewer |
|--------|------|--------------------|
| Model | `LeanVerifier/Model/AffineBinary.lean` | Unit theorems use `norm_num` (R-01; F-002) |
| Robustness lemmas | `LeanVerifier/Property/ClassificationRobustness.lean` | Score perturbation bound |
| Central theorem | `LeanVerifier/Theorem/AffineMarginRobustness.lean` | Explicit zero-boundary case split |
| Axiom inventory | `LeanVerifier/Audit/PublicAxioms.lean` | `#print axioms` anchors |

TCB layers: `docs/trusted-computing-base.md` (four-layer rewrite; F-020).

---

## 5. Axiom / checker report locations

| Report | How to produce / where it lands |
|--------|----------------------------------|
| Library + generated axiom audit | `python scripts/audit_axioms.py` (full; **no** `--skip-lean`) |
| Packaged RC axiom report | emitted by `scripts/package_rc_artifacts.sh` into the artifact dir |
| Public inventory module | `LeanVerifier/Audit/PublicAxioms.lean` |
| lean4checker | `scripts/run_lean4checker.sh`; CI job `lean4checker` |

Allowlist: `{propext, Classical.choice, Quot.sound}` only.

---

## 6. Decision-boundary analysis (draft)

| Topic | Intended treatment | Source |
|-------|--------------------|--------|
| Positive class | `classify = true` iff `score > 0` | `AffineBinary.lean` |
| Zero score | Never `true`; explicit case in `affine_margin_robustness` | theorem proof |
| Strict margin hypothesis | `ε * weightL1 M < |score M x|` | theorem type |
| Equality margin | **Not certified**; insufficient margin → CLI exit 2 | docstring + CLAIMS negative evidence |
| Other norms / FP | Out of assurance scope | `docs/assurance-boundary.md` |

Reviewer must confirm the English CLAIMS language does not imply certification of the equality case or of floating-point execution.

---

## 7. Negative examples (pointers)

| Case | Expected | Pointer |
|------|----------|---------|
| Margin not strict | Non-certify / exit 2 | CLI semantics; CLAIMS LV-C003 |
| Mutated weights / digest mismatch | verify fail | `tests/mutation/test_mutations.py` |
| Inserted `sorry` / custom `axiom` | axiom audit fail | same mutation suite |
| Legacy neural / web paths | unsupported | `experimental/`; CLAIMS LV-X* |

---

## Sign-off

| Field | Value |
|-------|-------|
| Formal interpretation approved? | **No** |
| Reviewer handle | TBD |
| Date | — |

Do not change the approval field to Yes without an external Lean reviewer signature and filed findings disposition.
