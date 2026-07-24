# Program board

## Active workstreams

| ID | Title | Owner | Status | Blocker |
|----|-------|-------|--------|---------|
| LV-00 | Truth reset | Mateo | done (in tree) | |
| gov-main | Rename + ruleset + CODEOWNERS | Mateo / admin | templates done | GitHub rename + remote ruleset |
| LV-01 | Build baseline | Mateo | done (in tree) | Local Lean reinstall may need SSL workaround |
| LV-02 | Affine kernel | Mateo | done | |
| LV-03 | Margin robustness | Mateo | done | Formal reviewer sign-off pending |
| LV-04 | Schemas | Mateo | done | |
| LV-05 | Generator | Mateo | done | Security reviewer sign-off pending |
| LV-06 | CLI | Mateo | done | |
| LV-07 | Tests | Mateo | done | |
| LV-08 | Legacy isolation | Mateo | done | |
| LV-09 | Assurance docs | Mateo | done | |
| LV-10 | RC pipeline | Mateo | scaffolding | Live attestations on GitHub |
| LV-11 | Independent review | — | blocked | External reviewers |

## Admin actions (GitHub)

- [ ] Rename default branch `master` → `main` (requires admin)
- [ ] Set `origin/HEAD` to `main`
- [ ] Activate branch ruleset on `main` (required reviews, CODEOWNERS, no force-push)
- [ ] Confirm CODEOWNERS usernames include non-implementer formal + security reviewers
