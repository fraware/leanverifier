# Risk register

| ID | Risk | Severity | Mitigation | Status |
|----|------|----------|------------|--------|
| R1 | Overclaimed public docs / web exposure | Critical | LV-00 docs; quarantine webapp; SECURITY warning | Mitigated in tree |
| R2 | RC toolchain + CI targeting wrong branch | High | Stable Lean pin; CI on `main`; admin rename | Partial (admin rename pending) |
| R3 | False `verified` without margin/axiom gates | Critical | Exit 2 on insufficient margin; axiom audit | Mitigated in design |
| R4 | Generator injection / path escape | Critical | Grammar + path policy + security tests | Mitigated in design |
| R5 | Single-reviewer merge on formal/kernel PRs | High | CODEOWNERS + reviewer separation rule | Needs real reviewer usernames |
| R6 | External review delay blocks v0.1.0 | Medium | LV-11 invitation template | Open |
