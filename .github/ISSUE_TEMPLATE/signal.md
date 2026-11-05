---
name: Signal
about: Test one macro signal against the volatility baselines
title: "Signal: "
labels: signal
---

Signal | P1
Depends on: #8, #9, #10 | Blocks: #18

**Goal:**

**Deliverable:** `macro/features/<name>.py`, its lookahead test, results, a scorecard row, and `research/<name>.md`.

**Tasks**
- Pull the series, confirm release timing, add the release lag to `data/README.md`
- Features (3–6), fit on past data only
- Lookahead test with `assert_no_lookahead`
- Run the harness at h ∈ {5, 21, 63} against the VIX + RV baseline

**Pre-registered threshold:** write yours in a comment on this issue before running.

**Done when:**
- [ ] Tests, ruff and format checks pass
- [ ] Lookahead test included
- [ ] No holdout data touched
- [ ] README signal table and `data/README.md` updated
- [ ] PR merged after one review; issue labeled `advancing` or `not-advancing`
