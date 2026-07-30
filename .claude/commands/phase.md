---
description: Run a numbered phase of docs/PLAN.md with its deliverable as the success criterion
---

Run phase $1 of `docs/PLAN.md` in this repo.

Before doing anything: read that phase's entry in PLAN.md §5 and state its deliverable back to me
in one line. Then check `runs/` for the previous phase's `report.json` so you know what you are
comparing against.

Rules:
- One variable per run. Config changes go in `configs/`, never hardcoded in training code.
- Do not report the phase complete unless the deliverable metric moved in the right direction AND
  the fluency probe and leakage check both pass.
- If it did not work, say so plainly and give me the three most likely causes ranked by how cheap
  they are to test.
- If two results are within seed variance, say "within noise" rather than ranking them.
