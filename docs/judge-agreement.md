# Judge-vs-human agreement

_No human grades yet._ Run `python scripts/grade.py` to grade the
answers in the eval trace, then re-run this script to populate the table.

## Mechanical over-strictness cross-check

Answers the judge scored correctness < 4 that nonetheless contain every required value (must-include = 1.0) — candidate judge over-strictness, pending human confirmation:

| id | judge | answer |
|---|---|---|
| g001 | 3 | ±250µV [2] |
| g004 | 3 | 538µA [1] |
| g015 | 3 | -80 MHz (G = 100) -45 MHz (G = 1) [5] |
| g016 | 3 | switched capacitor (sampled data) filter [2] |
| g029 | 3 | 1 -Mbyte [2] |
| g030 | 3 | 256 Kbytes [1] |
