# Judge-vs-human agreement

_No human grades yet._ Run `python scripts/grade.py` to grade the
answers in the eval trace, then re-run this script to populate the table.

## Mechanical over-strictness cross-check

Answers the judge scored correctness < 4 that nonetheless contain every required value (must-include = 1.0) — candidate judge over-strictness, pending human confirmation:

| id | judge | answer |
|---|---|---|
| g004 | 3 | 538µA [1] |
| g015 | 3 | -80 MHz (G = 100) -45 MHz (G = 1) [5] |
| g016 | 3 | switched capacitor (sampled data) filter [2] |
| g019 | 3 | Both devices have a maximum input offset voltage of 25 μ V [1] and [2]. |
| g029 | 3 | 1 Mbyte [2] |
| g030 | 3 | 256 Kbytes [1] |
| g038 | 3 | 420μA [7] |
| g055 | 3 | 8.8 nV/√Hz (0.1-Hz to 10-Hz) [1] |
| g061 | 3 | 2.5V (±1.25V) to 5.5V (±2.75V) [3] |
| g066 | 3 | less than 1 µA [1] |
| g071 | 3 | Arm Cortex-M4 |
| g077 | 3 | 16 Kbytes [4] |
| g083 | 1 | SN65HVD230 supports up to 1 Mbps [5,8], while THVD1450 supports up to 500 kbps [ |
| g085 | 3 | The OPA2993 achieves its rail-to-rail input common-mode range using paralleled c |
| g092 | 3 | Yes, the MCP3008 inputs can be configured as single-ended [1]. |
