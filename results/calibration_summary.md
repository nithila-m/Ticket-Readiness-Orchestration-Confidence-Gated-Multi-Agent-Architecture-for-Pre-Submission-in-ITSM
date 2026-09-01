# Agent 1 confidence calibration — Metric 6
n = 149 tickets (rows with a returned category and confidence)
Expected Calibration Error (ECE): **0.097**
Brier score: **0.125**

Lower is better for both. This is a single run on 150 synthetic tickets - report it as a directional first check, not a final calibration claim, consistent with how the rest of the evaluation suite is reported.

| Confidence bin | n | Mean confidence | Observed accuracy | Gap |
|---|---|---|---|---|
| (-0.001, 0.2] | 2 | 0.000 | 0.000 | +0.000 |
| (0.4, 0.6] | 3 | 0.533 | 0.333 | -0.200 |
| (0.6, 0.8] | 4 | 0.788 | 1.000 | +0.212 |
| (0.8, 1.0] | 140 | 0.957 | 0.864 | -0.093 |