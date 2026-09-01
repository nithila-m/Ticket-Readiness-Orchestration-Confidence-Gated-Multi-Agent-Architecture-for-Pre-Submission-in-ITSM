# Confidence recalibration (isotonic regression, cross-validated)
n = 149 tickets, 5-fold cross-validation
Raw ECE: **0.089**
Calibrated ECE (held-out folds): **0.014**

The calibrated ECE above is computed only on held-out folds - never the same points the correction was fit on - so this number reflects genuine generalization, not overfitting to 150 points. The final mapping saved to calibration_mapping.json is fit on the full set for downstream use and is a plain, auditable lookup table, not a black-box model.