# SECOM Model Card

## Intended use

This project provides an interview and research prototype for screening anonymous semiconductor process measurements associated with pass/fail outcomes. It is not a fab release system, root-cause engine or production model.

## Dataset

- UCI SECOM, record 179, CC BY 4.0
- 1,567 production entities
- 590 anonymous sensor columns
- 104 failures (`1`) and 1,463 passes (`-1` in the source, mapped to `0` internally)

The public dataset does not identify a fab, tool, recipe or process step.

## Validation design

The model comparison uses two validation strategies with shared model definitions and metrics:

1. Fixed 5-fold repeated stratified cross-validation with two repeats, representing an approximately IID sampling assumption.
2. Five-fold expanding-window temporal validation. Records are sorted by timestamp, split at unique timestamp boundaries, and each test window contains observations later than the training window.

Median imputation, missing indicators, constant-feature filtering, scaling and PCA are fitted independently inside each training fold. Fold exports include train/test sizes, failure prevalence, temporal boundaries, and a descriptive 95% confidence interval half-width for mean PR-AUC.

Models:

- Prior-probability dummy baseline
- Class-weighted logistic regression
- PCA plus class-weighted logistic regression
- Class-weighted random forest

PR-AUC is the primary metric because failures are rare. Balanced accuracy, ROC-AUC, recall and precision are reported as supporting metrics. Default classification thresholds are not optimized.

## Current result

On the reproducible 5x2 comparison, random forest has the strongest mean PR-AUC, while its default 0.5 threshold predicts no failures. Logistic regression produces lower ranking performance but non-zero failure recall. This demonstrates why ranking metrics and operating thresholds must be evaluated separately.

Results may vary with dependency versions. Run the app or `secom_validation_crosscheck` to regenerate exact values. The temporal comparison is a validation-assumption cross-check, not external or live-fab validation.

## Limitations

- No external or live-fab validation
- No confirmed process semantics for anonymous sensors
- Small minority class
- Repeated folds are correlated and are not independent experiments
- Temporal windows may still reflect a single public dataset and one historical production context
- No threshold optimization or cost model
- Statistical association and feature importance do not establish causality
