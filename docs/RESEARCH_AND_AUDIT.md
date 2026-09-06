# Dataset and Code Audit

Date: 2026-09-05

## Dataset decision

The primary dataset is **UCI SECOM**, the official UCI Machine Learning Repository record 179. It is described by UCI as data from a semiconductor manufacturing process and is licensed CC BY 4.0.

- Source: https://archive.ics.uci.edu/dataset/179/secom
- DOI: https://doi.org/10.24432/C54305
- Download: https://archive.ics.uci.edu/static/public/179/secom.zip
- Shape: 1,567 production entities and 590 columns in `secom.data`; UCI reports 591 documented features including the outcome-related field.
- Labels: `-1` pass and `1` fail; 104 failures.
- Missing values: source `NaN` values are preserved and reported, not silently deleted.

The application does not claim that anonymous sensor columns identify a particular fab, tool, recipe or process step. Those fields are unavailable in the public source and are not fabricated.

## GitHub research

The following projects were evaluated as references rather than copied dependencies:

- `carlosqsilva/pyspc`, about 242 stars, GPL-3.0. Broad SPC API, but copyleft makes it unsuitable as an unreviewed dependency for a closed portfolio product.
- `huft-jonathan/pyshewhart`, about 33 stars, MIT. Useful reference for Shewhart, CUSUM, attribute charts and Western Electric rules.
- `cap1tan/wafermap`, about 34 stars, MIT. Useful geometry-aware wafer visualization reference.
- `MichaelHotaling/klarfkit`, about 29 stars, MIT. Useful KLARF parser/normalization reference.
- `yzhao062/pyod`, about 10k stars, BSD-2-Clause. Strong generic detector registry and score API; the current project uses scikit-learn Isolation Forest while keeping the detector boundary replaceable.
- `evidentlyai/evidently`, about 7.9k stars, Apache-2.0. The report/test separation and machine-readable results are good patterns for the next monitoring module.
- `MAIF/eurybia`, about 224 stars, Apache-2.0. Useful reference for baseline/current drift reports and drift-classifier thinking; the project implements a smaller PSI/KS report rather than copying Eurybia code.
- `M1ndSmith/ASPC`, about 13 stars, MIT. Useful reference for an upload-driven SPC/capability/reporting platform shape.
- `aeoncyr/predictive-quality-control-semiconductor`, about 2 stars, CC0-1.0. SECOM-oriented learning project with PCA/modeling scripts; useful as a legally permissive reference, but the app keeps the official UCI download as the primary data source.
- `plotly/dash`, about 24k stars, MIT. Better future choice than Streamlit for multi-user reactive workflows; Streamlit remains appropriate for this interview prototype.
- `j4ggr/DaSPi`, small BSD-3-Clause project. Strong capability/MSA/ANOVA concepts, but each statistic should be independently validated before adoption.

Repositories without a declared data license, including several wafer-map and SECOM mirror repositories, are not copied. A repository code license also does not automatically license the underlying dataset. The generated `docs/github_reference_research.csv` stores GitHub metadata for repeatable review, not vendored source code.

### Unlicensed repository review

Repositories marked `NOASSERTION` were also inspected for public method descriptions and product ideas, without copying source code or downloading bundled datasets:

- `rajeshmore1/Semiconductor-manufacturing-process`: reinforces the value of feature selection under SECOM's high dimensionality and class imbalance. Its notebook is not copied.
- `dzecro/secom-wafer-fault-analysis`: publicly describes missingness analysis, Welch tests with Benjamini-Hochberg correction, PCA and an R Shiny dashboard. The statistical-screening idea was independently implemented with SciPy/Pandas; no R source was copied.
- `MD-Junayed000/WM811k-Silicon-Wafer-Map-Dataset-and-Defect-Detection-Using-CNNs-with-Grad-CAM`: useful evidence that wafer-map classification should pair image predictions with spatial explanations. The bundled RAR dataset and notebook are not downloaded because repository/data terms are not sufficiently clear.
- `AshifurRedwan/Wafer_Map_Defect_Pattern_Classification`: compares CNN workflows with and without augmentation, which is useful for future imbalanced wafer-image experiments. No code or data is used.
- `04pallav/1PCA-Classification-Methods-in-R-for-analysing-semiconductor-manufacturing-dataset`: suggests PCA plus classifier comparison, but only an unlicensed PDF is present; it is treated as a research lead, not reusable material.

The independent implementation threshold is deliberate: method names and published statistical techniques can inform the design, but unlicensed expression, code, notebooks, reports and bundled data are not reusable by default.

## Audit fixes applied

- Added a dedicated SECOM adapter for whitespace sensor files and quoted timestamps.
- Added row-alignment and label-domain checks.
- Added timestamp parsing with explicit day-first semantics.
- Added missingness and class-balance reporting.
- Added a reusable SECOM report helper that returns quality, missingness, model metrics and holdout predictions as machine-readable objects.
- Added PSI and Kolmogorov-Smirnov drift reporting for baseline/current comparisons, inspired by monitoring platforms but implemented locally.
- Added independent Welch pass/fail sensor screening with Benjamini-Hochberg false-discovery correction and standardized effect sizes.
- Added a classification path with train/test separation, stratification, median imputation inside the model pipeline, scaling and class weighting.
- Added a fixed four-model comparison using 5-fold repeated stratified cross-validation with two repeats; all preprocessing and PCA remain inside each training fold.
- Replaced the simplified sample-standard-deviation chart with Phase-I/II I-MR limits based on moving ranges and Western Electric rules WE-1 through WE-4.
- Removed the assumption that every dataset has lot, equipment and continuous yield columns.
- Made generic cleaning safe for schema validation instead of raising `KeyError` first.
- Added validation for uploaded-file parsing, empty equipment filters and invalid specification limits.
- Made the official download script idempotent for reproducible local setup.
- Added a GitHub research metadata script for license-aware platform and dataset scouting.
- Added real-data and edge-case tests covering shape, labels, timestamp, model metrics, missingness reporting, single-class rejection and download idempotency.
- Added GitHub Actions CI, direct-dependency locking, an application smoke test for both modes and a model card.

## Remaining limits

- UCI SECOM has anonymous sensor semantics and no wafer-coordinate data.
- The logistic model is a baseline, not a production yield predictor.
- PSI and KS drift flags are screening metrics; they identify distribution changes but do not prove process root cause.
- Sensor association tests are exploratory and univariate; adjusted significance and effect size do not establish causality or reveal anonymous sensor semantics.
- Cp/Cpk and SPC remain available only in the synthetic process-quality template, where continuous measurements and specifications are present.
- A future wafer-map module should use MIR-WM811K only after confirming redistribution terms; its public source does not clearly state an SPDX/Creative Commons license.
