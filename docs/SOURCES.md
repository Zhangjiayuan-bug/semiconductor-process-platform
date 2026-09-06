# Reusable Open-Source References

## Real Dataset

- [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), McCann and Johnston (2008), DOI [10.24432/C54305](https://doi.org/10.24432/C54305). The official UCI page describes it as semiconductor manufacturing data with 1,567 entities, 591 documented features, 104 failures and CC BY 4.0 licensing. The raw `secom.data` file contains 590 sensor columns; the project stores it under `data/raw/uci_secom` and keeps the source citation visible in the app.

The project uses standard Python libraries directly and keeps the business logic in this repository. The following GitHub projects were evaluated as optional references:

- [huft-jonathan/pyshewhart](https://github.com/huft-jonathan/pyshewhart), MIT: Shewhart/SPC chart implementation reference.
- [MAIF/eurybia](https://github.com/MAIF/eurybia), Apache-2.0: baseline/current data drift reporting reference.
- [evidentlyai/evidently](https://github.com/evidentlyai/evidently), Apache-2.0: report/test separation and machine-readable monitoring output reference.
- [M1ndSmith/ASPC](https://github.com/M1ndSmith/ASPC), MIT: upload-driven SPC, capability and reporting platform reference.
- [aeoncyr/predictive-quality-control-semiconductor](https://github.com/aeoncyr/predictive-quality-control-semiconductor), CC0-1.0: SECOM-oriented modeling reference. The project still uses the official UCI SECOM download as the primary source.
- [cap1tan/wafermap](https://github.com/cap1tan/wafermap), MIT: wafer-map plotting reference for a planned module.
- [MichaelHotaling/klarfkit](https://github.com/MichaelHotaling/klarfkit), MIT: KLARF and wafer defect-map data reference for a planned module.

No source repository is copied into the application. `docs/github_reference_research.csv` contains GitHub repository metadata collected by `scripts_github_research.py`. Licenses and upstream attribution must be rechecked before vendoring code or distributing a derivative.

## Research-only references

The following repositories have no detected license. They were inspected only for public method descriptions and were not copied, vendored or used as data sources:

- [dzecro/secom-wafer-fault-analysis](https://github.com/dzecro/secom-wafer-fault-analysis): Welch test, multiple-testing correction, PCA and dashboard workflow ideas.
- [rajeshmore1/Semiconductor-manufacturing-process](https://github.com/rajeshmore1/Semiconductor-manufacturing-process): high-dimensional feature-selection and class-imbalance problem framing.
- [MD-Junayed000/WM811k-Silicon-Wafer-Map-Dataset-and-Defect-Detection-Using-CNNs-with-Grad-CAM](https://github.com/MD-Junayed000/WM811k-Silicon-Wafer-Map-Dataset-and-Defect-Detection-Using-CNNs-with-Grad-CAM): image-classification and spatial-explanation workflow idea; bundled data is not used.
- [AshifurRedwan/Wafer_Map_Defect_Pattern_Classification](https://github.com/AshifurRedwan/Wafer_Map_Defect_Pattern_Classification): augmentation comparison idea for imbalanced wafer-map classes.
