# IBSR-18 3D U-Net Brain Tissue Segmentation

A modular, reproducible deep-learning pipeline for **3D brain tissue segmentation on the IBSR-18 T1-weighted MRI dataset**, built with PyTorch and MONAI.

The project focuses on both **medical image analysis** and **research-quality software engineering**: reproducible data preparation, configurable experiments, preprocessing validation, 3D patch-based training, sliding-window inference, quantitative evaluation, native-space prediction restoration, automated quality control, testing, linting, and containerization.

## Results at a glance

The selected model is a **residual 3D U-Net** trained using native voxel spacing without N4 bias-field correction.

| Metric                   | Validation Dice |
| ------------------------ | --------------: |
| Background               |          0.9718 |
| CSF                      |          0.8437 |
| Gray matter              |          0.9055 |
| White matter             |          0.9028 |
| **Mean foreground Dice** |      **0.8840** |

The model was selected after four controlled experiments:

| Experiment    | Architecture          | Spacing        | N4  | Mean foreground Dice |
| ------------- | --------------------- | -------------- | --- | -------------------: |
| **1 — Final** | Residual 3D U-Net     | Native         | No  |           **0.8840** |
| 2             | Residual 3D U-Net     | 1 mm isotropic | No  |               0.8748 |
| 3             | Conventional 3D U-Net | Native         | No  |               0.8493 |
| 4             | Residual 3D U-Net     | Native         | Yes |               0.8777 |

The experiments indicate that **residual connections had the largest impact among the evaluated design choices**, while 1 mm isotropic resampling and N4 correction did not improve the final validation score.

---

## Project overview

Brain tissue segmentation assigns each voxel to anatomical tissue classes. In this project, the target classes are:

* **0 — Background**
* **1 — Cerebrospinal fluid (CSF)**
* **2 — Gray matter (GM)**
* **3 — White matter (WM)**

The pipeline operates directly on 3D MRI volumes and uses patch-based training to accommodate the memory requirements of volumetric neural networks.

The project was designed as a complete research workflow rather than a single training script:

```text
MRI volumes
    │
    ▼
Data validation
    │
    ▼
Preprocessing
    │
    ├── RAS orientation
    ├── optional spacing normalization
    ├── intensity normalization
    └── foreground cropping
    │
    ▼
3D patch sampling + augmentation
    │
    ▼
Residual 3D U-Net
    │
    ▼
Validation / checkpoint selection
    │
    ▼
Sliding-window inference
    │
    ▼
Native-space reconstruction
    │
    ▼
NIfTI segmentation
```

---

## Dataset

The project uses the **IBSR-18 T1-weighted brain MRI dataset**.

### Dataset splits

| Split      | Subjects                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------- |
| Training   | IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06, IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18 |
| Validation | IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17                                              |
| Test       | IBSR_02, IBSR_10, IBSR_15                                                                |

The test subjects do not contain ground-truth segmentations in this project configuration. Consequently, quantitative Dice evaluation is performed on the validation set, while the test set is used for **unseen-subject inference and qualitative/generalization assessment**.

### Label mapping

| Label | Tissue       |
| ----: | ------------ |
|     0 | Background   |
|     1 | CSF          |
|     2 | Gray matter  |
|     3 | White matter |

The primary evaluation metric is mean foreground Dice:

$$
\mathrm{Mean\ Foreground\ Dice}
=
\frac{
\mathrm{Dice}_{CSF}
+
\mathrm{Dice}_{GM}
+
\mathrm{Dice}_{WM}
}{3}
$$

---

## Model

The final model is a **3D U-Net with two residual units per convolutional block**.

The architecture was selected through a controlled comparison against a conventional U-Net without residual units.

### Final configuration

* Architecture: Residual 3D U-Net
* Residual units: 2
* Input: 3D T1-weighted MRI
* Output classes: 4
* Loss: Dice + cross-entropy
* Patch size: `96 × 96 × 96`
* Batch size: 1
* Native voxel spacing
* N4 bias correction: disabled
* Sliding-window inference
* Gaussian blending during inference

---

## Preprocessing

The validation and inference pipeline includes:

1. NIfTI loading
2. Single-channel conversion
3. RAS orientation
4. Optional voxel-spacing normalization
5. Intensity normalization
6. Foreground cropping
7. Tensor conversion

Training additionally uses randomized 3D patch sampling and data augmentation.

The final selected configuration deliberately preserves **native voxel spacing**, because resampling to 1 mm isotropic spacing produced lower validation performance.

---

## N4 bias-field experiment

MRI intensity inhomogeneity was investigated using a precomputed N4 bias-field correction pipeline.

The finalized preprocessing configuration used:

* SimpleITK N4BiasFieldCorrection
* Shrink factor: `4`
* Maximum iterations: `[50, 50, 50, 50]`
* Convergence threshold: `0.001`
* B-spline control points: `[4, 4, 4]`
* Foreground mask: `image > 0`

All **18 IBSR-18 subjects passed automated N4 preprocessing QC**, including geometry preservation, finite-value checks, reconstruction consistency, and bias-field smoothness checks.

However, N4 correction achieved a mean foreground Dice of **0.8777**, compared with **0.8840** for the selected native/no-N4 model. It was therefore not included in the final inference pipeline.

More details are available in [`docs/experiments.md`](docs/experiments.md).

---

## Quantitative validation

The final checkpoint was evaluated on the five held-out validation subjects.

| Subject       |   CSF Dice |    GM Dice |    WM Dice |
| ------------- | ---------: | ---------: | ---------: |
| IBSR_11       |     0.8183 |     0.9032 |     0.9212 |
| IBSR_12       |     0.8624 |     0.8882 |     0.9029 |
| IBSR_13       |     0.8032 |     0.9043 |     0.8790 |
| IBSR_14       |     0.8448 |     0.9206 |     0.9149 |
| IBSR_17       |     0.8899 |     0.9115 |     0.8959 |
| **Aggregate** | **0.8437** | **0.9055** | **0.9028** |

**Mean foreground Dice: 0.8840**

The results show stronger performance for GM and WM than CSF, which is consistent with the greater difficulty of delineating relatively small CSF structures and thin sulcal regions.

---

## Test-set inference

The final checkpoint was applied to the three unseen test subjects:

* IBSR_02
* IBSR_10
* IBSR_15

Predictions are generated using 3D sliding-window inference with:

* ROI size: `96 × 96 × 96`
* Sliding-window batch size: `1`
* Overlap: `0.25`
* Gaussian blending

Predictions are then restored from the foreground-cropped representation to the **original native image space** using the preprocessing metadata recorded by MONAI.

Generated predictions:

```text
outputs/
└── predictions/
    └── test/
        ├── IBSR_02/
        │   └── IBSR_02_pred.nii.gz
        ├── IBSR_10/
        │   └── IBSR_10_pred.nii.gz
        └── IBSR_15/
            └── IBSR_15_pred.nii.gz
```

### Native-space sanity checks

All three test predictions passed:

* native spatial-shape verification
* affine consistency with the source MRI
* finite-value checks
* valid label verification
* non-empty foreground verification
* `uint8` output verification

The predictions therefore have valid native-space geometry and can be directly used as NIfTI segmentation outputs.

Because the test split has no ground-truth segmentations in this project configuration, **test Dice is not reported**.

---

## Qualitative test assessment

Qualitative inspection was performed in axial, coronal, and sagittal views.

### IBSR_02

The segmentation is generally anatomically plausible, with continuous cortical coverage and recognizable GM/WM organization. A disconnected inferior WM prediction was observed in the coronal view and is considered a false-positive region.

### IBSR_10

IBSR_10 produced the strongest qualitative result among the three test subjects. Cortical coverage is continuous, GM/WM boundaries are relatively detailed, and the ventricular CSF is well represented. No obvious disconnected components were observed in the inspected views.

### IBSR_15

IBSR_15 was more challenging. Qualitative limitations included:

* comparatively coarse WM regions
* apparent WM over-segmentation
* under-segmentation of ventricular CSF
* a localized cortical coverage gap
* small off-brain prediction islands

These observations illustrate subject-dependent generalization limitations despite the strong aggregate validation score.

No obvious rectangular or grid-like seams attributable to sliding-window inference were observed.

---

## Repository structure

```text
ibsr18-3d-unet/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── lint.yml
├── configs/
│   ├── train.yaml
│   └── finetune.yaml
├── data/
│   └── README.md
├── docs/
│   ├── architecture.md
│   └── experiments.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_visualization.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── analyze_labels.py
│   ├── precompute_n4.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_test.py
│   ├── visualize_predictions.py
│   └── check_test_predictions.py
├── src/
│   └── ibsr_unet/
│       ├── config/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── visualization/
│       └── utils/
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
├── pre-commit-config.yaml
├── README.md
└── LICENSE
```

---

## Installation

The project targets Python 3.11.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install the project:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

The project uses PyTorch and MONAI for volumetric deep learning and medical image processing.

---

## Configuration

Training and experiment settings are stored in YAML configuration files:

```text
configs/
├── train.yaml
└── finetune.yaml
```

This keeps model, data, optimization, and preprocessing settings separate from the implementation.

---

## Training

After preparing the dataset and configuring the desired experiment:

```bash
python scripts/train.py --config configs/train.yaml
```

Checkpoints and training outputs are written under:

```text
outputs/
```

---

## Evaluation

Validation evaluation:

```bash
python scripts/evaluate.py
```

The evaluation pipeline reports per-class Dice scores and mean foreground Dice.

---

## Test inference

Generate predictions for the three test subjects:

```bash
python scripts/predict_test.py
```

Then run native-space sanity checks:

```bash
python scripts/check_test_predictions.py
```

---

## Reproducibility and engineering practices

The repository includes:

* configurable YAML experiments
* deterministic/reproducible utilities
* modular dataset and transformation components
* separate training and inference pipelines
* unit tests
* Ruff linting
* pre-commit hooks
* GitHub Actions CI
* Docker support
* structured logging
* native-space NIfTI reconstruction
* automated preprocessing QC

The goal is to make the project reproducible and maintainable rather than tying the workflow to a single notebook.

---

## Limitations

Several limitations should be considered when interpreting the results:

1. The dataset is small, with only 18 subjects.
2. The public test split used here does not provide ground-truth labels, preventing quantitative test evaluation.
3. Qualitative inspection identified subject-dependent failure modes, particularly for CSF delineation and GM/WM boundaries.
4. Some isolated off-brain predictions remain in difficult test cases.
5. The model was evaluated on IBSR-18 and should not be assumed to generalize directly to other scanners, acquisition protocols, or clinical populations.

---

## Conclusion

The final pipeline demonstrates a complete 3D medical image segmentation workflow, from dataset validation and controlled preprocessing experiments through training, quantitative validation, native-space inference, automated prediction checks, and qualitative assessment.

The selected model is a **residual 3D U-Net operating at native voxel spacing without N4 correction**
