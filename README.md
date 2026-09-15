# IBSR-18 3D U-Net Brain Tissue Segmentation

A modular, reproducible deep-learning pipeline for **3D brain tissue segmentation on the IBSR-18 T1-weighted MRI dataset**, built with PyTorch and MONAI.

The project combines **medical image analysis** with **research-quality software engineering**: reproducible data preparation, configurable experiments, preprocessing validation, 3D patch-based training, sliding-window inference, quantitative evaluation, native-space prediction restoration, automated quality control, testing, linting, and containerization.

---

## Results at a glance

The selected model is a **residual 3D U-Net** trained using **native voxel spacing without N4 bias-field correction**.

The final checkpoint was selected from a 400-epoch training run. Its best validation performance occurred at **epoch 391**.

| Metric                   | Validation Dice |
| ------------------------ | --------------: |
| Background               |          0.9803 |
| CSF                      |          0.8938 |
| Gray matter              |          0.9340 |
| White matter             |          0.9276 |
| **Mean foreground Dice** |      **0.9185** |

### Training-duration study

The same native-spacing residual 3D U-Net was progressively trained for longer durations because validation performance was still improving:

| Training duration | Best validation mean foreground Dice | Best epoch |
| ----------------: | -----------------------------------: | ---------: |
|        100 epochs |                               0.8840 |         99 |
|        200 epochs |                               0.9054 |        199 |
|        300 epochs |                               0.9114 |        290 |
|    **400 epochs** |                           **0.9185** |    **391** |

This progression shows that a substantial portion of the improvement came from allowing the same architecture to continue learning rather than introducing increasingly complex preprocessing.

### Controlled experiment comparison

| Experiment                      | Architecture          | Spacing        | N4     | Mean foreground Dice |
| ------------------------------- | --------------------- | -------------- | ------ | -------------------: |
| Native baseline — 100 epochs    | Residual 3D U-Net     | Native         | No     |               0.8840 |
| Native — 200 epochs             | Residual 3D U-Net     | Native         | No     |               0.9054 |
| Native — 300 epochs             | Residual 3D U-Net     | Native         | No     |               0.9114 |
| **Native — 400 epochs (final)** | **Residual 3D U-Net** | **Native**     | **No** |           **0.9185** |
| 1 mm isotropic                  | Residual 3D U-Net     | 1 mm isotropic | No     |               0.8748 |
| Conventional U-Net              | Conventional 3D U-Net | Native         | No     |               0.8493 |
| N4 preprocessing                | Residual 3D U-Net     | Native         | Yes    |               0.8777 |

The experiments indicate that **residual connections and sufficient training duration were more beneficial than the evaluated 1 mm isotropic resampling or N4 preprocessing configurations**. The final model therefore uses native voxel spacing and no N4 correction.

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

The test subjects do not contain ground-truth segmentations in this project configuration. Consequently, quantitative Dice evaluation is performed on the validation set, while the test set is used for **unseen-subject inference and qualitative generalization assessment**.

### Label mapping

| Label | Tissue       |
| ----: | ------------ |
|     0 | Background   |
|     1 | CSF          |
|     2 | Gray matter  |
|     3 | White matter |

The primary evaluation metric is **mean foreground Dice**, calculated across CSF, GM, and WM:

```math
\frac{\mathrm{Dice}_{CSF} + \mathrm{Dice}_{GM} + \mathrm{Dice}_{WM}}{3}
```

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

The final checkpoint is:

```text
outputs/checkpoints/best_model.pt
```

It corresponds to **epoch 391 of the 400-epoch training run**.

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

### Native-spacing design choice

The native-spacing configuration was investigated progressively:

```text
100 epochs  → 0.8840
200 epochs  → 0.9054
300 epochs  → 0.9114
400 epochs  → 0.9185
```

The best validation score in the 400-epoch run occurred at epoch 391, after which the validation score fluctuated below the best checkpoint.

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

However, N4 correction achieved a mean foreground Dice of **0.8777**, compared with **0.9185** for the final native/no-N4 model. It was therefore not included in the final inference pipeline.

More details are available in [`docs/experiments.md`](docs/experiments.md).

---

## Quantitative validation

The final epoch-391 checkpoint was independently evaluated on the five held-out validation subjects.

| Subject       |   CSF Dice |    GM Dice |    WM Dice |
| ------------- | ---------: | ---------: | ---------: |
| IBSR_11       |     0.8819 |     0.9338 |     0.9443 |
| IBSR_12       |     0.8886 |     0.9187 |     0.9267 |
| IBSR_13       |     0.8655 |     0.9326 |     0.9051 |
| IBSR_14       |     0.9113 |     0.9446 |     0.9374 |
| IBSR_17       |     0.9217 |     0.9404 |     0.9245 |
| **Aggregate** | **0.8938** | **0.9340** | **0.9276** |

Overall validation performance:

| Metric              |       Dice |
| ------------------- | ---------: |
| Background          |     0.9803 |
| CSF                 |     0.8938 |
| GM                  |     0.9340 |
| WM                  |     0.9276 |
| **Mean foreground** | **0.9185** |

The model performs strongly across GM and WM while CSF remains the most challenging foreground class. This is expected to some extent because CSF includes relatively small structures and thin sulcal regions that are more difficult to delineate consistently.

---

## Training behavior

The final training run used 400 epochs. The best validation result occurred at **epoch 391**:

```text
Epoch 391/400
train_loss = 0.2335
val_loss   = 0.2027

CSF Dice = 0.8938
GM Dice  = 0.9340
WM Dice  = 0.9276

Mean foreground Dice = 0.9185
```

The following epochs did not surpass this checkpoint:

```text
Epoch 392 → 0.9128
Epoch 393 → 0.9134
Epoch 394 → 0.9131
Epoch 395 → 0.9131
Epoch 396 → 0.9093
Epoch 397 → 0.9085
Epoch 398 → 0.9140
Epoch 399 → 0.9057
Epoch 400 → 0.9142
```

Therefore, **epoch 391** was retained as the final model rather than the final training epoch.

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

IBSR_10 produced the strongest qualitative result among the three test subjects. Cortical coverage is continuous, GM/WM boundaries are relatively detailed, and ventricular CSF is well represented. No obvious disconnected components were observed in the inspected views.

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
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── lint.yml
│
├── configs/
│   ├── train.yaml
│   └── finetune.yaml
│
├── data/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   └── experiments.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_visualization.ipynb
│
├── scripts/
│   ├── prepare_data.py
│   ├── analyze_labels.py
│   ├── precompute_n4.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_test.py
│   ├── visualize_predictions.py
│   ├── visualize_test_predictions.py
│   └── check_test_predictions.py
│
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
│
├── tests/
│
├── Dockerfile
├── Makefile
├── pyproject.toml
├── pre-commit-config.yaml
├── README.md
└── LICENSE
```

---

## Installation

The project targets **Python 3.11**.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install the project:

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

For qualitative visualization:

```bash
python scripts/visualize_test_predictions.py
```

---

## Testing and code quality

Run the unit tests with:

```bash
python -m pytest
```

Run Ruff on the maintained inference and visualization scripts:

```bash
ruff check \
    scripts/check_test_predictions.py \
    scripts/predict_test.py \
    scripts/visualize_test_predictions.py \
    scripts/visualize_predictions.py
```

The repository also includes:

* GitHub Actions CI
* Ruff linting
* pre-commit hooks
* Docker support

---

## Reproducibility and engineering practices

The repository includes:

* configurable YAML experiments
* deterministic/reproducibility utilities
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
* checkpoint-based model selection

The goal is to make the project reproducible and maintainable rather than tying the workflow to a single notebook or training script.

---

## Limitations

Several limitations should be considered when interpreting the results:

1. The dataset is small, with only 18 subjects.
2. The public test split used here does not provide ground-truth labels, preventing quantitative test evaluation.
3. The validation set contains only five subjects, so the reported validation score should not be interpreted as a broad estimate of clinical performance.
4. Qualitative inspection identified subject-dependent failure modes, particularly for CSF delineation and GM/WM boundaries.
5. Some isolated off-brain predictions remain in difficult test cases.
6. The model was evaluated on IBSR-18 and should not be assumed to generalize directly to other scanners, acquisition protocols, datasets, or clinical populations.
7. The final checkpoint was selected using the held-out validation set; an external dataset would be required for a stronger assessment of generalization.

---

## Conclusion

This project demonstrates a complete 3D medical image segmentation workflow, from dataset validation and controlled preprocessing experiments through training, quantitative validation, native-space inference, automated prediction checks, and qualitative assessment.

The final model is a **residual 3D U-Net operating at native voxel spacing without N4 bias-field correction**.

Through progressively longer training runs, the validation mean foreground Dice improved from:

```text
100 epochs  → 0.8840
200 epochs  → 0.9054
300 epochs  → 0.9114
400 epochs  → 0.9185
```

The final selected checkpoint achieved a **mean foreground Dice of 0.9185 at epoch 391**, with class-wise Dice scores of **0.8938 for CSF, 0.9340 for GM, and 0.9276 for WM**.

The project demonstrates not only a strong segmentation result on IBSR-18, but also an end-to-end approach to **reproducible medical AI research and maintainable scientific software engineering**.
