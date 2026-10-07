# IBSR-18 3D U-Net Brain Tissue Segmentation

A modular and reproducible deep-learning pipeline for **3D brain tissue segmentation on the IBSR-18 T1-weighted MRI dataset**, built with **PyTorch and MONAI**.

The project combines **medical image analysis** with **research-oriented software engineering**, including reproducible data preparation, configurable experiments, preprocessing validation, 3D patch-based training, sliding-window inference, quantitative evaluation, native-space prediction restoration, automated quality control, testing, linting, and containerization.

---

## Results at a glance

The final selected model is a **residual 3D U-Net** trained at **native voxel spacing without N4 bias-field correction**.

The 400-epoch training run produced its best validation checkpoint at **epoch 391**, selected using the original 0.25-overlap validation evaluation.

### Final model

| Configuration | Value |
|---|---|
| Architecture | Residual 3D U-Net |
| Training duration | 400 epochs |
| Selected checkpoint | Epoch 391 |
| Voxel spacing | Native |
| N4 bias correction | No |
| Patch size | `96 × 96 × 96` |
| Loss | Dice + cross-entropy |

The selected checkpoint achieved a **mean foreground Dice of 0.9185** on the five-subject validation set using the 0.25-overlap evaluation. The same checkpoint achieved **0.9188** when subsequently evaluated with 0.50 sliding-window overlap. This was an inference-time change only; the model weights and training procedure were unchanged.

### Controlled experiments

| Experiment | Mean foreground Dice |
|---|---:|
| Native residual 3D U-Net — 100 epochs | 0.8840 |
| Native residual 3D U-Net — 200 epochs | 0.9054 |
| Native residual 3D U-Net — 300 epochs | 0.9114 |
| **Native residual 3D U-Net — 400 epochs** | **0.9185** |
| 1 mm isotropic | 0.8748 |
| Conventional 3D U-Net | 0.8493 |
| N4 preprocessing | 0.8777 |
| CSF-weighted loss | 0.9167 |

These experiments indicate that, under the evaluated configurations, the selected native-spacing residual U-Net provided the strongest validation result. The detailed subject-level validation results and experimental analysis are provided below.

---

## Project overview

Brain tissue segmentation assigns each voxel of a brain MRI volume to an anatomical tissue class.

This project predicts four classes:

| Label | Class |
|---:|---|
| 0 | Background |
| 1 | Cerebrospinal fluid (CSF) |
| 2 | Gray matter (GM) |
| 3 | White matter (WM) |

The pipeline operates directly on 3D MRI volumes and uses patch-based training to accommodate the memory requirements of volumetric neural networks.

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
    │
    ▼
Automated QC + qualitative inspection
```

The project was designed as a complete research workflow rather than a single training script.

---

## Dataset

The project uses the **IBSR-18 T1-weighted brain MRI dataset**.

### Dataset splits

| Split | Subjects |
|---|---|
| Training | IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06, IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18 |
| Validation | IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17 |
| Test | IBSR_02, IBSR_10, IBSR_15 |

The test subjects do not contain ground-truth segmentations in this project configuration. Therefore:

- quantitative Dice evaluation is performed on the validation set;
- the test split is used for unseen-subject inference;
- test predictions are assessed through geometry checks and qualitative inspection;
- test Dice is **not reported**.

### Label mapping

| Label | Tissue |
|---:|---|
| 0 | Background |
| 1 | CSF |
| 2 | Gray matter |
| 3 | White matter |

The primary evaluation metric is **mean foreground Dice**, calculated across CSF, GM, and WM:

```text
Mean foreground Dice =
    (Dice_CSF + Dice_GM + Dice_WM) / 3
```

---

## Model

The final model is a **3D U-Net with residual units** implemented using MONAI.

### Final model configuration

- Architecture: Residual 3D U-Net
- Input channels: 1
- Output classes: 4
- Residual units: 2
- Channels: `[16, 32, 64, 128, 256]`
- Patch size: `96 × 96 × 96`
- Batch size: 1
- Loss: Dice + cross-entropy
- Voxel spacing: Native
- N4 bias correction: Disabled

### Checkpoint selection and final inference configuration

The selected checkpoint is:

```text
outputs/checkpoints/best_model.pt
```
It corresponds to **epoch 391 of the 400-epoch training run**.

The checkpoint was selected using the validation mean foreground Dice from the original **0.25 sliding-window overlap** evaluation, which achieved:

```text
Mean foreground Dice = 0.9185
```

The same checkpoint was subsequently evaluated using a 0.50 sliding-window overlap for the final inference configuration:

ROI size: `96 × 96 × 96`
Sliding-window batch size: `1`
Overlap: `0.50`
Gaussian blending
No N4 correction
No test-time augmentation

This produced a mean foreground Dice of **0.9188** on the validation set. The 0.50 result is therefore an **inference-time evaluation of the selected checkpoint**, not a newly trained model.

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

The final configuration preserves **native voxel spacing**. A controlled 1 mm isotropic experiment produced lower validation performance than the native-spacing configuration.

### Native-spacing training progression

```text
100 epochs  → 0.8840
200 epochs  → 0.9054
300 epochs  → 0.9114
400 epochs  → 0.9185
```

The best validation result in the 400-epoch run occurred at epoch 391.

---

## N4 bias-field experiment

MRI intensity inhomogeneity was investigated using a precomputed N4 bias-field correction pipeline.

The evaluated N4 configuration used:

- SimpleITK N4BiasFieldCorrection
- Shrink factor: `4`
- Maximum iterations: `[50, 50, 50, 50]`
- Convergence threshold: `0.001`
- B-spline control points: `[4, 4, 4]`
- Foreground mask: `image > 0`

All **18 IBSR-18 subjects passed automated N4 preprocessing QC**, including geometry preservation, finite-value checks, reconstruction consistency, and bias-field smoothness checks.

The N4 experiment achieved a mean foreground Dice of **0.8777**, compared with **0.9185** for the selected native-spacing/no-N4 configuration.

N4 preprocessing was therefore not included in the final model pipeline.

Detailed experiment results are documented in [`docs/experiments.md`](docs/experiments.md).

---

## Quantitative validation

The validation results below provide the detailed quantitative evaluation of the selected epoch-391 checkpoint.

**Checkpoint selection:** The checkpoint was selected using the original 0.25 sliding-window overlap evaluation, where the mean foreground Dice was 0.9185.

**Final inference evaluation:** The same checkpoint was subsequently evaluated at 0.50 overlap. No model weights, architecture, preprocessing, training data, or checkpoint selection procedure were changed.

### Validation results — overlap 0.25

| Subject | CSF Dice | GM Dice | WM Dice |
|---|---:|---:|---:|
| IBSR_11 | 0.8819 | 0.9338 | 0.9443 |
| IBSR_12 | 0.8886 | 0.9187 | 0.9267 |
| IBSR_13 | 0.8655 | 0.9326 | 0.9051 |
| IBSR_14 | 0.9113 | 0.9446 | 0.9374 |
| IBSR_17 | 0.9217 | 0.9404 | 0.9245 |
| **Aggregate** | **0.8938** | **0.9340** | **0.9276** |

Overall:

| Metric | Dice |
|---|---:|
| Background | 0.9803 |
| CSF | 0.8938 |
| GM | 0.9340 |
| WM | 0.9276 |
| **Mean foreground** | **0.9185** |

### Inference-overlap evaluation

The same epoch-391 checkpoint was evaluated again using a sliding-window overlap of `0.50`.

This was an **inference-only evaluation**. Model weights, training data, architecture, preprocessing, and checkpoint selection were unchanged.

| Subject | Background | CSF Dice | GM Dice | WM Dice |
|---|---:|---:|---:|---:|
| IBSR_11 | 0.9820 | 0.8819 | 0.9338 | 0.9443 |
| IBSR_12 | 0.9737 | 0.8886 | 0.9187 | 0.9267 |
| IBSR_13 | 0.9810 | 0.8655 | 0.9326 | 0.9051 |
| IBSR_14 | 0.9843 | 0.9113 | 0.9446 | 0.9374 |
| IBSR_17 | 0.9816 | 0.9235 | 0.9424 | 0.9263 |
| **Aggregate** | **0.9805** | **0.8942** | **0.9344** | **0.9280** |

The resulting mean foreground Dice was:

```text
Overlap 0.25 → 0.9185
Overlap 0.50 → 0.9188
Δ mean foreground Dice = +0.0003

The resulting mean foreground Dice was:

```text
Overlap 0.25 → 0.9185
Overlap 0.50 → 0.9188

Δ mean foreground Dice = +0.0003
```

The final inference configuration therefore uses **0.50 sliding-window overlap**.

The small improvement should be interpreted as an inference-level refinement rather than a substantive change in model capability.

---

## CSF-weighted loss experiment

CSF was the lowest-performing foreground class in the baseline evaluation, so a controlled class-weighting experiment was performed.

The experiment preserved:

- Residual 3D U-Net
- Native voxel spacing
- No N4 correction
- `96 × 96 × 96` patches
- Batch size 1
- 400 training epochs

The evaluated class weights were:

```text
Background = 1.0
CSF        = 1.5
GM         = 1.0
WM         = 1.0
```

The best weighted checkpoint occurred at **epoch 371**.

| Metric | Baseline | CSF-weighted | Difference |
|---|---:|---:|---:|
| CSF | 0.8938 | **0.8953** | +0.0015 |
| GM | 0.9340 | 0.9315 | -0.0025 |
| WM | 0.9276 | 0.9233 | -0.0043 |
| **Mean foreground** | **0.9185** | 0.9167 | **-0.0018** |

The weighting increased CSF Dice slightly but reduced GM and WM Dice, resulting in a lower mean foreground Dice.

The weighted checkpoint is retained separately:

```text
outputs/experiments/csf_weighted/checkpoints/best_model.pt
```

It was **not selected as the final model**.

This experiment is retained as a controlled negative result demonstrating that increasing the CSF loss contribution alone did not improve the primary evaluation metric under the tested configuration.

---

## Test-set inference

The final checkpoint was applied to the three unseen test subjects:

- IBSR_02
- IBSR_10
- IBSR_15

The final inference protocol is:

- ROI size: `96 × 96 × 96`
- Sliding-window batch size: `1`
- Overlap: `0.50`
- Gaussian blending
- No N4 correction
- No test-time augmentation
- Epoch-391 checkpoint

Predictions are restored from the foreground-cropped representation to the **original native image space** using the preprocessing metadata recorded during inference.

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

The test prediction QC verifies:

- native spatial shape
- affine consistency with the source MRI
- finite values
- valid labels `[0, 1, 2, 3]`
- non-empty foreground
- output dtype
- voxel-level prediction statistics
- connected-component structure
- slice continuity

The automated QC treats MRI foreground containment as a **diagnostic check rather than a brain-mask ground truth**, because non-zero MRI voxels are not equivalent to a true anatomical brain mask.

Because the test split has no ground-truth segmentations, **test Dice is not reported**.

---

## Qualitative test assessment

Qualitative inspection was performed using axial, coronal, and sagittal views of the unseen test subjects.

The predictions show broadly plausible whole-brain tissue organization, including cortical coverage, major ventricular CSF spaces, central white matter, and inferior brain structures.

The main recurring limitation is **coarse GM/WM boundary delineation**, particularly around cortical regions and smaller CSF spaces.

### IBSR_02

The prediction shows:

- broad whole-brain coverage;
- recognizable cortical GM;
- organized deep WM;
- plausible major ventricular CSF;
- reasonable preservation of large CSF spaces.

The GM/WM interface is relatively coarse in places, and small sulci are simplified.

A disconnected inferior WM component was observed during QC. Subsequent anatomical inspection indicated that disconnected WM regions should not automatically be treated as false positives: some have coherent anatomical locations and may represent legitimate inferior or cerebellar white matter.

### IBSR_10

IBSR_10 shows relatively stable whole-brain organization:

- continuous cortical coverage;
- recognizable GM/WM organization;
- clear ventricular CSF;
- coherent central white matter;
- no obvious large off-brain prediction.

Small isolated regions remain candidates for further 3D inspection, but no major sliding-window seams were observed.

### IBSR_15

IBSR_15 was more challenging and showed:

- coarser GM/WM boundaries;
- subject-dependent variation in WM segmentation;
- simplified small CSF spaces;
- several disconnected WM components requiring anatomical review.

Detailed investigation of the disconnected components is documented in [`docs/test_qc.md`](docs/test_qc.md).

These findings demonstrate why automated connected-component removal was **not** applied blindly: a disconnected component is not necessarily a segmentation error.

### Overall qualitative assessment

Across the three unseen subjects:

- large-scale brain coverage is generally plausible;
- cortical GM coverage is broadly preserved;
- major ventricular CSF structures are recognizable;
- GM/WM organization is generally coherent;
- fine cortical GM/WM boundaries remain a limitation;
- small sulci and CSF spaces are simplified;
- some disconnected components require anatomical review;
- no obvious large sliding-window grid artifacts were observed.

Qualitative inspection is supportive but does not replace quantitative evaluation against ground truth.

---

## Test prediction QC

The repository includes automated checks for generated test predictions.

The QC pipeline examines:

```text
Geometry
├── native shape
├── affine consistency
└── voxel spacing

Numerical validity
├── finite values
├── valid class labels
└── expected output dtype

Segmentation structure
├── foreground volume
├── class volumes
├── connected components
└── slice continuity

MRI relationship
└── prediction voxels relative to non-zero MRI foreground
```

The output is written to:

```text
outputs/qc/test_prediction_qc.csv
```

The QC report is intended as a **sanity-checking and diagnostic tool**, not as a substitute for ground-truth evaluation.

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
│   └── eval_overlap_050.yaml
│
├── data/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   └── test_qc.md
│
├── scripts/
│   ├── prepare_data.py
│   ├── analyze_labels.py
│   ├── precompute_n4.py
│   ├── qc_n4.py
│   ├── smoke_test_n4.py
│   ├── smoke_test_experiment4.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── predict_test.py
│   ├── visualize_predictions.py
│   ├── visualize_test_predictions.py
│   ├── check_test_predictions.py
│   ├── inspect_ibsr15_csf.py
│   └── inspect_ibsr15_wm_component.py
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
├── .dockerignore
├── .gitignore
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

## Docker

The repository includes a CUDA-enabled Docker environment for reproducible execution.

The container is based on NVIDIA CUDA 12.6.3 with Python 3.11 and PyTorch 2.14.0 + CUDA 12.6.

Build the image:

```bash
docker build -t ibsr18-3d-unet .
```

Run the container:

```bash
docker run --rm ibsr18-3d-unet
```

For NVIDIA GPU access:

```bash
docker run --rm --gpus all ibsr18-3d-unet
```

The GPU-enabled container has been validated with an NVIDIA GeForce GTX 1650, with PyTorch reporting:

```text
PyTorch: 2.14.0+cu126
CUDA: True
GPU: NVIDIA GeForce GTX 1650
```

The Docker image does not include the raw IBSR-18 MRI data. Dataset files should be prepared separately according to `data/README.md`.

## Configuration

Training and experiment settings are stored in YAML configuration files:

```text
configs/
├── train.yaml
└── eval_overlap_050.yaml
```

The main configuration separates:

- dataset paths;
- preprocessing;
- patch size;
- model architecture;
- optimization;
- loss configuration;
- inference settings;
- output locations.

The final baseline configuration is stored in:

```text
configs/train.yaml
```

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

The final selected checkpoint is:

```text
outputs/checkpoints/best_model.pt
```

---

## Evaluation

Validation evaluation can be run with:

```bash
python scripts/evaluate.py
```

The evaluation pipeline reports per-class Dice scores and mean foreground Dice.

To evaluate a specific checkpoint:

```bash
python scripts/evaluate.py \
    --config configs/train.yaml \
    --checkpoint outputs/checkpoints/best_model.pt
```

The final reported inference configuration uses a sliding-window overlap of `0.50`. The checkpoint itself was selected using the original 0.25-overlap validation evaluation.

---

## Test inference

Generate predictions for the three test subjects:

```bash
python scripts/predict_test.py
```

Then run prediction sanity checks:

```bash
python scripts/check_test_predictions.py
```

For qualitative visualization:

```bash
python scripts/visualize_test_predictions.py
```

---

## Testing and code quality

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the unit tests:

```bash
python -m pytest
```

Run Ruff:

```bash
ruff check .
```

Run pre-commit hooks:

```bash
pre-commit run --all-files
```

The repository also includes:

- GitHub Actions CI
- Ruff linting
- pre-commit hooks
- Docker support
- unit tests

---

## Reproducibility and engineering practices

The repository is structured around reproducible research and maintainable scientific software.

Key practices include:

- configurable YAML experiments;
- fixed random seeds;
- reproducibility utilities;
- modular dataset and transformation components;
- separate training and inference pipelines;
- checkpoint-based model selection;
- unit tests;
- automated linting;
- pre-commit hooks;
- GitHub Actions CI;
- Docker support;
- structured logging;
- native-space NIfTI reconstruction;
- automated preprocessing QC;
- automated prediction sanity checks;
- documented positive and negative experiments.

The goal is to make the workflow reproducible and maintainable rather than tying the project to a single notebook or training script.

---

## Documentation

More detailed technical documentation is available in:

- [`docs/architecture.md`](docs/architecture.md) — model architecture, data flow, preprocessing, inference, QC, and software design.
- [`docs/experiments.md`](docs/experiments.md) — controlled experiments, ablations, validation results, TTA evaluation, and interpretation of experimental findings.
- [`docs/test_qc.md`](docs/test_qc.md) — automated test-prediction QC, geometry checks, connected components, and diagnostic interpretation.
- `data/README.md` — dataset organization and preparation.

---

## Limitations

Several limitations should be considered when interpreting the results:

1. The dataset contains only 18 subjects.

2. The public test split does not provide ground-truth labels, preventing quantitative test evaluation.

3. The validation set contains only five subjects, so the validation score should not be interpreted as a broad estimate of clinical performance.

4. Qualitative inspection identified subject-dependent limitations, particularly around fine GM/WM boundaries and small CSF structures.

5. Some disconnected prediction components occur in difficult test cases and require anatomical interpretation rather than automatic removal.

6. MRI foreground containment checks use non-zero image voxels as a conservative proxy and should not be interpreted as a true anatomical brain mask.

7. The model was evaluated on IBSR-18 and should not be assumed to generalize directly to other scanners, acquisition protocols, datasets, or clinical populations.

8. The final checkpoint was selected using the held-out validation set. Evaluation on an independent external dataset would provide a stronger assessment of generalization.

9. The 0.50 sliding-window overlap produced only a small improvement over 0.25.

10. The CSF-weighted loss experiment slightly improved CSF Dice but reduced GM, WM, and overall mean foreground Dice under the tested configuration.

---

## Conclusion

This project demonstrates an end-to-end workflow for **3D medical image segmentation**, covering dataset validation, controlled preprocessing experiments, model development, training, checkpoint selection, quantitative validation, native-space inference, automated QC, and qualitative assessment.

The selected model is a **Residual 3D U-Net operating at native voxel spacing without N4 bias-field correction**. The best validation checkpoint was obtained at **epoch 391 of the 400-epoch training run**, based on the original 0.25-overlap evaluation.

The same checkpoint was subsequently evaluated with 0.50 sliding-window overlap for the final inference configuration, producing a small improvement in validation Dice. Controlled experiments with voxel-spacing normalization, conventional U-Net architecture, N4 preprocessing, and CSF-weighted loss did not outperform the selected baseline under the evaluated configurations.

The project emphasizes not only segmentation performance, but also **controlled experimentation, reproducibility, software quality, quantitative validation, diagnostic QC, and careful interpretation of model predictions**.

The main limitations are the small dataset size, absence of quantitative labels for the test split, subject-dependent segmentation challenges, and the lack of external validation.

---

## License

See [`LICENSE`](LICENSE).
