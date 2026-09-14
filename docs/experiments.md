# Experiments

This document records the controlled experiments used to evaluate the IBSR-18 3D brain tissue segmentation pipeline.

The experiments progressively evaluate:

1. residual versus conventional 3D U-Net architecture,
2. native versus 1 mm isotropic voxel spacing, and
3. the effect of N4 MRI bias-field correction.

All experiments use the same train/validation split and evaluation protocol unless explicitly stated otherwise.

## Dataset

The project uses the IBSR-18 T1-weighted brain MRI dataset.

### Splits

* **Training:** IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06, IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18
* **Validation:** IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17
* **Test:** IBSR_02, IBSR_10, IBSR_15

The public test split does not contain ground-truth segmentation labels, so quantitative evaluation is performed on the validation set.

### Segmentation labels

| Label | Tissue            |
| ----: | ----------------- |
|     0 | Background        |
|     1 | CSF               |
|     2 | Gray matter (GM)  |
|     3 | White matter (WM) |

The primary metric is the mean Dice score across the three foreground tissue classes:

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

# Experiment 1 — Residual 3D U-Net, Native Spacing

### Objective

Establish the main segmentation baseline using a residual 3D U-Net while preserving each subject's native voxel spacing.

### Configuration

* **Architecture:** 3D U-Net
* **Residual units:** 2
* **Voxel spacing:** Native
* **Target spacing:** None
* **Training epochs:** 100
* **Best checkpoint:** Epoch 98
* **N4 bias correction:** Disabled
* **Runtime N4:** Disabled
* **Precomputed N4:** Disabled

The preprocessing pipeline includes RAS reorientation, percentile-based intensity normalization, foreground cropping, random 3D patch sampling, and spatial/intensity augmentation.

### Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9826 |
| CSF                 |     0.8359 |
| GM                  |     0.9104 |
| WM                  |     0.8949 |
| **Mean foreground** | **0.8804** |

### Per-subject foreground Dice

| Subject | Mean foreground Dice |
| ------- | -------------------: |
| IBSR_11 |               0.8806 |
| IBSR_12 |               0.8926 |
| IBSR_13 |               0.8501 |
| IBSR_14 |               0.8969 |
| IBSR_17 |               0.8818 |

### Result

The residual 3D U-Net provides the strongest baseline and is used as the reference configuration for subsequent preprocessing experiments.

---

# Experiment 2 — Residual 3D U-Net, 1 mm Isotropic Spacing

### Objective

Evaluate whether resampling all subjects to a common 1 mm isotropic voxel spacing improves segmentation performance.

### Configuration

The model and training procedure are kept consistent with Experiment 1.

* **Architecture:** 3D U-Net
* **Residual units:** 2
* **Voxel spacing:** 1 × 1 × 1 mm
* **N4 bias correction:** Disabled
* **Training epochs:** 100
* **Best checkpoint:** Epoch 99

### Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9813 |
| CSF                 |     0.8259 |
| GM                  |     0.9073 |
| WM                  |     0.8911 |
| **Mean foreground** | **0.8748** |

### Comparison with Experiment 1

$$
0.8748 - 0.8804 = -0.0056
$$

Resampling to 1 mm isotropic spacing reduced mean foreground Dice by **0.0056**.

### Result

For this dataset and model configuration, 1 mm isotropic resampling did not improve segmentation performance. The native-spacing configuration is therefore retained for subsequent experiments.

---

# Experiment 3 — Conventional 3D U-Net, Native Spacing

### Objective

Measure the contribution of residual units by comparing the residual U-Net against an otherwise comparable conventional 3D U-Net.

### Configuration

* **Architecture:** 3D U-Net
* **Residual units:** 0
* **Voxel spacing:** Native
* **N4 bias correction:** Disabled
* **Training epochs:** 100
* **Best checkpoint:** Epoch 99

### Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9810 |
| CSF                 |     0.7730 |
| GM                  |     0.8985 |
| WM                  |     0.8764 |
| **Mean foreground** | **0.8493** |

### Comparison with Experiment 1

| Metric              | Conventional U-Net | Residual U-Net | Improvement |
| ------------------- | -----------------: | -------------: | ----------: |
| CSF                 |             0.7730 |         0.8359 |     +0.0629 |
| GM                  |             0.8985 |         0.9104 |     +0.0119 |
| WM                  |             0.8764 |         0.8949 |     +0.0185 |
| **Mean foreground** |         **0.8493** |     **0.8804** | **+0.0311** |

### Result

Adding residual units produced a substantial improvement in segmentation performance, with the largest gain observed for CSF.

The residual 3D U-Net is therefore retained as the primary model architecture.

---

# Experiment 4 — Residual 3D U-Net with Precomputed N4 Correction

### Objective

Evaluate whether MRI bias-field correction improves tissue segmentation when applied as deterministic preprocessing.

### Motivation

T1-weighted MRI can contain low-frequency intensity inhomogeneity (bias field). N4 bias-field correction was evaluated as a preprocessing step while keeping the model architecture and native voxel spacing unchanged.

### N4 preprocessing

N4 correction was performed once per subject and stored as precomputed volumes to avoid repeating the computationally expensive correction during every training epoch.

Final N4 configuration:

* **Implementation:** SimpleITK N4BiasFieldCorrectionImageFilter
* **Shrink factor:** 4
* **Maximum iterations:** `[50, 50, 50, 50]`
* **Convergence threshold:** 0.001
* **B-spline control points:** `[4, 4, 4]`
* **Foreground mask:** `image > 0`
* **Runtime N4 during training:** Disabled
* **Precomputed N4 during training:** Enabled
* **Target spacing:** Native

The corrected image was generated using the estimated multiplicative bias field:

$$
I_{corrected}
=
\exp\left(
\log(\max(I, \epsilon))
-
\log(B)
\right)
$$

with the original zero-valued background restored.

### N4 preprocessing quality control

All **18 IBSR-18 subjects** passed automated QC before training.

The QC verified:

* image geometry preservation,
* matching foreground voxel counts,
* finite image and bias-field values,
* valid positive bias fields,
* reconstruction consistency,
* spatial smoothness of the estimated bias field.

The reconstruction check used:

$$
I_{N4}(x) \times B(x) \approx I_{raw}(x)
$$

with a P99 relative-error acceptance threshold of `1 × 10⁻⁴`.

All 18 subjects passed this criterion, with observed P99 relative reconstruction errors approximately in the range:

$$
3.4\times10^{-7}
\; \text{to} \;
4.4\times10^{-7}
$$

The bias-field P95 spatial gradient also remained below the defined review threshold for every subject.

### Segmentation configuration

* **Architecture:** Residual 3D U-Net
* **Residual units:** 2
* **Voxel spacing:** Native
* **Precomputed N4:** Enabled
* **Runtime N4:** Disabled
* **Training epochs:** 100
* **Best checkpoint:** Epoch 100

### Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9708 |
| CSF                 |     0.8408 |
| GM                  |     0.8984 |
| WM                  |     0.8939 |
| **Mean foreground** | **0.8777** |

### Per-subject results

| Subject | Mean foreground Dice |
| ------- | -------------------: |
| IBSR_11 |               0.8783 |
| IBSR_12 |               0.8767 |
| IBSR_13 |               0.8545 |
| IBSR_14 |               0.8976 |
| IBSR_17 |               0.8814 |

### Comparison with Experiment 1

| Metric              | Native baseline | N4 corrected |           Δ |
| ------------------- | --------------: | -----------: | ----------: |
| CSF                 |          0.8359 |       0.8408 |     +0.0049 |
| GM                  |          0.9104 |       0.8984 |     −0.0120 |
| WM                  |          0.8949 |       0.8939 |     −0.0010 |
| **Mean foreground** |      **0.8804** |   **0.8777** | **−0.0027** |

### Result

N4 correction produced a small reduction in mean foreground Dice:

$$
0.8777 - 0.8804 = -0.0027
$$

The effect is therefore approximately **0.27 percentage points** and substantially smaller than the improvement obtained from residual connections.

The result suggests that, for this dataset and model configuration, N4 bias-field correction is **approximately neutral but does not provide a measurable segmentation benefit**.

Importantly, the absence of improvement is not attributed to a preprocessing failure: the finalized N4 pipeline passed automated QC for all 18 subjects.

---

# Overall Comparison

| Experiment | Architecture       | Spacing        | N4  | Mean FG Dice | Best Epoch |
| ---------- | ------------------ | -------------- | --- | -----------: | ---------: |
| **1**      | Residual U-Net     | Native         | No  |   **0.8804** |         98 |
| **2**      | Residual U-Net     | 1 mm isotropic | No  |       0.8748 |         99 |
| **3**      | Conventional U-Net | Native         | No  |       0.8493 |         99 |
| **4**      | Residual U-Net     | Native         | Yes |       0.8777 |        100 |

## Main findings

### 1. Residual architecture had the largest effect

The residual U-Net improved mean foreground Dice from **0.8493 to 0.8804**, an absolute improvement of **0.0311**.

### 2. Native spacing performed slightly better

The native-spacing residual U-Net achieved **0.8804**, compared with **0.8748** after resampling to 1 mm isotropic spacing.

### 3. N4 correction did not improve the final segmentation score

The N4 configuration achieved **0.8777**, compared with **0.8804** for the native-spacing baseline.

Although N4 did not improve the score, the preprocessing itself was validated successfully across all 18 subjects.

## Selected configuration

Based on the controlled experiments, the current best-performing configuration is:

* **Residual 3D U-Net**
* **2 residual units**
* **Native voxel spacing**
* **No N4 preprocessing**
* **Mean foreground Dice: 0.8804**

This configuration is used as the primary model for subsequent final evaluation and qualitative visualization.
