# Experiments

This document records the controlled experiments used to evaluate the IBSR-18 3D brain tissue segmentation pipeline.

The experiments investigate three main questions:

1. Does a residual 3D U-Net outperform a conventional 3D U-Net?
2. Does resampling MRI volumes to 1 mm isotropic spacing improve segmentation?
3. Does N4 MRI bias-field correction improve segmentation performance?

All experiments use the same train/validation split and evaluation protocol unless explicitly stated otherwise.

---

# Dataset

The project uses the IBSR-18 T1-weighted brain MRI dataset.

## Splits

### Training

IBSR_01, IBSR_03, IBSR_04, IBSR_05, IBSR_06, IBSR_07, IBSR_08, IBSR_09, IBSR_16, IBSR_18

### Validation

IBSR_11, IBSR_12, IBSR_13, IBSR_14, IBSR_17

### Test

IBSR_02, IBSR_10, IBSR_15

The test split used by this project does not contain ground-truth segmentation labels. Therefore, quantitative model selection is performed exclusively on the validation set.

The test subjects are subsequently used for unseen-subject inference and qualitative generalization assessment.

## Segmentation labels

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

Background Dice is reported for completeness but is excluded from the primary model-selection metric.

---

# Common training configuration

Unless otherwise specified, the experiments use:

* 3D U-Net architecture
* 100 training epochs
* native or explicitly configured target spacing
* `96 × 96 × 96` training patches
* batch size of 1
* Dice + cross-entropy loss
* PyTorch
* MONAI
* foreground cropping
* intensity normalization
* randomized spatial/intensity augmentation
* validation after training
* best validation checkpoint selection

The validation pipeline does not use training-time augmentation.

---

# Experiment 1 — Residual 3D U-Net, Native Spacing

## Objective

Establish the primary segmentation baseline using a residual 3D U-Net while preserving each subject's native voxel spacing.

## Configuration

| Parameter           | Value    |
| ------------------- | -------- |
| Architecture        | 3D U-Net |
| Residual units      | 2        |
| Voxel spacing       | Native   |
| Target spacing      | None     |
| Training epochs     | 100      |
| Selected checkpoint | Epoch 99 |
| N4 bias correction  | Disabled |
| Runtime N4          | Disabled |
| Precomputed N4      | Disabled |

The preprocessing pipeline includes RAS reorientation, intensity normalization, foreground cropping, random 3D patch sampling, and spatial/intensity augmentation.

## Final validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9718 |
| CSF                 |     0.8437 |
| GM                  |     0.9055 |
| WM                  |     0.9028 |
| **Mean foreground** | **0.8840** |

The final checkpoint was selected based on validation performance.

## Per-subject validation results

The final checkpoint was evaluated independently on IBSR_11, IBSR_12, IBSR_13, IBSR_14, and IBSR_17.

| Subject |    CSF |     GM |     WM |
| ------- | -----: | -----: | -----: |
| IBSR_11 | 0.8183 | 0.9032 | 0.9212 |
| IBSR_12 | 0.8624 | 0.8882 | 0.9029 |
| IBSR_13 | 0.8032 | 0.9043 | 0.8790 |
| IBSR_14 | 0.8448 | 0.9206 | 0.9149 |
| IBSR_17 | 0.8899 | 0.9115 | 0.8959 |

## Result

The residual 3D U-Net provides the strongest overall configuration among the evaluated experiments and is selected as the final model.

---

# Experiment 2 — Residual 3D U-Net, 1 mm Isotropic Spacing

## Objective

Evaluate whether resampling all subjects to a common 1 mm isotropic voxel spacing improves segmentation performance.

## Configuration

The model and training procedure are kept consistent with Experiment 1.

| Parameter           | Value        |
| ------------------- | ------------ |
| Architecture        | 3D U-Net     |
| Residual units      | 2            |
| Voxel spacing       | 1 × 1 × 1 mm |
| N4 bias correction  | Disabled     |
| Training epochs     | 100          |
| Selected checkpoint | Epoch 99     |

## Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9813 |
| CSF                 |     0.8259 |
| GM                  |     0.9073 |
| WM                  |     0.8911 |
| **Mean foreground** | **0.8748** |

## Comparison with Experiment 1

$$
0.8748 - 0.8840 = -0.0092
$$

Resampling to 1 mm isotropic spacing reduced mean foreground Dice by **0.0092**.

## Result

For this dataset and model configuration, 1 mm isotropic resampling did not improve segmentation performance.

The native-spacing configuration was therefore retained.

---

# Experiment 3 — Conventional 3D U-Net, Native Spacing

## Objective

Measure the contribution of residual units by comparing the residual U-Net against an otherwise comparable conventional 3D U-Net.

## Configuration

| Parameter           | Value    |
| ------------------- | -------- |
| Architecture        | 3D U-Net |
| Residual units      | 0        |
| Voxel spacing       | Native   |
| N4 bias correction  | Disabled |
| Training epochs     | 100      |
| Selected checkpoint | Epoch 99 |

## Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9810 |
| CSF                 |     0.7730 |
| GM                  |     0.8985 |
| WM                  |     0.8764 |
| **Mean foreground** | **0.8493** |

## Comparison with the final residual model

| Metric              | Conventional U-Net | Residual U-Net | Improvement |
| ------------------- | -----------------: | -------------: | ----------: |
| CSF                 |             0.7730 |         0.8437 |     +0.0707 |
| GM                  |             0.8985 |         0.9055 |     +0.0070 |
| WM                  |             0.8764 |         0.9028 |     +0.0264 |
| **Mean foreground** |         **0.8493** |     **0.8840** | **+0.0347** |

## Result

Residual connections produced a substantial improvement in segmentation performance.

The largest class-specific improvement occurred for CSF, while GM and WM also improved.

This experiment supports retaining the residual architecture as the final model.

---

# Experiment 4 — Residual 3D U-Net with Precomputed N4 Correction

## Objective

Evaluate whether MRI bias-field correction improves tissue segmentation when applied as deterministic preprocessing.

## Motivation

T1-weighted MRI can contain low-frequency intensity inhomogeneity, commonly referred to as bias field.

N4 bias-field correction was therefore evaluated as a preprocessing intervention while keeping the model architecture and native voxel spacing unchanged.

## N4 preprocessing

N4 correction was performed once per subject and stored as precomputed volumes to avoid repeating the computationally expensive correction during every training epoch.

Final N4 configuration:

| Parameter               | Value                                      |
| ----------------------- | ------------------------------------------ |
| Implementation          | SimpleITK N4BiasFieldCorrectionImageFilter |
| Shrink factor           | 4                                          |
| Maximum iterations      | `[50, 50, 50, 50]`                         |
| Convergence threshold   | 0.001                                      |
| B-spline control points | `[4, 4, 4]`                                |
| Foreground mask         | `image > 0`                                |
| Runtime N4              | Disabled                                   |
| Precomputed N4          | Enabled                                    |
| Target spacing          | Native                                     |

The corrected image was generated from the estimated multiplicative bias field:

$$
I_{\mathrm{corrected}}
=
\exp
\left(
\log(\max(I,\epsilon))
-
\log(B)
\right)
$$

with the original zero-valued background restored.

---

## N4 preprocessing quality control

All **18 IBSR-18 subjects passed automated N4 preprocessing QC** before the segmentation experiment was started.

The QC process checked:

* image geometry preservation
* foreground voxel preservation
* finite image and bias-field values
* positive bias-field values
* reconstruction consistency
* spatial smoothness of the estimated bias field

The reconstruction check verified:

$$
I_{\mathrm{N4}}(x) \times B(x)
\approx
I_{\mathrm{raw}}(x)
$$

using a P99 relative-error acceptance threshold of:

$$
1\times10^{-4}
$$

All subjects passed this criterion.

The estimated bias fields also remained below the predefined spatial-gradient review threshold.

This confirms that the finalized N4 preprocessing pipeline itself was technically valid before evaluating its effect on segmentation.

---

## Segmentation configuration

| Parameter           | Value             |
| ------------------- | ----------------- |
| Architecture        | Residual 3D U-Net |
| Residual units      | 2                 |
| Voxel spacing       | Native            |
| Precomputed N4      | Enabled           |
| Runtime N4          | Disabled          |
| Training epochs     | 100               |
| Selected checkpoint | Epoch 100         |

## Validation results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9708 |
| CSF                 |     0.8408 |
| GM                  |     0.8984 |
| WM                  |     0.8939 |
| **Mean foreground** | **0.8777** |

## Comparison with the final native/no-N4 model

| Metric              | Native baseline | N4 corrected |           Δ |
| ------------------- | --------------: | -----------: | ----------: |
| CSF                 |          0.8437 |       0.8408 |     −0.0029 |
| GM                  |          0.9055 |       0.8984 |     −0.0071 |
| WM                  |          0.9028 |       0.8939 |     −0.0089 |
| **Mean foreground** |      **0.8840** |   **0.8777** | **−0.0063** |

## Result

N4 correction did not improve the final segmentation performance.

$$
0.8777 - 0.8840 = -0.0063
$$

The N4 configuration therefore performed **0.0063 lower in mean foreground Dice** than the selected native/no-N4 model.

The result is not attributed to a failure of the N4 preprocessing implementation: all 18 subjects passed the independent preprocessing QC procedure.

For this dataset and model configuration, the simpler native/no-N4 preprocessing pipeline is therefore preferred.

---

# Overall Comparison

| Experiment    | Architecture       | Spacing        | N4  | Mean FG Dice | Selected Epoch |
| ------------- | ------------------ | -------------- | --- | -----------: | -------------: |
| **1 — Final** | Residual U-Net     | Native         | No  |   **0.8840** |         **99** |
| 2             | Residual U-Net     | 1 mm isotropic | No  |       0.8748 |             99 |
| 3             | Conventional U-Net | Native         | No  |       0.8493 |             99 |
| 4             | Residual U-Net     | Native         | Yes |       0.8777 |            100 |

## Main findings

### 1. Residual architecture had the largest effect

The conventional U-Net achieved a mean foreground Dice of 0.8493.

Adding residual units increased this to 0.8840:

$$
0.8840 - 0.8493 = 0.0347
$$

This is an absolute improvement of **0.0347 Dice points**.

The largest class-specific improvement was observed for CSF.

### 2. Native spacing performed better than 1 mm isotropic resampling

The native residual U-Net achieved:

$$
0.8840
$$

compared with:

$$
0.8748
$$

after resampling to 1 mm isotropic spacing.

The native configuration therefore performed **0.0092 Dice points better**.

### 3. N4 correction did not improve segmentation

The N4 configuration achieved:

$$
0.8777
$$

compared with:

$$
0.8840
$$

for the final native/no-N4 model.

N4 therefore reduced mean foreground Dice by **0.0063**.

Although N4 did not improve segmentation, the preprocessing implementation itself was independently validated on all 18 subjects.

---

# Final Model Selection

The selected configuration is:

* **Residual 3D U-Net**
* **2 residual units**
* **Native voxel spacing**
* **No N4 preprocessing**
* **100 training epochs**
* **Selected checkpoint: epoch 99**
* **Mean foreground validation Dice: 0.8840**

The selection is based on validation performance across the controlled experiments.

The final checkpoint is stored at:

```text
outputs/checkpoints/best_model.pt
```

---

# Final Validation Assessment

The final model achieved:

| Tissue              |       Dice |
| ------------------- | ---------: |
| CSF                 |     0.8437 |
| GM                  |     0.9055 |
| WM                  |     0.9028 |
| **Mean foreground** | **0.8840** |

The model performs particularly strongly on GM and WM. CSF remains the most challenging foreground class, reflecting the difficulty of delineating smaller ventricular and thin sulcal CSF structures.

---

# Test-Set Inference

After selecting the final checkpoint, inference was performed on the three unseen test subjects:

* IBSR_02
* IBSR_10
* IBSR_15

Inference used:

* ROI size: `96 × 96 × 96`
* sliding-window batch size: 1
* overlap: 0.25
* Gaussian blending
* final native-space reconstruction

The test predictions were restored from the cropped inference representation to the original native NIfTI geometry.

## Native-space sanity checks

All three subjects passed:

* spatial-shape matching
* affine matching
* finite-value validation
* label-range validation
* non-empty foreground validation
* `uint8` output validation

Therefore, the generated test predictions are structurally valid native-space NIfTI segmentations.

Because ground-truth segmentations are unavailable for the test subjects, no quantitative test Dice is reported.

---

# Qualitative Test Assessment

Qualitative inspection was performed in axial, coronal, and sagittal planes.

## IBSR_02

The overall segmentation is anatomically plausible, with continuous cortical coverage and recognizable GM/WM organization.

A disconnected inferior WM prediction was observed in the coronal view. This is considered a likely false-positive region rather than a coherent anatomical component.

## IBSR_10

IBSR_10 produced the strongest qualitative prediction among the three unseen subjects.

Observed strengths include:

* continuous cortical coverage
* detailed GM/WM organization
* well-formed ventricular CSF
* good agreement between predicted structures and MRI appearance
* no obvious disconnected islands in the inspected views

## IBSR_15

IBSR_15 was the most challenging qualitative case.

Observed limitations include:

* coarser WM regions
* apparent WM over-segmentation
* under-segmented ventricular CSF
* localized cortical coverage loss
* small off-brain prediction islands

These findings demonstrate that good aggregate validation performance does not guarantee uniformly strong generalization across unseen subjects.

No obvious rectangular or grid-aligned seams characteristic of sliding-window inference were observed.

Localized off-brain predictions were observed, but their exact cause cannot be established from qualitative inspection alone.

---

# Interpretation

The experiments support three conclusions.

First, **architecture had the strongest measured effect** among the evaluated factors. Residual connections improved mean foreground Dice substantially compared with the conventional U-Net.

Second, **native voxel spacing was preferable to 1 mm isotropic resampling** for this dataset and model configuration. Resampling did not provide an accuracy benefit.

Third, **N4 bias-field correction was technically successful but did not improve segmentation performance**. This is an important distinction: a preprocessing method can be implemented correctly without necessarily improving the downstream task.

The final pipeline therefore favors the simpler native/no-N4 configuration.

---

# Limitations

The experimental conclusions should be interpreted within the limitations of the dataset and evaluation setup:

1. IBSR-18 contains a relatively small number of subjects.
2. The test split used here has no available ground-truth labels in this project configuration.
3. The validation set is therefore the primary quantitative evaluation set.
4. Qualitative test inspection revealed subject-dependent errors.
5. CSF remains more difficult to segment than GM and WM.
6. Some isolated false-positive predictions occur in difficult cases.
7. The results should not be assumed to generalize directly to other MRI scanners, acquisition protocols, or clinical populations.

---

# Final Conclusion

The final selected pipeline is a native-resolution residual 3D U-Net without N4 preprocessing.

It achieved a **0.8840 mean foreground Dice** on the held-out validation set, with class-specific Dice scores of:

* CSF: **0.8437**
* GM: **0.9055**
* WM: **0.9028**

The controlled experiments demonstrate that residual architecture improved performance substantially, whereas 1 mm isotropic resampling and N4 bias-field correction did not improve the final validation result.

The completed workflow additionally validates unseen test predictions through native-space reconstruction, automated structural sanity checks, and qualitative inspection.

The resulting project therefore provides both a reproducible **medical image segmentation experiment** and a modular **research software engineering implementation**.
