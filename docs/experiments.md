# Experiments

This document records the controlled experiments used to evaluate the IBSR-18 3D brain tissue segmentation pipeline.

The experiments investigate four main questions:

1. Does a residual 3D U-Net outperform a conventional 3D U-Net?
2. Does resampling MRI volumes to 1 mm isotropic spacing improve segmentation?
3. Does N4 MRI bias-field correction improve segmentation performance?
4. Does extending training beyond 100 epochs continue to improve validation performance?

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

The primary metric is the mean Dice score across the three foreground tissue classes.

**Mean Foreground Dice** = average Dice score across CSF, GM, and WM:

```math
\frac{\mathrm{Dice}_{CSF} + \mathrm{Dice}_{GM} + \mathrm{Dice}_{WM}}{3}
```

Background Dice is reported for completeness but is excluded from the primary model-selection metric.

---

# Common Training Configuration

Unless otherwise specified, the experiments use:

* 3D U-Net architecture
* residual or conventional convolutional blocks as specified by each experiment
* native or explicitly configured target spacing
* `96 × 96 × 96` training patches
* batch size of 1
* Dice + cross-entropy loss
* PyTorch
* MONAI
* foreground cropping
* intensity normalization
* randomized spatial/intensity augmentation
* validation-based checkpoint selection

The validation and inference pipelines do not use training-time augmentation.

The final model uses native voxel spacing and does not apply N4 bias-field correction.

---

# Experiment 1 — Residual 3D U-Net, Native Spacing

## Objective

Establish the primary segmentation configuration using a residual 3D U-Net while preserving each subject's native voxel spacing.

## Configuration

| Parameter          | Value    |
| ------------------ | -------- |
| Architecture       | 3D U-Net |
| Residual units     | 2        |
| Voxel spacing      | Native   |
| Target spacing     | None     |
| N4 bias correction | Disabled |
| Runtime N4         | Disabled |
| Precomputed N4     | Disabled |

The preprocessing pipeline includes RAS reorientation, intensity normalization, foreground cropping, random 3D patch sampling, and spatial/intensity augmentation.

## Initial 100-epoch result

The initial training run used 100 epochs.

| Parameter            | Value      |
| -------------------- | ---------- |
| Training epochs      | 100        |
| Best checkpoint      | Epoch 99   |
| Mean foreground Dice | **0.8840** |

The validation results were:

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9718 |
| CSF                 |     0.8437 |
| GM                  |     0.9055 |
| WM                  |     0.9028 |
| **Mean foreground** | **0.8840** |

The result indicated that the model was still capable of improving with additional optimization, motivating longer training runs using the same architecture and preprocessing configuration.

---

# Training-Duration Study

Because the native-spacing residual U-Net continued to improve after 100 epochs, training was progressively extended to determine whether additional optimization improved validation performance.

No architecture or preprocessing change was introduced between these runs.

| Training duration | Best mean foreground Dice | Best epoch |
| ----------------: | ------------------------: | ---------: |
|        100 epochs |                    0.8840 |         99 |
|        200 epochs |                    0.9054 |        199 |
|        300 epochs |                    0.9114 |        290 |
|    **400 epochs** |                **0.9185** |    **391** |

## 100 → 200 epochs

Extending training from 100 to 200 epochs improved mean foreground Dice from:

```math
0.8840 \rightarrow 0.9054
```

Absolute improvement:

```math
0.9054 - 0.8840 = 0.0214
```

## 200 → 300 epochs

Further extending training to 300 epochs improved the best validation score to:

```math
0.9114
```

Absolute improvement over the 200-epoch run:

```math
0.9114 - 0.9054 = 0.0060
```

## 300 → 400 epochs

The 400-epoch run reached a best validation mean foreground Dice of:

```math
0.9185
```

at epoch 391.

Absolute improvement over the 300-epoch run:

```math
0.9185 - 0.9114 = 0.0071
```

## Overall improvement

From the initial 100-epoch run to the final 400-epoch run:

```math
0.9185 - 0.8840 = 0.0345
```

Thus, extending training produced a **0.0345 absolute improvement in mean foreground Dice** without changing the architecture, voxel spacing, or preprocessing configuration.

---

# Final 400-Epoch Run

The final native-spacing residual U-Net was trained for 400 epochs.

The best validation checkpoint occurred at **epoch 391**:

```text
Epoch 391/400

train_loss = 0.2335
val_loss   = 0.2027

CSF Dice = 0.8938
GM Dice  = 0.9340
WM Dice  = 0.9276

Mean foreground Dice = 0.9185
```

The subsequent epochs did not exceed the epoch-391 result:

| Epoch | Mean foreground Dice |
| ----: | -------------------: |
|   392 |               0.9128 |
|   393 |               0.9134 |
|   394 |               0.9131 |
|   395 |               0.9131 |
|   396 |               0.9093 |
|   397 |               0.9085 |
|   398 |               0.9140 |
|   399 |               0.9057 |
|   400 |               0.9142 |

Therefore, **epoch 391 was selected as the final checkpoint** rather than the final training epoch.

The checkpoint is stored at:

```text
outputs/checkpoints/best_model.pt
```

---

# Final Validation Results

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

This represents an absolute improvement of:

| Metric              | Initial 100 epochs | Final 400-epoch run | Improvement |
| ------------------- | -----------------: | ------------------: | ----------: |
| CSF                 |             0.8437 |              0.8938 |     +0.0501 |
| GM                  |             0.9055 |              0.9340 |     +0.0285 |
| WM                  |             0.9028 |              0.9276 |     +0.0248 |
| **Mean foreground** |         **0.8840** |          **0.9185** | **+0.0345** |

The final model performs particularly strongly on GM and WM, while CSF remains the most challenging foreground class.

---

# Experiment 2 — Residual 3D U-Net, 1 mm Isotropic Spacing

## Objective

Evaluate whether resampling all subjects to a common 1 mm isotropic voxel spacing improves segmentation performance.

## Configuration

The model and training procedure were kept consistent with the native-spacing residual U-Net.

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

## Comparison with the initial native-spacing run

The 1 mm isotropic configuration achieved:

```math
0.8748
```

compared with:

```math
0.8840
```

for the 100-epoch native-spacing residual U-Net.

The difference was:

```math
0.8748 - 0.8840 = -0.0092
```

Thus, 1 mm isotropic resampling reduced mean foreground Dice by **0.0092**.

The comparison with the final 400-epoch native model is even larger:

```math
0.9185 - 0.8748 = 0.0437
```

## Result

For this dataset and model configuration, 1 mm isotropic resampling did not improve segmentation performance.

The final pipeline therefore retains native voxel spacing.

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

## Comparison with the initial residual model

| Metric              | Conventional U-Net | Residual U-Net | Improvement |
| ------------------- | -----------------: | -------------: | ----------: |
| CSF                 |             0.7730 |         0.8437 |     +0.0707 |
| GM                  |             0.8985 |         0.9055 |     +0.0070 |
| WM                  |             0.8764 |         0.9028 |     +0.0264 |
| **Mean foreground** |         **0.8493** |     **0.8840** | **+0.0347** |

Residual connections produced a substantial improvement in segmentation performance.

The largest class-specific improvement occurred for CSF, while GM and WM also improved.

## Result

The comparison supports the use of residual units in the final architecture.

The final model subsequently benefited further from extended training, reaching **0.9185 mean foreground Dice** after 400 epochs.

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

```math
I_{\mathrm{corrected}}
=
\exp
\left(
\log(\max(I,\epsilon))
-
\log(B)
\right)
```

with the original zero-valued background restored.

---

## N4 Preprocessing Quality Control

All **18 IBSR-18 subjects passed automated N4 preprocessing QC** before the segmentation experiment was started.

The QC process checked:

* image geometry preservation
* foreground voxel preservation
* finite image and bias-field values
* positive bias-field values
* reconstruction consistency
* spatial smoothness of the estimated bias field

The reconstruction check verified:

```math
I_{\mathrm{N4}}(x) \times B(x)
\approx
I_{\mathrm{raw}}(x)
```

using a P99 relative-error acceptance threshold of:

```math
1\times10^{-4}
```

All subjects passed this criterion.

The estimated bias fields also remained below the predefined spatial-gradient review threshold.

This confirms that the finalized N4 preprocessing pipeline itself was technically valid before evaluating its effect on segmentation.

---

## Segmentation Configuration

| Parameter           | Value             |
| ------------------- | ----------------- |
| Architecture        | Residual 3D U-Net |
| Residual units      | 2                 |
| Voxel spacing       | Native            |
| Precomputed N4      | Enabled           |
| Runtime N4          | Disabled          |
| Training epochs     | 100               |
| Selected checkpoint | Epoch 100         |

## Validation Results

| Class               |       Dice |
| ------------------- | ---------: |
| Background          |     0.9708 |
| CSF                 |     0.8408 |
| GM                  |     0.8984 |
| WM                  |     0.8939 |
| **Mean foreground** | **0.8777** |

## Comparison with the 100-epoch native/no-N4 model

| Metric              | Native baseline | N4 corrected |           Δ |
| ------------------- | --------------: | -----------: | ----------: |
| CSF                 |          0.8437 |       0.8408 |     −0.0029 |
| GM                  |          0.9055 |       0.8984 |     −0.0071 |
| WM                  |          0.9028 |       0.8939 |     −0.0089 |
| **Mean foreground** |      **0.8840** |   **0.8777** | **−0.0063** |

N4 correction therefore reduced mean foreground Dice by:

```math
0.8777 - 0.8840 = -0.0063
```

## Comparison with the final 400-epoch model

The final native/no-N4 model achieved:

```math
0.9185
```

while the N4 experiment achieved:

```math
0.8777
```

The difference is:

```math
0.9185 - 0.8777 = 0.0408
```

This comparison should be interpreted carefully because the N4 experiment used the shorter 100-epoch training schedule, whereas the final native model was trained for 400 epochs.

The controlled N4 comparison against the 100-epoch native baseline nevertheless shows that N4 did not provide a measurable benefit under the evaluated training configuration.

## Result

N4 correction did not improve segmentation performance.

Importantly, this result is **not attributed to a failure of the N4 preprocessing implementation**: all 18 subjects passed the independent preprocessing QC procedure.

For this dataset and model configuration, the simpler native/no-N4 preprocessing pipeline is therefore preferred.

---

# Overall Comparison

The main controlled experiment results are:

| Experiment         | Architecture       | Spacing        | N4     |  Epochs | Mean FG Dice |
| ------------------ | ------------------ | -------------- | ------ | ------: | -----------: |
| Native baseline    | Residual U-Net     | Native         | No     |     100 |       0.8840 |
| Native extended    | Residual U-Net     | Native         | No     |     200 |       0.9054 |
| Native extended    | Residual U-Net     | Native         | No     |     300 |       0.9114 |
| **Final**          | **Residual U-Net** | **Native**     | **No** | **400** |   **0.9185** |
| 1 mm isotropic     | Residual U-Net     | 1 mm isotropic | No     |     100 |       0.8748 |
| Conventional U-Net | Conventional U-Net | Native         | No     |     100 |       0.8493 |
| N4 preprocessing   | Residual U-Net     | Native         | Yes    |     100 |       0.8777 |

---

# Main Findings

## 1. Extended training substantially improved the native residual model

The same architecture and preprocessing configuration improved consistently as training was extended:

```math
0.8840
\rightarrow
0.9054
\rightarrow
0.9114
\rightarrow
0.9185
```

for 100, 200, 300, and 400 epochs respectively.

The overall improvement from the initial 100-epoch run to the final model was:

```math
0.9185 - 0.8840 = 0.0345
```

This demonstrates that the initial 100-epoch result should be regarded as a baseline rather than the final performance of the native residual architecture.

## 2. Residual architecture improved performance

The conventional U-Net achieved:

```math
0.8493
```

while the comparable 100-epoch residual U-Net achieved:

```math
0.8840
```

The improvement was:

```math
0.8840 - 0.8493 = 0.0347
```

The largest class-specific improvement was observed for CSF.

## 3. Native spacing performed better than 1 mm isotropic resampling

The 100-epoch native residual U-Net achieved:

```math
0.8840
```

compared with:

```math
0.8748
```

after resampling to 1 mm isotropic spacing.

The native configuration therefore performed:

```math
0.8840 - 0.8748 = 0.0092
```

Dice points better.

## 4. N4 correction did not improve segmentation

The N4 configuration achieved:

```math
0.8777
```

compared with:

```math
0.8840
```

for the comparable 100-epoch native/no-N4 configuration.

N4 therefore reduced mean foreground Dice by:

```math
0.0063
```

Although N4 did not improve segmentation, the preprocessing implementation itself was independently validated on all 18 subjects.

---

# Final Model Selection

The final selected configuration is:

* **Residual 3D U-Net**
* **2 residual units**
* **Native voxel spacing**
* **No N4 preprocessing**
* **400 training epochs**
* **Best checkpoint: epoch 391**
* **Mean foreground validation Dice: 0.9185**

The final validation scores are:

| Tissue              |       Dice |
| ------------------- | ---------: |
| CSF                 |     0.8938 |
| GM                  |     0.9340 |
| WM                  |     0.9276 |
| **Mean foreground** | **0.9185** |

The selected checkpoint is:

```text
outputs/checkpoints/best_model.pt
```

The model was selected based on the highest validation mean foreground Dice achieved during the controlled training and experiment process.

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
* native-space reconstruction

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

The experiments support four main conclusions.

First, **training duration had a substantial effect** on the native-spacing residual U-Net. The model improved from 0.8840 at 100 epochs to 0.9185 at the best checkpoint of the 400-epoch run.

Second, **residual connections improved performance** compared with the conventional U-Net, providing a 0.0347 absolute improvement in mean foreground Dice in the 100-epoch comparison.

Third, **native voxel spacing was preferable to 1 mm isotropic resampling** for the evaluated dataset and model configuration.

Fourth, **N4 bias-field correction was technically successful but did not improve segmentation performance**. This distinction is important: a preprocessing method can be implemented correctly and pass independent QC without necessarily improving the downstream segmentation task.

The final pipeline therefore favors the simpler native/no-N4 configuration with extended training.

---

# Limitations

The experimental conclusions should be interpreted within the limitations of the dataset and evaluation setup:

1. IBSR-18 contains a relatively small number of subjects.
2. The validation set contains only five subjects.
3. The test split used here has no available ground-truth labels in this project configuration.
4. The validation set is therefore the primary quantitative evaluation set.
5. Qualitative test inspection revealed subject-dependent errors.
6. CSF remains more difficult to segment than GM and WM.
7. Some isolated false-positive predictions occur in difficult cases.
8. The results should not be assumed to generalize directly to other MRI scanners, acquisition protocols, datasets, or clinical populations.
9. The final checkpoint was selected using the validation set; an external dataset would provide a stronger assessment of generalization.

---

# Final Conclusion

The final selected pipeline is a **native-resolution residual 3D U-Net without N4 preprocessing**, trained for 400 epochs with checkpoint selection based on validation performance.

The progression of the native residual model was:

```text
100 epochs  → 0.8840
200 epochs  → 0.9054
300 epochs  → 0.9114
400 epochs  → 0.9185
```

The best checkpoint occurred at **epoch 391**, achieving:

* **CSF Dice: 0.8938**
* **GM Dice: 0.9340**
* **WM Dice: 0.9276**
* **Mean foreground Dice: 0.9185**

The controlled experiments show that:

* residual architecture improved segmentation substantially;
* extended training produced a further improvement without changing the architecture;
* 1 mm isotropic resampling did not improve performance;
* N4 bias-field correction did not improve performance despite passing independent preprocessing QC.

The completed workflow additionally validates unseen test predictions through native-space reconstruction, automated structural sanity checks, and qualitative inspection.

The resulting project therefore provides both a reproducible **medical image segmentation experiment** and a modular **research software engineering implementation**.
