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

$$
0.8840 \rightarrow 0.9054
$$

Absolute improvement:

$$
0.9054 - 0.8840 = 0.0214
$$

## 200 → 300 epochs

Further extending training to 300 epochs improved the best validation score to:

$$
0.9114
$$

Absolute improvement over the 200-epoch run:

$$
0.9114 - 0.9054 = 0.0060
$$

## 300 → 400 epochs

The 400-epoch run reached a best validation mean foreground Dice of:

$$
0.9185
$$

at epoch 391.

Absolute improvement over the 300-epoch run:

$$
0.9185 - 0.9114 = 0.0071
$$

## Overall improvement

From the initial 100-epoch run to the final 400-epoch run:

$$
0.9185 - 0.8840 = 0.0345
$$

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

$$
0.8748
$$

compared with:

$$
0.8840
$$

for the 100-epoch native-spacing residual U-Net.

The difference was:

$$
0.8748 - 0.8840 = -0.0092
$$

Thus, 1 mm isotropic resampling reduced mean foreground Dice by **0.0092**.

The comparison with the final 400-epoch native model is even larger:

$$
0.9185 - 0.8748 = 0.0437
$$

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

The comparison suppor
