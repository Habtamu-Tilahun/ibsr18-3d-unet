# Test Prediction Quality Control

## Scope

This project uses IBSR_02, IBSR_10, and IBSR_15 as an unlabeled test set. The test cases do **not** have reference segmentation masks, so quantitative Dice/IoU performance is not reported for these subjects.

The test QC therefore checks native-space integrity and performs targeted qualitative anatomical review.

## Automated checks

`scripts/check_test_predictions.py` verifies:

- native MRI/prediction shape agreement;
- affine agreement;
- finite prediction values;
- allowed labels `{0, 1, 2, 3}`;
- non-empty foreground;
- `uint8` prediction dtype;
- native voxel spacing and derived tissue volumes;
- connected-component summaries for CSF, GM, and WM;
- foreground slice continuity;
- predicted foreground outside the nonzero MRI region.

The last item is **diagnostic rather than a hard failure**. The nonzero MRI region is a spatial proxy and is not a validated brain mask. A small number of predicted foreground voxels outside this proxy therefore does not by itself establish a segmentation error.

## Final test results

All three test predictions passed the hard spatial/format checks and were restored to their native `(256, 128, 256)` grid with matching affine information.

| Subject | Foreground voxels | Outside MRI foreground | Interpretation |
|---|---:|---:|---|
| IBSR_02 | 1,113,490 | 2,687 (0.2413%) | Small diagnostic discrepancy; no major anatomical concern identified |
| IBSR_10 | 916,972 | 6,619 (0.7218%) | Largest proxy discrepancy; no major anatomical concern identified |
| IBSR_15 | 1,299,583 | 7,066 (0.5437%) | Small diagnostic discrepancy; targeted anatomical review completed |

These percentages are calculated relative to predicted foreground voxels and should not be interpreted as false-positive rates.

## IBSR_15 qualitative review

IBSR_15 received additional review because its WM prediction contained multiple connected components. The disconnected components were investigated individually rather than removed automatically.

### WM components

The largest WM component contains **386,402 voxels**. A second component contains approximately **38,851 voxels**, corresponding to about 9.1% of the predicted WM volume.

Its approximate voxel-space bounding box was:

```text
x: 78–170
y: 18–64
z: 75–131
```

The component centroid was approximately:

```text
[124.67, 43.29, 95.41]
```

Its approximate physical bounding box was:

```text
[-148.16, 27.00, 62.78]
to
[-71.15, 96.00, 109.65]
```

The component was located in an inferior/posterior anatomical region and was predominantly surrounded by GM. Approximately 99.91% of its voxels were contained within the largest connected predicted brain mask. The closest main WM component was approximately 5 voxels away.

Its coherent 3D extent, anatomical location, and surrounding tissue did not provide convincing evidence that it was random prediction noise or an outside-brain false positive. It was therefore **not merged or removed**.

A further WM component containing approximately **540 voxels** was also inspected against the MRI.

Its approximate voxel-space bounding box was:

```text
x: 108–130
y: 71–79
z: 78–126
```

The centroid was approximately:

```text
[114.4, 74.6, 100.7]
```

Its approximate physical extent was:

```text
19.3 × 13.5 × 41 mm
```

A 6-neighbor boundary analysis found approximately:

```text
Adjacent GM voxels: 397
Adjacent background voxels: 304
Adjacent CSF voxels: 0
Adjacent WM voxels: 0
```

The component formed an elongated, spatially structured region within the MRI volume. There was no convincing evidence that it represented an outside-brain false positive or a sliding-window inference artifact. However, its exact anatomical identity and connectivity to the main WM component could not be established from qualitative prediction inspection alone.

It was therefore **retained as part of the raw prediction rather than removed using an arbitrary component-size threshold**.

### CSF

IBSR_15 contains **9,096 predicted CSF voxels** (approximately 9.56 mL) in 11 connected components. The largest two components contain 5,165 and 1,880 voxels.

Inspection of representative axial and sagittal views showed CSF predictions in plausible ventricular/deep CSF locations, without obvious widespread leakage into brain parenchyma or visible patch-shaped inference artifacts.

Fine sulcal/peripheral CSF appears less completely represented than larger CSF spaces. This is treated as a qualitative limitation rather than a quantitative failure because no test reference segmentation is available.

### Overall qualitative assessment

Across IBSR_02, IBSR_10, and IBSR_15, the predictions show plausible whole-brain coverage, recognizable cortical GM and deep WM organization, major ventricular/deep CSF structures, and no convincing large sliding-window seams.

The recurring qualitative limitation is **coarse/irregular GM-WM boundary detail and simplification of very small sulcal CSF spaces**.

These observations are qualitative and should not be interpreted as test-set accuracy measurements.

## Interpretation policy

Connected-component count is not used as a standalone pass/fail criterion. Anatomical structures such as ventricular CSF spaces and separate inferior/posterior WM structures can naturally form multiple components.

In particular, a disconnected component is not automatically a segmentation error. Components may reflect genuine anatomical separation, thin or uncertain connections, or prediction uncertainty. Conversely, connectedness alone does not establish anatomical correctness.

The repository therefore retains raw model predictions without automatic component filtering. QC reports component statistics and highlights cases requiring anatomical review rather than applying blind post-processing rules.
