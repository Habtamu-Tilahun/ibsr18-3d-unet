from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PRED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "test"
)

TEST_SUBJECTS = (
    "IBSR_02",
    "IBSR_10",
    "IBSR_15",
)


def main() -> None:
    print("=" * 70)
    print("IBSR-18 TEST PREDICTION SANITY CHECK")
    print("=" * 70)

    all_passed = True

    for subject_id in TEST_SUBJECTS:

        print()
        print(f"{subject_id}")
        print("-" * 70)

        image_path = (
            RAW_DIR
            / subject_id
            / f"{subject_id}.nii.gz"
        )

        prediction_path = (
            PRED_DIR
            / subject_id
            / f"{subject_id}_pred.nii.gz"
        )

        # ---------------------------------------------------------
        # Existence
        # ---------------------------------------------------------

        if not image_path.exists():
            print(
                f"  FAIL: MRI not found: {image_path}"
            )
            all_passed = False
            continue

        if not prediction_path.exists():
            print(
                f"  FAIL: prediction not found: "
                f"{prediction_path}"
            )
            all_passed = False
            continue

        # ---------------------------------------------------------
        # Load
        # ---------------------------------------------------------

        image = nib.load(
            str(image_path)
        )

        prediction = nib.load(
            str(prediction_path)
        )

        image_data = np.asarray(
            image.dataobj
        )

        prediction_data = np.asarray(
            prediction.dataobj
        )

        if (
            image_data.ndim == 4
            and image_data.shape[-1] == 1
        ):
            image_spatial_shape = (
                image_data.shape[:3]
            )
        else:
            image_spatial_shape = (
                image_data.shape
            )

        print(
            f"  MRI shape:         {image_spatial_shape}"
        )

        print(
            f"  Prediction shape:  {prediction_data.shape}"
        )

        # ---------------------------------------------------------
        # Shape
        # ---------------------------------------------------------

        if prediction_data.shape != image_spatial_shape:

            print(
                "  FAIL: spatial shape mismatch."
            )

            all_passed = False

        else:

            print(
                "  PASS: spatial shape matches MRI."
            )

        # ---------------------------------------------------------
        # Affine
        # ---------------------------------------------------------

        affine_matches = np.allclose(
            image.affine,
            prediction.affine,
            atol=1e-5,
        )

        if not affine_matches:

            print(
                "  FAIL: affine mismatch."
            )

            print(
                "  MRI affine:"
            )
            print(image.affine)

            print(
                "  Prediction affine:"
            )
            print(prediction.affine)

            all_passed = False

        else:

            print(
                "  PASS: affine matches MRI."
            )

        # ---------------------------------------------------------
        # Finite values
        # ---------------------------------------------------------

        if not np.all(
            np.isfinite(prediction_data)
        ):

            print(
                "  FAIL: prediction contains "
                "NaN or Inf values."
            )

            all_passed = False

        else:

            print(
                "  PASS: prediction contains "
                "only finite values."
            )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        labels = np.unique(
            prediction_data
        )

        print(
            f"  Labels:            {labels.tolist()}"
        )

        valid_labels = np.all(
            np.isin(
                labels,
                [0, 1, 2, 3],
            )
        )

        if not valid_labels:

            print(
                "  FAIL: unexpected segmentation labels."
            )

            all_passed = False

        else:

            print(
                "  PASS: labels are within {0,1,2,3}."
            )

        # ---------------------------------------------------------
        # Foreground
        # ---------------------------------------------------------

        foreground = (
            prediction_data > 0
        )

        foreground_voxels = int(
            foreground.sum()
        )

        total_voxels = prediction_data.size

        foreground_fraction = (
            foreground_voxels
            / total_voxels
        )

        print(
            f"  Foreground voxels: {foreground_voxels:,}"
        )

        print(
            f"  Foreground ratio:  "
            f"{foreground_fraction:.4%}"
        )

        if foreground_voxels == 0:

            print(
                "  FAIL: prediction contains "
                "no foreground voxels."
            )

            all_passed = False

        else:

            print(
                "  PASS: prediction contains foreground."
            )

        # ---------------------------------------------------------
        # Per-class voxel counts
        # ---------------------------------------------------------

        print(
            "  Per-class voxel counts:"
        )

        for label in range(4):

            count = int(
                np.sum(
                    prediction_data == label
                )
            )

            print(
                f"    {label}: {count:,}"
            )

        # ---------------------------------------------------------
        # Data type
        # ---------------------------------------------------------

        print(
            f"  Prediction dtype:  "
            f"{prediction_data.dtype}"
        )

        if prediction_data.dtype != np.uint8:

            print(
                "  WARNING: prediction is not uint8."
            )

        else:

            print(
                "  PASS: prediction dtype is uint8."
            )

    # -----------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------

    print()
    print("=" * 70)

    if all_passed:

        print(
            "ALL TEST PREDICTION SANITY CHECKS PASSED."
        )

    else:

        print(
            "ONE OR MORE TEST PREDICTION CHECKS FAILED."
        )

        raise SystemExit(1)

    print("=" * 70)


if __name__ == "__main__":
    main()