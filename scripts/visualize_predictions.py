"""
Generate qualitative visualizations of IBSR-18 segmentation predictions.

This script:
1. Loads the selected Experiment 1 checkpoint.
2. Runs inference on the validation subjects.
3. Uses the established sliding-window inference configuration.
4. Saves predicted segmentations as NIfTI files.
5. Generates axial, coronal, and sagittal qualitative comparisons.

Selected model:
    Experiment 1
    Residual 3D U-Net
    Native spacing
    No N4 preprocessing
    Best validation mean foreground Dice: 0.8804

Expected output:

outputs/
├── predictions/
│   ├── IBSR_11/
│   │   └── prediction.nii.gz
│   ├── IBSR_12/
│   │   └── prediction.nii.gz
│   ├── IBSR_13/
│   │   └── prediction.nii.gz
│   ├── IBSR_14/
│   │   └── prediction.nii.gz
│   └── IBSR_17/
│       └── prediction.nii.gz
│
└── visualizations/
    ├── IBSR_11_prediction.png
    ├── IBSR_12_prediction.png
    ├── IBSR_13_prediction.png
    ├── IBSR_14_prediction.png
    └── IBSR_17_prediction.png
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ibsr_unet.data.datamodule import IBSRDataModule
from ibsr_unet.inference.predictor import sliding_window_predict
from ibsr_unet.models.unet import build_unet
from ibsr_unet.visualization.plots import plot_prediction_comparison


CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "best_model.pt"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

VISUALIZATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
)


# ---------------------------------------------------------------------
# Expected selected experiment
# ---------------------------------------------------------------------

EXPECTED_BEST_DICE = 0.8840
EXPECTED_BEST_EPOCH = 99


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    """
    Load the YAML configuration file.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Expected the YAML configuration to contain a dictionary."
        )

    return config


# ---------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------


def build_datamodule(
    config: dict[str, Any],
) -> IBSRDataModule:
    """
    Build the IBSR-18 data module using the same preprocessing
    configuration as Experiment 1.

    Experiment 1:
        - native spacing
        - no runtime N4
        - no precomputed N4
    """

    data_config = config["data"]
    training_config = config["training"]

    # -------------------------------------------------------------
    # Resolve data paths.
    # -------------------------------------------------------------

    data_dir = PROJECT_ROOT / data_config["root_dir"]
    splits_dir = PROJECT_ROOT / data_config["splits_dir"]

    # -------------------------------------------------------------
    # Native spacing.
    #
    # Experiment 1 does not resample the images.
    # -------------------------------------------------------------

    spacing_config = data_config.get("spacing")

    if spacing_config is None:
        target_spacing = None
    else:
        target_spacing = tuple(
            float(value)
            for value in spacing_config
        )

    # -------------------------------------------------------------
    # Patch size.
    # -------------------------------------------------------------

    patch_size = tuple(
        int(value)
        for value in data_config.get(
            "patch_size",
            [96, 96, 96],
        )
    )

    # -------------------------------------------------------------
    # Data-loader configuration.
    # -------------------------------------------------------------

    num_samples = int(
        training_config.get(
            "num_samples",
            1,
        )
    )

    batch_size = int(
        training_config.get(
            "batch_size",
            1,
        )
    )

    num_workers = int(
        training_config.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        training_config.get(
            "pin_memory",
            torch.cuda.is_available(),
        )
    )

    # -------------------------------------------------------------
    # Create data module.
    # -------------------------------------------------------------

    return IBSRDataModule(
        data_dir=data_dir,
        splits_dir=splits_dir,
        patch_size=patch_size,
        num_samples=num_samples,
        target_spacing=target_spacing,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build the 3D U-Net using the architecture specified in train.yaml.
    """

    model_config = config["model"]

    model = build_unet(
        in_channels=int(
            model_config["in_channels"]
        ),
        out_channels=int(
            model_config["out_channels"]
        ),
        channels=tuple(
            int(value)
            for value in model_config["channels"]
        ),
        strides=tuple(
            int(value)
            for value in model_config["strides"]
        ),
        num_res_units=int(
            model_config["num_res_units"]
        ),
    )

    return model.to(device)


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load the trained model checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


# ---------------------------------------------------------------------
# Subject ID
# ---------------------------------------------------------------------


def get_subject_id(
    batch: dict[str, Any],
) -> str:
    """
    Extract the IBSR subject ID from a DataLoader batch.

    Depending on DataLoader collation, subject_id may be returned
    as a string, list, or tuple.
    """

    subject_id = batch["subject_id"]

    if isinstance(subject_id, (list, tuple)):
        if not subject_id:
            raise ValueError(
                "Batch contains an empty subject_id."
            )

        return str(subject_id[0])

    return str(subject_id)


# ---------------------------------------------------------------------
# NIfTI saving
# ---------------------------------------------------------------------


def save_prediction_nifti(
    prediction: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> None:
    """
    Save a predicted segmentation as NIfTI using the reference MRI's
    spatial metadata.

    The prediction is saved as uint8 because the segmentation contains
    discrete class labels:

        0 = Background
        1 = CSF
        2 = GM
        3 = WM
    """

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference MRI not found: {reference_path}"
        )

    reference = nib.load(
        str(reference_path)
    )

    prediction = np.asarray(
        prediction,
        dtype=np.uint8,
    )

    # -------------------------------------------------------------
    # Verify geometry before saving.
    # -------------------------------------------------------------

    reference_shape = reference.shape

    if prediction.shape != reference_shape:
        raise ValueError(
            "Prediction/reference shape mismatch: "
            f"prediction={prediction.shape}, "
            f"reference={reference_shape}"
        )

    # -------------------------------------------------------------
    # Preserve the reference affine and header.
    # -------------------------------------------------------------

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)

    prediction_image = nib.Nifti1Image(
        prediction,
        affine=reference.affine,
        header=header,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        prediction_image,
        str(output_path),
    )


# ---------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------


def tensor_to_numpy(
    value: Any,
) -> np.ndarray:
    """
    Convert a tensor-like value to a NumPy array.
    """

    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(value)


def remove_singleton_channel(
    array: np.ndarray,
) -> np.ndarray:
    """
    Remove a leading singleton channel dimension if present.
    """

    if (
        array.ndim == 4
        and array.shape[0] == 1
    ):
        return array[0]

    return array


# ---------------------------------------------------------------------
# Visualization generation
# ---------------------------------------------------------------------


def generate_visualizations(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    prediction_dir: Path,
    visualization_dir: Path,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
) -> None:
    """
    Run inference on the validation set, save NIfTI predictions,
    and generate qualitative visualizations.
    """

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    print()
    print("Generating qualitative results...")
    print("-" * 70)

    print(
        "Inference configuration:"
    )
    print(
        f"  ROI size:       {roi_size}"
    )
    print(
        f"  SW batch size:  {sw_batch_size}"
    )
    print(
        f"  Overlap:        {overlap}"
    )
    print(
        "  Target spacing: native"
    )
    print(
        "  N4 preprocessing: disabled"
    )

    print("-" * 70)

    with torch.no_grad():

        for batch in dataloader:

            # -----------------------------------------------------
            # Subject ID.
            # -----------------------------------------------------

            subject_id = get_subject_id(batch)

            # -----------------------------------------------------
            # Move image to device.
            # -----------------------------------------------------

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            # -----------------------------------------------------
            # Ground truth.
            # -----------------------------------------------------

            label_np = remove_singleton_channel(
                tensor_to_numpy(
                    batch["label"]
                )[0]
            )

            # -----------------------------------------------------
            # MRI volume.
            # -----------------------------------------------------

            image_np = (
                images[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            # -----------------------------------------------------
            # Sliding-window inference.
            # -----------------------------------------------------

            predictions = sliding_window_predict(
                model=model,
                images=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )

            # -----------------------------------------------------
            # Convert logits to class labels.
            # -----------------------------------------------------

            prediction_labels = torch.argmax(
                predictions,
                dim=1,
            )

            prediction_np = (
                prediction_labels[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # -----------------------------------------------------
            # Validate prediction labels.
            # -----------------------------------------------------

            unique_labels = np.unique(
                prediction_np
            )

            if not np.all(
                np.isin(
                    unique_labels,
                    [0, 1, 2, 3],
                )
            ):
                raise ValueError(
                    f"{subject_id}: unexpected prediction labels: "
                    f"{unique_labels}"
                )

            # -----------------------------------------------------
            # Locate reference MRI.
            #
            # The data module returns the subject ID and the
            # dataset uses the standard IBSR directory structure.
            # -----------------------------------------------------

            reference_path = (
                PROJECT_ROOT
                / "data"
                / "raw"
                / subject_id
                / f"{subject_id}.nii.gz"
            )

            # -----------------------------------------------------
            # Save predicted segmentation.
            # -----------------------------------------------------

            subject_prediction_dir = (
                prediction_dir
                / subject_id
            )

            prediction_path = (
                subject_prediction_dir
                / "prediction.nii.gz"
            )

            save_prediction_nifti(
                prediction=prediction_np,
                reference_path=reference_path,
                output_path=prediction_path,
            )

            print(
                f"Saved prediction: {prediction_path}"
            )

            # -----------------------------------------------------
            # Generate qualitative figure.
            # -----------------------------------------------------

            visualization_path = (
                visualization_dir
                / f"{subject_id}_prediction.png"
            )

            plot_prediction_comparison(
                image=image_np,
                label=label_np,
                prediction=prediction_np,
                subject_id=subject_id,
                output_path=visualization_path,
            )

            print(
                f"Saved visualization: "
                f"{visualization_path}"
            )

            print()

    print("-" * 70)
    print("Qualitative evaluation complete.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """
    Main entry point.
    """

    # -------------------------------------------------------------
    # Select device.
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # -------------------------------------------------------------
    # Load configuration.
    # -------------------------------------------------------------

    config = load_config(
        CONFIG_PATH
    )

    # -------------------------------------------------------------
    # Build data module.
    # -------------------------------------------------------------

    data_module = build_datamodule(
        config
    )

    data_module.setup()

    validation_loader = (
        data_module.val_dataloader()
    )

    # -------------------------------------------------------------
    # Build model.
    # -------------------------------------------------------------

    model = build_model(
        config=config,
        device=device,
    )

    # -------------------------------------------------------------
    # Load selected checkpoint.
    # -------------------------------------------------------------

    print()
    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    checkpoint_dice = checkpoint.get(
        "val_mean_dice"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint_epoch}"
    )

    if checkpoint_dice is not None:
        print(
            "Checkpoint validation mean Dice: "
            f"{checkpoint_dice:.4f}"
        )

    # -------------------------------------------------------------
    # Verify that the selected checkpoint is the expected model.
    #
    # Do not silently visualize a different checkpoint.
    # -------------------------------------------------------------

    if checkpoint_epoch != EXPECTED_BEST_EPOCH:
        raise RuntimeError(
            "The checkpoint is not the selected Experiment 1 "
            f"checkpoint. Expected epoch {EXPECTED_BEST_EPOCH}, "
            f"but found epoch {checkpoint_epoch}."
        )

    if checkpoint_dice is not None:
        if not np.isclose(
            float(checkpoint_dice),
            EXPECTED_BEST_DICE,
            atol=1e-4,
        ):
            raise RuntimeError(
                "Checkpoint validation Dice does not match the "
                "selected Experiment 1 model. "
                f"Expected {EXPECTED_BEST_DICE:.4f}, "
                f"found {float(checkpoint_dice):.4f}."
            )

    print(
        "Selected model verified:"
    )
    print(
        f"  Experiment:       1"
    )
    print(
        f"  Best epoch:       {EXPECTED_BEST_EPOCH}"
    )
    print(
        f"  Mean foreground Dice: "
        f"{EXPECTED_BEST_DICE:.4f}"
    )
    print(
        "  Preprocessing:    native spacing, no N4"
    )

    # -------------------------------------------------------------
    # Inference configuration.
    # -------------------------------------------------------------

    inference_config = config.get(
        "inference",
        {},
    )

    roi_size = tuple(
        int(value)
        for value in inference_config.get(
            "roi_size",
            config["data"].get(
                "patch_size",
                [96, 96, 96],
            ),
        )
    )

    sw_batch_size = int(
        inference_config.get(
            "sw_batch_size",
            1,
        )
    )

    overlap = float(
        inference_config.get(
            "overlap",
            0.25,
        )
    )

    # -------------------------------------------------------------
    # Generate results.
    # -------------------------------------------------------------

    generate_visualizations(
        model=model,
        dataloader=validation_loader,
        device=device,
        prediction_dir=PREDICTION_DIR,
        visualization_dir=VISUALIZATION_DIR,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )


if __name__ == "__main__":
    main()