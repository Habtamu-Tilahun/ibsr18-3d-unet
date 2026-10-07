"""Utilities for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Seeds Python's ``random`` module, NumPy, and PyTorch. When CUDA is
    available, all CUDA devices are seeded as well.

    Parameters
    ----------
    seed:
        Random seed used by the experiment.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
