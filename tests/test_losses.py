"""Tests for segmentation loss functions."""

import torch

from ibsr_unet.training import DiceCrossEntropyLoss


def test_dice_ce_loss_returns_scalar() -> None:
    """The loss should return a scalar tensor."""

    loss_fn = DiceCrossEntropyLoss()

    prediction = torch.randn(
        2,
        4,
        32,
        32,
        32,
    )

    target = torch.randint(
        low=0,
        high=4,
        size=(2, 1, 32, 32, 32),
    )

    loss = loss_fn(prediction, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dice_ce_loss_supports_target_without_channel() -> None:
    """The loss should also accept [B, D, H, W] targets."""

    loss_fn = DiceCrossEntropyLoss()

    prediction = torch.randn(
        2,
        4,
        32,
        32,
        32,
    )

    target = torch.randint(
        low=0,
        high=4,
        size=(2, 32, 32, 32),
    )

    loss = loss_fn(prediction, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dice_ce_loss_backward() -> None:
    """The combined loss should support backpropagation."""

    loss_fn = DiceCrossEntropyLoss()

    prediction = torch.randn(
        2,
        4,
        16,
        16,
        16,
        requires_grad=True,
    )

    target = torch.randint(
        low=0,
        high=4,
        size=(2, 1, 16, 16, 16),
    )

    loss = loss_fn(prediction, target)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()