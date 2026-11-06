from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


LossFunction = Callable[["Tensor", "Tensor"], "Tensor"]


def cross_entropy(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    import torch.nn.functional as F

    target = target.view(-1).long()

    return F.cross_entropy(
        prediction,
        target,
    )


LOSS_REGISTRY: dict[str, LossFunction] = {
    "cross_entropy": cross_entropy,
}
