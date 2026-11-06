from collections.abc import Callable

from torch import Tensor
import torch.nn.functional as F


LossFunction = Callable[[Tensor, Tensor], Tensor]


def cross_entropy(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    
    target = target.view(-1).long()

    return F.cross_entropy(
        prediction,
        target,
    )


LOSS_REGISTRY: dict[str, LossFunction] = {
    "cross_entropy": cross_entropy,
}