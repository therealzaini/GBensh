from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


MetricFunction = Callable[..., float]


def accuracy(
    prediction: Tensor,
    target: Tensor,
    *,
    evaluator: Any | None = None,
) -> float:

    predicted_classes = prediction.argmax(dim=-1)
    target = target.view(-1)

    return (
        predicted_classes
        .eq(target)
        .float()
        .mean()
        .item()
    )


METRIC_REGISTRY: dict[str, MetricFunction] = {
    "accuracy": accuracy,
}
