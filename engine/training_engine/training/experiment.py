"""Values passed between experiment setup, execution, and presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device as TorchDevice, nn
    from torch.optim import Optimizer

    from engine.training_engine.training.dataset_bundle import DatasetBundle


@dataclass
class PreparedExperiment:
    model: nn.Module
    bundle: DatasetBundle
    optimizer: Optimizer
    device: TorchDevice


@dataclass(frozen=True)
class ExperimentResult:
    model: str
    dataset: str
    seed: int
    device: str
    metric: str
    history: list[dict[str, float | int]]
    test_loss: float
    test_metric: float
