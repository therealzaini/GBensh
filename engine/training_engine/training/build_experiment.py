"""Construct an experiment using factories supplied by the application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from engine.training_engine.config.experiment_config import ExperimentConfig
from engine.training_engine.training.experiment import PreparedExperiment
from engine.training_engine.utils.device_utils import resolve_device
from engine.training_engine.utils.seed_utils import set_seed

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

    from engine.training_engine.training.dataset_bundle import DatasetBundle


def prepare_experiment(
    *,
    model_factory: Callable[..., nn.Module],
    dataset_loader: Callable[[Path], DatasetBundle],
    optimizer_factory: Callable[..., Optimizer],
    config: ExperimentConfig,
    seed: int,
    training_device: str,
    dataset_root: Path,
) -> PreparedExperiment:
    device = resolve_device(training_device)
    set_seed(seed)
    bundle = dataset_loader(dataset_root)
    model = model_factory(
        in_channels=bundle.num_features,
        out_channels=bundle.num_classes,
        **config.model_config,
    ).to(device)
    optimizer = optimizer_factory(model.parameters(), **config.optimizer_args)

    return PreparedExperiment(model, bundle, optimizer, device)
