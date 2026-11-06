"""Run one experiment independently of the interactive shell."""

from pathlib import Path

from engine.paths import RAW_DATASETS_DIR
from engine.repositories.datasets import DATASETS
from engine.repositories.losses import LOSS_REGISTRY
from engine.repositories.metrics import METRIC_REGISTRY
from engine.repositories.models import MODELS
from engine.repositories.optimizers import OPTIMIZERS
from engine.repositories.trainers import TRAINER_REGISTRY
from engine.training_engine.config.experiment_config import load_experiment_config
from engine.training_engine.training.build_experiment import prepare_experiment
from engine.training_engine.training.experiment import ExperimentResult
from engine.training_engine.utils.seed_utils import resolve_seed


def _lookup(registry, name, kind):
    try:
        return registry[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown {kind} '{name}'. Available: {', '.join(registry)}."
        ) from exc


def _implementation(metadata, key, description):
    implementation = metadata.get(key)
    if not callable(implementation):
        raise NotImplementedError(f"{description} is not implemented yet.")
    return implementation


def run_planted_seed(
    model_str: str,
    dataset_str: str,
    seed: int | None = None,
    training_device: str = "auto",
    *,
    seed_index: int | None = None,
    config_root: Path | None = None,
    dataset_root: Path = RAW_DATASETS_DIR,
) -> ExperimentResult:
    """Validate, prepare, and train a model, returning data for any caller.

    Supply exactly one of ``seed`` and ``seed_index``. ``config_root`` can
    override the directory containing model/dataset JSON configurations.
    """
    model_metadata = _lookup(MODELS, model_str, "model")
    dataset_metadata = _lookup(DATASETS, dataset_str, "dataset")
    if dataset_metadata["task"] not in model_metadata["task"]:
        raise ValueError(
            f"Model {model_str.upper()} is incompatible with "
            f"the {dataset_str.upper()} task."
        )

    model_factory = _implementation(model_metadata, "object", f"Model '{model_str}'")
    dataset_loader = _implementation(
        dataset_metadata, "loader", f"Dataset '{dataset_str}'"
    )
    resolved_seed = resolve_seed(seed, seed_index)
    if training_device not in ("auto", "cpu", "cuda"):
        raise ValueError("training_device must be 'auto', 'cpu', or 'cuda'.")

    config_dir = (
        config_root / model_str
        if config_root is not None
        else model_metadata["config_dir"]
    )
    config = load_experiment_config(config_dir / f"{dataset_str}.json")
    optimizer_factory = _lookup(OPTIMIZERS, config.optimizer, "optimizer")
    trainer = _lookup(TRAINER_REGISTRY, dataset_metadata.get("trainer"), "trainer")
    loss_function = _lookup(LOSS_REGISTRY, dataset_metadata.get("default_loss"), "loss")
    metric_name = dataset_metadata.get("metric")
    metric_function = _lookup(METRIC_REGISTRY, metric_name, "metric")

    experiment = prepare_experiment(
        model_factory=model_factory,
        dataset_loader=dataset_loader,
        optimizer_factory=optimizer_factory,
        config=config,
        seed=resolved_seed,
        training_device=training_device,
        dataset_root=dataset_root,
    )
    training_result = trainer(
        model=experiment.model,
        bundle=experiment.bundle,
        optimizer=experiment.optimizer,
        loss_function=loss_function,
        metric_function=metric_function,
        device=experiment.device,
        epochs=config.epochs,
    )

    return ExperimentResult(
        model=model_str,
        dataset=dataset_str,
        seed=resolved_seed,
        device=str(experiment.device),
        metric=metric_name,
        **training_result,
    )
