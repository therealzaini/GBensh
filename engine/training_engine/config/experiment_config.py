"""Read experiment settings without constructing any training components."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.training_engine.config.config import EPOCHS


@dataclass(frozen=True)
class ExperimentConfig:
    model_config: dict[str, Any]
    optimizer: str
    optimizer_args: dict[str, Any]
    epochs: int = EPOCHS


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def load_experiment_config(path: Path) -> ExperimentConfig:
    try:
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"No experiment configuration found at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}.") from exc

    config = _require_object(config, "Configuration")
    model_config = _require_object(config.get("model_config"), "model_config")
    training_config = _require_object(config.get("training_config"), "training_config")
    optimizer = training_config.get("optimizer")
    if not isinstance(optimizer, str) or not optimizer:
        raise ValueError("training_config.optimizer must be a nonempty string.")
    optimizer_args = _require_object(
        training_config.get("optimizer_args", {}), "training_config.optimizer_args"
    )
    epochs = training_config.get("epochs", EPOCHS)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("training_config.epochs must be a positive integer.")
    if "in_channels" in model_config or "out_channels" in model_config:
        raise ValueError("in_channels and out_channels are supplied by the dataset.")

    return ExperimentConfig(model_config, optimizer, optimizer_args, epochs)
