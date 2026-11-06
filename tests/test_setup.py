import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

from engine.training_engine.config.experiment_config import ExperimentConfig
from engine.training_engine.training.build_experiment import prepare_experiment
from engine.training_engine.utils.device_utils import resolve_device
from engine.training_engine.utils.seed_utils import generate_seeds, resolve_seed


class SetupTests(unittest.TestCase):
    def test_factories_receive_dataset_dimensions_and_seed_precedes_construction(self):
        events = []
        bundle = types.SimpleNamespace(num_features=7, num_classes=3)
        model = Mock()
        model.to.return_value = model
        model.parameters.return_value = sentinel.parameters
        model_factory = Mock(side_effect=lambda **kwargs: events.append("model") or model)
        loader = Mock(side_effect=lambda root: events.append("dataset") or bundle)
        optimizer = Mock(side_effect=lambda *args, **kwargs: events.append("optimizer") or sentinel.optimizer)
        config = ExperimentConfig({"hidden_channels": 16}, "adam", {"lr": 0.01}, 2)

        with patch("engine.training_engine.training.build_experiment.resolve_device", return_value="cpu"), \
             patch("engine.training_engine.training.build_experiment.set_seed", side_effect=lambda seed: events.append("seed")) as seed:
            experiment = prepare_experiment(
                model_factory=model_factory, dataset_loader=loader,
                optimizer_factory=optimizer, config=config,
                seed=42, training_device="cpu", dataset_root=Path("data"),
            )

        self.assertEqual(events, ["seed", "dataset", "model", "optimizer"])
        seed.assert_called_once_with(42)
        loader.assert_called_once_with(Path("data"))
        model_factory.assert_called_once_with(in_channels=7, out_channels=3, hidden_channels=16)
        model.to.assert_called_once_with("cpu")
        optimizer.assert_called_once_with(sentinel.parameters, lr=0.01)
        self.assertIs(experiment.model, model)
        self.assertIs(experiment.bundle, bundle)
        self.assertIs(experiment.optimizer, sentinel.optimizer)


class SeedTests(unittest.TestCase):
    def test_original_seed_schedule_is_preserved(self):
        self.assertEqual(generate_seeds()[:3], [1416840650, 64878673, 3455414167])
        self.assertEqual(len(generate_seeds()), 100)
        self.assertEqual(generate_seeds(num_seeds=0), [])
        self.assertEqual(resolve_seed(seed_index=99), generate_seeds()[99])
        for seed in (0, 42, 2**32 - 1):
            self.assertEqual(resolve_seed(seed=seed), seed)

    def test_invalid_seed_inputs_are_rejected(self):
        for kwargs in ({}, {"seed": 1, "seed_index": 0}, {"seed": -1},
                       {"seed": 2**32}, {"seed": True}, {"seed": 1.5},
                       {"seed_index": -1}, {"seed_index": 100}, {"seed_index": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                resolve_seed(**kwargs)


class DeviceTests(unittest.TestCase):
    def fake_torch(self, cuda_available):
        return types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
            device=lambda name: name,
        )

    def test_auto_device_and_explicit_cpu(self):
        for available, expected in ((True, "cuda"), (False, "cpu")):
            with self.subTest(available=available), patch.dict("sys.modules", {"torch": self.fake_torch(available)}):
                self.assertEqual(resolve_device("auto"), expected)
                self.assertEqual(resolve_device("cpu"), "cpu")

    def test_explicit_cuda_requires_available_gpu(self):
        with patch.dict("sys.modules", {"torch": self.fake_torch(False)}):
            with self.assertRaisesRegex(RuntimeError, "no CUDA GPU"):
                resolve_device("cuda")
        with patch.dict("sys.modules", {"torch": self.fake_torch(True)}):
            self.assertEqual(resolve_device("cuda"), "cuda")

    def test_unknown_device_is_rejected_before_import(self):
        with self.assertRaises(ValueError):
            resolve_device("gpu")


if __name__ == "__main__":
    unittest.main()
