import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, sentinel

from engine.cli_engine.gbensh_cmd import GBensh
from engine.training_engine.config.config import EPOCHS
from engine.training_engine.config.experiment_config import load_experiment_config
from engine.training_engine.training.experiment import PreparedExperiment
from engine.training_engine.utils.seed_utils import generate_seeds


service = importlib.import_module(
    "engine.training_engine.training.commands.run_planted_seed"
)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config_root = Path(directory.name)
        self.config_path = self.config_root / "gcn" / "cora.json"
        self.config_path.parent.mkdir()
        self.config = {
            "model_config": {"hidden_channels": 16, "num_layers": 2, "dropout": 0.5},
            "training_config": {
                "optimizer": "adam",
                "optimizer_args": {"lr": 0.01, "weight_decay": 5e-4},
                "epochs": 2,
            },
        }
        self.write_config()
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.prepare = self.contexts.enter_context(patch.object(service, "prepare_experiment"))
        self.prepare.return_value = PreparedExperiment(
            sentinel.model, sentinel.bundle, sentinel.optimizer, "cpu"
        )
        self.history = [{"epoch": 1, "val_metric": 0.7}]
        self.trainer = Mock(return_value={
            "history": self.history, "test_loss": 0.3, "test_metric": 0.8,
        })
        self.contexts.enter_context(patch.dict(
            service.TRAINER_REGISTRY, {"full_batch_node_trainer": self.trainer}
        ))

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def run_experiment(self, **kwargs):
        return service.run_planted_seed(
            "gcn", "cora", config_root=self.config_root, **kwargs
        )

    def test_headless_run_dispatches_configured_components_and_returns_results(self):
        result = self.run_experiment(seed=42, training_device="cpu", dataset_root=Path("data"))
        self.prepare.assert_called_once()
        setup = self.prepare.call_args.kwargs
        self.assertEqual(setup["seed"], 42)
        self.assertEqual(setup["training_device"], "cpu")
        self.assertEqual(setup["dataset_root"], Path("data"))
        self.assertEqual(setup["config"].optimizer_args["lr"], 0.01)
        self.assertIs(setup["model_factory"], service.MODELS["gcn"]["object"])
        self.assertIs(setup["dataset_loader"], service.DATASETS["cora"]["loader"])
        self.assertIs(setup["optimizer_factory"], service.OPTIMIZERS["adam"])
        self.trainer.assert_called_once_with(
            model=sentinel.model, bundle=sentinel.bundle,
            optimizer=sentinel.optimizer, device="cpu", epochs=2,
            loss_function=service.LOSS_REGISTRY["cross_entropy"],
            metric_function=service.METRIC_REGISTRY["accuracy"],
        )
        self.assertEqual((result.model, result.dataset, result.seed, result.device),
                         ("gcn", "cora", 42, "cpu"))
        self.assertEqual((result.metric, result.test_loss, result.test_metric),
                         ("accuracy", 0.3, 0.8))
        self.assertIs(result.history, self.history)

    def test_seed_index_uses_existing_schedule(self):
        result = self.run_experiment(seed_index=2)
        self.assertEqual(result.seed, generate_seeds()[2])
        self.assertEqual(self.prepare.call_args.kwargs["seed"], result.seed)

    def test_existing_config_keeps_default_epoch_count(self):
        del self.config["training_config"]["epochs"]
        self.write_config()
        self.run_experiment(seed=42)
        self.assertEqual(self.trainer.call_args.kwargs["epochs"], EPOCHS)

    def test_unsupported_requests_fail_before_setup(self):
        cases = [
            ("missing", "cora", "Unknown model"),
            ("gcn", "missing", "Unknown dataset"),
            ("gcn", "mutag", "incompatible"),
            ("gat", "cora", "not implemented"),
            ("gcn", "pubmed", "not implemented"),
        ]
        for model, dataset, message in cases:
            with self.subTest(model=model, dataset=dataset):
                with self.assertRaisesRegex((ValueError, NotImplementedError), message):
                    service.run_planted_seed(model, dataset, seed=42)
        self.prepare.assert_not_called()
        self.trainer.assert_not_called()

    def test_invalid_seed_or_device_fails_before_setup(self):
        for kwargs in ({}, {"seed": -1}, {"seed_index": -1},
                       {"seed": 1, "seed_index": 0}, {"seed": 1, "training_device": "gpu"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_experiment(**kwargs)
        self.prepare.assert_not_called()

    def test_unknown_training_components_fail_before_setup(self):
        self.config["training_config"]["optimizer"] = "unknown"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
            self.run_experiment(seed=42)
        self.config["training_config"]["optimizer"] = "adam"
        self.write_config()
        for key, message in (("trainer", "trainer"), ("default_loss", "loss"), ("metric", "metric")):
            with self.subTest(key=key), patch.dict(service.DATASETS["cora"], {key: "unknown"}):
                with self.assertRaisesRegex(ValueError, f"Unknown {message}"):
                    self.run_experiment(seed=42)
        self.prepare.assert_not_called()

    def test_cli_uses_same_service_and_formats_result(self):
        output = io.StringIO()
        shell = GBensh(stdout=output)
        with patch.object(service, "load_experiment_config", return_value=load_experiment_config(self.config_path)):
            shell.onecmd("plant --model gcn --dataset cora --seed 42 --device cpu")
        self.assertIn("GCN on CORA | seed=42 | device=cpu", output.getvalue())
        self.assertIn("test accuracy=0.8000 | test loss=0.3000", output.getvalue())
        self.trainer.assert_called_once()

    def test_cli_reports_unsupported_experiment_without_training(self):
        output = io.StringIO()
        self.assertFalse(GBensh(stdout=output).onecmd("plant --model gat --dataset cora --seed 42"))
        self.assertIn("Model 'gat' is not implemented yet", output.getvalue())
        self.prepare.assert_not_called()

    def test_cli_parse_errors_do_not_start_experiments(self):
        with contextlib.redirect_stderr(io.StringIO()):
            GBensh().onecmd("plant --model gcn --dataset cora --seed 42 --seed-index 0")
        self.prepare.assert_not_called()


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "experiment.json"

    def test_missing_or_invalid_json_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "No experiment configuration"):
            load_experiment_config(self.path)
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            load_experiment_config(self.path)

    def test_invalid_configuration_is_rejected(self):
        invalid = [None, [], {}, {"model_config": [], "training_config": {}},
                   {"model_config": {}, "training_config": {}}]
        for epochs in (0, -1, True, 1.5, "2"):
            invalid.append({"model_config": {}, "training_config": {"optimizer": "adam", "epochs": epochs}})
        invalid.append({"model_config": {}, "training_config": {"optimizer": "adam", "optimizer_args": []}})
        invalid.append({"model_config": {"in_channels": 2}, "training_config": {"optimizer": "adam"}})
        for config in invalid:
            with self.subTest(config=config):
                self.path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_experiment_config(self.path)


class ImportBoundaryTests(unittest.TestCase):
    def test_help_and_service_validation_do_not_import_training_dependencies(self):
        code = '''
import importlib.abc
import sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_geometric', 'numpy', 'ogb'}:
            raise AssertionError('Unexpected training import: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
from engine.cli_engine.gbensh_cmd import GBensh
from engine.training_engine.training.commands.run_planted_seed import run_planted_seed
GBensh().onecmd('plant --help')
try:
    run_planted_seed('gat', 'cora', seed=42)
except NotImplementedError:
    pass
else:
    raise AssertionError('Expected an unsupported-model error')
'''
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: plant", result.stdout)


if __name__ == "__main__":
    unittest.main()
