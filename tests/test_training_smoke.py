"""Exercise real CPU training when the project's dependencies are installed."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_TRAINING_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "torch_geometric")
)


@unittest.skipUnless(HAS_TRAINING_DEPS, "PyTorch and PyG are required for CPU training")
class TrainingSmokeTests(unittest.TestCase):
    def test_two_epoch_experiment_trains_on_an_in_memory_graph(self):
        import torch
        from torch_geometric.data import Data, InMemoryDataset

        from engine.repositories.datasets import DATASETS
        from engine.repositories.models import MODELS, build_gcn
        from engine.training_engine.training.commands.run_planted_seed import run_planted_seed
        from engine.training_engine.training.dataset_bundle import DatasetBundle

        data = Data(
            x=torch.tensor([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.],
                            [0., 0., 1.], [1., 0., 1.], [0., 1., 0.]]),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                                     [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]),
            y=torch.tensor([0, 0, 1, 1, 0, 1]),
            train_mask=torch.tensor([True, True, True, False, False, False]),
            val_mask=torch.tensor([False, False, False, True, False, False]),
            test_mask=torch.tensor([False, False, False, False, True, True]),
        )
        dataset = InMemoryDataset()
        dataset.data, dataset.slices = dataset.collate([data])
        bundle = DatasetBundle(
            name="cora", task="node_classification", dataset=dataset, data=data,
            train_split=data.train_mask, val_split=data.val_mask,
            test_split=data.test_mask, num_features=3, num_classes=2,
        )
        captured = {}

        def capture_model(**kwargs):
            model = build_gcn(**kwargs)
            captured["model"] = model
            captured["before"] = [parameter.detach().clone() for parameter in model.parameters()]
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gcn").mkdir()
            (root / "gcn" / "cora.json").write_text(json.dumps({
                "model_config": {"hidden_channels": 4, "num_layers": 2, "dropout": 0.0},
                "training_config": {"optimizer": "adam", "optimizer_args": {"lr": 0.01}, "epochs": 2},
            }), encoding="utf-8")
            with patch.dict(MODELS["gcn"], {"object": capture_model}), \
                 patch.dict(DATASETS["cora"], {"loader": lambda root: bundle}):
                result = run_planted_seed("gcn", "cora", seed=42, training_device="cpu", config_root=root)

        self.assertEqual([entry["epoch"] for entry in result.history], [1, 2])
        self.assertEqual(result.device, "cpu")
        self.assertTrue(math.isfinite(result.test_loss))
        self.assertTrue(0.0 <= result.test_metric <= 1.0)
        self.assertTrue(any(
            not torch.equal(before, after.detach())
            for before, after in zip(captured["before"], captured["model"].parameters())
        ))


if __name__ == "__main__":
    unittest.main()
