from pathlib import Path

from typing import cast
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from engine.training_engine.training.dataset_bundle import DatasetBundle


def load_cora(root: Path) -> DatasetBundle:

    dataset = Planetoid(
        root=str(root / "cora"),
        name="Cora",
        transform=NormalizeFeatures(),
    )

    data = cast(Data, dataset[0])

    return DatasetBundle(
        name="cora",
        task="node_classification",

        dataset=dataset,
        data=data,

        train_split=data["train_mask"],
        val_split=data["val_mask"],
        test_split=data["test_mask"],

        num_features=dataset.num_features,
        num_classes=dataset.num_classes,

        evaluator=None,
    )