from pathlib import Path

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def load_cora(root: Path):

    dataset = Planetoid(
        root=str(root / "cora"),
        name="Cora",
        transform=NormalizeFeatures(),
    )

    data = dataset[0]

    return dataset, data