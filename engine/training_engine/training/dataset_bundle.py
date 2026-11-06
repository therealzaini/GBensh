from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import Dataset


@dataclass
class DatasetBundle:
    name: str
    task: str

    dataset: Dataset
    data: Data

    train_split: Tensor
    val_split: Tensor
    test_split: Tensor

    num_features: int
    num_classes: int | None

    evaluator: Any | None = None