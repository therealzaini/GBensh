import json 
import torch

from engine.paths import RAW_DATASETS_DIR
from engine.paths import MODELS_CONFIG_DIR

from engine.repositories.datasets import DATASETS
from engine.repositories.models import MODELS
from engine.repositories.optimizers import OPTIMIZERS

from engine.training_engine.training.dataset_bundle import DatasetBundle

from engine.training_engine.utils.seed_utils import set_seed

def run_planted_seed(model_str, dataset_str, seed, training_device):

    set_seed(seed)

    if training_device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif training_device == 'cpu':
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise NotImplementedError("No CUDA GPU has been detected.")
        device = torch.device("cuda")

    dataset_bundle: DatasetBundle = DATASETS[dataset_str]["loader"](RAW_DATASETS_DIR)

    json_file = MODELS_CONFIG_DIR / model_str / f"{dataset_str}.json"

    with open(json_file, "r") as file:
        config = json.load(file)

    model_config, training_config = config["model_config"], config["training_config"]

    
    model = MODELS[model_str]["object"](
        in_channels=dataset_bundle.num_features,
        out_channels=dataset_bundle.num_classes,
        **model_config
    ).to(device)

    optimizer = OPTIMIZERS[training_config["optimizer"]](
        model.parameters(),
        **training_config["optimizer_args"]
    )

    


    
