from hashlib import sha256
from engine.training_engine.config.config import NUM_SEEDS, MASTER_SEED

import random
import numpy as np
import torch

def generate_seeds(master_seed: int=MASTER_SEED, num_seeds: int=NUM_SEEDS) -> list[int]:
    if not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer.")
    if num_seeds < 0:
        raise ValueError("num_seeds must be non-negative.")

    seeds = []
    for i in range(num_seeds):
        payload = f"{master_seed}:{i}".encode("utf-8")
        digest = sha256(payload).digest()
        seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        seeds.append(seed)

    return seeds

def set_seed(seed: int) -> None:
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False