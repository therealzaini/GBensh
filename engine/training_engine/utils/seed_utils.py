from hashlib import sha256
import random

from engine.training_engine.config.config import NUM_SEEDS, MASTER_SEED


def generate_seeds(master_seed: int = MASTER_SEED, num_seeds: int = NUM_SEEDS) -> list[int]:
    if type(master_seed) is not int:
        raise TypeError("master_seed must be an integer.")
    if type(num_seeds) is not int:
        raise TypeError("num_seeds must be an integer.")
    if num_seeds < 0:
        raise ValueError("num_seeds must be non-negative.")

    seeds = []
    for i in range(num_seeds):
        payload = f"{master_seed}:{i}".encode("utf-8")
        digest = sha256(payload).digest()
        seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        seeds.append(seed)

    return seeds


def resolve_seed(seed: int | None = None, seed_index: int | None = None) -> int:
    """Resolve either an explicit NumPy-compatible seed or a schedule index."""
    if (seed is None) == (seed_index is None):
        raise ValueError("Provide exactly one of seed or seed_index.")

    if seed is not None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
        return seed

    if type(seed_index) is not int or not 0 <= seed_index < NUM_SEEDS:
        raise ValueError(f"seed_index must be an integer between 0 and {NUM_SEEDS - 1}.")
    return generate_seeds()[seed_index]


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
