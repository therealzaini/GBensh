from engine.repositories.models import (
    MODELS_LIST,
)

from engine.repositories.datasets import (
    DATASETS_LIST,
)

from argparse import ArgumentParser

def parse_planting_args():

    parser = ArgumentParser(prog="plant",
                                        description="Runs a single model on a single dataset with a single seed.")

    parser.add_argument(
        "--model", type=str, required=True, choices=MODELS_LIST
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASETS_LIST
    )
    parser.add_argument(
        "--device", type=str, default='auto', choices=['cpu', 'cuda', 'auto']
    )

    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seed-index", type=int, default=None)

    return parser