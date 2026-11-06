# The Graph Benchmarking Shell (GBensh)

Welcome to **GBensh**, the Graph Benchmarking Shell.

GBensh is an interactive command-line interface for running benchmarks
and experiments on graph neural network models across a variety of datasets.
It is intended primarily for research and educational use.

## Installation

GBensh uses [uv](https://docs.astral.sh/uv/) to manage Python versions,
dependencies, and the project's virtual environment.

The only prerequisite is `uv`. If it is not already installed on your
system, please refer to the official
[uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/)
for instructions appropriate for your operating system.

First, clone the repository and navigate to the project root:

```bash
git clone https://github.com/therealzaini/GBensh
cd GBensh
```
Then launch GBensh with:
```bash
uv run gbensh
```

On first launch, `uv` will automatically set up the required Python
environment and install the project dependencies.
No manual virtual-environment creation or activation is required.

Future versions of GBensh may also support direct installation via `pip`.

## Usage

Start the shell with `uv run gbensh`, then run an experiment:

```text
(GBensh) plant --model gcn --dataset cora --seed 42 --device auto
```

Use `--seed-index 0` instead of `--seed 42` to select the first seed from
the reproducible schedule. Indices range from 0 to 99. An explicit seed
must be an integer between 0 and 2**32 - 1.

`plant --help` lists the options. `quit` (or `q`) exits the shell.
The implemented combination is currently **GCN on Cora**. Other names
in the registries are placeholders and produce an explanatory error.

Each run prints its seed, device, test metric, and test loss after training.
The existing defaults use Cora's public split, normalized features, and
10,000 training epochs, with test evaluation on the final model.
Set `training_config.epochs` in the model/dataset JSON file to override
the epoch count. Model settings and optimizer arguments are read from
the same file, for example `engine/training_engine/config/gcn/cora.json`.

Experiments can also be run from Python without starting the shell:

```python
from engine.training_engine.training.commands.run_planted_seed import run_planted_seed

result = run_planted_seed("gcn", "cora", seed=42, training_device="cpu")
print(result.test_metric)
print(result.history)
```

The API accepts `config_root` and `dataset_root` as `pathlib.Path` values
to select configuration and data directories. `config_root` contains one
directory per model and one JSON file per dataset.

## Overview and Architecture

The experiment service composes the components for a run. Training code
receives those components as arguments and does not access CLI arguments
or look up model and dataset names.

| Component | Responsibility |
| --- | --- |
| `engine/cli_engine` | Parse commands and present errors and results. |
| `training/commands/run_planted_seed.py` | Validate the request, resolve registry entries, and invoke setup and training. |
| `config/experiment_config.py` | Load and validate model and training settings. |
| `training/build_experiment.py` | Seed the run, load data, and construct the model and optimizer using supplied factories. |
| `training/trainers` | Optimize and evaluate the supplied model using supplied data, loss, metric, and epoch count. |
| `engine/repositories` | Map names to metadata and factories; defer training imports until execution. |
| `training/experiment.py` | Define the prepared experiment and the result returned to callers. |

Paths beginning with `training/` or `config/` above are relative to
`engine/training_engine`. The shell delegates to the same experiment
service that Python callers use. Command help and request validation
do not import PyTorch or load datasets.

## Development checks

Run the tests with:

```bash
uv run python -m unittest discover -s tests -v
```

Most tests use supplied factories and require only the Python standard
library. The CPU training smoke test uses a tiny in-memory graph and
requires PyTorch and PyG; it is skipped when they are unavailable.
Tests do not download datasets or require a GPU.

## License

GBensh is released under the Apache License 2.0.
See [LICENSE](LICENSE) for details.

