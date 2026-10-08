from pathlib import Path


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent

TRAINING_ENGINE_DIR = ENGINE_DIR / "training_engine"
CLI_ENGINE_DIR = ENGINE_DIR / "cli_engine"

MODELS_CONFIG_DIR = TRAINING_ENGINE_DIR / "config"

VENV_DIR = PROJECT_ROOT / ".venv"

RAW_DATASETS_DIR = TRAINING_ENGINE_DIR / "raw_datasets"