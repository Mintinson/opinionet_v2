"""Configuration module for OpinioNet models and training."""

from pathlib import Path
from typing import Any, Dict

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
SUBMIT_DIR = PROJECT_ROOT / "submit"

# Pretrained model configurations
PRETRAINED_MODELS: Dict[str, Dict[str, Any]] = {
    # HuggingFace Hub models (will auto-download)
    "roberta": {
        "name": "roberta",
        "path": "hfl/chinese-roberta-wwm-ext",  # HuggingFace model
        "lr": 6e-6,
        "version": "large",
        "focal": False,
    },
    "wwm": {
        "name": "wwm",
        "path": "hfl/chinese-bert-wwm-ext",  # HuggingFace model
        "lr": 6e-6,
        "version": "large",
        "focal": False,
    },
    "ernie": {
        "name": "ernie",
        "path": "nghuyong/ernie-3.0-base-zh",  # HuggingFace model
        "lr": 8e-6,
        "version": "large",
        "focal": False,
    },
    # Local models (if you have them downloaded)
    "roberta_local": {
        "name": "roberta_local",
        "path": str(MODELS_DIR / "chinese_roberta_wwm_ext_pytorch"),
        "lr": 6e-6,
        "version": "large",
        "focal": False,
    },
    "wwm_local": {
        "name": "wwm_local",
        "path": str(MODELS_DIR / "chinese_wwm_ext_pytorch"),
        "lr": 6e-6,
        "version": "large",
        "focal": False,
    },
    "ernie_local": {
        "name": "ernie_local",
        "path": str(MODELS_DIR / "ERNIE"),
        "lr": 8e-6,
        "version": "large",
        "focal": False,
    },
    "roberta_focal": {
        "name": "roberta_focal",
        "path": "hfl/chinese-roberta-wwm-ext",
        "lr": 6e-6,
        "version": "large",
        "focal": True,
    },
    "wwm_focal": {
        "name": "wwm_focal",
        "path": "hfl/chinese-bert-wwm-ext",
        "lr": 6e-6,
        "version": "large",
        "focal": True,
    },
    # Large models (768 -> 1024 hidden size)
    "roberta_large": {
        "name": "roberta_large",
        "path": "hfl/chinese-roberta-wwm-ext-large",
        "lr": 3e-6,  # Lower LR for large models
        "version": "large",
        "focal": False,
    },
    "macbert_large": {
        "name": "macbert_large",
        "path": "hfl/chinese-macbert-large",
        "lr": 3e-6,
        "version": "large",
        "focal": False,
    },
    "ernie_large": {
        "name": "ernie_large",
        "path": "nghuyong/ernie-3.0-xbase-zh",  # ERNIE 3.0 xbase is larger
        "lr": 4e-6,
        "version": "large",
        "focal": False,
    },
}
