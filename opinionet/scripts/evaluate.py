"""
Script for evaluating OpinioNet models.

This is a evaluation script using the refactored OpinioNet package.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import draccus
import pandas as pd
import torch
from transformers import BertConfig, BertTokenizer

from opinionet.config import PRETRAINED_MODELS
from opinionet.data.dataset import ReviewDataset
from opinionet.evaluation.metrics import evaluate_model
from opinionet.models.opinnet import OpinionNet
from opinionet.overwatch.overwatch import initialize_overwatch
from opinionet.scripts.train import TrainingConfig


# fmt: off
@dataclass
class EvaluationConfig(TrainingConfig):

    ckpt_dir: Optional[str] = None # Directory containing the trained model checkpoint, the script will look for `config.yaml` in this directory
    model_path: str = ""     # name to the trained model checkpoint. if empty, use default (best) from TrainingConfig the parent directory will be the config file output directory
    review_path: str =  'data/TEST/Test_reviews.csv' # Path to the evaluation dataset
    eval_labels_path: Optional[str] = None  # Path to test labels CSV file (optional, for validation)
    output: str = 'submit/Result.csv'  # Path to save evaluation results
# fmt: on

overwatch = initialize_overwatch(__name__)


def load_config_from_checkpoint(
    ckpt_dir: Path, old_config: EvaluationConfig
) -> EvaluationConfig:
    """Load evaluation configuration from a checkpoint directory."""
    import yaml

    config_path = ckpt_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    old_config_dict = asdict(old_config)
    # Update only the fields that exist in the old configuration
    config_dict.update({k: v for k, v in old_config_dict.items() if k not in config_dict})
    eval_config = EvaluationConfig(**config_dict)

    return eval_config


@draccus.wrap()
def evaluate(cfg: EvaluationConfig):
    if not cfg.ckpt_dir:
        msg = "⚠️ ckpt_dir not specified, attempting to use model_path directly."
        overwatch.error(msg)
        raise RuntimeError(msg)
    ckpt_dir = Path(cfg.ckpt_dir)
    if not ckpt_dir.exists():
        msg = f"⚠️ Specified ckpt_dir does not exist: {ckpt_dir}"
        overwatch.error(msg)
        raise FileNotFoundError(msg)
    cfg = load_config_from_checkpoint(ckpt_dir, cfg)
    if not cfg.model_path:
        # Use default best checkpoint from training
        cfg.model_path = str(ckpt_dir / f"{cfg.base_model}_best.pt")

    overwatch.info(
        f"🔍 Evaluating OpinioNet model from {cfg.model_path} on {cfg.review_path}"
    )
    overwatch.info(f"   Base model: {cfg.base_model}")
    overwatch.info(f"   Data type: {cfg.data_type}")
    overwatch.info(f"   Device: {cfg.device}")
    overwatch.info("   === Optimization Switches ===")
    overwatch.info(
        f"   Biaffine: {cfg.use_biaffine} (hidden_size={cfg.biaffine_hidden_size})"
    )

    model_config = PRETRAINED_MODELS[cfg.base_model]
    pretrained_path = model_config["path"]

    overwatch.info("📚 Loading model...")
    tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)

    overwatch.info("📊 Loading evaluation data...")
    test_dataset = ReviewDataset(
        reviews_path=cfg.review_path,
        labels_path=cfg.eval_labels_path,
        tokenizer=tokenizer,
        data_type=cfg.data_type,
    )

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=4,
    )
    overwatch.info(f"   Loaded {len(test_dataset)} samples for evaluation.")
    overwatch.info("")

    # Create model
    overwatch.info("🔧 Initializing model...")
    bert_config = BertConfig.from_pretrained(pretrained_path)
    model = OpinionNet(
        bert_config,
        use_biaffine=cfg.use_biaffine,
        biaffine_hidden_size=cfg.biaffine_hidden_size,
    )

    device = torch.device(cfg.device)
    chekkpoint = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(chekkpoint)

    model = model.to(device)
    overwatch.info(f"   ✓ Loaded checkpoint from {cfg.model_path}")
    overwatch.info(f"   ✓ Model moved to {device}")
    overwatch.info("")

    overwatch.info("🔍 Starting evaluation...")

    predictions, ground_truths, f1, precision, recall = evaluate_model(
        model=model, dataloader=test_loader, device=cfg.device, desc="Evaluating"
    )

    if cfg.eval_labels_path:
        overwatch.info("📈 Evaluation Results:")
        overwatch.info(f"   F1 Score: {f1:.4f}")
        overwatch.info(f"   Precision: {precision:.4f}")
        overwatch.info(f"   Recall: {recall:.4f}")

    # Save Results
    model_parent = Path(cfg.model_path).parent
    output_path = model_parent / cfg.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overwatch.info(f"💾 Saving results to {output_path}...")

    # Format results for submission using functional style
    from opinionet.data.dataset import ID2LAPTOP, ID2MAKEUP, ID2P

    reviews_df = pd.read_csv(cfg.review_path, encoding="utf-8")
    categories = ID2MAKEUP if cfg.data_type == "makeup" else ID2LAPTOP

    def _format_prediction(pred_tuple: tuple, review_id: int) -> dict:
        """Convert prediction tuple to result dictionary."""
        a_s, a_e, o_s, o_e, cat, pol = pred_tuple
        return {
            "id": review_id,
            "A_start": a_s - 1 if a_s > 0 else "",
            "A_end": a_e if a_e > 0 else "",
            "O_start": o_s - 1 if o_s > 0 else "",
            "O_end": o_e if o_e > 0 else "",
            "Categories": categories[cat] if 0 <= cat < len(categories) else "",
            "Polarities": ID2P[pol] if 0 <= pol < len(ID2P) else "",
        }

    results = [
        _format_prediction(pred_tuple, review_id)
        for pred_set, review_id in zip(predictions, reviews_df["id"])
        for pred_tuple in pred_set
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding="utf-8")
    overwatch.info(f"   ✓ Results saved ({len(results)} predictions)")
    overwatch.info("✅ Evaluation complete!")


if __name__ == "__main__":
    evaluate()  # pyright: ignore[reportCallIssue]
