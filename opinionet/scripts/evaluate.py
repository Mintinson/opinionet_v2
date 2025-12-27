"""
Script for evaluating OpinioNet models.

This is a evaluation script using the refactored OpinioNet package.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import draccus
import pandas as pd
import torch
from transformers import BertConfig, BertTokenizer

from opinionet.config import PRETRAINED_MODELS
from opinionet.data.dataset import ReviewDataset
from opinionet.evaluation.metrics import evaluate_model
from opinionet.models.opinnet import OpinionNet
from opinionet.overwatch.overwatch import initialize_overwatch


# fmt: off
@dataclass
class EvaluationConfig:
    model_path: str  # Path to the trained model
    base_model: Literal["roberta", "wwm", "ernie", "roberta_large", "macbert_large", "ernie_large"] = "roberta"  # Base pretrained model used
    review_path: str =  'data/TEST/Test_reviews.csv' # Path to the evaluation dataset
    labels_path: Optional[str] = None  # Path to test labels CSV file (optional, for validation)
    data_type : Literal["laptop", "makeup"] = "makeup"  # Type of dataset
    output: str = 'submit/Result.csv'  # Path to save evaluation results
    batch_size: int = 8  # Batch size for evaluation
    device: str = "cuda"  # Device to use for evaluation
    threshold: float = 0.1  # 'NMS threshold for filtering predictions'

    use_biaffine: bool = False  # Enable Biaffine attention for pointer network
    biaffine_hidden_size: int = 150  # Hidden size for biaffine layer
# fmt: on

overwatch = initialize_overwatch(__name__)


@draccus.wrap()
def evaluate(cfg: EvaluationConfig):
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
        labels_path=cfg.labels_path,
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
    model = OpinionNet(bert_config, use_biaffine=cfg.use_biaffine, biaffine_hidden_size=cfg.biaffine_hidden_size)

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

    if cfg.labels_path:
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
