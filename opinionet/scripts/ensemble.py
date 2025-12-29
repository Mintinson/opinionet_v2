"""
Script for evaluating an Ensemble of OpinioNet models.
Performs soft-voting (probability averaging) to boost performance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple

import draccus
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from opinionet.config import PRETRAINED_MODELS
from opinionet.data.dataset import ID2LAPTOP, ID2MAKEUP, ID2P, ReviewDataset
from opinionet.models.opinnet import OpinionNet
from opinionet.overwatch.overwatch import initialize_overwatch


# fmt: off
@dataclass
class EnsembleConfig:
    # List of model paths to ensemble.
    # Example: ["models/roberta_makeup/seed502/best.pt", "models/roberta_makeup/seed42/best.pt"]
    model_paths: List[str] = field(default_factory=list)

    base_model: Literal["roberta", "wwm", "ernie", "roberta_large", "macbert_large", "ernie_large"] = "roberta"  # Base pretrained model used (must be same for all)
    review_path: str = 'data/TEST/Test_reviews.csv' # Path to the evaluation dataset
    data_type : Literal["laptop", "makeup"] = "makeup"  # Type of dataset
    output: str = 'submit_ensemble/Result.csv'  # Path to save evaluation results
    batch_size: int = 8  # Batch size for evaluation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.1  # NMS threshold

    use_biaffine: bool = False  # Enable Biaffine attention for pointer network
    biaffine_hidden_size: int = 150  # Hidden size for biaffine layer
# fmt: on

overwatch = initialize_overwatch(__name__)


def get_trained_model(
    model_path: str, bert_config: BertConfig, device: torch.device
) -> OpinionNet:
    """
    Load a single trained OpinioNet model from the specified path.
    And properly configure it according to the config.yaml in the same directory.
    """
    config_path = Path(model_path).parent / "config.yaml"
    if not config_path.exists():
        msg = f"❌ Configuration file not found for model at {model_path}"
        overwatch.error(msg)
        raise FileNotFoundError(msg)
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    model = OpinionNet(bert_config, **config_dict)  # Initialize structure
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    # models.append(model)
    # except Exception as e:
    #     overwatch.error(f"   ❌ Failed to load model at {path}: {e}")
    #     return
    return model


def average_probs(probs_list: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    Average the probability outputs from multiple models.

    Args:
        probs_list: A list where each element is the output 'probs' tuple from a single model.
                    Each 'probs' tuple contains multiple tensors (AS, AE, OS, OE, OBJ, C, P).

    Returns:
        A single tuple of tensors containing the averaged probabilities.
    """
    if not probs_list:
        return tuple()

    # Number of distinct probability tensors (e.g., 7 for OpinioNet)
    num_tensors = len(probs_list[0])
    averaged_tensors = []

    for i in range(num_tensors):
        # Stack the i-th tensor from all models along a new dimension 0
        # Shape becomes: [num_models, batch_size, seq_len, ...]
        stacked = torch.stack([p[i] for p in probs_list], dim=0)

        # Calculate mean across the models dimension
        avg = torch.mean(stacked, dim=0)
        averaged_tensors.append(avg)

    return tuple(averaged_tensors)


@draccus.wrap()
def ensemble_evaluate(cfg: EnsembleConfig):
    if not cfg.model_paths:
        overwatch.error(
            "❌ No model paths provided provided! Use --model_paths path1 path2 ..."
        )
        return

    overwatch.info(f"🚀 Starting Ensemble Evaluation with {len(cfg.model_paths)} models")
    overwatch.info(f"   Base model: {cfg.base_model}")
    overwatch.info(f"   Data type: {cfg.data_type}")
    overwatch.info(f"   Review path: {cfg.review_path}")
    overwatch.info(
        f"   Biaffine: {cfg.use_biaffine} (hidden_size={cfg.biaffine_hidden_size})"
    )

    # 1. Load Tokenizer & Config (Assuming all models share the same base config)
    model_config = PRETRAINED_MODELS[cfg.base_model]
    pretrained_path = model_config["path"]

    overwatch.info("📚 Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
    bert_config = BertConfig.from_pretrained(pretrained_path)

    # 2. Prepare Data Loader
    overwatch.info("📊 Loading evaluation data...")
    test_dataset = ReviewDataset(
        reviews_path=cfg.review_path,
        labels_path=None,  # No labels needed for inference
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

    # 3. Load All Models
    overwatch.info("🔧 Loading models into memory...")
    models = []
    device = torch.device(cfg.device)

    for i, path in enumerate(cfg.model_paths):
        overwatch.info(f"   [{i + 1}/{len(cfg.model_paths)}] Loading {path}...")

        # try:
        #     overwatch.info(f"   [{i + 1}/{len(cfg.model_paths)}] Loading {path}...")
        #     model = OpinionNet(
        #         bert_config,
        #         use_biaffine=cfg.use_biaffine,
        #         biaffine_hidden_size=cfg.biaffine_hidden_size,
        #     )  # Initialize structure
        #     checkpoint = torch.load(path, map_location=device)
        #     model.load_state_dict(checkpoint)
        #     model.to(device)
        #     model.eval()
        #     models.append(model)
        # except Exception as e:
        #     overwatch.error(f"   ❌ Failed to load model at {path}: {e}")
        #     return
        model = get_trained_model(path, bert_config, device)
        models.append(model)

    overwatch.info(f"   ✓ Successfully loaded {len(models)} models.")

    # 4. Inference Loop
    overwatch.info("🔍 Running inference and fusing probabilities...")
    all_predictions = []

    with torch.no_grad():
        for _, inputs, _ in tqdm(
            test_loader, total=len(test_loader), desc="Ensemble Inference"
        ):
            # Move inputs to device
            input_ids, attention_mask, token_mask = [item.to(device) for item in inputs]

            batch_probs_list = []

            # Forward pass through ALL models
            for model in models:
                # model() returns (probs, logits). We only need probs for inference.
                probs, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_mask=token_mask,
                    data_type=cfg.data_type,
                )
                batch_probs_list.append(probs)

            # --- SOFT VOTING: Average the Probabilities ---
            avg_probs = average_probs(batch_probs_list)

            # Decode using the averaged probabilities
            # (We can use any model instance for decoding as the logic is static)
            candidates = models[0].generate_candidates(avg_probs)
            filtered_preds = models[0].nms_filter(candidates, threshold=cfg.threshold)

            for b in range(len(filtered_preds)):
                pred = set([x[0] for x in filtered_preds[b]])

                all_predictions.append(pred)

            # all_predictions.extend(filtered_preds)

    # 5. Save Results
    overwatch.info("💾 Formatting and saving results...")

    reviews_df = pd.read_csv(cfg.review_path, encoding="utf-8")
    categories = ID2MAKEUP if cfg.data_type == "makeup" else ID2LAPTOP

    def _extract_term(review_text: str, start: int, end: int) -> str:
        """Extract term from review text using positions."""
        if start <= 0 or end <= 0:
            return "_"
        return review_text[start - 1:end]

    def _format_prediction(pred_tuple: tuple, review_text: str, review_id: int) -> dict:
        """Convert prediction tuple to submission format."""
        a_s, a_e, o_s, o_e, cat, pol = pred_tuple
        return {
            "id": review_id,
            "AspectTerm": _extract_term(review_text, a_s, a_e),
            "OpinionTerm": _extract_term(review_text, o_s, o_e),
            "Category": categories[cat] if 0 <= cat < len(categories) else "_",
            "Polarity": ID2P[pol] if 0 <= pol < len(ID2P) else "_",
        }

    # Match predictions with review IDs
    if len(all_predictions) != len(reviews_df):
        overwatch.warning(
            f"⚠ Warning: Number of predictions ({len(all_predictions)}) does not match number of reviews ({len(reviews_df)})."
        )

    # Generate results with actual text extraction
    results = [
        _format_prediction(pred_tuple, row["Reviews"], row["id"])
        for (_, row), pred_set in zip(reviews_df.iterrows(), all_predictions)
        for pred_tuple in (pred_set if pred_set else [(0, 0, 0, 0, -1, -1)])
    ]

    # Ensure all review IDs are present (even if no predictions)
    existing_ids = {r["id"] for r in results}
    missing_results = [
        {"id": rid, "AspectTerm": "_", "OpinionTerm": "_", "Category": "_", "Polarity": "_"}
        for rid in reviews_df["id"]
        if rid not in existing_ids
    ]
    results.extend(missing_results)

    # use the first model's directory to save
    output_path = Path(cfg.model_paths[0]).parent / cfg.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by id and save without header, without BOM
    results_df = pd.DataFrame(results).sort_values("id")
    results_df.to_csv(output_path, index=False, encoding="utf-8", header=False)

    overwatch.info(f"   ✓ Ensemble results saved to {output_path} ({len(results)} predictions)")
    overwatch.info("✅ Ensemble evaluation complete!")


if __name__ == "__main__":
    ensemble_evaluate()  # pyright: ignore[reportCallIssue]
