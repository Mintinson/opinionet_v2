"""
Evaluation utilities for OpinioNet.

Provides metrics calculation and evaluation functions.
"""

from typing import Any, List, Set, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_f1_score(
    predictions: List[Set], ground_truths: List[Set]
) -> Tuple[float, float, float]:
    """
    Calculate F1, precision, and recall scores.

    Args:
        predictions: List of prediction sets
        ground_truths: List of ground truth sets

    Returns:
        Tuple of (f1, precision, recall)
    """
    total_pred = 0
    total_gt = 0
    total_correct = 0

    for pred, gt in zip(predictions, ground_truths):
        total_pred += len(pred)
        total_gt += len(gt)
        total_correct += len(pred.intersection(gt))

    if total_pred == 0 or total_gt == 0 or total_correct == 0:
        return 0.0, 0.0, 0.0

    precision = total_correct / total_pred
    recall = total_correct / total_gt
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def evaluate_model(
    model: Any, dataloader: DataLoader, device: str = "cuda", desc: str = "Evaluating"
) -> Tuple[List, List, float, float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The OpinioNet model
        dataloader: Data loader
        device: Device to evaluate on
        desc: Description for progress bar

    Returns:
        Tuple of (predictions, ground_truths, f1, precision, recall)
    """
    model.eval()

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for raw, inputs, targets in tqdm(dataloader, desc=desc):
            rv_raw, lb_raw = raw

            # Move to device
            inputs = [item.to(device) for item in inputs]

            # Forward pass
            input_ids, attention_mask, token_mask = inputs
            probs, _ = model(
                input_ids=input_ids, attention_mask=attention_mask, token_mask=token_mask
            )

            # Generate predictions
            pred_results = model.generate_candidates(probs)
            pred_results = model.nms_filter(pred_results, threshold=0.1)

            # Collect results
            for b in range(len(pred_results)):
                pred = set([x[0] for x in pred_results[b]])
                gt = set(lb_raw[b]) if lb_raw[b] is not None else set()

                all_predictions.append(pred)
                all_ground_truths.append(gt)

    # Calculate metrics
    f1, precision, recall = calculate_f1_score(all_predictions, all_ground_truths)

    return all_predictions, all_ground_truths, f1, precision, recall


__all__ = [
    "calculate_f1_score",
    "evaluate_model",
]
