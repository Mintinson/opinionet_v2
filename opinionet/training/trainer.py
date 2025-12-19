"""
Training utilities and core training logic.

Provides training and evaluation loops with PyTorch patterns.
"""

from typing import Any, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def f1_score(P: float, G: float, S: float) -> Tuple[float, float, float]:
    """
    Calculate F1 score, precision, and recall.

    Args:
        P: Number of predicted entities
        G: Number of ground truth entities
        S: Number of correct predictions (intersection)

    Returns:
        Tuple of (f1, precision, recall)
    """
    if P == 0 or G == 0 or S == 0:
        return 0.0, 0.0, 0.0
    precision = S / P
    recall = S / G
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return f1, precision, recall


def evaluate_sample(gt: Sequence, pred: Sequence) -> Tuple[int, int, int]:
    """
    Evaluate a single sample.

    Args:
        gt: Ground truth set
        pred: Predicted set

    Returns:
        Tuple of (P, G, S) - predicted count, ground truth count, correct count
    """
    gt_set = set(gt) if not isinstance(gt, set) else gt
    pred_set = set(pred) if not isinstance(pred, set) else pred

    p = len(pred_set)
    g = len(gt_set)
    s = len(gt_set.intersection(pred_set))

    return p, g, s


def train_epoch(
    model: Any,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[Any] = None,
    device: str = "cuda",
    data_type: str = "laptop",
    epoch_str: str = "",
) -> Tuple[float, float, float, float]:
    """
    Train for one epoch.

    Args:
        model: The OpinioNet model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device to train on
        data_type: Dataset type ('laptop' or 'makeup')

    Returns:
        Tuple of (loss, f1, precision, recall)
    """
    model.train()

    cum_loss = 0.0
    total_P, total_G, total_S = 0.0, 0.0, 0.0

    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Training {epoch_str}")

    for _, (raw, inputs, targets) in enumerate(pbar):
        rv_raw, lb_raw = raw

        # Move data to device
        inputs = [item.to(device) for item in inputs]
        targets = [item.to(device) for item in targets]

        # Forward pass
        input_ids, attention_mask, token_mask = inputs
        probs, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_mask=token_mask,
            data_type=data_type,
        )

        # Compute loss
        loss = model.compute_loss(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # Update weights
        if scheduler is not None:
            scheduler.step()

        # Generate predictions
        pred_results = model.generate_candidates(probs)
        pred_results = model.nms_filter(pred_results, threshold=0.1)

        # Accumulate metrics
        cum_loss += loss.item() * len(rv_raw)
        total_samples += len(rv_raw)

        for b in range(len(pred_results)):
            if lb_raw[b] is not None:
                gt = lb_raw[b]
                pred = [x[0] for x in pred_results[b]]
                p, g, s = evaluate_sample(gt, pred)
                total_P += p
                total_G += g
                total_S += s

        # Update progress bar
        if total_G > 0:
            current_f1, _, _ = f1_score(total_P, total_G, total_S)
            pbar.set_postfix(
                {"loss": f"{cum_loss / total_samples:.4f}", "f1": f"{current_f1:.4f}"}
            )

    # Calculate final metrics
    avg_loss = cum_loss / total_samples if total_samples > 0 else 0.0
    final_f1, final_pr, final_rc = f1_score(total_P, total_G, total_S)

    return avg_loss, final_f1, final_pr, final_rc


def eval_epoch(
    model: Any,
    dataloader: DataLoader,
    device: str = "cuda",
    data_type: str = "laptop",
    epoch_str: str = "",
) -> Tuple[float, float, float, float]:
    """
    Evaluate for one epoch.

    Args:
        model: The OpinioNet model
        dataloader: Evaluation data loader
        device: Device to evaluate on
        data_type: Dataset type ('laptop' or 'makeup')

    Returns:
        Tuple of (loss, f1, precision, recall)
    """
    model.eval()

    cum_loss = 0.0
    total_P, total_G, total_S = 0.0, 0.0, 0.0

    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Evaluating {epoch_str}")

    with torch.no_grad():
        for _, (raw, inputs, targets) in enumerate(pbar):
            rv_raw, lb_raw = raw

            # Move data to device
            inputs = [item.to(device) for item in inputs]
            targets = [item.to(device) for item in targets]

            # Forward pass
            input_ids, attention_mask, token_mask = inputs
            probs, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_mask=token_mask,
                data_type=data_type,
            )

            # Compute loss
            loss = model.compute_loss(logits, targets)

            # Generate predictions
            pred_results = model.generate_candidates(probs)
            pred_results = model.nms_filter(pred_results, threshold=0.1)

            # Accumulate metrics
            cum_loss += loss.item() * len(rv_raw)
            total_samples += len(rv_raw)

            for b in range(len(pred_results)):
                if lb_raw[b] is not None:
                    gt = lb_raw[b]
                    pred = [x[0] for x in pred_results[b]]
                    p, g, s = evaluate_sample(gt, pred)
                    total_P += p
                    total_G += g
                    total_S += s

            # Update progress bar
            if total_G > 0:
                current_f1, _, _ = f1_score(total_P, total_G, total_S)
                pbar.set_postfix(
                    {"loss": f"{cum_loss / total_samples:.4f}", "f1": f"{current_f1:.4f}"}
                )

    # Calculate final metrics
    avg_loss = cum_loss / total_samples if total_samples > 0 else 0.0
    final_f1, final_pr, final_rc = f1_score(total_P, total_G, total_S)

    return avg_loss, final_f1, final_pr, final_rc


__all__ = [
    "f1_score",
    "evaluate_sample",
    "train_epoch",
    "eval_epoch",
]
