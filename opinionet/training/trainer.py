"""
Training utilities and core training logic.

Provides training and evaluation loops with PyTorch patterns.
Includes optimization techniques: R-Drop, FGM adversarial training.
"""

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class FGM:
    """Fast Gradient Method for adversarial training.

    Adds perturbation to word embeddings during training to improve robustness.

    Usage:
        fgm = FGM(model)
        # In training loop:
        loss.backward()  # Normal backward
        fgm.attack()     # Add perturbation
        loss_adv = model(...)  # Forward with perturbation
        loss_adv.backward()    # Accumulate gradients
        fgm.restore()    # Restore original embeddings
        optimizer.step()
    """

    def __init__(
        self, model: nn.Module, epsilon: float = 1.0, emb_name: str = "word_embeddings"
    ) -> None:
        """
        Args:
            model: The model to attack
            epsilon: Perturbation magnitude
            emb_name: Name of the embedding layer to perturb
        """
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup: dict = {}

    def attack(self) -> None:
        """Add adversarial perturbation to embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm  # pyright: ignore[reportOperatorIssue]
                    param.data.add_(r_at)

    def restore(self) -> None:
        """Restore original embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def compute_kl_loss(
    p_logits: torch.Tensor, q_logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute symmetric KL divergence for R-Drop with numerical stability.

    Args:
        p_logits: First forward pass logits
        q_logits: Second forward pass logits
        mask: Optional mask for valid positions

    Returns:
        KL divergence loss (mean reduced for numerical stability)
    """
    # Add numerical stability with temperature scaling and clamping
    p_probs = F.softmax(p_logits, dim=-1).clamp(min=1e-8)
    q_probs = F.softmax(q_logits, dim=-1).clamp(min=1e-8)

    p_log_probs = torch.log(p_probs)
    q_log_probs = torch.log(q_probs)

    # KL(p||q) = sum(p * (log(p) - log(q)))
    p_loss = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
    q_loss = (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1)

    if mask is not None:
        p_loss = p_loss.masked_fill(mask, 0.0)
        q_loss = q_loss.masked_fill(mask, 0.0)
        # Compute mean over valid positions only
        num_valid = (~mask).float().sum().clamp(min=1.0)
        kl_loss = (p_loss.sum() + q_loss.sum()) / (2 * num_valid)
    else:
        # Symmetric KL with mean reduction
        kl_loss = (p_loss.mean() + q_loss.mean()) / 2

    return kl_loss


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
    # === Optimization switches ===
    use_rdrop: bool = False,
    rdrop_alpha: float = 0.1,
    use_fgm: bool = False,
    fgm_epsilon: float = 0.5,
    nms_threshold: float = 0.1,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Train for one epoch with optional R-Drop and FGM adversarial training.

    Args:
        model: The OpinioNet model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device to train on
        data_type: Dataset type ('laptop' or 'makeup')
        epoch_str: String for progress bar description
        use_rdrop: Enable R-Drop regularization (forward twice, minimize KL)
        rdrop_alpha: Weight for R-Drop KL loss
        use_fgm: Enable FGM adversarial training
        fgm_epsilon: Perturbation magnitude for FGM
        nms_threshold: NMS threshold for filtering predictions
        max_grad_norm: Maximum gradient norm for clipping (0 = disabled)

    Returns:
        Tuple of (loss, f1, precision, recall)
    """
    model.train()

    cum_loss = 0.0
    total_P, total_G, total_S = 0.0, 0.0, 0.0
    total_samples = 0

    # Initialize FGM if enabled
    fgm = FGM(model, epsilon=fgm_epsilon) if use_fgm else None

    pbar = tqdm(dataloader, desc=f"Training {epoch_str}")

    for _, (raw, inputs, targets) in enumerate(pbar):
        rv_raw, lb_raw = raw

        # Move data to device
        inputs = [item.to(device) for item in inputs]
        targets = [item.to(device) for item in targets]

        input_ids, attention_mask, token_mask = inputs

        optimizer.zero_grad()

        # Forward pass
        probs, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_mask=token_mask,
            data_type=data_type,
        )

        # Compute base loss
        loss = model.compute_loss(logits, targets)

        # R-Drop: Forward twice and minimize KL divergence
        if use_rdrop:
            probs2, logits2 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_mask=token_mask,
                data_type=data_type,
            )
            loss2 = model.compute_loss(logits2, targets)

            # Compute KL divergence for pointer logits (indices 0-3) and classification (5-6)
            kl_loss = torch.tensor(0.0, device=device)
            token_mask_bool = (1 - token_mask).bool()

            # For pointer networks, we need a mask for each "row" (start position)
            # Since logits[i] is [batch, seq, seq], after view(-1, seq) it becomes [batch*seq, seq]
            # We need a mask of shape [batch*seq] indicating which rows are valid
            # A row is invalid if the corresponding token position is masked
            pointer_row_mask = token_mask_bool.view(-1)  # [batch * seq]

            for i in [0, 1, 2, 3]:  # Pointer logits [batch, seq, seq]
                # Flatten to [batch * seq, seq] for KL computation
                batch_size, seq_len, _ = logits[i].shape
                l1 = logits[i].view(-1, seq_len)
                l2 = logits2[i].view(-1, seq_len)
                kl_loss = kl_loss + compute_kl_loss(l1, l2, mask=pointer_row_mask)

            for i in [5, 6]:  # Category and polarity logits [batch, seq, num_classes]
                kl_loss = kl_loss + compute_kl_loss(logits[i], logits2[i], mask=token_mask_bool)

            # Combined loss: average CE + alpha * KL (scale KL by number of terms)
            loss = (loss + loss2) / 2 + rdrop_alpha * kl_loss / 6.0

        # Check for NaN loss and skip if detected
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent explosion (important for Large models)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # FGM adversarial training
        if use_fgm and fgm is not None:
            fgm.attack()
            # Forward with perturbed embeddings
            _, logits_adv = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_mask=token_mask,
                data_type=data_type,
            )
            loss_adv = model.compute_loss(logits_adv, targets)
            # Check adversarial loss for NaN
            if not (torch.isnan(loss_adv) or torch.isinf(loss_adv)):
                loss_adv.backward()
                # Clip gradients again after adversarial backward
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            fgm.restore()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Generate predictions
        pred_results = model.generate_candidates(probs)
        pred_results = model.nms_filter(pred_results, threshold=nms_threshold)

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
    nms_threshold: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    Evaluate for one epoch.

    Args:
        model: The OpinioNet model
        dataloader: Evaluation data loader
        device: Device to evaluate on
        data_type: Dataset type ('laptop' or 'makeup')
        epoch_str: String for progress bar description
        nms_threshold: NMS threshold for filtering predictions

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
            pred_results = model.nms_filter(pred_results, threshold=nms_threshold)

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
    "FGM",
    "compute_kl_loss",
]
