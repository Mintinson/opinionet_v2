from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from transformers import BertConfig, BertTokenizer
from typing_extensions import Literal

from opinionet.config import PRETRAINED_MODELS
from opinionet.data.dataset import get_pretrain_loaders
from opinionet.models.opinnet import OpinionNet
from opinionet.overwatch.overwatch import initialize_overwatch
from opinionet.training.scheduler import GradualWarmupScheduler
from opinionet.training.trainer import eval_epoch, evaluate_sample, f1_score


# fmt: off
@dataclass
class PretrainConfig:
    base_model: Literal["roberta", "wwm", "ernie"] = "roberta"  # Base pretrained model to use
    epochs: int = 25  # Number of pretraining epochs
    batch_size: int = 12  # Batch size for pretraining
    lr: Optional[float] = None  # Learning rate (if not specified, use model default)
    output_dir: str = "pretrained_models/"  # Directory to save pretrained model
    val_split: float = 0.15  # Validation split ratio
    seed : int = 502  # Random seed for reproducibility
    reviews_path: str = "data/PRETRAIN/Pretrain_reviews.csv"  # Path to reviews CSV file
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use for pretraining
    seed: int = 502  # Random seed for reproducibility
# fmt: on

overwatch = initialize_overwatch(__name__)


def _cycle_loader(dataloader):
    """Create an infinite iterator that cycles through a DataLoader."""
    while True:
        yield from dataloader


def _process_mlm_step(
    model,
    batch,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    device: Union[torch.device, str],
):
    """Process a single MLM training step.

    Args:
        model: OpinioNet model
        batch: Batch data (corpus_ids, corpus_attn, lm_label)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on

    Returns:
        Tuple of (loss_value, batch_size)
    """
    corpus_ids, corpus_attn, lm_label = batch
    corpus_ids = corpus_ids.to(device)
    corpus_attn = corpus_attn.to(device)
    lm_label = lm_label.to(device)

    lm_loss = model.forward_mlm(corpus_ids, corpus_attn, lm_label)

    optimizer.zero_grad()
    lm_loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return lm_loss.item(), len(corpus_ids)


def _process_opinion_step(
    model,
    batch,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    device: Union[torch.device, str],
):
    """Process a single opinion mining training step.

    Args:
        model: OpinioNet model
        batch: Batch data (raw, inputs, targets)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on

    Returns:
        Tuple of (loss_value, batch_size, predictions, ground_truths)
    """
    raw, inputs, targets = batch
    rv_raw, lb_raw = raw

    inputs = [item.to(device) for item in inputs]
    targets = [item.to(device) for item in targets]

    input_ids, attention_mask, token_mask = inputs
    probs, logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_mask=token_mask,
        data_type="makeup",
    )

    loss: torch.Tensor = model.compute_loss(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    # Generate and filter predictions
    predictions = model.generate_candidates(probs)
    predictions = model.nms_filter(predictions, 0.1)

    return loss.item(), len(rv_raw), predictions, lb_raw


def _compute_batch_metrics(predictions, ground_truths):
    """Compute metrics for a batch of predictions.

    Args:
        predictions: List of predictions for each sample
        ground_truths: List of ground truth labels for each sample

    Returns:
        Tuple of (P, G, S) - predicted count, ground truth count, correct count
    """
    metrics = [
        evaluate_sample(gt, [x[0] for x in pred])
        for pred, gt in zip(predictions, ground_truths)
        if gt is not None
    ]

    if not metrics:
        return 0, 0, 0

    total_p = sum(p for p, _, _ in metrics)
    total_g = sum(g for _, g, _ in metrics)
    total_s = sum(s for _, _, s in metrics)

    return total_p, total_g, total_s


def train_pretrain_epoch(
    model, makeup_loader, corpus_loader, optimizer, scheduler: LRScheduler, device="cuda"
):
    """
    Train one epoch with both MLM and opinion mining tasks.

    Args:
        model: OpinioNet model with MLM capability
        makeup_loader: DataLoader for makeup review data
        corpus_loader: DataLoader for unlabeled corpus (MLM)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on

    Returns:
        Tuple of (opinion_loss, mlm_loss, f1, precision, recall)
    """
    from tqdm import tqdm

    model.train()

    # Initialize accumulators
    total_opinion_loss = 0.0
    total_mlm_loss = 0.0
    total_opinion_samples = 0
    total_mlm_samples = 0
    total_P, total_G, total_S = 0, 0, 0

    # Create cycling iterators
    makeup_iter = _cycle_loader(makeup_loader)
    corpus_iter = _cycle_loader(corpus_loader)

    max_steps = max(len(makeup_loader), len(corpus_loader))
    pbar = tqdm(range(max_steps), desc="Pretraining")

    for step in pbar:
        # Process MLM step
        mlm_batch = next(corpus_iter)
        mlm_loss, mlm_batch_size = _process_mlm_step(
            model, mlm_batch, optimizer, scheduler, device
        )
        total_mlm_loss += mlm_loss * mlm_batch_size
        total_mlm_samples += mlm_batch_size

        # Process opinion mining step
        opinion_batch = next(makeup_iter)
        opinion_loss, opinion_batch_size, predictions, ground_truths = (
            _process_opinion_step(model, opinion_batch, optimizer, scheduler, device)
        )
        total_opinion_loss += opinion_loss * opinion_batch_size
        total_opinion_samples += opinion_batch_size

        # Accumulate metrics
        batch_p, batch_g, batch_s = _compute_batch_metrics(predictions, ground_truths)
        total_P += batch_p
        total_G += batch_g
        total_S += batch_s

        # Update progress bar
        if total_G > 0:
            current_f1, _, _ = f1_score(total_P, total_G, total_S)
            pbar.set_postfix(
                {
                    "loss": f"{total_opinion_loss / total_opinion_samples:.4f}",
                    "lm_loss": f"{total_mlm_loss / total_mlm_samples:.4f}",
                    "f1": f"{current_f1:.4f}",
                }
            )

    # Compute final metrics
    avg_opinion_loss = (
        total_opinion_loss / total_opinion_samples if total_opinion_samples > 0 else 0.0
    )
    avg_mlm_loss = total_mlm_loss / total_mlm_samples if total_mlm_samples > 0 else 0.0
    final_f1, final_pr, final_rc = f1_score(total_P, total_G, total_S)

    return avg_opinion_loss, avg_mlm_loss, final_f1, final_pr, final_rc


@draccus.wrap()
def pretrain(cfg: PretrainConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Get model configuration
    model_config = PRETRAINED_MODELS[cfg.base_model]
    model_path = model_config["path"]
    lr = cfg.lr if cfg.lr is not None else model_config["lr"]
    focal = model_config["focal"]
    version = model_config.get("version", "large")

    overwatch.info(f"🛠️  Pretraining OpinioNet with {cfg.base_model} model")
    overwatch.info(f"   Model path: {model_path}")
    overwatch.info(f"   Learning rate: {lr}")
    overwatch.info(f"   Focal loss: {focal}")
    overwatch.info(f"   Version: {version}")
    overwatch.info(f"   Device: {cfg.device}")

    # Load tokenizer
    overwatch.info("📚 Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

    # Load data
    overwatch.info(f"📊 Loading datasets from {cfg.reviews_path}...")

    makeup_train_loader, makeup_val_loader, corpus_loader = get_pretrain_loaders(
        tokenizer=tokenizer, batch_size=cfg.batch_size, val_split=cfg.val_split
    )

    overwatch.info(f"   Makeup train samples: {len(makeup_train_loader.dataset)}")  # pyright: ignore[reportArgumentType]
    overwatch.info(f"   Makeup val samples: {len(makeup_val_loader.dataset)}")  # pyright: ignore[reportArgumentType]
    overwatch.info(f"   Corpus samples: {len(corpus_loader.dataset)}")  # pyright: ignore[reportArgumentType]
    overwatch.info("")

    # Create model
    overwatch.info("🤖 Creating model...")
    bert_config = BertConfig.from_pretrained(model_path)
    model = OpinionNet(config=bert_config, focal=focal, version=version)

    # Load pretrained weights
    try:
        model.bert = model.bert.from_pretrained(model_path)
        overwatch.info("   ✓ Loaded pretrained BERT weights")
    except Exception as e:
        overwatch.error(f"   ⚠ Could not load pretrained weights: {e}")

    # Move to device
    device = torch.device(cfg.device)
    model = model.to(device)
    overwatch.info(f"   ✓ Model moved to {device}")

    # Setup optimizer and scheduler
    overwatch.info("⚙️  Setting up optimizer and scheduler...")
    optimizer = Adam(model.parameters(), lr=lr)
    warmup_steps = 2 * max(len(makeup_train_loader), len(corpus_loader))
    scheduler = GradualWarmupScheduler(optimizer, total_epoch=warmup_steps)

    # Pretraining loop
    overwatch.info(f"\n🏋️  Pretraining for {cfg.epochs} epochs...")
    overwatch.info("=" * 60)

    best_val_f1 = 0.0
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        overwatch.info(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        overwatch.info("-" * 60)

        # Pretrain for one epoch
        train_loss, train_lm_loss, train_f1, train_pr, train_rc = train_pretrain_epoch(
            model=model,
            makeup_loader=makeup_train_loader,
            corpus_loader=corpus_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
        )
        overwatch.info(
            f"Train - Loss: {train_loss:.4f}, LM Loss: {train_lm_loss:.4f}, "
            f"F1: {train_f1:.4f}, Precision: {train_pr:.4f}, Recall: {train_rc:.4f}"
        )
        # Validate
        overwatch.info("Validating on makeup data...")
        val_loss, val_f1, val_pr, val_rc = eval_epoch(
            model=model,
            dataloader=makeup_val_loader,
            device=cfg.device,
            data_type="makeup",
        )

        overwatch.info(
            f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, "
            f"Precision: {val_pr:.4f}, Recall: {val_rc:.4f}"
        )

        # Update best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            # Save model if F1 is good enough
            if best_val_f1 >= 0.75:
                output_path = Path(cfg.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                model_save_path = output_path / f"pretrained_{model_config['name']}.pt"
                torch.save(model.state_dict(), model_save_path)
                overwatch.info(f"   ✓ Saved best model to {model_save_path}")
        else:
            # Early stopping if F1 doesn't improve
            overwatch.info("   ⚠ F1 did not improve, stopping early")
            break
    overwatch.info("\n" + "=" * 60)
    overwatch.info("✅ Pretraining complete!")
    overwatch.info(f"   Best F1: {best_val_f1:.4f}")
    overwatch.info(f"   Best Loss: {best_val_loss:.4f}")

    if best_val_f1 >= 0.75:
        overwatch.info(f"   Model saved to: {cfg.output_dir}")


if __name__ == "__main__":
    pretrain()  # pyright: ignore[reportCallIssue]
