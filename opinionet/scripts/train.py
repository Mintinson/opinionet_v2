from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import draccus
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizer

from opinionet.config import PRETRAINED_MODELS
from opinionet.data.dataset import get_data_loaders
from opinionet.models.opinnet import OpinionNet
from opinionet.overwatch.overwatch import initialize_overwatch
from opinionet.training.scheduler import GradualWarmupScheduler
from opinionet.training.trainer import eval_epoch, train_epoch
from opinionet.utils.common import Metrics


# fmt: off
@dataclass
class TrainingConfig:
    base_model: Literal["roberta", "wwm", "ernie"] = "roberta"  # Base pretrained model to use
    data_type : Literal["laptop", "makeup"] = "makeup"  # Type of dataset
    reviews_path: str = "data/TRAIN/Train_reviews.csv"  # Path to reviews CSV file
    labels_path: str = "data/TRAIN/Train_labels.csv" # Path to labels CSV file
    epochs: int = 5  # Number of training epochs
    batch_size: int = 8  # Batch size for training
    lr : Optional[float] = None # Learning rate (None to use default from PRETRAINED_MODELS)
    warmup_epochs: int = 2  # Number of warmup epochs
    val_split:float = 0.20 # Validation split ratio
    output_dir: str = "models/"  # Directory to save model checkpoints
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Device to use for training
    seed :int = 502  # Random seed for reproducibility
    focal : bool = False  # Whether to use focal loss
    max_num_ckpt_saves: int = 3  # Maximum number of model checkpoints to save
# fmt: on

overwatch = initialize_overwatch(__name__)


def save_cfg_yaml(cfg: TrainingConfig, output_path: Path) -> None:
    """Save training configuration to a YAML file.

    Args:
        cfg (TrainingConfig): Training configuration dataclass.
        output_path (Path): Path to save the YAML file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(asdict(cfg), f, allow_unicode=True)
    overwatch.info(f"   Saved training configuration to {output_path}")


@draccus.wrap()
def train(cfg: TrainingConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model_config = PRETRAINED_MODELS[cfg.base_model]
    model_path = model_config["path"]
    lr = cfg.lr if cfg.lr else model_config["lr"]
    focal = cfg.focal if cfg.focal else model_config["focal"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(cfg.output_dir) / f"{cfg.base_model}_{cfg.data_type}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    overwatch.add_file_handler(output_dir / "training.log")

    overwatch.info(f"🚀 Training OpinioNet with {cfg.base_model} model")
    overwatch.info(f"   Model path: {model_path}")
    overwatch.info(f"   Data type: {cfg.data_type}")
    overwatch.info(f"   Learning rate: {lr}")
    overwatch.info(f"   Batch Size: {cfg.batch_size}")
    overwatch.info(f"   Focal loss: {focal}")
    overwatch.info(f"   Device: {cfg.device}")
    save_cfg_yaml(cfg, output_dir / "config.yaml")

    overwatch.info("📚 Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

    # Load data
    overwatch.info(
        f"📊 Loading datasets from {cfg.reviews_path} and {cfg.labels_path}..."
    )
    train_loader, val_loader = get_data_loaders(
        reviews_path=cfg.reviews_path,
        labels_path=cfg.labels_path,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        data_type=cfg.data_type,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    overwatch.info(f"   Train samples: {len(train_loader.dataset)}")  # pyright: ignore[reportArgumentType]
    overwatch.info(f"   Val samples: {len(val_loader.dataset)}")  # pyright: ignore[reportArgumentType]
    overwatch.info("=" * 80)

    # Create model
    overwatch.info("🤖 Creating model...")
    bert_config = BertConfig.from_pretrained(model_path)
    model = OpinionNet(config=bert_config, focal=focal)

    # Load pretrained weights
    try:
        model.bert = model.bert.from_pretrained(model_path)
        overwatch.info("   ✓ Loaded pretrained BERT weights")
    except Exception as e:
        overwatch.warning(f"   ⚠ Could not load pretrained weights: {e}")
    # Move model to device AFTER loading pretrained weights
    device = torch.device(cfg.device)
    model = model.to(device)
    # Setup optimizer and scheduler
    overwatch.info("⚙️  Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = GradualWarmupScheduler(optimizer, total_epoch=cfg.warmup_epochs)

    # Training loop
    overwatch.info("=" * 40 + f"🏋️  Training for {cfg.epochs} epochs..." + "=" * 40)

    best_f1 = 0.0
    best_epoch = 0

    train_data = Metrics()
    val_data = Metrics()

    for epoch in range(cfg.epochs):
        train_loss, train_f1, train_pr, train_rc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
            data_type=cfg.data_type,
            epoch_str=f"{epoch + 1}/{cfg.epochs}",
        )
        train_data.loss.append(train_loss)
        train_data.f1.append(train_f1)
        train_data.precision.append(train_pr)
        train_data.recall.append(train_rc)
        overwatch.info(
            f"Train {epoch + 1}/{cfg.epochs} - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, "
            f"Precision: {train_pr:.4f}, Recall: {train_rc:.4f}"
        )
        if cfg.max_num_ckpt_saves > 0:
            # Save checkpoint
            model_save_path = output_dir / f"{cfg.base_model}_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), model_save_path)
            overwatch.info(f"   ✓ Saved model checkpoint to {model_save_path}")

            # Manage old checkpoints
            all_ckpt_files = sorted(
                output_dir.glob(f"{cfg.base_model}_epoch*.pt"),
                key=lambda x: x.stat().st_mtime,
            )
            if len(all_ckpt_files) > cfg.max_num_ckpt_saves:
                num_to_delete = len(all_ckpt_files) - cfg.max_num_ckpt_saves
                for ckpt_file in all_ckpt_files[:num_to_delete]:
                    ckpt_file.unlink()
                    overwatch.info(f"   🗑 Deleted old checkpoint {ckpt_file}")

        # Evaluate
        val_loss, val_f1, val_pr, val_rc = eval_epoch(
            model=model,
            dataloader=val_loader,
            device=cfg.device,
            data_type=cfg.data_type,
            epoch_str=f"{epoch + 1}/{cfg.epochs}",
        )
        val_data.loss.append(val_loss)
        val_data.f1.append(val_f1)
        val_data.precision.append(val_pr)
        val_data.recall.append(val_rc)
        overwatch.info(
            f"Val {epoch + 1}/{cfg.epochs}   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, "
            f"Precision: {val_pr:.4f}, Recall: {val_rc:.4f}"
        )
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            model_save_path = output_dir / f"{cfg.base_model}_best.pt"
            torch.save(model.state_dict(), model_save_path)
            overwatch.info(f"   ✓ Saved best model to {model_save_path}")

    overwatch.info("\n" + "=" * 80)
    overwatch.info("✅ Training complete!")
    overwatch.info(f"   Best F1: {best_f1:.4f} (Epoch {best_epoch})")

    # Optionally, save training and validation metrics to a file
    metrics_save_path = output_dir / "training_metrics.npy"
    np.save(metrics_save_path, {"train": train_data, "val": val_data})  # pyright: ignore[reportArgumentType]

    overwatch.info(f"   Model saved to: {output_dir}")


if __name__ == "__main__":
    train()  # pyright: ignore[reportCallIssue]
