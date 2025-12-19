"""Visualization script for training and validation metrics."""

import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from opinionet.utils.common import Metrics


def plot(title: str, loss, f1, precision, recall, save_path: Path) -> None:
    epochs = range(1, len(loss) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    epochs = range(len(loss))
    plot_configs = [
        (
            axes[0, 0],
            loss,
            "Loss",
            "tab:red",
            "Training Loss",
            (0, max(loss) * 1.1),
        ),
        (axes[0, 1], f1, "F1 Score", "tab:blue", "F1 Score", (0, 1.0)),
        (
            axes[1, 0],
            precision,
            "Precision",
            "tab:green",
            "Precision",
            (0, 1.0),
        ),
        (axes[1, 1], recall, "Recall", "tab:orange", "Recall", (0, 1.0)),
    ]

    for ax, data, title, color, ylabel, ylim in plot_configs:
        # 绘制曲线
        ax.plot(
            epochs,
            data,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=f"{ylabel} (Final: {data[-1]:.3f})",
        )

        # 设置标题和标签
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")

        # 美化网格和边框
        ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        # 设置坐标轴范围
        ax.set_ylim(ylim)
        # ax.set_xlim(1, 50)

        # 添加图例
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

        # 添加最终值的标注
        final_val = data[-1]
        ax.annotate(
            f"{final_val:.3f}",
            xy=(50, final_val),
            xytext=(52, final_val),
            fontsize=10,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为主标题留出空间

    # 6. 保存和显示
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"✅ Save metrics plot to '{save_path}'")


def visualize_data(data_path: Union[str, Path]) -> None:
    """Visualize data stored in a NumPy file.

    Args:
        data_path: Path to the NumPy file containing data.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
    parent_dir = data_path.parent

    train_data_path = parent_dir / "training_metrics.png"
    val_data_path = parent_dir / "validation_metrics.png"

    data: dict[str, Metrics] = np.load(data_path, allow_pickle=True).item()
    train_metrics = data["train"]
    eval_metrics = data["val"]

    plot(
        title="Training Metrics Over Epochs",
        loss=train_metrics.loss,
        f1=train_metrics.f1,
        precision=train_metrics.precision,
        recall=train_metrics.recall,
        save_path=train_data_path,
    )
    plot(
        title="Validation Metrics Over Epochs",
        loss=eval_metrics.loss,
        f1=eval_metrics.f1,
        precision=eval_metrics.precision,
        recall=eval_metrics.recall,
        save_path=val_data_path,
    )


def visualize():
    parser = argparse.ArgumentParser(
        description="Visualize training and validation metrics from a NumPy file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the NumPy file containing training and validation metrics.",
    )
    args = parser.parse_args()
    visualize_data(args.data_path)


if __name__ == "__main__":
    visualize()
