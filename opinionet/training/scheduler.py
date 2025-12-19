from typing import Optional

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up learning rate in training.

    This scheduler linearly increases the learning rate from 0 to the base learning rate
    over a specified number of epochs. After the warmup phase, it transitions to a
    secondary scheduler (after_scheduler) if provided, or maintains the base learning rate.
    It specifically handles the transition to ReduceLROnPlateau which has a different
    step signature.

    Attributes:
        optimizer: Wrapped optimizer.
        total_epoch: Number of epochs for the linear warmup phase.
        after_scheduler: Scheduler to transition to after warmup (e.g., StepLR, ReduceLROnPlateau).
        finished: Boolean flag indicating if the transition to after_scheduler has occurred.
    """

    def __init__(
        self, optimizer: Optimizer, total_epoch: int, after_scheduler: Optional[_LRScheduler] = None
    ) -> None:
        """
        Initialize the warmup scheduler.

        Args:
            optimizer: Optimizer instance.
            total_epoch: Number of epochs to reach the base learning rate.
            after_scheduler: Optional scheduler to use after the warmup phase.
        """
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """
        Compute the current learning rate based on the warmup progress.

        Returns:
            List of learning rates for each parameter group.
        """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr for base_lr in self.base_lrs]

        # Warmup phase: linear increase
        return [base_lr * (self.last_epoch / self.total_epoch) for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None) -> None:
        """
        Update the learning rate.

        Args:
            epoch: Current epoch number. If None, uses internal counter.
            metrics: Validation metrics required if after_scheduler is ReduceLROnPlateau.
        """
        if type(self.after_scheduler) is not ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super().step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def step_ReduceLROnPlateau(self, metrics: Optional[float], epoch: Optional[int] = None) -> None:  # noqa: N802
        """
        Specialized step logic for transitioning to ReduceLROnPlateau.

        Args:
            metrics: Metric value used by ReduceLROnPlateau to decide LR reduction.
            epoch: Current epoch number.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at epoch end

        if self.last_epoch <= self.total_epoch:
            # warmup phase
            warmup_lr = [
                base_lr * (self.last_epoch / self.total_epoch) for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            # After warmup
            assert self.after_scheduler is not None, "after_scheduler is not set!"
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch) # pyright: ignore[reportCallIssue]


__all__ = ["GradualWarmupScheduler"]
