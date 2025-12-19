"""overwatch.py.

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""

import logging
import logging.config
import os
from collections.abc import Callable, MutableMapping
from contextlib import nullcontext
from logging import LoggerAdapter
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

# Overwatch Default Format String
RICH_FORMATTER = "%(filename)s:%(lineno)d | >> %(message)s"
FILE_FORMATTER = "%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s"
DATEFMT = "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "formatter": "simple-console",
            "class": "rich.logging.RichHandler",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": False,
            "show_time": True,
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    """Custom LoggerAdapter that adds contextual prefixes to log messages.

    This adapter formats log messages with hierarchical prefixes to indicate
    the context level of the message, providing visual structure to logs.

    Attributes:
        CTX_PREFIXES (ClassVar[dict[int, str]]): Mapping of context levels to prefix strings.

    """

    CTX_PREFIXES: ClassVar[dict[int, str]] = {
        0: "[*] ",
        **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]},
    }

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Process a log message by adding contextual prefix.

        Args:
            msg (str): The original log message.
            kwargs (MutableMapping[str, Any]): Additional keyword arguments for logging.
                The 'ctx_level' key is extracted to determine the prefix level.

        Returns:
            tuple[str, MutableMapping[str, Any]]: Processed message with prefix and
                updated kwargs with 'ctx_level' removed.

        """
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    """Overwatch wrapper for distributed training with logging and accelerate integration.

    This class provides centralized logging and distributed training utilities by wrapping
    the accelerate library's PartialState and providing convenient access to distributed
    training properties and logging methods.

    Attributes:
        logger (ContextAdapter): Enhanced logger with contextual formatting.
        distributed_state (PartialState): Accelerate's distributed state manager.
        debug (Callable): Logger debug method.
        info (Callable): Logger info method.
        warning (Callable): Logger warning method.
        error (Callable): Logger error method.
        critical (Callable): Logger critical method.

    """

    def __init__(self, name: str, log_file: Optional[str] = None) -> None:
        """Initialize DistributedOverwatch with logging and distributed state.

        Sets up a contextual logger and accelerate's PartialState for distributed training.
        Configures logging levels to show INFO on main process and ERROR on others.

        Args:
            name (str): Name for the logger instance.
            log_file (Optional[str]): Path to a file for logging. Only active on main process.

        """
        try:
            from accelerate import PartialState  # pyright: ignore[reportMissingImports]  # noqa: I001, PLC0415
        except ImportError as e:
            raise ImportError(
                "The 'accelerate' library is required for DistributedOverwatch. "
                "Please install it via 'pip install accelerate'."
            ) from e

        # Note that PartialState is always safe to initialize regardless of `accelerate launch` or `torchrun`
        #   =>> However, might be worth actually figuring out if we need the `accelerate` dependency at all!
        self.logger, self.distributed_state = (
            ContextAdapter(logging.getLogger(name), extra={}),
            PartialState(),
        )

        # Add File Handler if specified (only on main process)
        if log_file and self.distributed_state.is_main_process:
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(FILE_FORMATTER, datefmt=DATEFMT))
            self.logger.logger.addHandler(fh)

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        self.logger.setLevel(
            logging.INFO if self.distributed_state.is_main_process else logging.ERROR
        )

    def add_file_handler(self, file: Union[str,  Path]) -> None:
        """Add a logging handler to the logger.

        Args:
            handler (logging.Handler): The logging handler to add.

        """
        if self.distributed_state.is_main_process:
            fh = logging.FileHandler(file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(FILE_FORMATTER, datefmt=DATEFMT))
            self.logger.logger.addHandler(fh)

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        """Get decorator for executing functions only on the main process.

        Returns:
            Callable[..., Any]: Decorator that restricts execution to main process.

        """
        return self.distributed_state.on_main_process

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        """Get decorator for executing functions only on the local main process.

        Returns:
            Callable[..., Any]: Decorator that restricts execution to local main process.

        """
        return self.distributed_state.on_local_main_process

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        """Get context manager for executing code on main process first.

        Returns:
            Callable[..., Any]: Context manager for main-process-first execution.

        """
        return self.distributed_state.main_process_first

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        """Get context manager for executing code on local main process first.

        Returns:
            Callable[..., Any]: Context manager for local-main-process-first execution.

        """
        return self.distributed_state.local_main_process_first

    def is_rank_zero(self) -> bool:
        """Check if current process is the main process.

        Returns:
            bool: True if this is the main process, False otherwise.

        """
        return self.distributed_state.is_main_process

    def rank(self) -> int:
        """Get the global rank of the current process.

        Returns:
            int: Global process rank.

        """
        return self.distributed_state.process_index

    def local_rank(self) -> int:
        """Get the local rank of the current process.

        Returns:
            int: Local process rank within the node.

        """
        return self.distributed_state.local_process_index

    def world_size(self) -> int:
        """Get the total number of processes in the distributed setup.

        Returns:
            int: Total number of processes.

        """
        return self.distributed_state.num_processes


class PureOverwatch:
    """Simple Overwatch wrapper for non-distributed environments.

    This class provides the same interface as DistributedOverwatch but for
    single-process environments. All distributed-related methods return
    sensible defaults for single-process execution.

    Attributes:
        logger (ContextAdapter): Enhanced logger with contextual formatting.
        debug (Callable): Logger debug method.
        info (Callable): Logger info method.
        warning (Callable): Logger warning method.
        error (Callable): Logger error method.
        critical (Callable): Logger critical method.

    """

    def __init__(self, name: str, log_file: Optional[str] = None) -> None:
        """Initialize PureOverwatch with logging only.

        Sets up a contextual logger for single-process environments.

        Args:
            name (str): Name for the logger instance.
            log_file (Optional[str]): Path to a file for logging.

        """
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        # Add File Handler if specified
        if log_file:
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(FILE_FORMATTER, datefmt=DATEFMT))
            self.logger.logger.addHandler(fh)

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)

    def add_file_handler(self, file: Union[str,  Path]) -> None:
        """Add a logging handler to the logger.

        Args:
            handler (logging.Handler): The logging handler to add.

        """
        fh = logging.FileHandler(file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(FILE_FORMATTER, datefmt=DATEFMT))
        self.logger.logger.addHandler(fh)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        """Get an identity decorator that returns functions unchanged.

        Returns:
            Callable[..., Any]: Identity decorator function.

        """

        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        """Get identity decorator (no-op for single process).

        Returns:
            Callable[..., Any]: Identity decorator that does nothing.

        """
        return self.get_identity_ctx()

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        """Get identity decorator (no-op for single process).

        Returns:
            Callable[..., Any]: Identity decorator that does nothing.

        """
        return self.get_identity_ctx()

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        """Get null context manager (no-op for single process).

        Returns:
            Callable[..., Any]: Null context manager.

        """
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        """Get null context manager (no-op for single process).

        Returns:
            Callable[..., Any]: Null context manager.

        """
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        """Check if current process is rank zero (always True for single process).

        Returns:
            bool: Always True for single process.

        """
        return True

    @staticmethod
    def rank() -> int:
        """Get process rank (always 0 for single process).

        Returns:
            int: Always 0 for single process.

        """
        return 0

    @staticmethod
    def world_size() -> int:
        """Get world size (always 1 for single process).

        Returns:
            int: Always 1 for single process.

        """
        return 1


def initialize_overwatch(
    name: str, log_file: Optional[str] = None
) -> DistributedOverwatch | PureOverwatch:
    """Initialize appropriate Overwatch instance based on environment.

    Automatically detects whether running in distributed or single-process mode
    by checking the WORLD_SIZE environment variable and returns the appropriate
    OverWatch implementation.

    Args:
        name (str): Name for the logger instance.
        log_file (Optional[str]): Path to a file for logging.

    Returns:
        DistributedOverwatch | PureOverwatch: DistributedOverwatch if WORLD_SIZE > 1,
            otherwise PureOverwatch.

    """
    return (
        DistributedOverwatch(name, log_file)
        if int(os.environ.get("WORLD_SIZE", "-1")) != -1
        else PureOverwatch(name, log_file)
    )


__all__ = [
    "DistributedOverwatch",
    "PureOverwatch",
    "initialize_overwatch",
]

if __name__ == "__main__":
    ow = initialize_overwatch("test")
    ow.info("This is an info message.")
    ow.info("This is an info message.", ctx_level=2)
    ow.debug("This is a debug message.")
    ow.warning("This is a warning message.")
    ow.error("This is an error message.", ctx_level=1)
    ow.critical("This is a critical message.")
