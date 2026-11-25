# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import atexit
import functools
import logging
import os
import sys
import uuid
import wandb
from datetime import datetime
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from numpy import ndarray

from sam3.train.utils.train_utils import get_machine_local_and_dist_rank, makedir
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

Scalar = Union[Tensor, ndarray, int, float]


def _make_default_wandb_run_name(project: str, run_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = os.path.basename(os.path.normpath(run_dir or "")) or "run"
    return f"{project}-{dir_name}-{timestamp}"


def _sanitize_scalar(data: Scalar):
    """Convert scalar-like tensors/arrays into plain Python numbers before logging."""
    if isinstance(data, Tensor):
        data = data.detach()
        if data.numel() == 1:
            return data.item()
        return data.cpu()
    if isinstance(data, ndarray):
        if data.size == 1:
            return data.item()
        return data
    return data


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    makedir(log_dir)
    summary_writer_method = SummaryWriter
    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


def make_wandb_logger(project: str, name: Optional[str] = None, config: Optional[Dict] = None, **kwargs: Any):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    return WandbLogger(
        project=project,
        name=name,
        config=config,
        **kwargs
    )


def make_wandb_logger_multi_axis(project: str, name: Optional[str] = None, config: Optional[Dict] = None, **kwargs: Any):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    return WandbLoggerMultiAxis(
        project=project,
        name=name,
        config=config,
        **kwargs
    )

class TensorBoardWriterWrapper:
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        data = _sanitize_scalar(data)
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        hparams = {k: _sanitize_scalar(v) for k, v in hparams.items()}
        meters = {k: _sanitize_scalar(v) for k, v in meters.items()}
        self._writer.add_hparams(hparams, meters)


# class Logger:
#     """
#     A logger class that can interface with multiple loggers. It now supports tensorboard only for simplicity, but you can extend it with your own logger.
#     """

#     def __init__(self, logging_conf):
#         # allow turning off TensorBoard with "should_log: false" in config
#         tb_config = logging_conf.tensorboard_writer
#         tb_should_log = tb_config and tb_config.pop("should_log", True)
#         self.tb_logger = instantiate(tb_config) if tb_should_log else None

#     def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
#         if self.tb_logger:
#             self.tb_logger.log_dict(payload, step)

#     def log(self, name: str, data: Scalar, step: int) -> None:
#         if self.tb_logger:
#             self.tb_logger.log(name, data, step)

#     def log_hparams(
#         self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
#     ) -> None:
#         if self.tb_logger:
#             self.tb_logger.log_hparams(hparams, meters)


class WandbLogger:
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        group: Optional[str] = None,
        config: Optional[Dict] = None,
        dir: str = "./wandb_logs",
        entity: Optional[str] = None,
        resume: str = "allow",
        **kwargs: Any,
    ) -> None:
        self._enabled = False
        _, self._rank = get_machine_local_and_dist_rank()
        
        if self._rank == 0:
            makedir(dir)
            if not name:
                name = _make_default_wandb_run_name(project, dir)
            wandb.init(
                project=project,
                name=name,
                group=group,
                config=config,
                dir=dir,
                entity=entity,
                resume=resume,
                **kwargs
            )
            self._enabled = True
            logging.info(f"Wandb initialized: {wandb.run.get_url() if wandb.run else 'Enabled'}")
        else:
            logging.debug(f"Wandb disabled on rank {self._rank}")

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._enabled:
            return
        payload = {k: _sanitize_scalar(v) for k, v in payload.items()}
        wandb.log(payload, step=step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._enabled:
            return
        data = _sanitize_scalar(data)
        wandb.log({name: data}, step=step)

    def log_image(self, name: str, image: Any, step: int, caption: str = "") -> None:
        if not self._enabled:
            return
        wandb.log({name: wandb.Image(image, caption=caption)}, step=step)

    def log_summary(self, metrics: Dict[str, Scalar]) -> None:
        if not self._enabled:
            return
        for k, v in metrics.items():
            wandb.run.summary[k] = _sanitize_scalar(v)

    def save_file(self, file_path: str, policy: str = "now") -> None:
        if not self._enabled:
            return
        wandb.save(file_path, policy=policy)

    def finish(self) -> None:
        if self._enabled:
            wandb.finish()


class WandbLoggerMultiAxis:
    """
    WandB logger that keeps separate semantic x-axes:
    - train step metrics -> `train_step`
    - validation batch/debug metrics -> `val_step`
    - epoch snapshots -> `epoch` (duplicated under `Epoch_Stats/*`)

    This avoids mixing `train_step`, `val_step`, and `epoch` into a single
    monotonically increasing wandb `step`, while keeping the original trainer
    code unchanged.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        group: Optional[str] = None,
        config: Optional[Dict] = None,
        dir: str = "./wandb_logs",
        entity: Optional[str] = None,
        resume: str = "allow",
        **kwargs: Any,
    ) -> None:
        self._enabled = False
        self._last_train_step = -1
        self._last_val_step = -1
        self._last_epoch = -1
        _, self._rank = get_machine_local_and_dist_rank()

        if self._rank == 0:
            makedir(dir)
            if not name:
                name = _make_default_wandb_run_name(project, dir)
            wandb.init(
                project=project,
                name=name,
                group=group,
                config=config,
                dir=dir,
                entity=entity,
                resume=resume,
                **kwargs,
            )
            self._enabled = True
            self._define_metrics()
            logging.info(
                f"Wandb multi-axis initialized: {wandb.run.get_url() if wandb.run else 'Enabled'}"
            )
        else:
            logging.debug(f"Wandb disabled on rank {self._rank}")

    def _define_metrics(self) -> None:
        if not self._enabled:
            return

        wandb.define_metric("train_step")
        wandb.define_metric("val_step")
        wandb.define_metric("epoch")

        train_step_patterns = [
            "Step_Stats/train/*",
            "Step_Losses/Losses/train_*",
            "Losses/train_*",
            "Losses/val_*",
            "Meters*",
            "Optim/*",
            "Trainer/where",
            "Trainer/steps_train",
            "Trainer/steps_val",
        ]
        for pattern in train_step_patterns:
            wandb.define_metric(pattern, step_metric="train_step")

        val_step_patterns = [
            "Step_Stats/val/*",
            "Step_Losses/Losses/val_*",
        ]
        for pattern in val_step_patterns:
            wandb.define_metric(pattern, step_metric="val_step")

        wandb.define_metric("Epoch_Stats/*", step_metric="epoch")

    def _update_train_step(self, step: int) -> int:
        step = int(step)
        self._last_train_step = max(self._last_train_step, step)
        return self._last_train_step

    def _update_val_step(self, step: int) -> int:
        step = int(step)
        self._last_val_step = max(self._last_val_step, step)
        return self._last_val_step

    def _update_epoch(self, epoch: Any) -> Optional[int]:
        if epoch is None:
            return None
        epoch = int(epoch)
        self._last_epoch = max(self._last_epoch, epoch)
        return self._last_epoch

    def _infer_axis_from_name(self, name: str) -> Optional[str]:
        if (
            name.startswith("Step_Stats/train/")
            or name.startswith("Step_Losses/Losses/train_")
            or name.startswith("Losses/train_")
            or name.startswith("Optim/")
            or name.startswith("Trainer/")
        ):
            return "train_step"
        if (
            name.startswith("Step_Stats/val/")
            or name.startswith("Step_Losses/Losses/val_")
        ):
            return "val_step"
        if name.startswith("Epoch_Stats/"):
            return "epoch"
        return None

    def _log_payload(self, payload: Dict[str, Scalar]) -> None:
        if not self._enabled:
            return
        wandb.log(payload)

    def _make_epoch_payload(self, payload: Dict[str, Scalar], epoch: int) -> Dict[str, Scalar]:
        epoch_payload = {f"Epoch_Stats/{k}": v for k, v in payload.items()}
        epoch_payload["epoch"] = epoch
        return epoch_payload

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if not self._enabled:
            return

        payload = {k: _sanitize_scalar(v) for k, v in payload.items()}
        epoch = self._update_epoch(payload.get("epoch", payload.get("Trainer/epoch")))

        if "Trainer/steps_train" in payload:
            train_step = self._update_train_step(payload["Trainer/steps_train"])
            train_payload = dict(payload)
            train_payload["train_step"] = train_step
            self._log_payload(train_payload)
            if epoch is not None:
                self._log_payload(self._make_epoch_payload(payload, epoch))
            return

        if "Trainer/steps_val" in payload:
            val_step = self._update_val_step(payload["Trainer/steps_val"])
            train_anchor = self._last_train_step if self._last_train_step >= 0 else val_step
            val_payload = dict(payload)
            val_payload["train_step"] = train_anchor
            val_payload["val_step"] = val_step
            self._log_payload(val_payload)
            if epoch is not None:
                self._log_payload(self._make_epoch_payload(payload, epoch))
            return

        axis = self._infer_axis_from_name(next(iter(payload.keys()))) if payload else None
        generic_payload = dict(payload)
        if axis == "val_step":
            generic_payload["val_step"] = self._update_val_step(step)
        elif axis == "epoch":
            generic_payload["epoch"] = self._update_epoch(step)
        else:
            generic_payload["train_step"] = self._update_train_step(step)
        self._log_payload(generic_payload)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if not self._enabled:
            return

        data = _sanitize_scalar(data)
        payload = {name: data}
        axis = self._infer_axis_from_name(name)

        if axis == "val_step":
            payload["val_step"] = self._update_val_step(step)
        elif axis == "epoch":
            payload["epoch"] = self._update_epoch(step)
        else:
            payload["train_step"] = self._update_train_step(step)

        self._log_payload(payload)

    def log_image(self, name: str, image: Any, step: int, caption: str = "") -> None:
        if not self._enabled:
            return
        payload = {name: wandb.Image(image, caption=caption)}
        axis = self._infer_axis_from_name(name)
        if axis == "val_step":
            payload["val_step"] = self._update_val_step(step)
        else:
            payload["train_step"] = self._update_train_step(step)
        self._log_payload(payload)

    def log_summary(self, metrics: Dict[str, Scalar]) -> None:
        if not self._enabled:
            return
        for k, v in metrics.items():
            wandb.run.summary[k] = _sanitize_scalar(v)

    def save_file(self, file_path: str, policy: str = "now") -> None:
        if not self._enabled:
            return
        wandb.save(file_path, policy=policy)

    def finish(self) -> None:
        if self._enabled:
            wandb.finish()


class Logger:
    def __init__(self, logging_conf):
        # --- TensorBoard ---
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

        # --- Wandb ---
        wb_config = logging_conf.wandb_writer
        wb_should_log = wb_config and wb_config.pop("should_log", True)
        self.wb_logger = instantiate(wb_config) if wb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log_dict(payload, step)
        if self.wb_logger:
            self.wb_logger.log_dict(payload, step)
    
    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger:
            self.tb_logger.log(name, data, step)
        if self.wb_logger:
            self.wb_logger.log(name, data, step)

    def log_image(self, name: str, image: Any, step: int, caption: str = "") -> None:
        if self.wb_logger:
            self.wb_logger.log_image(name, image, step, caption)
    
    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        if self.tb_logger:
            self.tb_logger.log_hparams(hparams, meters)

    def finish(self):
        if self.wb_logger:
            self.wb_logger.finish()

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # we tune the buffering value so that the logs are updated
    # frequently.
    log_buffer_kb = 10 * 1024  # 10KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io


def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # Cleanup any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)

    # we log to file as well if user wants
    if log_filename and rank == 0:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


def shutdown_logging():
    """
    After training is done, we ensure to shut down all the logger streams.
    """
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()
