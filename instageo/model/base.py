"""Base Module for Prithvi Models."""

import logging
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch

from instageo.model.model import PrithviSeg

log = logging.getLogger(__name__)


class PrithviBaseModule(pl.LightningModule):
    """Base PyTorch Lightning Module for Prithvi Models."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        load_pretrained_weights: bool = True,
        temporal_step: int = 1,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        model_name: str = "pritvhi-eo1",
        weight_clip_range: Optional[List[float]] = None,
    ) -> None:
        """Initialize the base module.

        Args:
            image_size (int): Size of input image.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            load_pretrained_weights (bool): Flag to load pretrained weights.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            weight_decay (float): Weight decay for L2 regularization.
            scheduler (bool): Flag to use a learning rate scheduler.
            model_name (str): Model architecture chosen.
            weight_clip_range (Optional[List[float]]): Range for weight clipping
                [min_value, max_value]. If None, no clipping is applied.
        """
        super().__init__()
        self.net = PrithviSeg(
            image_size=image_size,
            num_classes=self.num_classes,
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
            variant=model_name,
            load_pretrained_weights=load_pretrained_weights,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.weight_clip_range = weight_clip_range

    @property
    def num_classes(self) -> int:
        """Get number of classes.

        Returns:
            int: Number of classes.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.net(x)

    def clip_weights(self) -> None:
        """Clip model weights to prevent them from growing too large.

        Only clips weights if weight_clip_range is not None.
        The range should be a list of [min_value, max_value].
        """
        if self.weight_clip_range is not None:
            min_val, max_val = self.weight_clip_range
            with torch.no_grad():
                for param in self.parameters():
                    param.data.clamp_(min_val, max_val)

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        """Configure the model's optimizers and learning rate schedulers.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler]]:
            A tuple containing the list of optimizers and the list of LR schedulers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=0
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer], []

    def _shared_step(self, batch: Any, step_type: str) -> torch.Tensor:
        """Shared logic for training, validation and test steps.

        Args:
            batch (Any): Input batch data.
            step_type (str): Type of step ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        loss = self._shared_step(batch, "train")

        # Log learning rate
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Clip weights after each training step
        self.clip_weights()

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        with torch.no_grad():
            return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        with torch.no_grad():
            return self._shared_step(batch, "test")

    def _shared_epoch_end(self, step_type: str) -> None:
        """Shared logic for training, validation and test epoch ends.

        Args:
            step_type (str): Type of step ('train', 'val', or 'test').
        """
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        """Compute and log metrics at the end of training epoch."""
        self._shared_epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        """Compute and log metrics at the end of validation epoch."""
        self._shared_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """Compute and log metrics at the end of test epoch."""
        self._shared_epoch_end("test")

    def predict_step(self, batch: Any) -> torch.Tensor:
        """Perform a prediction step.

        Args:
            batch (Any): Input batch data.

        Returns:
            torch.Tensor: The model predictions.
        """
        raise NotImplementedError
