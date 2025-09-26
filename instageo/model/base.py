"""Base Module for Prithvi Models."""

import logging
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from instageo.model.model import PrithviSeg

log = logging.getLogger(__name__)


class PrithviBaseModule(pl.LightningModule):
    """Base PyTorch Lightning Module for Prithvi Models."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        weight_clip_range: Optional[List[float]] = None,
        image_size: int = 224,
        temporal_step: int = 1,
        freeze_backbone: bool = True,
        model_name: str = "prithvi_eo_v1_100",
        load_pretrained_weights: bool = True,
        depth: int = -1,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the base module.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for L2 regularization.
            scheduler (bool): Flag to use a learning rate scheduler.
            weight_clip_range (Optional[List[float]]): Range for weight clipping
                [min_value, max_value]. If None, no clipping is applied.
            image_size (int): Size of input image.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            model_name (str): Model architecture chosen.
            load_pretrained_weights (bool): Flag to load pretrained weights.
            depth (int): Depth of the model.
            num_classes (int): Number of classes.
            **kwargs: Additional keyword arguments to pass to PrithviSeg.
        """
        super().__init__()
        self.net = PrithviSeg(
            image_size=image_size,
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
            variant=model_name,
            load_pretrained_weights=load_pretrained_weights,
            depth=depth,
            num_classes=num_classes,
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
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
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


class PrithviDistillationBaseModule(PrithviBaseModule):
    """Base Module for Prithvi Distillation Models."""

    def __init__(
        self,
        teacher_ckpt_path: str,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        load_pretrained_weights: bool = True,
        freeze_backbone: bool = False,
        temporal_step: int = 1,
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        depth: int = 12,
        model_name: str = "prithvi_eo_v1_100",
        weight_clip_range: Optional[List[float]] = None,
        distillation_loss: nn.Module = nn.MSELoss(),
        num_classes: int = 2,
        class_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the base distillation module.

        Args:
            teacher_ckpt_path (str): Path to the teacher checkpoint.
            image_size (int): Size of input image.
            learning_rate (float): Learning rate for the optimizer.
            load_pretrained_weights (bool): Flag to load pretrained weights.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
            scheduler (bool): Flag to use a learning rate scheduler.
            model_name (str): Model architecture chosen.
            depth (int): Depth of the teacher model.
            weight_clip_range (Optional[List[float]]): Range for weight clipping
                [min_value, max_value]. If None, no clipping is applied.
            distillation_loss (nn.Module): Loss function for distillation.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            num_classes (int): Number of classes.
            class_weights (List[float]): Class weights.
            **kwargs: Additional keyword arguments to pass to PrithviSeg.
        """
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler=scheduler,
            weight_clip_range=weight_clip_range,
            image_size=image_size,
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
            load_pretrained_weights=False,
            depth=depth,
            num_classes=num_classes,
            model_name=model_name,
            **kwargs,
        )
        self.ignore_index = ignore_index
        self.distillation_loss = distillation_loss
        self.strict_loading = False

        # Initialize teacher model
        self.teacher = self._init_teacher(
            teacher_ckpt_path=teacher_ckpt_path,
            image_size=image_size,
            learning_rate=learning_rate,
            temporal_step=temporal_step,
            ignore_index=ignore_index,
            weight_decay=weight_decay,
            model_name=model_name,
            scheduler=scheduler,
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.teacher.eval()
        self.teacher.freeze()
        self.num_parameters = sum(p.numel() for p in self.parameters())

        # Load teacher's weights into student
        teacher_state_dict = self.teacher.net.prithvi_encoder.state_dict()
        student_state_dict = self.net.prithvi_encoder.state_dict()
        if load_pretrained_weights:
            # Filter out incompatible keys
            shared_state_dict = {
                k: v
                for k, v in teacher_state_dict.items()
                if k in student_state_dict and v.shape == student_state_dict[k].shape
            }

            # Load compatible weights
            self.net.prithvi_encoder.load_state_dict(shared_state_dict, strict=False)

        # Initialize segmentation head with random weights
        for m in self.net.segmentation_head.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def num_classes(self) -> int:
        """Get number of classes. To be implemented by subclasses.

        Returns:
            int: Number of classes.
        """
        raise NotImplementedError("Subclasses must implement num_classes property")

    def _init_teacher(
        self,
        teacher_ckpt_path: str,
        image_size: int,
        learning_rate: float,
        temporal_step: int,
        ignore_index: int,
        weight_decay: float,
        model_name: str,
        scheduler: bool,
        num_classes: int,
        class_weights: Optional[List[float]] = None,
    ) -> Any:
        """Initialize the teacher model.

        Args:
            teacher_ckpt_path (str): Path to the teacher checkpoint.
            image_size (int): Size of input image.
            learning_rate (float): Learning rate for the optimizer.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
            model_name (str): Model architecture chosen.
            scheduler (bool): Flag to use a learning rate scheduler.
            num_classes (int): Number of classes.
            class_weights (List[float]): Class weights.

        Returns:
            Any: Initialized teacher model.
        """
        raise NotImplementedError("Subclasses must implement _init_teacher")

    def state_dict(self) -> dict:
        """Customize the state dict of the model.

        Remove the teacher weights from the state dict.

        Returns:
            dict: State dict of the model without the teacher weights.
        """
        # Don't save the teacher
        return {k: v for k, v in super().state_dict().items() if "teacher" not in k}
