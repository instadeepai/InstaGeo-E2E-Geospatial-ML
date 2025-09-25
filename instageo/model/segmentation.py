"""Segmentation Module for Prithvi Models."""

from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from instageo.model.base import PrithviBaseModule, PrithviDistillationBaseModule
from instageo.model.metrics import RunningAUC, RunningConfusionMatrix


class PrithviSegmentationModule(PrithviBaseModule):
    """Prithvi Segmentation PyTorch Lightning Module."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        load_pretrained_weights: bool = True,
        num_classes: int = 2,
        temporal_step: int = 1,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        model_name: str = "prithvi_eo_v1_100",
        weight_clip_range: Optional[List[float]] = None,
        depth: int = -1,
    ) -> None:
        """Initialize the segmentation module.

        Args:
            image_size (int): Size of input image.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            load_pretrained_weights (bool): Flag to load pretrained weights.
            num_classes (int): Number of classes for segmentation.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            class_weights (List[float]): Class weights for mitigating class imbalance.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
            scheduler (bool): Flag to use a learning rate scheduler.
            model_name (str): Model architecture chosen.
            weight_clip_range (Optional[List[float]]): Range for weight clipping
                [min_value, max_value]. If None, no clipping is applied.
            depth (int): Depth of the model.
        """
        self._num_classes = num_classes
        super().__init__(
            image_size=image_size,
            learning_rate=learning_rate,
            freeze_backbone=freeze_backbone,
            load_pretrained_weights=load_pretrained_weights,
            temporal_step=temporal_step,
            weight_decay=weight_decay,
            scheduler=scheduler,
            model_name=model_name,
            weight_clip_range=weight_clip_range,
            depth=depth,
            num_classes=num_classes,
        )
        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight_tensor, reduction="none"
        )
        self.ignore_index = ignore_index

        # Initialize streaming metrics
        self.train_metrics = RunningConfusionMatrix(num_classes, ignore_index)
        self.val_metrics = RunningConfusionMatrix(num_classes, ignore_index)
        self.test_metrics = RunningConfusionMatrix(num_classes, ignore_index)
        self.train_auc = RunningAUC(num_classes)
        self.val_auc = RunningAUC(num_classes)
        self.test_auc = RunningAUC(num_classes)

    @property
    def num_classes(self) -> int:
        """Get number of classes.

        Returns:
            int: Number of classes.
        """
        return self._num_classes

    def _shared_step(self, batch: Any, step_type: str) -> torch.Tensor:
        """Shared logic for training, validation and test steps.

        Args:
            batch (Any): Input batch data.
            step_type (str): Type of step ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        labels = labels.long()
        mask = labels.ne(self.ignore_index)
        loss = self.criterion(outputs, labels)
        loss = loss[mask].mean()

        # Get predictions and probabilities
        preds = torch.argmax(outputs, dim=1)
        probs = torch.nn.functional.softmax(outputs.detach(), dim=1)

        # Get valid indices
        valid_indices = mask.reshape(-1).nonzero().squeeze()

        if valid_indices.numel() > 0:
            # Flatten and select valid pixels
            preds = preds.reshape(-1)[valid_indices]
            labels = labels.reshape(-1)[valid_indices]
            probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))[valid_indices]

            # Ensure consistent dimensionality when valid_indices is a single element
            if valid_indices.numel() == 1:
                preds = preds.unsqueeze(0)
                labels = labels.unsqueeze(0)
                probs = probs.unsqueeze(0)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.astype(np.int64)
        preds = preds.astype(np.int64)
        probs = probs.cpu().numpy()

        # Update metrics
        metrics = getattr(self, f"{step_type}_metrics")
        metrics.update(labels, preds)

        # Only compute AUC metrics during testing
        if step_type == "test":
            auc_metrics = getattr(self, f"{step_type}_auc")
            auc_metrics.update(labels, probs)

        # Log the loss
        self.log(
            f"{step_type}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def _shared_epoch_end(self, step_type: str) -> None:
        """Shared logic for training, validation and test epoch ends.

        Args:
            step_type (str): Type of step ('train', 'val', or 'test').
        """
        metrics = getattr(self, f"{step_type}_metrics").compute()

        # Log overall metrics
        self.log(f"{step_type}_Acc", metrics["accuracy"], logger=True)
        self.log(f"{step_type}_IoU", metrics["jaccard"], logger=True)
        self.log(f"{step_type}_F1", metrics["f1"], logger=True)
        self.log(f"{step_type}_Precision", metrics["precision"], logger=True)
        self.log(f"{step_type}_Recall", metrics["recall"], logger=True)

        # Only compute and log ROC-AUC metrics during testing

        if step_type == "test":
            auc_metrics = getattr(self, f"{step_type}_auc").score()
            self.log(f"{step_type}_roc_auc", auc_metrics["roc_auc_macro"], logger=True)

        # Log per-class metrics
        for idx, value in enumerate(metrics["jaccard_per_class"]):
            self.log(f"{step_type}_IoU_{idx}", value, logger=True)
        for idx, value in enumerate(metrics["f1_per_class"]):
            self.log(f"{step_type}_F1_{idx}", value, logger=True)

        # Reset metrics for next epoch
        getattr(self, f"{step_type}_metrics").reset()
        if step_type == "test":
            getattr(self, f"{step_type}_auc").reset()

    def predict_step(self, batch: Any) -> torch.Tensor:
        """Perform a prediction step.

        Args:
            batch (Any): Input batch data.

        Returns:
            torch.Tensor: The model predictions.
        """
        prediction = self.forward(batch)
        probabilities = torch.nn.functional.softmax(prediction, dim=1)[:, 1, :, :]
        return probabilities


class PrithviDistillationSegmentationModule(
    PrithviDistillationBaseModule, PrithviSegmentationModule
):
    """Prithvi Knowledge Distillation PyTorch Lightning Module."""

    def __init__(
        self,
        teacher_ckpt_path: str,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        num_classes: int = 2,
        temporal_step: int = 1,
        class_weights: List[float] = [1, 2],
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        model_name: str = "prithvi_eo_v1_100",
        depth: int = 12,
        load_pretrained_weights: bool = True,
        scheduler: bool = True,
        weight_clip_range: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DistillationModule.

        Args:
            teacher_ckpt_path (str): Path to the teacher model checkpoint.
            image_size (int): Image size.
            learning_rate (float): Learning rate.
            num_classes (int): Number of classes.
            temporal_step (int): Temporal step.
            class_weights (List[float]): Class weights.
            ignore_index (int): Ignore index.
            weight_decay (float): Weight decay.
            model_name (str): Model name.
            depth (int): Depth.
            load_pretrained_weights (bool): Load pretrained weights.
            scheduler (bool): Scheduler.
            weight_clip_range (Optional[List[float]]): Weight clip range.
        """
        self._num_classes = num_classes
        super().__init__(
            teacher_ckpt_path=teacher_ckpt_path,
            image_size=image_size,
            learning_rate=learning_rate,
            load_pretrained_weights=load_pretrained_weights,
            temporal_step=temporal_step,
            ignore_index=ignore_index,
            weight_decay=weight_decay,
            scheduler=scheduler,
            depth=depth,
            model_name=model_name,
            weight_clip_range=weight_clip_range,
            distillation_loss=nn.KLDivLoss(reduction="batchmean"),
            num_classes=num_classes,
            class_weights=class_weights,
        )

        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight_tensor, reduction="none"
        )
        self.ignore_index = ignore_index

        # Initialize streaming metrics
        self.train_metrics = RunningConfusionMatrix(num_classes, ignore_index)
        self.val_metrics = RunningConfusionMatrix(num_classes, ignore_index)
        self.test_metrics = RunningConfusionMatrix(num_classes, ignore_index)
        self.train_auc = RunningAUC(num_classes)
        self.val_auc = RunningAUC(num_classes)
        self.test_auc = RunningAUC(num_classes)

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
    ) -> PrithviSegmentationModule:
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
            PrithviSegmentationModule: Initialized teacher model.
        """
        model = PrithviSegmentationModule(
            image_size=image_size,
            learning_rate=learning_rate,
            temporal_step=temporal_step,
            ignore_index=ignore_index,
            weight_decay=weight_decay,
            model_name=model_name,
            scheduler=scheduler,
            num_classes=num_classes,
            class_weights=class_weights,
            load_pretrained_weights=False,
        )

        # load teacher checkpoint
        state_dict = torch.load(teacher_ckpt_path, map_location=torch.device("cpu"))

        # modify to new checkpoint format to make sure all teacher weights follow
        # the same naming convention
        state_dict["state_dict"] = OrderedDict(
            (k.replace("prithvi_100M_backbone", "prithvi_encoder"), v)
            for k, v in state_dict["state_dict"].items()
        )
        model.load_state_dict(state_dict["state_dict"])
        return model

    @property
    def num_classes(self) -> int:
        """Get number of classes.

        Returns:
            int: Number of classes.
        """
        return self._num_classes

    def _compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined distillation and standard loss.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Compute cross entropy loss
        ce_loss = self.criterion(student_logits, labels.long())

        valid_pixels = labels.ne(self.ignore_index).reshape(-1)
        student_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, self._num_classes)[
            valid_pixels
        ]
        teacher_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, self._num_classes)[
            valid_pixels
        ]
        labels = labels.reshape(-1)[valid_pixels]
        ce_loss = ce_loss.reshape(-1)[valid_pixels].mean()
        # Compute distillation loss
        distill_loss = self.distillation_loss(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
        )
        total_loss = ce_loss + distill_loss
        metrics = {
            "loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "distill_loss": distill_loss.item(),
        }

        return total_loss, metrics

    def _shared_step(self, batch: Any, step_type: str) -> torch.Tensor:
        """Perform a training or validation step.

        Args:
            batch (Any): Input batch data.
            step_type (str): Type of step to perform.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        with torch.no_grad():
            teacher_logits = self.teacher.net(inputs)

        student_logits = self.net(inputs)
        loss, loss_metrics = self._compute_loss(student_logits, teacher_logits, labels)

        # Get predictions and probabilities
        preds = torch.argmax(student_logits, dim=1)
        probs = torch.nn.functional.softmax(student_logits.detach(), dim=1)

        # Get valid indices
        no_ignore = labels.ne(self.ignore_index)
        valid_indices = no_ignore.reshape(-1).nonzero().squeeze()

        if valid_indices.numel() > 0:
            # Flatten and select valid pixels
            preds = preds.reshape(-1)[valid_indices]
            labels = labels.reshape(-1)[valid_indices]
            probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))[valid_indices]

            # Ensure consistent dimensionality when valid_indices is a single element
            if valid_indices.numel() == 1:
                preds = preds.unsqueeze(0)
                labels = labels.unsqueeze(0)
                probs = probs.unsqueeze(0)

            # Update metrics - ensure integer type
            metrics = getattr(self, f"{step_type}_metrics")
            metrics.update(
                labels.cpu().numpy().astype(np.int64),
                preds.cpu().numpy().astype(np.int64),
            )

            # Only compute AUC metrics during testing
            if step_type == "test":
                auc_metrics = getattr(self, f"{step_type}_auc")
                auc_metrics.update(labels.cpu().numpy().astype(np.int64), probs.cpu().numpy())

        for loss_name, loss_value in loss_metrics.items():
            self.log(
                f"{step_type}_{loss_name}",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss
