"""Segmentation Module for Prithvi Models."""

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

from instageo.model.base import PrithviBaseModule
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
        class_weights: List[float] = [1, 2],
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        model_name: str = "pritvhi-eo1",
        weight_clip_range: Optional[List[float]] = None,
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

        # Update metrics
        metrics = getattr(self, f"{step_type}_metrics")
        metrics.update(labels, preds)
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
        auc_metrics = getattr(self, f"{step_type}_auc").score()

        # Log overall metrics
        self.log(f"{step_type}_Acc", metrics["accuracy"], logger=True)
        self.log(f"{step_type}_IoU", metrics["jaccard"], logger=True)
        self.log(f"{step_type}_F1", metrics["f1"], logger=True)
        self.log(f"{step_type}_Precision", metrics["precision"], logger=True)
        self.log(f"{step_type}_Recall", metrics["recall"], logger=True)
        self.log(f"{step_type}_roc_auc", auc_metrics["roc_auc_macro"], logger=True)

        # Log per-class metrics
        for idx, value in enumerate(metrics["jaccard_per_class"]):
            self.log(f"{step_type}_IoU_{idx}", value, logger=True)
        for idx, value in enumerate(metrics["f1_per_class"]):
            self.log(f"{step_type}_F1_{idx}", value, logger=True)

        # Reset metrics for next epoch
        getattr(self, f"{step_type}_metrics").reset()
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
