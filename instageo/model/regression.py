"""Regression Module for Prithvi Models."""

from collections import OrderedDict
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from instageo.model.base import PrithviBaseModule, PrithviDistillationBaseModule
from instageo.model.metrics import RunningRegressionMetrics


class LogScaler:
    """LogScaler class for scaling and unscaling the data."""

    def __init__(self) -> None:
        """Log scaler class for scaling and unscaling the data."""
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform x to the log scale with the natural logarithm of (1 + x).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return torch.log1p(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform x from the log scale with the natural logarithm of (1 + x).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Inverse transformed tensor.
        """
        return torch.expm1(x)


class PrithviRegressionModule(PrithviBaseModule):
    """Prithvi Regression PyTorch Lightning Module."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        load_pretrained_weights: bool = True,
        temporal_step: int = 1,
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        model_name: str = "prithvi_eo_v1_100",
        use_log_scale: bool = False,
        plot_reg_results: bool = False,
        include_ee: bool = False,
        weight_clip_range: Optional[List[float]] = None,
        depth: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize the regression module.

        Args:
            image_size (int): Size of input image.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            load_pretrained_weights (bool): Flag to load pretrained weights.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
            scheduler (bool): Flag to use a learning rate scheduler.
            model_name (str): Model architecture chosen.
            use_log_scale (bool): Determines whether to use log scale for training the model.
            plot_reg_results (bool): Determines whether to plot the regression results.
            include_ee (bool): Determines whether to include expected error metrics.
            weight_clip_range (Optional[List[float]]): Range for weight clipping
                [min_value, max_value]. If None, no clipping is applied.
            depth (int): Depth of the model.
        """
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
            num_classes=1,
        )
        self.criterion = nn.MSELoss(reduction="none")
        self.ignore_index = ignore_index
        self.use_log_scale = use_log_scale
        self.plot_reg_results = plot_reg_results
        self.include_ee = include_ee
        self.log_scaler = LogScaler()

        # Initialize streaming metrics
        self.train_metrics = RunningRegressionMetrics(include_ee=include_ee)
        self.val_metrics = RunningRegressionMetrics(include_ee=include_ee)
        self.test_metrics = RunningRegressionMetrics(include_ee=include_ee)

        self.plot_outputs: List[np.ndarray] = []
        self.plot_labels: List[np.ndarray] = []

    @property
    def num_classes(self) -> int:
        """Get number of classes.

        Returns:
            int: Number of classes (1 for regression).
        """
        return 1

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
        outputs = outputs.squeeze(1)
        mask = labels.ne(self.ignore_index)
        if self.use_log_scale:
            labels = self.log_scaler.transform(labels)
        outputs = outputs[mask]
        labels = labels[mask]
        loss = self.criterion(outputs, labels)
        loss = loss.mean()

        # Get predictions
        preds = outputs.detach()
        if self.use_log_scale:
            preds = self.log_scaler.inverse_transform(preds)
            labels = self.log_scaler.inverse_transform(labels)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        # Update metrics
        metrics = getattr(self, f"{step_type}_metrics")
        metrics.update(labels, preds)

        # Add regression results to plot
        if step_type in ["val", "test"]:
            if self.plot_reg_results:
                self.plot_outputs.append(preds)
                self.plot_labels.append(labels)

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

        self.log(f"{step_type}_RMSE", metrics["rmse"], logger=True)
        self.log(f"{step_type}_MAE", metrics["mae"], logger=True)
        self.log(f"{step_type}_R2", metrics["r2_score"], logger=True)
        self.log(f"{step_type}_Pearson", metrics["pearson_corrcoef"], logger=True)
        if metrics["ee_percentage"] is not None:
            self.log(f"{step_type}_EE_Percentage", metrics["ee_percentage"], logger=True)

        # Plot regression results for validation and test stages
        if step_type in ["val", "test"]:
            if self.plot_reg_results:
                all_preds = np.concatenate(self.plot_outputs)
                all_labels = np.concatenate(self.plot_labels)
                if step_type == "val":
                    # We only plot for validation stage if it is the first epoch or if
                    # we have a new best model
                    best_rmse = self.trainer.checkpoint_callback.best_model_score
                    if best_rmse is None or metrics["rmse"] < best_rmse:
                        self.create_regression_plot(all_preds, all_labels, metrics, step_type)
                else:
                    self.create_regression_plot(all_preds, all_labels, metrics, step_type)

                # Clear the lists
                self.plot_outputs = []
                self.plot_labels = []

        # Reset metrics for next epoch
        getattr(self, f"{step_type}_metrics").reset()

    def create_regression_plot(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metrics: dict[str, torch.Tensor],
        step_type: str = "test",
    ) -> None:
        """Create a scatter plot comparing predictions vs targets for regression tasks.

        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth values
            metrics (dict): Dictionary containing computed metrics
            step_type (str): Stage of the plot ('val' or 'test')
        """
        # Convert tensors to numpy arrays and flatten
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Filter out NaN and infinite values
        mask = ~(
            np.isnan(predictions) | np.isnan(targets) | np.isinf(predictions) | np.isinf(targets)
        )
        predictions = predictions[mask]
        targets = targets[mask]
        if predictions.shape[0] == 0:
            return

        # Create figure with seaborn jointplot
        plt.figure(figsize=(10, 10))
        g = sns.jointplot(
            x=targets,
            y=predictions,
            kind="scatter",
            height=10,
            ratio=5,
            space=0.2,
            marginal_kws=dict(bins=50),
            marginal_ticks=True,
            alpha=0.1,  # Add transparency to scatter points
        )

        # Add 1:1 reference line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        g.ax_joint.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 reference line")

        # Add expected error lines
        if self.include_ee:
            ee_bias = metrics["ee_bias"]
            ee_coef = metrics["ee_coef"]
            x = np.linspace(min_val, max_val, 100)
            upper_ee = x + (ee_bias + ee_coef * x)
            lower_ee = x - (ee_bias + ee_coef * x)
            g.ax_joint.plot(x, upper_ee, "g--")
            g.ax_joint.plot(x, lower_ee, "g--")
            g.ax_joint.fill_between(
                x,
                upper_ee,
                lower_ee,
                color="g",
                alpha=0.05,
                label="Expected error window",
            )

        # Add labels and title
        g.ax_joint.set_xlabel("Ground truth values")
        g.ax_joint.set_ylabel("Predicted values")
        g.ax_joint.set_title(f"Model Predictions vs Ground Truth ({step_type.capitalize()})")

        # Add metrics to plot
        metrics_text = (
            f'Pearson Correlation: {metrics["pearson_corrcoef"]:.3f}\n'
            f'RMSE: {metrics["rmse"]:.3f}\n'
            f'MAE: {metrics["mae"]:.3f}\n'
            f'R2 Score: {metrics["r2_score"]:.3f}\n'
            f'Within EE: {metrics["ee_percentage"]:.1f}%'
            if self.include_ee
            else ""
        )
        g.ax_joint.text(
            0.02,
            0.95,
            metrics_text,
            transform=g.ax_joint.transAxes,
            bbox=dict(facecolor="lightblue", alpha=0.25),
        )

        # Add legend
        g.ax_joint.legend()

        # Save plot to file
        plot_path = f"regression_plot_{step_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Log plot to Neptune if available
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.experiment[f"regression_plot_{step_type}"].upload(plot_path)

    def predict_step(self, batch: Any) -> torch.Tensor:
        """Perform a prediction step.

        Args:
            batch (Any): Input batch data.

        Returns:
            torch.Tensor: The model predictions.
        """
        prediction = self.forward(batch)
        prediction = prediction.squeeze(1)
        if self.use_log_scale:
            prediction = self.log_scaler.inverse_transform(prediction)
        return prediction


class PrithviDistillationRegressionModule(PrithviDistillationBaseModule, PrithviRegressionModule):
    """Prithvi Distillation Regression PyTorch Lightning Module."""

    def __init__(
        self,
        teacher_ckpt_path: str,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        load_pretrained_weights: bool = True,
        temporal_step: int = 1,
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
        scheduler: bool = True,
        depth: int = 12,
        model_name: str = "prithvi_eo_v1_100",
        use_log_scale: bool = False,
        plot_reg_results: bool = False,
        include_ee: bool = False,
        weight_clip_range: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the regression module.

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
            use_log_scale (bool): Determines whether to use log scale for training the model.
            plot_reg_results (bool): Determines whether to plot the regression results.
            include_ee (bool): Determines whether to include expected error metrics.
            depth (int): Depth of the teacher model.
            weight_clip_range (Optional[List[float]]): Range for weight clipping
                [min_value, max_value]. If None, no clipping is applied.
        """
        super().__init__(
            teacher_ckpt_path=teacher_ckpt_path,
            image_size=image_size,
            learning_rate=learning_rate,
            freeze_backbone=False,
            load_pretrained_weights=load_pretrained_weights,
            temporal_step=temporal_step,
            ignore_index=ignore_index,
            weight_decay=weight_decay,
            scheduler=scheduler,
            depth=depth,
            model_name=model_name,
            weight_clip_range=weight_clip_range,
            distillation_loss=nn.MSELoss(reduction="none"),
        )
        self.criterion = nn.MSELoss(reduction="none")
        self.ignore_index = ignore_index
        self.use_log_scale = use_log_scale
        self.plot_reg_results = plot_reg_results
        self.include_ee = include_ee
        self.log_scaler = LogScaler()

        # Initialize streaming metrics
        self.train_metrics = RunningRegressionMetrics(include_ee=include_ee)
        self.val_metrics = RunningRegressionMetrics(include_ee=include_ee)
        self.test_metrics = RunningRegressionMetrics(include_ee=include_ee)

        self.plot_outputs: List[np.ndarray] = []
        self.plot_labels: List[np.ndarray] = []

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
    ) -> PrithviRegressionModule:
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
            class_weights (Optional[List[float]]): Class weights.

        Returns:
            PrithviRegressionModule: Initialized teacher model.
        """
        model = PrithviRegressionModule(
            image_size=image_size,
            learning_rate=learning_rate,
            temporal_step=temporal_step,
            ignore_index=ignore_index,
            weight_decay=weight_decay,
            model_name=model_name,
            scheduler=scheduler,
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
            int: Number of classes (1 for regression).
        """
        return 1

    def _compute_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined distillation and standard loss.

        Args:
            student_outputs: Outputs from student model
            teacher_outputs: Outputs from teacher model
            labels: Ground truth labels

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Compute MSE loss for regression

        mse_loss = self.criterion(student_outputs, labels)

        mse_loss = mse_loss.mean()
        distill_loss = self.distillation_loss(student_outputs, teacher_outputs)
        distill_loss = distill_loss.mean()

        # Combine losses
        total_loss = mse_loss + distill_loss

        metrics = {
            "loss": total_loss.item(),
            "mse_loss": mse_loss.item(),
            "distill_loss": distill_loss.item(),
        }

        return total_loss, metrics

    def _shared_step(self, batch: Any, step_type: str) -> torch.Tensor:
        """Shared logic for training, validation and test steps.

        Args:
            batch (Any): Input batch data.
            step_type (str): Type of step ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        mask = labels.ne(self.ignore_index)
        with torch.no_grad():
            teacher_outputs = self.teacher.net(inputs)
        if self.use_log_scale:
            labels = self.log_scaler.transform(labels)
            teacher_outputs = self.log_scaler.transform(teacher_outputs)
            # TODO: Account for teachers trained on log scale. We will need to
            # skip the log scale transformation for the teacher outputs.
        student_outputs = self.net(inputs)
        student_outputs = student_outputs.squeeze(1)[mask]
        teacher_outputs = teacher_outputs.squeeze(1)[mask]
        labels = labels[mask]
        loss, loss_metrics = self._compute_loss(student_outputs, teacher_outputs, labels)

        # Get predictions
        preds = student_outputs.detach()
        if self.use_log_scale:
            preds = self.log_scaler.inverse_transform(preds)
            labels = self.log_scaler.inverse_transform(labels)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        # Update metrics
        metrics = getattr(self, f"{step_type}_metrics")
        metrics.update(labels, preds)

        # Add regression results to plot
        if step_type in ["val", "test"]:
            if self.plot_reg_results:
                self.plot_outputs.append(preds)
                self.plot_labels.append(labels)

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
