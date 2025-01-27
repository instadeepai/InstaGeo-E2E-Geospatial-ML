# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Run Module Containing Training, Evaluation and Inference Logic."""

import json
import logging
import os
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from instageo.model.dataloader import (
    InstaGeoDataset,
    process_and_augment,
    process_data,
    process_test,
)
from instageo.model.infer_utils import chip_inference, sliding_window_inference
from instageo.model.model import PrithviSeg

pl.seed_everything(seed=1042, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def check_required_flags(required_flags: List[str], config: DictConfig) -> None:
    """Check if required flags are provided.

    Args:
        required_flags: A list of required command line arguments.

    Raises:
        An exception if at least one of the arguments is not set
    """
    for flag_name in required_flags:
        if getattr(config, flag_name) == "None":
            raise RuntimeError(f"Flag --{flag_name} is required.")


def get_device() -> str:
    """Selects available device."""
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401

        device = "tpu"
        logging.info("TPU is available. Using TPU...")
    except ImportError:
        if torch.cuda.is_available():
            device = "gpu"
            logging.info("GPU is available. Using GPU...")
        else:
            device = "cpu"
            logging.info("Neither GPU nor TPU is available. Using CPU...")
    return device


def eval_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation DataLoader Collate Function.

    Args:
        batch (Tuple[Tensor]): A list of tuples containing features and labels.

    Returns:
        Tuple of (x,y) concatenated into separate tensors
    """
    data = torch.cat([a[0][0] for a in batch], 0)
    labels = torch.cat([a[0][1] for a in batch], 0)
    return data, labels


def infer_collate_fn(batch: tuple[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference DataLoader Collate Function.

    Args:
        batch (Tuple[Tensor]): A list of tuples containing features and labels.

    Returns:
        Tuple of (x,y) concatenated into separate tensors
    """
    data = torch.stack([a[0][0] for a in batch], 0)
    labels = [a[0][1] for a in batch]
    filepaths = [a[1] for a in batch]
    return (data, labels), filepaths


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 1,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    This function is a convenient wrapper around the PyTorch DataLoader class,
    allowing for easy setup of various DataLoader parameters.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn (Optional[Callable]): Merges a list of samples to form a mini-batch.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned
            memory.

    Returns:
        DataLoader: An instance of the PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


class PrithviSegmentationModule(pl.LightningModule):
    """Prithvi Segmentation PyTorch Lightning Module."""

    def __init__(
        self,
        image_size: int = 224,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        num_classes: int = 2,
        temporal_step: int = 1,
        class_weights: List[float] = [1, 2],
        ignore_index: int = -100,
        weight_decay: float = 1e-2,
    ) -> None:
        """Initialization.

        Initialize the PrithviSegmentationModule, a PyTorch Lightning module for image
        segmentation.

        Args:
            image_size (int): Size of input image.
            num_classes (int): Number of classes for segmentation.
            temporal_step (int): Number of temporal steps for multi-temporal input.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Flag to freeze ViT transformer backbone weights.
            class_weights (List[float]): Class weights for mitigating class imbalance.
            ignore_index (int): Class index to ignore during loss computation.
            weight_decay (float): Weight decay for L2 regularization.
        """
        super().__init__()
        self.net = PrithviSeg(
            image_size=image_size,
            num_classes=num_classes,
            temporal_step=temporal_step,
            freeze_backbone=freeze_backbone,
        )
        weight_tensor = torch.tensor(class_weights).float() if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight_tensor
        )
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "train", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "val", loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Args:
            batch (Any): Input batch data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels.long())
        self.log_metrics(outputs, labels, "test", loss)
        return loss

    def predict_step(self, batch: Any) -> torch.Tensor:
        """Perform a prediction step.

        Args:
            batch (Any): Input batch data.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        prediction = self.forward(batch)
        probabilities = torch.nn.functional.softmax(prediction, dim=1)[:, 1, :, :]
        return probabilities

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0
        )
        return [optimizer], [scheduler]

    def log_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
        loss: torch.Tensor,
    ) -> None:
        """Log all metrics for any stage.

        Args:
            predictions(torch.Tensor): Prediction tensor from the model.
            labels(torch.Tensor): Label mask.
            stage (str): One of train, val and test stages.
            loss (torch.Tensor): Loss value.

        Returns:
            None.
        """
        out = self.compute_metrics(predictions, labels)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_aAcc",
            out["acc"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_mIoU",
            out["iou"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for idx, value in enumerate(out["iou_per_class"]):
            self.log(
                f"{stage}_IoU_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        for idx, value in enumerate(out["acc_per_class"]):
            self.log(
                f"{stage}_Acc_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        for idx, value in enumerate(out["precision_per_class"]):
            self.log(
                f"{stage}_Precision_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        for idx, value in enumerate(out["recall_per_class"]):
            self.log(
                f"{stage}_Recall_{idx}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def compute_metrics(
        self, pred_mask: torch.Tensor, gt_mask: torch.Tensor
    ) -> dict[str, List[float]]:
        """Calculate the Intersection over Union (IoU), Accuracy, Precision and Recall metrics.

        Args:
            pred_mask (np.array): Predicted segmentation mask.
            gt_mask (np.array): Ground truth segmentation mask.

        Returns:
            dict: A dictionary containing 'iou', 'overall_accuracy', and
                'accuracy_per_class', 'precision_per_class' and 'recall_per_class'.
        """
        pred_mask = torch.argmax(pred_mask, dim=1)
        no_ignore = gt_mask.ne(self.ignore_index).to(self.device)
        pred_mask = pred_mask.masked_select(no_ignore).cpu().numpy()
        gt_mask = gt_mask.masked_select(no_ignore).cpu().numpy()
        classes = np.unique(np.concatenate((gt_mask, pred_mask)))

        iou_per_class = []
        accuracy_per_class = []
        precision_per_class = []
        recall_per_class = []

        for clas in classes:
            pred_cls = pred_mask == clas
            gt_cls = gt_mask == clas

            intersection = np.logical_and(pred_cls, gt_cls)
            union = np.logical_or(pred_cls, gt_cls)
            true_positive = np.sum(intersection)
            false_positive = np.sum(pred_cls) - true_positive
            false_negative = np.sum(gt_cls) - true_positive

            if np.any(union):
                iou = np.sum(intersection) / np.sum(union)
                iou_per_class.append(iou)

            accuracy = true_positive / np.sum(gt_cls) if np.sum(gt_cls) > 0 else 0
            accuracy_per_class.append(accuracy)

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive) > 0
                else 0
            )
            precision_per_class.append(precision)

            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative) > 0
                else 0
            )
            recall_per_class.append(recall)

        # Overall IoU and accuracy
        mean_iou = np.mean(iou_per_class) if iou_per_class else 0.0
        overall_accuracy = np.sum(pred_mask == gt_mask) / gt_mask.size

        return {
            "iou": mean_iou,
            "acc": overall_accuracy,
            "acc_per_class": accuracy_per_class,
            "iou_per_class": iou_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
        }


def compute_mean_std(data_loader: DataLoader) -> Tuple[List[float], List[float]]:
    """Compute the mean and standard deviation of a dataset.

    Args:
        data_loader (DataLoader): PyTorch DataLoader.

    Returns:
        mean (list): List of means for each channel.
        std (list): List of standard deviations for each channel.
    """
    mean = 0.0
    var = 0.0
    nb_samples = 0

    for data, _ in data_loader:
        # Reshape data to (B, C, T*H*W)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        nb_samples += batch_samples

        # Sum over batch, height and width
        mean += data.mean(2).sum(0)

        var += data.var(2, unbiased=False).sum(0)

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    return mean.tolist(), std.tolist()  # type:ignore


@hydra.main(config_path="configs", version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runner Entry Point.

    Performs training, evaluation or inference/prediction depending on the selected mode.

    Arguments:
        cfg (DictConfig): Dict-like object containing necessary values used to configure runner.

    Returns:
        None.
    """
    log.info(f"Script: {__file__}")
    log.info(f"Imported hydra config:\n{OmegaConf.to_yaml(cfg)}")

    BANDS = cfg.dataloader.bands
    MEAN = cfg.dataloader.mean
    STD = cfg.dataloader.std
    IM_SIZE = cfg.dataloader.img_size
    TEMPORAL_SIZE = cfg.dataloader.temporal_dim

    batch_size = cfg.train.batch_size
    root_dir = cfg.root_dir
    valid_filepath = cfg.valid_filepath
    train_filepath = cfg.train_filepath
    test_filepath = cfg.test_filepath
    checkpoint_path = cfg.checkpoint_path

    if cfg.mode == "stats":
        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=[0] * len(MEAN),
                std=[1] * len(STD),
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        mean, std = compute_mean_std(train_loader)
        print(mean)
        print(std)
        exit(0)

    if cfg.mode == "train":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath"], cfg)
        train_dataset = InstaGeoDataset(
            filename=train_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )

        valid_dataset = InstaGeoDataset(
            filename=valid_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
        )
        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        valid_loader = create_dataloader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        model = PrithviSegmentationModule(
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
        )
        hydra_out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        checkpoint_callback = ModelCheckpoint(
            monitor="val_mIoU",
            dirpath=hydra_out_dir,
            filename="instageo_epoch-{epoch:02d}-val_iou-{val_mIoU:.2f}",
            auto_insert_metric_name=False,
            mode="max",
            save_top_k=3,
        )

        logger = TensorBoardLogger(hydra_out_dir, name="instageo")

        trainer = pl.Trainer(
            accelerator=get_device(),
            max_epochs=cfg.train.num_epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
        )

        # run training and validation
        trainer.fit(model, train_loader, valid_loader)

    elif cfg.mode == "eval":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_test,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                img_size=cfg.test.img_size,
                crop_size=cfg.test.crop_size,
                stride=cfg.test.stride,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, collate_fn=eval_collate_fn
        )
        model = PrithviSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
        )
        trainer = pl.Trainer(accelerator=get_device())
        result = trainer.test(model, dataloaders=test_loader)
        log.info(f"Evaluation results:\n{result}")

    elif cfg.mode == "sliding_inference":
        model = PrithviSegmentationModule.load_from_checkpoint(
            cfg.checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
        )
        model.eval()
        infer_filepath = os.path.join(root_dir, cfg.test_filepath)
        assert (
            os.path.splitext(infer_filepath)[-1] == ".json"
        ), f"Test file path expects a json file but got {infer_filepath}"
        output_dir = os.path.join(root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(infer_filepath)) as json_file:
            hls_dataset = json.load(json_file)
        for key, hls_tile_path in tqdm(
            hls_dataset.items(), desc="Processing HLS Dataset"
        ):
            try:
                hls_tile, _ = process_data(
                    hls_tile_path,
                    None,
                    bands=cfg.dataloader.bands,
                    no_data_value=cfg.dataloader.no_data_value,
                    constant_multiplier=cfg.dataloader.constant_multiplier,
                    mask_cloud=cfg.test.mask_cloud,
                    replace_label=cfg.dataloader.replace_label,
                    reduce_to_zero=cfg.dataloader.reduce_to_zero,
                )
            except rasterio.RasterioIOError:
                continue
            nan_mask = hls_tile == cfg.dataloader.no_data_value
            nan_mask = np.any(nan_mask, axis=0).astype(int)
            hls_tile, _ = process_and_augment(
                hls_tile,
                None,
                mean=cfg.dataloader.mean,
                std=cfg.dataloader.std,
                temporal_size=cfg.dataloader.temporal_dim,
                augment=False,
            )
            prediction = sliding_window_inference(
                hls_tile,
                model,
                window_size=(cfg.test.img_size, cfg.test.img_size),
                stride=cfg.test.stride,
                batch_size=cfg.train.batch_size,
                device=get_device(),
            )
            prediction = np.where(nan_mask == 1, np.nan, prediction)
            prediction_filename = os.path.join(output_dir, f"{key}_prediction.tif")
            with rasterio.open(hls_tile_path["tiles"]["B02_0"]) as src:
                crs = src.crs
                transform = src.transform
            with rasterio.open(
                prediction_filename,
                "w",
                driver="GTiff",
                height=prediction.shape[0],
                width=prediction.shape[1],
                count=1,
                dtype=str(prediction.dtype),
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(prediction, 1)

    # TODO: Add support for chips that are greater than image size used for training
    elif cfg.mode == "chip_inference":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"], cfg)
        output_dir = os.path.join(cfg.output_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=cfg.test.img_size,
                augment=False,
            ),
            bands=BANDS,
            replace_label=cfg.dataloader.replace_label,
            reduce_to_zero=cfg.dataloader.reduce_to_zero,
            no_data_value=cfg.dataloader.no_data_value,
            constant_multiplier=cfg.dataloader.constant_multiplier,
            include_filenames=True,
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=batch_size, collate_fn=infer_collate_fn
        )
        model = PrithviSegmentationModule.load_from_checkpoint(
            checkpoint_path,
            image_size=IM_SIZE,
            learning_rate=cfg.train.learning_rate,
            freeze_backbone=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            temporal_step=cfg.dataloader.temporal_dim,
            class_weights=cfg.train.class_weights,
            ignore_index=cfg.train.ignore_index,
            weight_decay=cfg.train.weight_decay,
        )
        chip_inference(test_loader, output_dir, model, device=get_device())


if __name__ == "__main__":
    main()
