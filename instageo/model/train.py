from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from absl import app, flags
from torch.utils.data import DataLoader, Dataset

from instageo.model.dataloader import InstaGeoDataset, process_and_augment
from instageo.model.model import PrithviSeg

FLAGS = flags.FLAGS

flags.DEFINE_string("root_dir", None, "Root directory of the dataset.")
flags.DEFINE_string("train_filepath", None, "File path for the training data.")
flags.DEFINE_string("valid_filepath", None, "File path for the validation data.")
flags.DEFINE_string("test_filepath", None, "File path for the test data.")
flags.DEFINE_string("checkpoint_path", None, "File path for model checkpoint.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs to train.")
flags.DEFINE_integer("batch_size", 4, "Batch size")
flags.DEFINE_enum(
    "mode", "train", ["train", "eval"], "Select one of training or evaluation mode."
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_required_flags(required_flags: List[str]) -> None:
    """Check if required flags are provided.

    Args:
        required_flags: A list of required command line arguments.

    Raises:
        An exception if at least one of the arguments is not set
    """
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


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


class PrithviSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 2, learning_rate: float = 1e-3) -> None:
        """Initialization.

        Initialize the PrithviSegmentationModel, a PyTorch Lightning module for image
        segmentation.

        Args:
            num_classes (int): Number of classes for segmentation.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.net = PrithviSeg()
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1, 8]).float().to(DEVICE), ignore_index=-1
        )
        self.learning_rate = learning_rate

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
        iou = self.computeIOU(outputs, labels)
        acc = self.computeAccuracy(outputs, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
        iou = self.computeIOU(outputs, labels)
        acc = self.computeAccuracy(outputs, labels)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
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
        iou = self.computeIOU(outputs, labels)
        acc = self.computeAccuracy(outputs, labels)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0
        )
        return [optimizer], [scheduler]

    @staticmethod
    def computeIOU(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Intersection over Union (IoU) metric.

        Args:
            output (torch.Tensor): The output predictions from the model.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The IoU score.
        """
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()

        no_ignore = target.ne(255).to(DEVICE)
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        intersection = torch.sum(output * target)
        union = torch.sum(target) + torch.sum(output) - intersection
        iou = (intersection + 0.0000001) / (union + 0.0000001)

        if iou != iou:
            print("failed, replacing with 0")
            iou = torch.tensor(0).float()

        return iou

    @staticmethod
    def computeAccuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the accuracy metric.

        Args:
            output (torch.Tensor): The output predictions from the model.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The accuracy score.
        """
        output = torch.argmax(output, dim=1).flatten()
        target = target.flatten()

        no_ignore = target.ne(255).to(DEVICE)
        output = output.masked_select(no_ignore)
        target = target.masked_select(no_ignore)
        correct = torch.sum(output.eq(target))

        return correct.float() / len(target)


def main(argv: Any) -> None:
    """Trainer Entry Point"""
    del argv
    # TODO (Ibrahim): Add a class for managing experiment configs
    BANDS = [1, 2, 3, 8, 11, 12]  # Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2
    MEAN = [
        775.2290211032589,
        1080.992780391705,
        1228.5855250417867,
        2497.2022620507532,
        2204.2139147975554,
        1610.8324823273745,
    ]
    STD = [
        1281.526139861424,
        1270.0297974547493,
        1399.4802505642526,
        1368.3446143747644,
        1291.6764008585435,
        1154.505683480695,
    ]
    IM_SIZE = 224
    TEMPORAL_SIZE = 1

    learning_rate = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    root_dir = FLAGS.root_dir
    valid_filepath = FLAGS.valid_filepath
    train_filepath = FLAGS.train_filepath
    test_filepath = FLAGS.test_filepath
    checkpoint_path = FLAGS.checkpoint_path

    if FLAGS.mode == "train":
        check_required_flags(["root_dir", "train_filepath", "valid_filepath"])
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
        )
        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        valid_loader = create_dataloader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        model = PrithviSegmentationModel(learning_rate=learning_rate)
        trainer = pl.Trainer(max_epochs=num_epochs)

        # run training and validation
        trainer.fit(model, train_loader, valid_loader)

    elif FLAGS.mode == "eval":
        check_required_flags(["root_dir", "test_filepath", "checkpoint_path"])
        test_dataset = InstaGeoDataset(
            filename=test_filepath,
            input_root=root_dir,
            preprocess_func=partial(
                process_and_augment,
                mean=MEAN,
                std=STD,
                temporal_size=TEMPORAL_SIZE,
                im_size=IM_SIZE,
            ),
            bands=BANDS,
        )
        test_loader = create_dataloader(test_dataset, batch_size=batch_size)
        model = PrithviSegmentationModel.load_from_checkpoint(checkpoint_path)
        trainer = pl.Trainer()
        trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    app.run(main)
