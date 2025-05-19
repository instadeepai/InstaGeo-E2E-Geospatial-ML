from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from instageo.model.metrics import RunningAUC, RunningConfusionMatrix
from instageo.model.run import (
    PrithviSegmentationModule,
    check_required_flags,
    compute_class_weights,
    compute_stats,
    get_device,
)


class MockPrithviSeg(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # Convert 5D input (B, C, T, H, W) to 4D (B, C, H, W) for the mock conv
        x = x.squeeze(2)  # Remove temporal dimension
        return self.conv(x)


class MockDataset(Dataset):
    def __init__(self, size=10, num_classes=2, temporal_step=1):
        self.size = size
        self.num_classes = num_classes
        self.temporal_step = temporal_step
        # Add temporal dimension to data and use 6 channels for spectral bands
        self.data = torch.randn(
            size, 6, temporal_step, 224, 224
        )  # Random images with temporal dimension and 6 spectral bands
        self.labels = torch.randint(0, num_classes, (size, 224, 224))  # Random labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def model():
    with patch("instageo.model.run.PrithviSeg", MockPrithviSeg):
        model = PrithviSegmentationModule(
            image_size=224,
            learning_rate=1e-4,
            freeze_backbone=True,
            num_classes=2,
            temporal_step=1,
            class_weights=[1, 2],
            ignore_index=-100,
            weight_decay=1e-2,
        )
        # Patch the optimizers method
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-4}]
        model.optimizers = MagicMock(return_value=mock_optimizer)
        return model


@pytest.fixture
def mock_dataloader():
    dataset = MockDataset(temporal_step=1)
    return DataLoader(dataset, batch_size=2)


def test_compute_class_weights():
    counts = {0: 100, 1: 50}
    weights = compute_class_weights(counts)
    assert len(weights) == 2
    assert (
        weights[0] < weights[1]
    )  # Class 1 should have higher weight due to fewer samples


def test_compute_stats(mock_dataloader):
    mean, std, class_weights = compute_stats(mock_dataloader)
    assert len(mean) == 6  # 6 spectral bands
    assert len(std) == 6
    assert len(class_weights) == 2  # 2 classes


def test_check_required_flags():
    config = DictConfig({"flag1": "value", "flag2": "None"})
    with pytest.raises(RuntimeError):
        check_required_flags(["flag2"], config)


def test_get_device():
    with patch("torch.cuda.is_available", return_value=True):
        assert get_device() == "gpu"

    with patch("torch.cuda.is_available", return_value=False):
        assert get_device() == "cpu"


def test_model_forward(model):
    # Add temporal dimension to input and use 6 channels for spectral bands
    x = torch.randn(
        2, 6, 1, 224, 224
    )  # batch_size, spectral_bands, temporal, height, width
    output = model(x)
    assert output.shape == (2, 2, 224, 224)  # batch_size, num_classes, height, width


def test_training_step(model, mock_dataloader):
    # Get a batch
    batch = next(iter(mock_dataloader))
    # Call training step directly
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    # Check that metrics were updated
    assert model.train_metrics.total > 0
    assert model.train_auc.n_pos.sum() > 0 or model.train_auc.n_neg.sum() > 0


def test_validation_step(model, mock_dataloader):
    # Get a batch
    batch = next(iter(mock_dataloader))
    # Call validation step directly
    loss = model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad
    # Check that metrics were updated
    assert model.val_metrics.total > 0
    assert model.val_auc.n_pos.sum() > 0 or model.val_auc.n_neg.sum() > 0


def test_test_step(model, mock_dataloader):
    # Get a batch
    batch = next(iter(mock_dataloader))
    # Call test step directly
    loss = model.test_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad
    # Check that metrics were updated
    assert model.test_metrics.total > 0
    assert model.test_auc.n_pos.sum() > 0 or model.test_auc.n_neg.sum() > 0


def test_predict_step(model):
    # Add temporal dimension to input and use 6 channels for spectral bands
    x = torch.randn(
        2, 6, 1, 224, 224
    )  # batch_size, spectral_bands, temporal, height, width
    probabilities = model.predict_step(x)
    assert probabilities.shape == (2, 224, 224)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)


def test_compute_metrics(model):
    # Create mock predictions and ground truth
    pred_mask = torch.randn(2, 2, 224, 224)  # 2 classes
    gt_mask = torch.randint(0, 2, (2, 224, 224))

    # Get predictions and probabilities
    preds = torch.argmax(pred_mask, dim=1)
    probs = torch.nn.functional.softmax(pred_mask, dim=1)

    # Flatten tensors
    preds = preds.reshape(-1).cpu().numpy()
    gt_mask = gt_mask.reshape(-1).cpu().numpy()
    probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1)).cpu().numpy()

    # Compute metrics using RunningConfusionMatrix
    cm = RunningConfusionMatrix(num_classes=2)
    cm.update(gt_mask, preds)
    metrics = cm.compute()

    # Compute AUC using RunningAUC
    auc = RunningAUC(num_classes=2)
    auc.update(gt_mask, probs)
    auc_metrics = auc.score()

    # Verify all expected metrics are present
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "jaccard" in metrics
    assert "roc_auc_macro" in auc_metrics


def test_configure_optimizers(model):
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert isinstance(
        schedulers[0], torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    )


def test_on_train_epoch_end(model, mock_dataloader):
    # First add some predictions and labels
    batch = next(iter(mock_dataloader))
    model.training_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert model.train_metrics.total > 0
    assert model.train_auc.n_pos.sum() > 0 or model.train_auc.n_neg.sum() > 0
    # Call epoch end
    model.on_train_epoch_end()
    # Verify that metrics are reset
    assert model.train_metrics.total == 0
    assert model.train_auc.n_pos.sum() == 0 and model.train_auc.n_neg.sum() == 0


def test_on_validation_epoch_end(model, mock_dataloader):
    # First add some predictions and labels
    batch = next(iter(mock_dataloader))
    model.validation_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert model.val_metrics.total > 0
    assert model.val_auc.n_pos.sum() > 0 or model.val_auc.n_neg.sum() > 0
    # Call epoch end
    model.on_validation_epoch_end()
    # Verify that metrics are reset
    assert model.val_metrics.total == 0
    assert model.val_auc.n_pos.sum() == 0 and model.val_auc.n_neg.sum() == 0


def test_on_test_epoch_end(model, mock_dataloader):
    # First add some predictions and labels
    batch = next(iter(mock_dataloader))
    model.test_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert model.test_metrics.total > 0
    assert model.test_auc.n_pos.sum() > 0 or model.test_auc.n_neg.sum() > 0
    # Call epoch end
    model.on_test_epoch_end()
    # Verify that metrics are reset
    assert model.test_metrics.total == 0
    assert model.test_auc.n_pos.sum() == 0 and model.test_auc.n_neg.sum() == 0
