from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from instageo.model.factory import create_model
from instageo.model.pipeline_utils import compute_class_weights
from instageo.model.run import check_required_flags, compute_stats, get_device


class MockPrithviSeg(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_classes = kwargs.get("num_classes", 2)
        self.conv = torch.nn.Conv2d(6, self.num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Convert 5D input (B, C, T, H, W) to 4D (B, C, H, W) for the mock conv
        x = x.squeeze(2)  # Remove temporal dimension
        return self.conv(x)


class MockDataset(Dataset):
    def __init__(self, size=10, num_classes=2, temporal_step=1, is_reg_task=False):
        self.size = size
        self.num_classes = num_classes
        self.temporal_step = temporal_step
        self.is_reg_task = is_reg_task
        # Add temporal dimension to data and use 6 channels for spectral bands
        self.data = torch.randn(
            size, 6, temporal_step, 224, 224
        )  # Random images with temporal dimension and 6 spectral bands
        if is_reg_task:
            self.labels = torch.rand(size, 224, 224)  # Random regression values
        else:
            self.labels = torch.randint(0, num_classes, (size, 224, 224))  # Random labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def mock_config():
    return DictConfig(
        {
            "teacher_ckpt_path": None,
            "is_reg_task": False,
            "mode": "train",
            "dataloader": {
                "img_size": 224,
                "temporal_dim": 1,
            },
            "train": {
                "learning_rate": 1e-4,
                "class_weights": [1, 2],
                "ignore_index": -100,
                "weight_decay": 1e-2,
                "scheduler": True,
                "distillation": False,
            },
            "model": {
                "freeze_backbone": True,
                "num_classes": 2,
                "model_name": "prithvi_eo_v1_100",
                "load_pretrained_weights": True,
                "weight_clip_range": None,
                "depth": -1,
                "use_log_scale": False,
                "plot_reg_results": False,
                "include_ee_metric": False,
            },
        }
    )


@pytest.fixture
def mock_reg_config(mock_config):
    mock_config.is_reg_task = True
    mock_config.model.update(
        {
            "num_classes": 1,
            "use_log_scale": False,
            "plot_reg_results": False,
            "include_ee_metric": False,
        }
    )
    return mock_config


@pytest.fixture
def model(mock_config):
    with patch("instageo.model.base.PrithviSeg", MockPrithviSeg):
        model = create_model(mock_config)
        # Patch the optimizers method
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-4}]
        model.optimizers = MagicMock(return_value=mock_optimizer)
        return model


@pytest.fixture
def reg_model(mock_reg_config):
    with patch("instageo.model.base.PrithviSeg", MockPrithviSeg):
        model = create_model(mock_reg_config)
        # Patch the optimizers method
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [{"lr": 1e-4}]
        model.optimizers = MagicMock(return_value=mock_optimizer)
        return model


@pytest.fixture
def mock_dataloader():
    dataset = MockDataset(temporal_step=1)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def mock_reg_dataloader():
    dataset = MockDataset(temporal_step=1, is_reg_task=True)
    return DataLoader(dataset, batch_size=2)


def test_compute_class_weights():
    counts = {0: 100, 1: 50}
    weights = compute_class_weights(counts)
    assert len(weights) == 2
    assert weights[0] < weights[1]  # Class 1 should have higher weight due to fewer samples


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
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.backends.mps.is_available", return_value=False
    ):
        assert get_device() == "gpu"

    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        assert get_device() == "mps"

    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=False
    ):
        assert get_device() == "cpu"


def test_model_forward(model):
    """Test forward pass of segmentation model outputs correct
    shape with class probabilities ."""
    # Add temporal dimension to input and use 6 channels for spectral bands
    x = torch.randn(2, 6, 1, 224, 224)  # batch_size, spectral_bands, temporal, height, width
    output = model(x)
    assert output.shape == (2, 2, 224, 224)  # batch_size, num_classes, height, width


def test_reg_model_forward(reg_model):
    """Test forward pass of regression model outputs correct shape."""
    # Add temporal dimension to input and use 6 channels for spectral bands
    x = torch.randn(2, 6, 1, 224, 224)  # batch_size, spectral_bands, temporal, height, width
    output = reg_model(x)
    assert output.shape == (2, 1, 224, 224)  # batch_size, 1, height, width


def test_training_step(model, mock_dataloader):
    """Test that `training_step` updates confusion matrix and AUC metrics with gradients for segmentation model."""
    # Get a batch
    batch = next(iter(mock_dataloader))
    # Call training step directly
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    # Check that metrics were updated
    assert model.train_metrics.total > 0


def test_reg_training_step(reg_model, mock_reg_dataloader):
    """Test that `training_step` updates metrics and returns loss with gradients for regression model."""
    # Get a batch
    batch = next(iter(mock_reg_dataloader))
    # Call training step directly
    loss = reg_model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    # Check that metrics were updated
    assert reg_model.train_metrics.n > 0


def test_validation_step(model, mock_dataloader):
    """Test that `validation_step` updates confusion matrix and AUC metrics without gradients for segmentation model."""
    # Get a batch
    batch = next(iter(mock_dataloader))
    # Call validation step directly
    loss = model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad
    # Check that metrics were updated
    assert model.val_metrics.total > 0


def test_reg_validation_step(reg_model, mock_reg_dataloader):
    """Test that `validation_step` updates metrics and returns loss without gradients for regression model."""
    # Get a batch
    batch = next(iter(mock_reg_dataloader))
    # Call validation step directly
    loss = reg_model.validation_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad
    # Check that metrics were updated
    assert reg_model.val_metrics.n > 0


def test_test_step(model, mock_dataloader):
    """Test that `test_step` updates confusion matrix and AUC metrics without gradients for segmentation model."""
    # Get a batch
    batch = next(iter(mock_dataloader))
    # Call test step directly
    loss = model.test_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad
    # Check that metrics were updated
    assert model.test_metrics.total > 0
    assert model.test_auc.n_pos.sum() > 0 or model.test_auc.n_neg.sum() > 0


def test_reg_test_step(reg_model, mock_reg_dataloader):
    """Test that `test_step` updates metrics and returns loss without gradients for regression model."""
    # Get a batch
    batch = next(iter(mock_reg_dataloader))
    # Call test step directly
    loss = reg_model.test_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert not loss.requires_grad
    # Check that metrics were updated
    assert reg_model.test_metrics.n > 0


def test_predict_step(model):
    """Test that `predict_step` returns class probabilities with correct shape for segmentation model."""
    # Add temporal dimension to input and use 6 channels for spectral bands
    x = torch.randn(2, 6, 1, 224, 224)  # batch_size, spectral_bands, temporal, height, width
    probabilities = model.predict_step(x)
    assert probabilities.shape == (2, 224, 224)
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)


def test_reg_predict_step(reg_model):
    """Test that `predict_step` returns predictions with correct shape for regression model."""
    # Add temporal dimension to input and use 6 channels for spectral bands
    x = torch.randn(2, 6, 1, 224, 224)  # batch_size, spectral_bands, temporal, height, width
    predictions = reg_model.predict_step(x)
    assert predictions.shape == (2, 224, 224)


def test_configure_optimizers(model):
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


def test_on_train_epoch_end(model, mock_dataloader):
    """Test that `on_train_epoch_end` resets confusion matrix and AUC metrics after processing for segmentation model."""
    # First add some predictions and labels
    batch = next(iter(mock_dataloader))
    model.training_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert model.train_metrics.total > 0
    # Call epoch end
    model.on_train_epoch_end()
    # Verify that metrics are reset
    assert model.train_metrics.total == 0


def test_reg_on_train_epoch_end(reg_model, mock_reg_dataloader):
    """Test that `on_train_epoch_end` resets metrics after processing for regression model."""
    # First add some predictions and labels
    batch = next(iter(mock_reg_dataloader))
    reg_model.training_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert reg_model.train_metrics.n > 0
    # Call epoch end
    reg_model.on_train_epoch_end()
    # Verify that metrics are reset
    assert reg_model.train_metrics.n == 0


def test_on_validation_epoch_end(model, mock_dataloader):
    """Test that `on_validation_epoch_end` resets confusion matrix and AUC metrics after processing for segmentation model."""
    # First add some predictions and labels
    batch = next(iter(mock_dataloader))
    model.validation_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert model.val_metrics.total > 0
    # Call epoch end
    model.on_validation_epoch_end()
    # Verify that metrics are reset
    assert model.val_metrics.total == 0


def test_reg_on_validation_epoch_end(reg_model, mock_reg_dataloader):
    """Test that `on_validation_epoch_end` resets metrics after processing for regression model."""
    # First add some predictions and labels
    batch = next(iter(mock_reg_dataloader))
    reg_model.validation_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert reg_model.val_metrics.n > 0
    # Call epoch end
    reg_model.on_validation_epoch_end()
    # Verify that metrics are reset
    assert reg_model.val_metrics.n == 0


def test_on_test_epoch_end(model, mock_dataloader):
    """Test that `on_test_epoch_end` resets confusion matrix and AUC metrics after processing for segmentation model."""
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


def test_reg_on_test_epoch_end(reg_model, mock_reg_dataloader):
    """Test that `on_test_epoch_end` resets metrics after processing for regression model."""
    # First add some predictions and labels
    batch = next(iter(mock_reg_dataloader))
    reg_model.test_step(batch, 0)
    # Verify that metrics are not empty before epoch end
    assert reg_model.test_metrics.n > 0
    # Call epoch end
    reg_model.on_test_epoch_end()
    # Verify that metrics are reset
    assert reg_model.test_metrics.n == 0
