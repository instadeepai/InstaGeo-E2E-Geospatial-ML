"""Very concise tests for pipeline_utils.py."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig

from instageo.model.pipeline_utils import (
    check_required_flags,
    compute_class_weights,
    create_dataloader,
    eval_collate_fn,
    get_device,
    infer_collate_fn,
)


def test_check_required_flags():
    """Test flag validation."""
    config = DictConfig({"flag1": "value1", "flag2": "None"})
    check_required_flags(["flag1"], config)  # Should pass
    with pytest.raises(RuntimeError):
        check_required_flags(["flag2"], config)  # Should fail


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_get_device(mock_mps, mock_cuda):
    """Test device selection."""
    # Test GPU available
    mock_cuda.return_value = True
    mock_mps.return_value = False
    assert get_device() == "gpu"

    # Test MPS available (Apple Silicon)
    mock_cuda.return_value = False
    mock_mps.return_value = True
    assert get_device() == "mps"

    # Test CPU only (neither CUDA nor MPS available)
    mock_cuda.return_value = False
    mock_mps.return_value = False
    assert get_device() == "cpu"


def test_collate_functions():
    """Test collate functions."""
    batch = [((torch.tensor([[1, 2]]), torch.tensor([0])), "file1")]

    data, labels = eval_collate_fn(batch)
    assert data.shape == (1, 2)

    (data, labels), filepaths = infer_collate_fn(batch)
    assert data.shape == (1, 1, 2)
    assert len(labels) == 1
    assert len(filepaths) == 1
    assert filepaths[0] == "file1"


def test_create_dataloader():
    """Test DataLoader creation."""
    dataset = MagicMock()
    dataloader = create_dataloader(dataset, batch_size=4)
    assert dataloader.batch_size == 4


def test_compute_class_weights():
    """Test class weight computation."""
    counts = {0: 100, 1: 50}
    weights = compute_class_weights(counts)
    assert len(weights) == 2
    assert weights[1] > weights[0]  # Less frequent class gets higher weight
