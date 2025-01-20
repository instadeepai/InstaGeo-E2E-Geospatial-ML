from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from instageo.model.infer_utils import chip_inference, sliding_window_inference


@pytest.fixture
def model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def predict_step(self, input):
            h, w = input.shape[-2:]
            return torch.ones([h, w])

    return Model()


def test_sliding_window_inference(model):
    hls_tile = torch.zeros((2, 18, 672, 672))
    prediction = sliding_window_inference(
        hls_tile, model, window_size=(224, 224), stride=224, batch_size=2, device="cpu"
    )
    assert np.unique(prediction) == 1


@patch("instageo.model.infer_utils.save_prediction")
@patch("instageo.model.infer_utils.rasterio.open")
@patch("instageo.model.infer_utils.ThreadPoolExecutor")
@patch("instageo.model.infer_utils.tqdm", side_effect=lambda x, **kwargs: x)
def test_chip_inference_basic(
    mock_tqdm, mock_executor, mock_rasterio, mock_save_prediction
):
    # Mock model
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock()
    model.return_value = None

    # Mock the model's behavior during inference
    def mock_forward(data):
        return torch.ones(data.size(0), 3, 10, 10)  # Simulate batch of predictions

    model.side_effect = mock_forward

    # Mock DataLoader
    input_data = torch.randn(3, 3, 10, 10)  # Simulate a batch of 3 samples
    file_names = ["file1.tif", "file2.tif", "file3.tif"]
    dataloader = MagicMock(spec=DataLoader)
    dataloader.__iter__.return_value = [((input_data, None), file_names)]

    # Mock rasterio.open
    mock_raster = MagicMock()
    mock_raster.__enter__.return_value.profile = {"count": 3, "dtype": "uint8"}
    mock_rasterio.return_value = mock_raster

    # Mock ThreadPoolExecutor
    mock_executor_instance = MagicMock()
    mock_executor_instance.submit = MagicMock(
        side_effect=lambda func, *args: func(*args)
    )
    mock_executor.return_value.__enter__.return_value = mock_executor_instance

    # Call the function
    output_folder = "mock_output"
    chip_inference(dataloader, output_folder, model, device="cpu", num_workers=3)

    # Assertions
    model.eval.assert_called_once()
    model.to.assert_called_once_with("cpu")
    mock_tqdm.assert_called_once_with(dataloader, desc="Running Inference")

    # Verify rasterio.open was called for each file
    calls = [call(file) for file in file_names]
    mock_rasterio.assert_has_calls(calls, any_order=True)

    # Verify ThreadPoolExecutor.submit was called for each prediction
    assert mock_executor_instance.submit.call_count == len(file_names)

    assert mock_save_prediction.call_count == 3, "save_prediction was not called!"


def test_chip_inference_handles_empty_dataloader():
    dataloader = MagicMock(spec=DataLoader)
    dataloader.__iter__.return_value = iter([])

    model = MagicMock()
    output_folder = "mock_output"

    with patch("instageo.model.infer_utils.save_prediction") as mock_save_prediction:
        chip_inference(dataloader, output_folder, model, device="gpu")
        mock_save_prediction.assert_not_called()
