import numpy as np
import pytest
import torch

from instageo.model.infer_utils import sliding_window_inference


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
    hls_tile = torch.zeros((2, 18, 448, 448))
    prediction = sliding_window_inference(
        hls_tile, model, window_size=(224, 224), stride=224, batch_size=2, device="cpu"
    )
    assert np.unique(prediction) == 1
