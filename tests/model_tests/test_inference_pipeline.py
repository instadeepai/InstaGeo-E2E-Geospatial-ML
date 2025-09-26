"""Very simple and concise tests for inference_pipeline.py."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from instageo.model.inference_pipeline import ChipInferenceConfig, RayEvaluationPipeline


@pytest.fixture
def temp_test_files():
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary test file
        test_file = os.path.join(temp_dir, "test_data.csv")
        with open(test_file, "w") as f:
            f.write("filepath,label\n")
            f.write("/fake/path1.tif,0\n")
            f.write("/fake/path2.tif,1\n")

        # Create temporary checkpoint file
        checkpoint_file = os.path.join(temp_dir, "model.ckpt")
        with open(checkpoint_file, "w") as f:
            f.write("fake checkpoint content")

        yield {
            "root_dir": temp_dir,
            "test_filepath": test_file,
            "checkpoint_path": checkpoint_file,
        }


def test_evaluation_config(temp_test_files):
    """Test ChipInferenceConfig creation with temporary files."""
    config = ChipInferenceConfig(
        root_dir=temp_test_files["root_dir"],
        test_filepath=temp_test_files["test_filepath"],
        checkpoint_path=temp_test_files["checkpoint_path"],
    )
    assert config.root_dir == temp_test_files["root_dir"]
    assert config.test_filepath == temp_test_files["test_filepath"]
    assert config.checkpoint_path == temp_test_files["checkpoint_path"]
    assert config.mode == "chip_inference"
    assert config.train.batch_size == 8  # batch_size is in train component


@patch("instageo.model.inference_pipeline.ray.init")
@patch("instageo.model.inference_pipeline.serve.start")
def test_ray_evaluation_pipeline_init(mock_serve_start, mock_ray_init, temp_test_files):
    """Test RayEvaluationPipeline initialization."""
    config = ChipInferenceConfig(
        root_dir=temp_test_files["root_dir"],
        test_filepath=temp_test_files["test_filepath"],
        checkpoint_path=temp_test_files["checkpoint_path"],
    )

    pipeline = RayEvaluationPipeline(config)

    assert pipeline.config == config
    assert pipeline.ray_initialized is False
    assert pipeline.serve_started is False
    assert pipeline.model_server_handle is None
    assert pipeline.neptune_logger is None
    assert pipeline.test_loader is None


@patch("instageo.model.inference_pipeline.ray.init")
@patch("instageo.model.inference_pipeline.serve.start")
@patch("instageo.model.inference_pipeline.serve.shutdown")
def test_ray_evaluation_pipeline_context_manager(
    mock_shutdown, mock_serve_start, mock_ray_init, temp_test_files
):
    """Test RayEvaluationPipeline as context manager."""
    config = ChipInferenceConfig(
        root_dir=temp_test_files["root_dir"],
        test_filepath=temp_test_files["test_filepath"],
        checkpoint_path=temp_test_files["checkpoint_path"],
    )

    with RayEvaluationPipeline(config) as pipeline:
        assert pipeline.config == config
        assert isinstance(pipeline, RayEvaluationPipeline)


@patch("instageo.model.inference_pipeline.ray.init")
@patch("instageo.model.inference_pipeline.serve.start")
@patch("instageo.model.inference_pipeline.serve.shutdown")
@patch("instageo.model.inference_pipeline.serve.run")
@patch("instageo.model.inference_pipeline.serve.get_deployment_handle")
def test_ray_evaluation_pipeline_evaluate(
    mock_get_handle,
    mock_serve_run,
    mock_shutdown,
    mock_serve_start,
    mock_ray_init,
    temp_test_files,
):
    """Test RayEvaluationPipeline evaluate method."""
    config = ChipInferenceConfig(
        root_dir=temp_test_files["root_dir"],
        test_filepath=temp_test_files["test_filepath"],
        checkpoint_path=temp_test_files["checkpoint_path"],
    )

    pipeline = RayEvaluationPipeline(config)

    # Mock the internal methods
    with patch.object(pipeline, "_setup_data_preprocessing") as mock_data:
        with patch.object(pipeline, "_initialize_ray_and_serve") as mock_ray:
            with patch.object(pipeline, "_deploy_model_server") as mock_deploy:
                with patch.object(pipeline, "_setup_neptune_logger") as mock_neptune:
                    with patch.object(pipeline, "_run_evaluation") as mock_eval:
                        with patch.object(pipeline, "_cleanup") as mock_cleanup:
                            mock_data.return_value = MagicMock()
                            mock_eval.return_value = {"status": "evaluation_completed"}

                            results = pipeline.evaluate()

                            assert results == {"status": "evaluation_completed"}
                            mock_cleanup.assert_called_once()


@patch("instageo.model.inference_pipeline.ray.init")
@patch("instageo.model.inference_pipeline.serve.start")
@patch("instageo.model.inference_pipeline.serve.shutdown")
@patch("instageo.model.inference_pipeline.serve.run")
@patch("instageo.model.inference_pipeline.serve.get_deployment_handle")
def test_ray_evaluation_pipeline_start_evaluation_pipeline(
    mock_get_handle,
    mock_serve_run,
    mock_shutdown,
    mock_serve_start,
    mock_ray_init,
    temp_test_files,
):
    """Test RayEvaluationPipeline start_evaluation_pipeline method."""
    config = ChipInferenceConfig(
        root_dir=temp_test_files["root_dir"],
        test_filepath=temp_test_files["test_filepath"],
        checkpoint_path=temp_test_files["checkpoint_path"],
    )

    pipeline = RayEvaluationPipeline(config)

    # Mock the internal methods
    with patch.object(pipeline, "_setup_data_preprocessing") as mock_data:
        with patch.object(pipeline, "_initialize_ray_and_serve") as mock_ray:
            with patch.object(pipeline, "_deploy_model_server") as mock_deploy:
                with patch.object(pipeline, "_setup_neptune_logger") as mock_neptune:
                    with patch.object(pipeline, "_run_evaluation") as mock_eval:
                        with patch.object(pipeline, "_cleanup") as mock_cleanup:
                            mock_data.return_value = MagicMock()
                            mock_eval.return_value = {"status": "evaluation_completed"}

                            pipeline.start_evaluation_pipeline()

                            mock_data.assert_called_once()
                            mock_ray.assert_called_once()
                            mock_deploy.assert_called_once()
                            mock_neptune.assert_called_once()


@patch("instageo.model.inference_pipeline.ray.init")
@patch("instageo.model.inference_pipeline.serve.start")
@patch("instageo.model.inference_pipeline.serve.shutdown")
def test_ray_evaluation_pipeline_cleanup(
    mock_shutdown, mock_serve_start, mock_ray_init, temp_test_files
):
    """Test RayEvaluationPipeline cleanup method."""
    config = ChipInferenceConfig(
        root_dir=temp_test_files["root_dir"],
        test_filepath=temp_test_files["test_filepath"],
        checkpoint_path=temp_test_files["checkpoint_path"],
    )

    pipeline = RayEvaluationPipeline(config)
    pipeline.serve_started = True
    pipeline.ray_initialized = True

    pipeline._cleanup()

    mock_shutdown.assert_called_once()
    assert pipeline.serve_started is False
    assert pipeline.ray_initialized is False
