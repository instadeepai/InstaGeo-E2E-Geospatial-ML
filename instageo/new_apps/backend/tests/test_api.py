import os
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

# Set a temporary directory for tests before importing the app
os.environ["DATA_FOLDER"] = tempfile.gettempdir()

from instageo.model.configs.config_dataclasses import ModelInfo
from instageo.new_apps.backend.app.main import app

client = TestClient(app)


def create_mock_model_info():
    """Create a mock ModelInfo object for testing."""
    return ModelInfo(
        model_key="aod-estimator",
        model_size="tiny",
        model_name="Aerosol Optical Depth Estimation",
        model_short_name="aod-estim",
        model_type="reg",
        data_source="HLS",
        temporal_step=30,
        num_params=4.14,
        chip_size=224,
    )


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_run_model_and_task_status():
    # Mock the job processing functions to return success immediately
    with patch(
        "instageo.new_apps.backend.app.tasks.process_data_extraction_with_task"
    ) as mock_data_process, patch(
        "instageo.new_apps.backend.app.tasks.process_model_prediction_with_task"
    ) as mock_model_process, patch(
        "instageo.new_apps.backend.app.main.ModelRegistry"
    ) as mock_model_registry:
        # Configure mocks to return successful results
        mock_data_process.return_value = {
            "status": "completed",
            "data": "test_processed_data",
            "output_path": "/tmp/test_output",
        }
        mock_model_process.return_value = {
            "status": "completed",
            "predictions": "test_predictions",
            "model_output": "/tmp/test_model_output",
        }

        # Mock ModelRegistry to return valid model info
        mock_registry_instance = mock_model_registry.return_value
        mock_model_info = create_mock_model_info()
        mock_registry_instance.get_model_metadata_for_size.return_value = mock_model_info
        mock_registry_instance.get_available_models.return_value = [mock_model_info]

        # Prepare a valid request matching TaskCreationRequest format
        payload = {
            "bboxes": [
                [10.0, 20.0, 30.0, 40.0],
                [4.0, 2.0, 3.0, 4.0],
            ],
            "model_key": "aod-estimator",
            "model_size": "tiny",
            "date": "2023-01-01",
            "cloud_coverage": 10,
            "temporal_tolerance": 30,
        }
        response = client.post("/api/run-model", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in (
            "data_processing",
            "model_prediction",
            "visualization_preparation",
            "completed",
            "failed",
        )
        assert "task_id" in data
        task_id = data["task_id"]

        # Check task status endpoint
        response = client.get(f"/api/task/{task_id}")
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["task_id"] == task_id
        assert "status" in status_data
        assert "stages" in status_data


def test_queues_status():
    response = client.get("/api/queues/status")
    assert response.status_code == 200
    data = response.json()
    assert "data_processing" in data
    assert "model_prediction" in data
    assert "visualization_preparation" in data


def test_get_all_tasks():
    # Mock the job processing functions
    with patch(
        "instageo.new_apps.backend.app.tasks.process_data_extraction_with_task"
    ) as mock_data_process, patch(
        "instageo.new_apps.backend.app.tasks.process_model_prediction_with_task"
    ) as mock_model_process, patch(
        "instageo.new_apps.backend.app.tasks.process_visualization_preparation_with_task"
    ) as mock_visualization_process, patch(
        "instageo.new_apps.backend.app.main.ModelRegistry"
    ) as mock_model_registry:
        mock_data_process.return_value = {
            "status": "completed",
            "data": "test_processed_data",
        }
        mock_model_process.return_value = {
            "status": "completed",
            "predictions": "test_predictions",
        }
        mock_visualization_process.return_value = {
            "status": "completed",
            "visualization": "test_visualization",
        }

        # Mock ModelRegistry to return valid model info
        mock_registry_instance = mock_model_registry.return_value
        mock_model_info = create_mock_model_info()
        mock_registry_instance.get_model_metadata_for_size.return_value = mock_model_info
        mock_registry_instance.get_available_models.return_value = [mock_model_info]

        # First create a task with correct format
        payload = {
            "bboxes": [
                [10.0, 20.0, 30.0, 40.0],
                [4.0, 2.0, 3.0, 4.0],
            ],
            "model_key": "aod-estimator",
            "model_size": "tiny",
            "date": "2023-01-01",
            "cloud_coverage": 10,
            "temporal_tolerance": 30,
        }
        response = client.post("/api/run-model", json=payload)
        assert response.status_code == 200

        # Now get all tasks
        response = client.get("/api/tasks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Should have at least one task
        assert len(data) >= 1

        # Check task structure
        task = data[0]
        assert "task_id" in task
        assert "status" in task
        assert "created_at" in task
        assert "bboxes" in task
        assert "model_type" in task
        assert "stages" in task


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "redis" in data["components"]
    assert "queues" in data["components"]
    assert "workers" in data["components"]


def test_visualize_task_id_endpoint():
    """Test the /api/visualize/{task_id} endpoint comprehensively."""

    # Test 1: Non-existent task should return 404
    fake_task_id = "non-existent-task-12345"
    response = client.get(f"/api/visualize/{fake_task_id}")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"].lower()

    # Test 2: Invalid task ID format
    invalid_task_id = ""
    response = client.get(f"/api/visualize/{invalid_task_id}")
    assert response.status_code == 404  # FastAPI returns 404 for empty path params

    # Test 3: Task ID with special characters
    special_task_id = "task-with-special-chars-!@#"
    response = client.get(f"/api/visualize/{special_task_id}")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@patch("instageo.new_apps.backend.app.tiler_service.InstaGeoTilerService._get_chips_cog_file")
@patch("instageo.new_apps.backend.app.tiler_service.InstaGeoTilerService._get_predictions_cog_file")
def test_visualize_task_id_with_mocked_cogs(mock_predictions_cog, mock_chips_cog):
    """Test the visualize endpoint with mocked COG files to verify response structure."""
    from pathlib import Path
    from unittest.mock import MagicMock

    # Create mock Path objects
    mock_chips_path = MagicMock(spec=Path)
    mock_chips_path.exists.return_value = True
    mock_chips_path.__str__.return_value = "/tmp/test_chips.tif"
    mock_chips_path.__fspath__.return_value = "/tmp/test_chips.tif"

    mock_predictions_path = MagicMock(spec=Path)
    mock_predictions_path.exists.return_value = True
    mock_predictions_path.__str__.return_value = "/tmp/test_predictions.tif"
    mock_predictions_path.__fspath__.return_value = "/tmp/test_predictions.tif"

    # Configure the mocks
    mock_chips_cog.return_value = mock_chips_path
    mock_predictions_cog.return_value = mock_predictions_path

    # Test with a valid task ID
    task_id = "test-task-with-cogs-123"
    response = client.get(f"/api/visualize/{task_id}")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "task_id" in data
    assert "satellite" in data
    assert "prediction" in data
    assert "status" in data

    assert data["task_id"] == task_id
    assert data["status"] == "ready"

    # Verify satellite data structure
    satellite = data["satellite"]
    assert "tiles_url" in satellite
    assert "tilejson_url" in satellite
    assert "preview_url" in satellite

    # Verify URLs contain correct task ID references
    assert f"{task_id}_chips" in satellite["tiles_url"]
    assert f"{task_id}_chips" in satellite["tilejson_url"]
    assert f"{task_id}_chips" in satellite["preview_url"]

    # Verify prediction data structure
    prediction = data["prediction"]
    assert "tiles_url" in prediction
    assert "tilejson_url" in prediction
    assert "preview_url" in prediction

    # Verify URLs contain correct task ID references
    assert f"{task_id}_predictions" in prediction["tiles_url"]
    assert f"{task_id}_predictions" in prediction["tilejson_url"]
    assert f"{task_id}_predictions" in prediction["preview_url"]
