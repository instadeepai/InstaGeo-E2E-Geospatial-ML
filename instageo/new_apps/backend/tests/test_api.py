import os
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

# Set a temporary directory for tests before importing the app
os.environ["DATA_FOLDER"] = tempfile.gettempdir()

from instageo.new_apps.backend.app.main import app

client = TestClient(app)


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
    ) as mock_model_process:
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

        # Prepare a valid request matching TaskCreationRequest format
        payload = {
            "bboxes": [
                [10.0, 20.0, 30.0, 40.0],
                [4.0, 2.0, 3.0, 4.0],
            ],  # List of bounding boxes
            "model_type": "aod",  # Required field
            "date": "2023-01-01",  # Required field
            "chip_size": 256,  # Optional field
            "cloud_coverage": 10,  # Optional field
        }
        response = client.post("/api/run-model", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in (
            "data_processing",
            "model_prediction",
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


def test_get_all_tasks():
    # Mock the job processing functions
    with patch(
        "instageo.new_apps.backend.app.tasks.process_data_extraction_with_task"
    ) as mock_data_process, patch(
        "instageo.new_apps.backend.app.tasks.process_model_prediction_with_task"
    ) as mock_model_process:
        mock_data_process.return_value = {
            "status": "completed",
            "data": "test_processed_data",
        }
        mock_model_process.return_value = {
            "status": "completed",
            "predictions": "test_predictions",
        }

        # First create a task with correct format
        payload = {
            "bboxes": [
                [10.0, 20.0, 30.0, 40.0],
                [4.0, 2.0, 3.0, 4.0],
            ],  # List of bounding boxes
            "model_type": "aod",  # Required field
            "date": "2023-01-01",  # Required field
            "chip_size": 256,  # Optional field
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
        assert "bboxes_count" in task
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
