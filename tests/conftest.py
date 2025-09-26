"""Pytest configuration for InstaGeo tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set testing flag to bypass EarthData authentication
    os.environ["TESTING"] = "true"

    # Set default EarthData credentials for testing
    os.environ["EARTHDATA_USERNAME"] = "test_user"
    os.environ["EARTHDATA_PASSWORD"] = "test_password"

    # Set other required environment variables
    os.environ["VCS_AUTHOR_EMAIL"] = "test@example.com"

    yield

    # Clean up
    os.environ.pop("TESTING", None)
    os.environ.pop("EARTHDATA_USERNAME", None)
    os.environ.pop("EARTHDATA_PASSWORD", None)
    os.environ.pop("VCS_AUTHOR_EMAIL", None)
