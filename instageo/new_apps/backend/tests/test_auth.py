"""Tests for authentication module."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
import requests
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import after environment variables are set in conftest.py
from instageo.new_apps.backend.app.auth import (
    get_current_user,
    get_jwks,
    get_user_info,
    is_task_owner,
    verify_access_token,
)
from instageo.new_apps.backend.app.models import Base, Task, User

# Test database setup
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    db = TestSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_user_info():
    """Mock user info response from Auth0."""
    return {
        "sub": "auth0|test-user-123",
        "email": "test@example.com",
        "name": "Test User",
    }


# TODO: Add tests for is_task_owner function


class TestGetCurrentUser:
    """Tests for get_current_user function."""

    def test_get_current_user_success(self, db_session, mock_user_info):
        """Test successful user retrieval."""
        claims = {"sub": "auth0|test-user-123", "access_token": "mock-token"}

        with patch("instageo.new_apps.backend.app.auth.get_user_info", return_value=mock_user_info):
            with patch("instageo.new_apps.backend.app.auth.add_user_to_db") as mock_add:
                mock_user = User(sub=claims["sub"], email=mock_user_info["email"])
                mock_add.return_value = mock_user

                result = get_current_user(db=db_session, claims=claims)
                assert result.sub == claims["sub"]
                mock_add.assert_called_once()

    def test_get_current_user_missing_sub(self, db_session):
        """Test user retrieval with missing sub claim."""
        claims = {"access_token": "mock-token"}

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(db=db_session, claims=claims)

        assert exc_info.value.status_code == 401
        assert "Token invalid" in str(exc_info.value.detail)

    def test_get_current_user_no_access_token(self, db_session):
        """Test user retrieval without access token."""
        claims = {"sub": "auth0|test-user-123"}

        with patch("instageo.new_apps.backend.app.auth.add_user_to_db") as mock_add:
            mock_user = User(sub=claims["sub"])
            mock_add.return_value = mock_user

            result = get_current_user(db=db_session, claims=claims)

            assert result.sub == claims["sub"]
            # add_user_to_db should be called with None for user_info
            mock_add.assert_called_once_with(db_session, claims["sub"], None)


class TestGetUserInfo:
    """Tests for get_user_info function."""

    @patch("instageo.new_apps.backend.app.auth.requests.get")
    def test_get_user_info_success(self, mock_get, mock_user_info):
        """Test successful user info retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = mock_user_info
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_user_info("mock-access-token")

        assert result == mock_user_info
        mock_get.assert_called_once()
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer mock-access-token"

    @patch("instageo.new_apps.backend.app.auth.requests.get")
    def test_get_user_info_retry_on_failure(self, mock_get):
        """Test user info retrieval with retry logic."""
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.json.return_value = mock_user_info
        mock_response.raise_for_status = Mock()

        mock_get.side_effect = [
            requests.exceptions.RequestException,
            mock_response,  # Second call succeeds
        ]

        result = get_user_info("mock-access-token")

        assert result == mock_user_info
        assert mock_get.call_count == 2  # Called twice due to retry

    @patch("instageo.new_apps.backend.app.auth.requests.get")
    def test_get_user_info_max_retries_exceeded(self, mock_get, mock_user_info):
        """Test user info retrieval with all retries failing."""
        mock_get.side_effect = requests.exceptions.RequestException

        with pytest.raises(HTTPException) as exc_info:
            get_user_info("mock-access-token")

        assert exc_info.value.status_code == 500
        assert "Failed to fetch user information" in str(exc_info.value.detail)
        assert mock_get.call_count == 3  # Max retries
