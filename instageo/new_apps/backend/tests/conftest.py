"""Pytest configuration for backend tests.

This file sets up environment variables BEFORE any test modules are imported.
This is critical because settings.py validates DATABASE_URL at module import time.
# """

import os

# Set environment variables BEFORE any imports that might use them
# This happens at pytest collection time, before test execution
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("AUTH0_DOMAIN", "test-domain.auth0.com")
os.environ.setdefault("AUTH0_AUDIENCE", "https://api.instageo.com")
os.environ.setdefault("DATA_FOLDER", "/tmp/test-instageo-data")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_DB", "0")
os.environ.setdefault("REDIS_PASSWORD", "")
os.environ.setdefault("REDIS_TTL", "86400")
