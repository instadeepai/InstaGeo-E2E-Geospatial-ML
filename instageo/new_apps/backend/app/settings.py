"""Settings module for the InstaGeo backend."""
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the InstaGeo backend."""

    auth0_domain: str = os.getenv("AUTH0_DOMAIN", "")
    api_audience: str = os.getenv("AUTH0_AUDIENCE", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_ttl: int = int(os.getenv("REDIS_TTL", 24 * 3600))  # Default: 1 day


settings = Settings()

if not settings.database_url:
    database_dir = "/app/database"
    os.makedirs(database_dir, exist_ok=True)
    settings.database_url = f"sqlite:///{database_dir}/instageo.db"
else:
    if settings.database_url.startswith("sqlite:///"):
        db_path = settings.database_url.replace("sqlite:///", "")
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
